"""Recursive rolling-origin model selection followed by an untouched final gate."""
import logging

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .cleaning import KEYS


NUMERIC = ["lag_1", "lag_3", "lag_season", "rolling_3", "rolling_6", "month", "weekday", "year"]
FEATURES = KEYS + NUMERIC


def features(history, cfg):
    frame = history.sort_values([*KEYS, "sale_date"]).copy()
    groups = frame.groupby(KEYS).units_sold
    for lag, name in ((1, "lag_1"), (3, "lag_3"), (cfg.season, "lag_season")):
        frame[name] = groups.shift(lag)
    for window in (3, 6):
        frame[f"rolling_{window}"] = groups.transform(lambda s: s.shift(1).rolling(window).mean())
    return calendar(frame).dropna(subset=NUMERIC)


def calendar(frame):
    return frame.assign(month=frame.sale_date.dt.month, weekday=frame.sale_date.dt.dayofweek,
                        year=frame.sale_date.dt.year)


def make_model(name, cfg):
    if name == "random_forest":
        estimator = RandomForestRegressor(n_estimators=150, min_samples_leaf=2,
                                          n_jobs=2, random_state=cfg.random_state)
    elif name == "hist_gb":
        estimator = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.05,
                                                   early_stopping=False, random_state=cfg.random_state)
    elif name == "xgboost":
        try:
            from xgboost import XGBRegressor
        except ImportError:
            logging.getLogger(__name__).warning("XGBoost is not installed; skipping candidate")
            return None
        estimator = XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05,
                                 objective="reg:squarederror", n_jobs=2, random_state=cfg.random_state)
    else:
        raise ValueError(f"Unsupported candidate: {name}")
    pre = ColumnTransformer([("ids", OneHotEncoder(handle_unknown="ignore", sparse_output=False), KEYS),
                             ("numeric", "passthrough", NUMERIC)])
    return Pipeline([("pre", pre), ("model", estimator)])


def fit(model, history, cfg):
    data = features(history, cfg)
    if data.empty:
        raise ValueError("Insufficient history for lag features")
    model.fit(data[FEATURES], data.units_sold)
    return model


def forecast(model, history, months, cfg):
    """model=None is a seasonal naive baseline, also forecast recursively."""
    series = {key: group.sort_values("sale_date").units_sold.astype(float).tolist()
              for key, group in history.groupby(KEYS, sort=True)}
    rows = []
    for target in months:
        batch = []
        for key, values in series.items():
            batch.append({"sale_date": target, **dict(zip(KEYS, key)),
                          "lag_1": values[-1], "lag_3": values[-3], "lag_season": values[-cfg.season],
                          "rolling_3": np.mean(values[-3:]), "rolling_6": np.mean(values[-6:])})
        frame = calendar(pd.DataFrame(batch))
        values = frame.lag_season.to_numpy() if model is None else np.maximum(0, model.predict(frame[FEATURES]))
        if not np.isfinite(values).all():
            raise ValueError("Non-finite forecast")
        for row, value in zip(batch, values):
            key = tuple(row[k] for k in KEYS)
            series[key].append(float(value))
            rows.append({"sale_date": target, **dict(zip(KEYS, key)), "predicted_units": float(value)})
    return pd.DataFrame(rows)


def score(actual, prediction):
    merged = actual.merge(prediction, on=["sale_date", *KEYS], how="outer", validate="one_to_one", indicator=True)
    if not merged._merge.eq("both").all():
        raise ValueError("Actual/prediction keys do not match")
    y, p = merged.units_sold.to_numpy(float), merged.predicted_units.to_numpy(float)
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("Non-finite evaluation values")
    error = np.abs(y - p)
    return {"mae": float(error.mean()), "rmse": float(np.sqrt(np.mean((y-p)**2))),
            "wape": float(error.sum()/np.abs(y).sum()) if np.abs(y).sum() else None}


def validate_panel(history, cfg):
    dates = pd.DatetimeIndex(sorted(history.sale_date.unique()))
    required = cfg.season + 6 + cfg.horizon * (cfg.backtest_folds + 1)
    if len(dates) < required:
        raise ValueError(f"Need at least {required} observed periods; got {len(dates)}")
    expected = pd.date_range(dates.min(), dates.max(), freq=cfg.frequency)
    for key, group in history.groupby(KEYS):
        if not pd.DatetimeIndex(group.sale_date.sort_values()).equals(expected):
            raise ValueError(f"{key}: incomplete panel; resolve gaps or active dates before training")
    return dates


def select_model(history, cfg):
    dates = validate_panel(history, cfg)
    # Earlier folds select candidates. The last horizon is used only for the final gate.
    folds = []
    for i in range(cfg.backtest_folds + 1, 1, -1):
        start = len(dates) - i * cfg.horizon
        folds.append((dates[:start], dates[start:start + cfg.horizon]))
    candidates, skipped = {}, []
    for name in cfg.candidate_models:
        fold_results = []
        for train_dates, test_dates in folds:
            model = make_model(name, cfg)
            if model is None:
                break
            train = history.loc[history.sale_date.isin(train_dates)]
            test = history.loc[history.sale_date.isin(test_dates)]
            fit(model, train, cfg)
            fold_results.append({"train_end": train_dates[-1].date().isoformat(),
                                 "test_start": test_dates[0].date().isoformat(),
                                 "test_end": test_dates[-1].date().isoformat(),
                                 "model": score(test, forecast(model, train, test_dates, cfg)),
                                 "baseline": score(test, forecast(None, train, test_dates, cfg))})
        if len(fold_results) == len(folds):
            candidates[name] = fold_results
        else:
            skipped.append(name)
    eligible = {name: rows for name, rows in candidates.items()
                if all(r["model"]["wape"] is not None for r in rows)}
    if not eligible:
        raise ValueError("No evaluable candidate; WAPE is undefined when all actuals are zero")
    winner = min(eligible, key=lambda name: np.mean([r["model"]["wape"] for r in eligible[name]]))
    train = history.loc[history.sale_date < dates[-cfg.horizon]]
    test = history.loc[history.sale_date >= dates[-cfg.horizon]]
    model = fit(make_model(winner, cfg), train, cfg)
    predicted = forecast(model, train, dates[-cfg.horizon:], cfg)
    metrics = score(test, predicted)
    baseline = score(test, forecast(None, train, dates[-cfg.horizon:], cfg))
    passed = (metrics["wape"] is not None and baseline["wape"] is not None
              and metrics["wape"] < baseline["wape"])
    result = {"model_type": winner, "passed": bool(passed), "metrics": metrics,
              "baseline": baseline, "backtests": candidates, "skipped": skipped,
              "test_start": dates[-cfg.horizon].date().isoformat(),
              "test_end": dates[-1].date().isoformat(),
              "evaluation": "Earlier recursive folds select candidate; final untouched horizon gates deployment"}
    # The rejected candidate is never refitted for deployment or saved.
    return (fit(model, history, cfg) if passed else None), result