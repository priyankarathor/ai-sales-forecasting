class SevenPeriodBaseline:
    """Predict the observed units from seven observation periods earlier."""

    def fit(self, features, target):
        return self

    def predict(self, features):
        return features["lag_7"].to_numpy(dtype=float)
