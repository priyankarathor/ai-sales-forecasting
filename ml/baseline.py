class SevenPeriodBaseline:
   

    def fit(self, features, target):
        return self

    def predict(self, features):
        return features["lag_7"].to_numpy(dtype=float)
