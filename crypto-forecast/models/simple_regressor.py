# models/simple_regressor.py

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from .base_model import BaseTimeSeriesModel


class RandomForestPriceModel(BaseTimeSeriesModel):
    """
    Απλό μοντέλο RandomForest για πρόβλεψη της τιμής της επόμενης ημέρας.
    """

    def __init__(self, name: str = "rf_price_model"):
        super().__init__(name)
        self.model = RandomForestRegressor(
            n_estimators=200,
            random_state=42,
        )

    def fit(self, X, y):
        """
        Εκπαίδευση μοντέλου με features X και target y.
        """
        self.model.fit(np.asarray(X), np.asarray(y))

    def predict(self, X):
        """
        Επιστρέφει την προβλεπόμενη τιμή.
        """
        return self.model.predict(np.asarray(X))
