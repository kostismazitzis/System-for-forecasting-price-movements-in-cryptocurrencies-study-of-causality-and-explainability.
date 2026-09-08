# models/base_model.py

from abc import ABC, abstractmethod
from pathlib import Path
import joblib

from config import settings


class BaseTimeSeriesModel(ABC):
    """
    Abstract βάση για μοντέλα πρόβλεψης.
    Περιλαμβάνει βασικές λειτουργίες:
     - fit()
     - predict()
     - save()
     - load()
    """

    def __init__(self, name: str):
        self.name = name
        self.model = None

    @abstractmethod
    def fit(self, X, y):
        ...

    @abstractmethod
    def predict(self, X):
        ...

    def save(self) -> Path:
        """
        Αποθηκεύει το μοντέλο σε αρχείο .joblib.
        """
        settings.models_dir.mkdir(parents=True, exist_ok=True)
        path = settings.models_dir / f"{self.name}.joblib"
        joblib.dump(self.model, path)
        return path

    def load(self) -> None:
        """
        Φορτώνει μοντέλο από το .joblib αρχείο.
        """
        path = settings.models_dir / f"{self.name}.joblib"
        self.model = joblib.load(path)
