"""Pruebas exclusivamente sobre los datos oficiales descargados de Kaggle."""
import pytest
from src.data import load_data


@pytest.fixture(scope="session")
def real_heart():
    return load_data()
