"""Esquema y transformaciones verificadas sobre datos reales."""
import pandas as pd
import pytest
from src.data import prepare_features, validate_schema
from src.modeling import split_data


def test_valid_schema(real_heart):
    assert validate_schema(real_heart)
    assert len(real_heart) == 918
    assert len(split_data(real_heart)[1]) > 0


def test_incorrect_target(real_heart):
    # Se renombra una columna para comprobar validación, no se crean registros.
    wrong = real_heart.rename(columns={"HeartDisease": "target"})
    with pytest.raises(ValueError, match="HeartDisease"):
        validate_schema(wrong)


def test_existing_zeros_treated_as_missing(real_heart):
    x = prepare_features(real_heart)
    for col in ("Cholesterol", "RestingBP"):
        found = real_heart[col].eq(0)
        if found.any():
            assert x.loc[found, col].isna().all()
