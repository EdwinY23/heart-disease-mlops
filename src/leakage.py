"""Demostración de fuga de preprocesamiento con observaciones reales únicamente."""
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from src.data import TARGET, prepare_features
from src.modeling import SEED, make_preprocessor


def compare_leakage(frame):
    x = prepare_features(frame)
    y = frame[TARGET].astype(int)
    tr, te = train_test_split(np.arange(len(frame)), test_size=0.2,
                              stratify=y, random_state=SEED)
    # INCORRECTO: aprender escalas, imputaciones y categorías usando también el test.
    preprocessing_on_all = make_preprocessor()
    transformed_full = preprocessing_on_all.fit_transform(x)
    flawed = SVC(C=1, probability=True, random_state=SEED)
    flawed.fit(transformed_full[tr], y.iloc[tr])
    auc_flawed = roc_auc_score(y.iloc[te], flawed.predict_proba(transformed_full[te])[:, 1])

    # CORRECTO: el preprocesamiento se aprende solo en entrenamiento.
    valid = Pipeline([("preprocessing", make_preprocessor()),
                      ("clf", SVC(C=1, probability=True, random_state=SEED))])
    valid.fit(x.iloc[tr], y.iloc[tr])
    auc_valid = roc_auc_score(y.iloc[te], valid.predict_proba(x.iloc[te])[:, 1])
    return {"AUC_preprocesamiento_con_fuga": float(auc_flawed),
            "AUC_sin_fuga": float(auc_valid)}
