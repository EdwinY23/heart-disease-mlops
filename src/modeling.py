"""Modelos comparables con CV estratificada y preprocesamiento sin fugas."""
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from src.data import CATEGORICAL, NUMERIC, TARGET, prepare_features

SEED = 42


def make_preprocessor():
    num = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("numeric", num, NUMERIC),
        ("categorical", cat, CATEGORICAL),
    ], remainder="drop")


def model_specs():
    return {
        "SVC": (SVC(probability=True, random_state=SEED), {
            "clf__C": [0.1, 1, 10], "clf__gamma": ["scale", 0.01],
        }),
        "LogisticRegression": (
            LogisticRegression(max_iter=2000, random_state=SEED),
            {"clf__C": [0.05, 0.2, 1]},
        ),
        "RandomForest": (
            RandomForestClassifier(random_state=SEED, n_jobs=1),
            {"clf__n_estimators": [100, 180], "clf__max_depth": [5, 9],
             "clf__min_samples_leaf": [2, 4]},
        ),
        "KNN": (
            KNeighborsClassifier(),
            {"clf__n_neighbors": [9, 15, 21],
             "clf__weights": ["uniform", "distance"]},
        ),
        "GradientBoosting": (
            GradientBoostingClassifier(random_state=SEED),
            {"clf__n_estimators": [70, 130], "clf__learning_rate": [0.05, 0.1]},
        ),
    }


def split_data(frame):
    x = prepare_features(frame)
    y = frame[TARGET].astype(int)
    return train_test_split(x, y, test_size=0.20, random_state=SEED, stratify=y)


def train_pipeline(x_train, y_train, model, param_grid, cv=5):
    pipe = Pipeline([("preprocessing", make_preprocessor()), ("clf", model)])
    folds = StratifiedKFold(n_splits=cv, shuffle=True, random_state=SEED)
    grid = GridSearchCV(
        pipe, param_grid=param_grid, scoring="roc_auc", cv=folds,
        n_jobs=2, refit=True, return_train_score=False,
    )
    grid.fit(x_train, y_train)
    return grid


def evaluate(model, x_test, y_test):
    scores = model.predict_proba(x_test)[:, 1]
    preds = (scores >= 0.50).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, preds, labels=[0, 1]).ravel()
    return {
        "AUC_test": float(roc_auc_score(y_test, scores)),
        "Accuracy_test": float(accuracy_score(y_test, preds)),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }


def fit_all(frame):
    """Selecciona el estimador final SOLO por AUC media de validación cruzada."""
    x_train, x_test, y_train, y_test = split_data(frame)
    search_results = {}
    rows = []
    for name, (estimator, params) in model_specs().items():
        print(f"Entrenando: {name}", flush=True)
        grid = train_pipeline(x_train, y_train, estimator, params)
        metrics = evaluate(grid.best_estimator_, x_test, y_test)
        rows.append({
            "modelo": name,
            "AUC_CV": float(grid.best_score_),
            "AUC_test": metrics["AUC_test"],
            "Accuracy_test": metrics["Accuracy_test"],
            "mejores_parametros": str(grid.best_params_),
            **{key: metrics[key] for key in ("TN", "FP", "FN", "TP")},
        })
        search_results[name] = grid
    import pandas as pd
    ranking = pd.DataFrame(rows).sort_values("AUC_CV", ascending=False).reset_index(drop=True)
    selected_name = ranking.loc[0, "modelo"]
    return {
        "ranking": ranking,
        "best_name": selected_name,
        "best_estimator": search_results[selected_name].best_estimator_,
        "x_train": x_train, "x_test": x_test,
        "y_train": y_train, "y_test": y_test,
        "grids": search_results,
    }
