"""Validación de Pipeline: ajustar solo en entrenamiento."""
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src.modeling import evaluate, train_pipeline, split_data


def test_cv_pipeline(real_heart):
    x_train, x_test, y_train, y_test = split_data(real_heart)
    grid = train_pipeline(
        x_train, y_train, LogisticRegression(max_iter=800), {"clf__C": [0.1, 1]}, cv=3
    )
    assert "preprocessing" in grid.best_estimator_.named_steps
    assert 0 <= grid.best_score_ <= 1
    result = evaluate(grid.best_estimator_, x_test, y_test)
    assert 0 <= result["AUC_test"] <= 1
    assert sum(result[k] for k in ("TP", "TN", "FP", "FN")) == len(y_test)
    assert 0 <= roc_auc_score(y_test, grid.best_estimator_.predict_proba(x_test)[:, 1]) <= 1
