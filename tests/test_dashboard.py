"""Validaciones del nuevo dashboard sobre el dataset original de Kaggle."""
from pathlib import Path

from dashboard.figures import descriptive_figures
from src.modeling import model_specs


def test_requested_models_present():
    specs = model_specs()
    assert all(n in specs for n in ("KNN", "RandomForest", "LogisticRegression"))
    forest = specs["RandomForest"][1]
    assert all(depth is not None for depth in forest["clf__max_depth"])
    assert min(forest["clf__min_samples_leaf"]) >= 2
    assert min(specs["KNN"][1]["clf__n_neighbors"]) >= 5


def test_eda_figures_use_real_dataset(real_heart):
    figures = descriptive_figures(real_heart)
    assert set(figures) == {
        "clases", "edad", "colesterol", "dolor", "correlaciones"
    }
    assert all(len(fig.data) > 0 for fig in figures.values())


def test_dash_structure_and_illustration_exist():
    for name in ("dashboard/app.py", "dashboard/assets/style.css",
                 "dashboard/assets/heart-hero.svg", "src/dashboard_site.py"):
        assert Path(name).is_file(), name
    source = Path("dashboard/app.py").read_text(encoding="utf-8")
    assert all(tab in source for tab in ('value="contexto"', 'value="eda"',
                                        'value="models"'))
