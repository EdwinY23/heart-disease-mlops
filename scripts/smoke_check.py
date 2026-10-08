"""Comprueba integridad básica sin descargar datos ni ejecutar modelos reales."""
from pathlib import Path

import nbformat

required = [
    "README.md", "src/data.py", "src/modeling.py", "src/leakage.py",
    "app/api.py", "docker/Dockerfile", "docker/requirements.txt",
    "k8s/deployment.yaml", "k8s/service.yaml", ".github/workflows/ci.yml",
    "scripts/train.py", "scripts/drift.py",
    "notebooks/1_model_leakage_demo.ipynb",
    "notebooks/2_model_pipeline_cv.ipynb",
]
for name in required:
    assert Path(name).is_file(), f"Falta: {name}"
for name in ("notebooks/1_model_leakage_demo.ipynb", "notebooks/2_model_pipeline_cv.ipynb"):
    nbformat.validate(nbformat.read(name, as_version=4))
print("Estructura y notebooks validados.")
