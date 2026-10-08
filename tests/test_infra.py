"""Comprobación estática mínima de entregables de infraestructura."""
from pathlib import Path

import yaml


def test_kubernetes_manifests():
    deployment = yaml.safe_load(Path("k8s/deployment.yaml").read_text())
    service = yaml.safe_load(Path("k8s/service.yaml").read_text())
    assert deployment["kind"] == "Deployment"
    assert service["kind"] == "Service"
    assert deployment["spec"]["template"]["spec"]["containers"][0]["readinessProbe"]
    assert service["spec"]["selector"]["app"] == "heart-model"


def test_workflows_and_docker_files_exist():
    for name in ["docker/Dockerfile", ".github/workflows/entrega-mlops.yml",
                 "scripts/drift.py"]:
        assert Path(name).is_file(), name
