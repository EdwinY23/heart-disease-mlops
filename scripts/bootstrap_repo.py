"""Expande únicamente el código del proyecto suministrado por sus autores."""
from pathlib import Path
from zipfile import ZipFile
import shutil

ARCHIVE = Path("Proyecto_MLOps_Yunis_Serrano_SOLO_DATOS_PROFESOR.zip")
if not ARCHIVE.exists():
    raise SystemExit(f"No está el archivo requerido: {ARCHIVE}")
staging = Path("_project_staging")
if staging.exists():
    shutil.rmtree(staging)
staging.mkdir()
try:
    with ZipFile(ARCHIVE) as z:
        for item in z.infolist():
            p = Path(item.filename)
            if item.is_dir():
                continue
            if p.is_absolute() or ".." in p.parts:
                raise ValueError("Ruta insegura en ZIP: " + item.filename)
        z.extractall(staging)
    candidates = [p for p in staging.rglob("requirements.txt")
                  if p.parent.joinpath("src/data.py").exists()
                  and p.parent.joinpath("scripts/train.py").exists()]
    if len(candidates) != 1:
        raise ValueError("El ZIP no contiene un único proyecto MLOps reconocible.")
    project = candidates[0].parent
    copied = 0
    for item in project.rglob("*"):
        if not item.is_file():
            continue
        relative = item.relative_to(project)
        if relative.parts[:2] == (".github", "workflows"):
            continue  # Conserva los workflows preparados en GitHub.
        destination = Path(relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, destination)
        copied += 1
    if copied < 20:
        raise ValueError("La estructura extraída parece incompleta.")
    print(f"Archivos del proyecto integrados: {copied}")
    ARCHIVE.unlink()
finally:
    shutil.rmtree(staging, ignore_errors=True)
