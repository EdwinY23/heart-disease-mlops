"""Integra el ZIP original de los autores sin generar datos artificiales."""
from pathlib import Path
from zipfile import ZipFile
import shutil

archives = sorted(Path(".").glob("Proyecto_MLOps_Yunis_Serrano_SOLO_DATOS_PROFESOR*.zip"))
if len(archives) != 1:
    raise SystemExit(
        "Se esperaba exactamente un ZIP de entrega; encontrados: "
        + ", ".join(str(p) for p in archives)
    )
archive = archives[0]
print(f"Archivo recibido: {archive}")
staging = Path("_project_staging")
if staging.exists():
    shutil.rmtree(staging)
staging.mkdir()

try:
    with ZipFile(archive) as z:
        for item in z.infolist():
            p = Path(item.filename)
            if p.is_absolute() or ".." in p.parts:
                raise ValueError("Ruta insegura en ZIP: " + item.filename)
        z.extractall(staging)

    candidates = [
        p.parent
        for p in staging.rglob("requirements.txt")
        if (p.parent / "src/data.py").exists()
        and (p.parent / "scripts/train.py").exists()
    ]
    if len(candidates) != 1:
        raise ValueError("El ZIP no contiene un único proyecto MLOps reconocible.")
    project = candidates[0]
    copied = 0
    for item in project.rglob("*"):
        if not item.is_file():
            continue
        relative = item.relative_to(project)
        if relative.parts[:2] == (".github", "workflows"):
            continue  # El workflow activo integra CI, pruebas y publicación.
        destination = Path(relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, destination)
        if relative.as_posix() in ("src/site.py", "README.md", "PUBLICAR_GITHUB.md"):
            source = destination.read_text(encoding="utf-8")
            source = source.replace(".github/workflows/ci.yml", ".github/workflows/entrega-mlops.yml")
            source = source.replace(".github/workflows/publicar-informe.yml", ".github/workflows/entrega-mlops.yml")
            destination.write_text(source, encoding="utf-8")
        copied += 1
    if copied < 20:
        raise ValueError("Estructura extraída incompleta.")
    print(f"Archivos integrados: {copied}")
    archive.unlink()
finally:
    shutil.rmtree(staging, ignore_errors=True)
