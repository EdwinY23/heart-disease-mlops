"""Ejecución de entrenamiento y generación de evidencias reales, usando Kaggle."""
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

from src.data import DATASET_ORIGINAL, describe_data, load_data
from src.leakage import compare_leakage
from src.modeling import SEED, fit_all
from src.site import make_site
from src.dashboard_site import render_dashboard


def main():
    for directory in ("artifacts", "reportes", "docs/assets"):
        Path(directory).mkdir(parents=True, exist_ok=True)

    dataset = load_data()
    if len(dataset) != 918:
        raise ValueError("La entrega exige los 918 registros del dataset original de Kaggle.")
    plt.switch_backend("Agg")
    description = describe_data(dataset)
    leakage = compare_leakage(dataset)
    results = fit_all(dataset)
    table = results["ranking"]
    table.to_csv("reportes/ranking_modelos.csv", index=False)
    joblib.dump(results["best_estimator"], "artifacts/model.joblib")
    payload = {
        "autores": ["Edwin Yunis", "Jairo Serrano"],
        "seed": SEED,
        "dataset": DATASET_ORIGINAL,
        "descripcion": description,
        "fuga_demostrativa": leakage,
        "modelo_seleccionado_por_cv": results["best_name"],
        "ranking": table.to_dict(orient="records"),
        "advertencia": (
            "Clasifica presencia de enfermedad cardíaca en un dataset histórico; "
            "no predice falla cardíaca futura ni sustituye evaluación médica."
        ),
    }
    Path("reportes/resultados.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    x_test, y_test = results["x_test"], results["y_test"]
    model = results["best_estimator"]
    # Análisis exploratorio con datos reales, antes de entrenar cualquier modelo.
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    dataset["HeartDisease"].value_counts().sort_index().plot.bar(ax=ax)
    ax.set_title("Distribución de HeartDisease")
    ax.set_xlabel("Clase: 0 = ausencia · 1 = presencia")
    ax.set_ylabel("Registros")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig("docs/assets/distribucion_clases.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for label, subset in dataset.groupby("HeartDisease"):
        ax.hist(subset["Age"], bins=16, alpha=0.55, label=f"HeartDisease={label}")
    ax.set_title("Distribución de edad por clase")
    ax.set_xlabel("Edad (años)")
    ax.set_ylabel("Frecuencia")
    ax.legend()
    fig.tight_layout()
    fig.savefig("docs/assets/edad_clases.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    RocCurveDisplay.from_estimator(model, x_test, y_test, ax=ax)
    ax.set_title("Curva ROC del modelo seleccionado por validación cruzada")
    fig.tight_layout()
    fig.savefig("docs/assets/roc_modelo.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    ConfusionMatrixDisplay.from_estimator(model, x_test, y_test, ax=ax, values_format="d")
    ax.set_title("Matriz de confusión — prueba reservada (20 %)")
    fig.tight_layout()
    fig.savefig("docs/assets/matriz_confusion.png", dpi=150)
    plt.close(fig)
    make_site(payload, table)
    Path('docs/index.html').replace('docs/informe_tecnico.html')
    render_dashboard(dataset, table, results, payload)
    write_academic_report(payload, table)
    print("\nResultados calculados sobre el dataset cargado:")
    print(table[["modelo", "AUC_CV", "AUC_test", "Accuracy_test"]].to_string(index=False))
    print("\nArchivos: artifacts/model.joblib, reportes/, docs/index.html")
    return payload



def write_academic_report(payload, table):
    d = payload["descripcion"]
    leak = payload["fuga_demostrativa"]
    ranking = "\n".join(
        f"| {row.modelo} | {row.AUC_CV:.3f} | {row.AUC_test:.3f} | "
        f"{row.Accuracy_test:.3f} | {row.TP} | {row.FN} |"
        for row in table.itertuples()
    )
    leader = table.iloc[0]
    text = f"""# Informe final — Proyecto Integrador de Machine Learning

**Autores:** Edwin Yunis y Jairo Serrano  
**Asignatura:** Machine Learning  
**Docente:** Dr. Lihki Rubio  
**Programa:** Maestría en Ingeniería Industrial

## 1. Propósito y alcance
Se entrenaron clasificadores binarios para reconocer la variable `HeartDisease` del dataset público *Heart Failure Prediction* publicado por fedesoriano en Kaggle. Es una clasificación retrospectiva de presencia de enfermedad cardíaca, **no** una predicción temporal de insuficiencia cardíaca futura ni un servicio médico validado.

## 2. Dataset y EDA
Se utilizaron **{d['n_registros']}** registros y **{d['n_variables']}** predictores originales; la clase positiva contiene **{d['positivos']}** casos ({d['prevalencia_observada']:.1%}). Se detectaron **{d['duplicados_completos']}** duplicados de fila completa. Además, `RestingBP=0` se observó **{d['restingbp_cero']}** veces y `Cholesterol=0`, **{d['cholesterol_cero']}** veces: ambos se convierten a valores faltantes en los predictores, sin modificar las etiquetas, y su imputación se estima exclusivamente dentro de cada pipeline.

Consultar `docs/assets/distribucion_clases.png` y `docs/assets/edad_clases.png`. La distribución de clases debe considerarse al interpretar Accuracy; el análisis del efecto de edad no implica causalidad.

## 3. Demostración de fuga de información
Con observaciones originales se comparó un flujo **incorrecto**, que estima el preprocesamiento antes de separar entrenamiento y prueba (AUC **{leak['AUC_preprocesamiento_con_fuga']:.3f}**), frente a un flujo **correcto**, que aprende transformaciones solo con entrenamiento (AUC **{leak['AUC_sin_fuga']:.3f}**). La diferencia puede ser pequeña o tener cualquier signo; el defecto metodológico es la información del conjunto de prueba utilizada durante el ajuste. No se crean características ni pacientes adicionales.

## 4. Diseño de entrenamiento
División aleatoria estratificada 80/20 (`SEED=42`), preprocesamiento y codificación dentro de `Pipeline`, y búsqueda de hiperparámetros mediante `GridSearchCV` con cinco particiones estratificadas. El conjunto de prueba permanece apartado hasta terminar la selección por rendimiento promedio de CV.

## 5. Resultados reales

| Clasificador | AUC CV | AUC prueba | Accuracy prueba | TP | FN |
| --- | ---: | ---: | ---: | ---: | ---: |
{ranking}

El clasificador elegido por mejor AUC media de validación cruzada fue **{leader.modelo}**, con AUC CV de **{leader.AUC_CV:.3f}**; su AUC en prueba fue **{leader.AUC_test:.3f}** y Accuracy **{leader.Accuracy_test:.3f}**. Las diferencias entre ambas métricas muestran que exactitud en un umbral dado y capacidad discriminativa son conceptos distintos. El mismo conjunto de prueba se usa para reportar una comparación descriptiva de candidatos, pero **no** para seleccionar el modelo final.

La curva ROC y la matriz de confusión figuran en `docs/assets/roc_modelo.png` y `docs/assets/matriz_confusion.png`.

## 6. Despliegue y aseguramiento
Se implementó FastAPI con validación Pydantic, Docker con dependencias especificadas, manifiestos de Kubernetes para Minikube y workflow GitHub Actions que ejecuta linting y pruebas automatizadas. El despliegue es **preparado para verificación local**, sin afirmar que se haya publicado realmente en un cluster.

## 7. Monitoreo
`scripts/drift.py` compara con Evidently las covariables originales de entrenamiento y prueba, sin alterarlas. Esto permite verificar el procedimiento de comparación retrospectiva, **no** demuestra que se haya observado deriva en un entorno productivo.

## 8. Conclusión y limitaciones
Se ilustra un ciclo coherente de MLOps, desde la auditoría de fugas hasta la preparación de un despliegue reproducible. El desempeño se limita a una partición del conjunto de Kaggle. Para aplicación clínica se requerirían datos externos, representatividad poblacional, calibración probabilística, análisis de sesgos, supervisión médica, gobernanza y validación prospectiva. La puntuación calculada es académica y no debe usarse para diagnosticar pacientes.

## Bibliografía
- fedesoriano (2021). Heart Failure Prediction Dataset. https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction
- Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR*, 12, 2825–2830.
- Documentaciones técnicas de FastAPI, Docker, Kubernetes, GitHub Actions y Evidently.

**Edwin Yunis · Jairo Serrano**
"""
    Path("reportes/INFORME_FINAL.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
