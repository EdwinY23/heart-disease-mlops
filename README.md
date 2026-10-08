# Proyecto Integrador de Aprendizaje Automático — MLOps

**Autores:** Edwin Yunis y Jairo Serrano  
**Asignatura:** Machine Learning · **Docente:** Dr. Lihki Rubio  
**Programa:** Maestría en Ingeniería Industrial

## Fuente única de datos

**[Heart Failure Prediction — Kaggle (fedesoriano)](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)**. Solo se utiliza el `heart.csv` original (918 registros, 11 variables predictoras y variable objetivo `HeartDisease`). `src/data.py` intenta descargar el ZIP mediante la API oficial de Kaggle. En caso de fallo, descarga manualmente el CSV desde el enlace del profesor y guárdalo en `data/heart.csv`. **No existe ruta de reemplazo por otra base, registros fabricados ni espejo externo.**

> El objetivo `HeartDisease` corresponde a presencia/ausencia de enfermedad cardíaca registrada. No está validado como pronóstico prospectivo de insuficiencia cardíaca ni herramienta clínica.

## Etapas de la guía

| Etapa | Archivo/evidencia |
| --- | --- |
| 0. Estructura | `src/`, `app/`, `docker/`, `k8s/`, `notebooks/`, `tests/` y `.github/workflows/` |
| 1. EDA y data leakage | `notebooks/1_model_leakage_demo.ipynb`, `src/leakage.py` |
| 2. Pipeline + GridSearchCV | `notebooks/2_model_pipeline_cv.ipynb`, `src/modeling.py` |
| 3. FastAPI + Docker | `app/api.py`, `docker/Dockerfile` |
| 4. Kubernetes | `k8s/deployment.yaml`, `k8s/service.yaml` |
| 5. CI/CD | `.github/workflows/entrega-mlops.yml`, pruebas automatizadas |
| 6. Evidently | `scripts/drift.py`, `reportes/drift_report.html` tras ejecución |

**Decisión de la entrega:** en el ejemplo docente aparece una variable fabricada (`leaky_feature`) para ilustrar fuga de datos. Por indicación de los autores de utilizar exclusivamente registros originales, ese fragmento **no se reproduce**. En su lugar se demuestra con el **mismo dataset real** la fuga causada por ajustar el preprocesamiento antes de dividir entrenamiento/prueba, otra práctica incorrecta que aparece en el ejemplo compartido. La diferencia de AUC no tiene por qué ser positiva.

El reporte Evidently compara sin alteraciones el subconjunto de entrenamiento con el subconjunto de prueba. **No se fabrican registros, no se simula cambio de edades y no se afirma monitoreo productivo.**

## Dashboard de tres vistas: Dash + Plotly

**Contexto:** presenta el problema clínico, objetivo binario, procedencia del dataset, 918 registros, variables y las etapas MLOps.

**EDA:** presenta gráficas interactivas sobre el CSV original: balance de clases, distribución por edad, colesterol válido por clase, dolor torácico y matriz de correlaciones; documenta las observaciones de ceros inverosímiles.

**Models:** compara **KNN, Random Forest y regresión logística**, usando la misma división de prueba que los demás clasificadores. La selección de hiperparámetros utiliza `Pipeline` y `GridSearchCV` estratificada con 5 folds. Se restringen parámetros de complejidad para **reducir**, sin prometer eliminar, sobreajuste. Se reportan AUC CV, AUC prueba, Accuracy, brecha AUC entrenamiento−CV, curvas ROC y matrices de confusión de cada candidato.

Para ejecutar la aplicación real Dash (servidor interactivo):

```bash
python -m pip install -r requirements.txt
python -m scripts.train
python -m dashboard.app
```

Abrir **http://localhost:8050**. La API FastAPI se sirve por separado en el puerto 8000.

**Versión de GitHub Pages:** `https://edwiny23.github.io/heart-disease-mlops/`. Es una presentación HTML interactiva de Plotly con las mismas tres ventanas, **no un servidor Dash**. GitHub Pages solo sirve contenido estático, mientras la aplicación Dash real se ejecuta localmente. El entrenamiento genera `reportes/modelos_dashboard.json` y `docs/index.html` con resultados medidos. La imagen cardiovascular del encabezado es una ilustración, no un registro médico.

## Ejecutar el proyecto

```bash
python -m pip install -r requirements.txt
python -m scripts.train
pytest -q
python -m pip install -r requirements-monitor.txt
python -m scripts.drift
uvicorn app.api:app --host 127.0.0.1 --port 8000
```

`python -m scripts.train` crea `artifacts/model.joblib`, `reportes/resultados.json`, `reportes/ranking_modelos.csv`, las figuras y `docs/index.html` con **métricas obtenidas del dataset original**. La evaluación usa división estratificada 80/20, 5 folds de validación cruzada, `Pipeline` y `GridSearchCV`. Compara SVC, Logistic Regression, Random Forest, KNN y Gradient Boosting. La selección es por AUC promedio en CV y la prueba reservada muestra AUC, Accuracy, ROC y matriz de confusión.

### Docker y Kubernetes

```bash
docker build -t heart-mlops:local -f docker/Dockerfile .
docker run --rm -p 8000:8000 heart-mlops:local
minikube start
minikube image load heart-mlops:local
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl port-forward service/heart-service 8000:80
```

La prueba HTTP automatizada utiliza una fila real del `heart.csv`, no un paciente inventado. Docker y Kubernetes se incluyen como infraestructura a verificar en el entorno que ejecute el proyecto; GitHub Pages no ejecuta FastAPI.

### GitHub Pages

El workflow `.github/workflows/entrega-mlops.yml` obtiene el dataset original, entrena, crea el reporte, comprueba Docker y publica el informe HTML en `gh-pages`. Consulta `PUBLICAR_GITHUB.md`. No compartas una página sin salidas verificadas.

### Limitaciones

Todos los resultados son retrospectivos sobre la misma fuente. No se dispone de muestras clínicas posteriores, eventos observados de un modelo ya desplegado ni datos reales de producción. Evidently compara particiones originales; no puede certificarse una deriva productiva. No usar la API en decisiones médicas.

**Edwin Yunis · Jairo Serrano**
