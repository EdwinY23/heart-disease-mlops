# Publicar el trabajo en GitHub Pages — Edwin Yunis y Jairo Serrano

1. Crear repositorio **PÚBLICO** `heart-disease-mlops` en la cuenta EdwinY23, rama `main`.
2. Subir los archivos de esta carpeta al repositorio, conservando la subcarpeta oculta `.github/workflows`.
3. En **Actions** esperar a que termine el workflow **Entrenar, verificar y publicar informe académico** (incluye entrenamiento del CSV de Kaggle, Evidently y Docker). Si falla, revisar el paso rojo. **No compartir un sitio vacío o sin métricas**.
4. En **Settings → Pages → Build and deployment**, elegir **Deploy from a branch**, rama `gh-pages`, carpeta `/(root)`, guardar.
5. El enlace público, una vez desplegado, será: `https://edwiny23.github.io/heart-disease-mlops/`.
6. Comprobar que todas las secciones Etapa 0 a Etapa 6, la tabla AUC/Accuracy, las imágenes ROC/matriz y el reporte Evidently estén presentes.

**Nota:** el código está en `main`. La rama `gh-pages` la crea GitHub Actions con el HTML generado sin celdas de código. La API y Kubernetes no se ejecutan en GitHub Pages; se incluyen los manifiestos y las pruebas. No se reporta Kubernetes local como ejecutado hasta probar Minikube.
