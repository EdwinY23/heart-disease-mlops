"""Reporte retrospectivo Evidently: solo particiones sin alterar de Kaggle."""
from pathlib import Path
from src.data import load_data
from src.modeling import split_data


def main():
    # Conjunto de referencia y actual provienen de la misma base histórica.
    # NO se modifican valores; NO representa monitoreo de tráfico de producción.
    frame = load_data()
    x_train, x_test, _, _ = split_data(frame)
    from evidently import Report
    from evidently.presets import DataDriftPreset
    report = Report([DataDriftPreset()])
    snapshot = report.run(reference_data=x_train.reset_index(drop=True),
                          current_data=x_test.reset_index(drop=True))
    Path("reportes").mkdir(exist_ok=True)
    snapshot.save_html("reportes/drift_report.html")
    print("Reporte Evidently: entrenamiento vs. prueba, sin modificación de registros")


if __name__ == "__main__":
    main()
