"""Dashboard Dash/Plotly ejecutable: Contexto, EDA y Models.

Uso: python -m scripts.train && python -m dashboard.app
No fabrica datos ni métricas clínicas.
"""
import json
from pathlib import Path
import pandas as pd
from dash import Dash, Input, Output, dcc, html
from dashboard.figures import (
    descriptive_figures, comparison_figure, model_roc_figure, confusion_figure
)
from src.data import describe_data, load_data

ROOT = Path(__file__).resolve().parents[1]
MODEL_JSON = ROOT / "reportes/modelos_dashboard.json"
if not MODEL_JSON.exists():
    raise RuntimeError("Ejecuta antes: python -m scripts.train")
records = json.loads(MODEL_JSON.read_text(encoding="utf-8"))
table = pd.read_csv(ROOT / "reportes/ranking_modelos.csv")
frame = load_data(path=ROOT / "data/heart.csv")
stats = describe_data(frame)
plots = descriptive_figures(frame)
names = {"KNN": "KNN", "RandomForest": "Random Forest",
         "LogisticRegression": "Regresión logística"}
models = list(names)
app = Dash(__name__, assets_folder=str(Path(__file__).parent / "assets"),
           suppress_callback_exceptions=True,
           title="ML Cardiovascular | Edwin Yunis y Jairo Serrano")
server = app.server


def card(*items):
    return html.Div(list(items), className="card")


def figure(fig):
    return card(dcc.Graph(figure=fig, config={"displayModeBar": False},
                          responsive=True))


def kpi(value, label):
    return html.Div([html.B(str(value)), html.Span(label)], className="kpi")


def contexto():
    return html.Div([
        html.H2("Contexto del proyecto"),
        card(
            html.H3("Problema y alcance"),
            html.P("Se clasifica HeartDisease (0: ausencia, 1: presencia) "
                   "con el dataset Heart Failure Prediction de Kaggle, "
                   "indicado por el profesor. Es una evaluación retrospectiva "
                   "y no una predicción de eventos cardíacos futuros."),
            html.Span("Kaggle · fedesoriano", className="mini-pill"),
            html.Span("Clasificación binaria", className="mini-pill"),
            html.Span("Datos originales", className="mini-pill"),
        ),
        html.Div([
            kpi(stats["n_registros"], "Observaciones"),
            kpi(stats["n_variables"], "Predictores"),
            kpi(f'{stats["prevalencia_observada"]:.1%}', "Clase positiva"),
            kpi(stats["duplicados_completos"], "Duplicados completos")
        ], className="kpi-grid"),
        card(
            html.H3("Ciclo de trabajo MLOps"),
            html.Div([
                html.Article([html.B("01 · EDA"),
                              html.P("Calidad y distribuciones clínicas.")]),
                html.Article([html.B("02 · Modelos"),
                              html.P("Pipeline, GridSearchCV y prueba reservada.")]),
                html.Article([html.B("03 · Despliegue"),
                              html.P("FastAPI, Docker, Kubernetes, CI/CD y Evidently.")])
            ], className="flow"),
            html.P("Uso académico únicamente; no constituye un diagnóstico.",
                   className="warning")
        ),
    ])


def eda():
    return html.Div([
        html.H2("Exploración descriptiva (EDA)"),
        html.P("Todas las figuras emplean observaciones originales. Los valores "
               "Cholesterol=0 se omiten únicamente en el diagrama de cajas.",
               className="description"),
        html.Div([
            kpi(int((frame["HeartDisease"] == 0).sum()), "HeartDisease = 0"),
            kpi(int((frame["HeartDisease"] == 1).sum()), "HeartDisease = 1"),
            kpi(stats["restingbp_cero"], "RestingBP = 0"),
            kpi(stats["cholesterol_cero"], "Cholesterol = 0"),
        ], className="kpi-grid"),
        html.Div([figure(plots[k]) for k in ("clases", "edad", "colesterol", "dolor")],
                 className="grid-2"),
        figure(plots["correlaciones"]),
        card(html.P("Las relaciones observadas son descriptivas, no causales."))
    ])


def modelos():
    focused = table[table["modelo"].isin(models)]
    rows = [html.Tr([
        html.Td(names[row.modelo]),
        html.Td(f"{row.AUC_CV:.3f}"),
        html.Td(f"{row.AUC_test:.3f}"),
        html.Td(f"{row.Accuracy_test:.3f}"),
        html.Td(f'{records[row.modelo]["auc_entrenamiento"] - row.AUC_CV:+.3f}')
    ]) for row in focused.itertuples()]
    return html.Div([
        html.H2("Modelos de Machine Learning"),
        card(html.H3("Entrenamiento seguro"),
             html.P("Separación estratificada 80/20 antes de preprocesar. "
                    "Pipeline con imputación y escalado internos; "
                    "GridSearchCV con cinco pliegues. Selección por AUC CV "
                    "sin consultar la prueba reservada."),
             html.P("Se limita la complejidad de Random Forest, se ajustan "
                    "vecinos en KNN y se regulariza la regresión logística. "
                    "No es posible garantizar ausencia total de sobreajuste.",
                    className="warning")),
        figure(comparison_figure(table)),
        card(html.H3("Comparación de modelos"),
             html.Div(html.Table([
                 html.Thead(html.Tr([html.Th(x) for x in (
                     "Clasificador", "AUC CV", "AUC prueba",
                     "Accuracy", "Brecha train−CV")])),
                 html.Tbody(rows)]), className="table-scroller")),
        card(html.Label("Selecciona un modelo"),
             dcc.Dropdown(id="model-selector", value="RandomForest",
                          clearable=False,
                          options=[{"label": names[n], "value": n} for n in models]),
             html.Div([
                 dcc.Graph(id="roc", config={"displayModeBar": False}),
                 dcc.Graph(id="confusion", config={"displayModeBar": False})
             ], className="grid-2"),
             html.P("Las métricas son retrospectivas; la brecha entre "
                    "AUC de entrenamiento y CV alerta sobre sobreajuste.",
                    className="description")),
    ])


app.layout = html.Div([
    html.Aside([
        html.Div([
            html.Div([html.Span("♡", className="logo-pulse"),
                      html.Span("CARDIO · ML")], className="logo"),
            html.P("Machine Learning · MLOps", className="side-label"),
            html.P("Autores: Edwin Yunis y Jairo Serrano", className="side-meta"),
        ]),
        html.P("Fuente: Kaggle · Heart Failure Prediction", className="side-footer")
    ], className="sidebar"),
    html.Main([
        html.Header([
            html.Div([
                html.P("PROYECTO INTEGRADOR", className="eyebrow"),
                html.H1("Enfermedad cardíaca: análisis y modelos"),
                html.P("Dashboard analítico con tres vistas y "
                       "validación metodológica reproducible."),
                html.Strong("Edwin Yunis · Jairo Serrano")
            ]),
            html.Img(src="/assets/heart-hero.svg",
                     alt="Ilustración cardiovascular y electrocardiograma")
        ], className="hero"),
        dcc.Tabs(id="tabs", value="contexto", children=[
            dcc.Tab(label="01 · Contexto", value="contexto",
                    className="dash-tab", selected_className="dash-tab--selected"),
            dcc.Tab(label="02 · EDA", value="eda",
                    className="dash-tab", selected_className="dash-tab--selected"),
            dcc.Tab(label="03 · Models", value="models",
                    className="dash-tab", selected_className="dash-tab--selected")
        ]),
        html.Div(id="main-content"),
        html.Footer("Edwin Yunis · Jairo Serrano · Machine Learning")
    ], className="content")
], className="page")


@app.callback(Output("main-content", "children"), Input("tabs", "value"))
def navigate(value):
    return {"contexto": contexto, "eda": eda, "models": modelos}[value]()


@app.callback(Output("roc", "figure"), Output("confusion", "figure"),
              Input("model-selector", "value"))
def model_details(selected):
    return model_roc_figure(records[selected]), confusion_figure(records[selected])


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8050)
