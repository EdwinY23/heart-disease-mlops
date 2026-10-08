"""Gráficas interactivas del dataset original Heart Failure Prediction (Kaggle).

No genera pacientes ni variables clínicas artificiales.
"""
import plotly.express as px
import plotly.graph_objects as go

COLORS = {"Sin enfermedad": "#34b3d1", "Con enfermedad": "#ed6386"}
BASE = dict(
    template="plotly_white",
    paper_bgcolor="#ffffff",
    plot_bgcolor="#ffffff",
    font=dict(family="Arial, sans-serif", color="#244054", size=12),
    margin=dict(l=30, r=22, t=58, b=32),
    legend=dict(orientation="h", y=-0.19, x=0),
)


def style(fig, title, height=340):
    fig.update_layout(**BASE, title=dict(text=title, x=0.03, font=dict(size=16)),
                      height=height)
    return fig


def descriptive_figures(frame):
    data = frame.copy()
    data["Estado"] = data["HeartDisease"].map({
        0: "Sin enfermedad", 1: "Con enfermedad"
    })
    counts = data["Estado"].value_counts().reindex(list(COLORS))
    fig1 = px.pie(
        names=counts.index, values=counts.values, hole=0.65,
        color=counts.index, color_discrete_map=COLORS
    )
    fig1.update_traces(textinfo="percent+label", textposition="outside",
                       hovertemplate="%{label}: %{value} casos (%{percent})<extra></extra>")
    style(fig1, "Distribución de la clase objetivo")

    fig2 = px.histogram(
        data, x="Age", color="Estado", nbins=22,
        barmode="overlay", opacity=0.75,
        color_discrete_map=COLORS,
        labels={"Age": "Edad (años)", "count": "Frecuencia"}
    )
    style(fig2, "Edad de los pacientes según clase")
    fig2.update_layout(bargap=0.07)
    fig2.update_yaxes(title_text="Pacientes")

    cholesterol = data[data["Cholesterol"] > 0]
    fig3 = px.box(
        cholesterol, x="Estado", y="Cholesterol", color="Estado",
        color_discrete_map=COLORS,
        labels={"Cholesterol": "Colesterol (mg/dL)"}, points=False
    )
    style(fig3, "Colesterol observado por clase (excluye ceros)")
    fig3.update_layout(showlegend=False, xaxis_title=None)

    fig4 = px.histogram(
        data, x="ChestPainType", color="Estado", barmode="group",
        color_discrete_map=COLORS,
        category_orders={"ChestPainType": ["ATA", "NAP", "ASY", "TA"]},
        labels={"ChestPainType": "Tipo de dolor torácico"}
    )
    style(fig4, "Tipo de dolor torácico y clase observada")
    fig4.update_yaxes(title_text="Pacientes")

    numeric = ["Age", "RestingBP", "Cholesterol", "FastingBS",
               "MaxHR", "Oldpeak", "HeartDisease"]
    corr = data[numeric].replace({"RestingBP": {0: None},
                                  "Cholesterol": {0: None}}).corr()
    fig5 = go.Figure(go.Heatmap(
        z=corr.to_numpy(), x=numeric, y=numeric,
        zmin=-1, zmax=1, colorscale="RdBu", reversescale=True,
        text=corr.round(2).to_numpy(), texttemplate="%{text}",
        hovertemplate="%{x} × %{y}: %{z:.2f}<extra></extra>"
    ))
    style(fig5, "Correlaciones entre variables numéricas", height=420)
    return {"clases": fig1, "edad": fig2, "colesterol": fig3,
            "dolor": fig4, "correlaciones": fig5}


def model_roc_figure(details):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines", name="Referencia",
        line=dict(color="#96a8b4", dash="dash")
    ))
    fig.add_trace(go.Scatter(
        x=details["fpr"], y=details["tpr"], mode="lines",
        name=f"AUC = {details['auc_test']:.3f}",
        line=dict(color="#21a8ca", width=3),
        fill="tozeroy", fillcolor="rgba(33,168,202,.08)"
    ))
    style(fig, "Curva ROC — prueba reservada", height=380)
    fig.update_layout(xaxis_title="Tasa de falsos positivos",
                      yaxis_title="Tasa de verdaderos positivos",
                      xaxis_range=[0, 1], yaxis_range=[0, 1.02])
    return fig


def confusion_figure(details):
    matrix = details["confusion"]
    fig = go.Figure(go.Heatmap(
        z=matrix, x=["Predicho 0", "Predicho 1"],
        y=["Real 0", "Real 1"], colorscale=[[0, "#e9f6fa"], [1, "#1a739a"]],
        showscale=False, text=matrix, texttemplate="%{text}",
        hovertemplate="%{y} / %{x}: %{z}<extra></extra>"
    ))
    style(fig, "Matriz de confusión — prueba reservada", height=380)
    fig.update_yaxes(autorange="reversed")
    return fig


def comparison_figure(ranking):
    selected = ranking[ranking["modelo"].isin(
        ["KNN", "RandomForest", "LogisticRegression"]
    )].copy()
    names = {"KNN": "KNN", "RandomForest": "Random Forest",
             "LogisticRegression": "Regresión logística"}
    selected["nombre"] = selected["modelo"].map(names)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=selected["nombre"], y=selected["AUC_CV"],
        name="AUC validación cruzada", marker_color="#29a7ce"
    ))
    fig.add_trace(go.Bar(
        x=selected["nombre"], y=selected["AUC_test"],
        name="AUC prueba final", marker_color="#ef829c"
    ))
    style(fig, "Comparación de los tres clasificadores", height=360)
    fig.update_layout(barmode="group", yaxis_range=[0, 1],
                      yaxis_title="ROC AUC")
    return fig
