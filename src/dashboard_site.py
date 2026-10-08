"""Produce un dashboard HTML interactivo Plotly para GitHub Pages.

GitHub Pages no ejecuta el servidor Dash. La aplicación Dash real está
en dashboard/app.py y el HTML conserva las mismas tres vistas con
gráficas interactivas calculadas desde la base oficial de Kaggle.
"""
import json
import shutil
from pathlib import Path

from plotly.offline import get_plotlyjs_version
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

from dashboard.figures import comparison_figure, descriptive_figures
from src.data import describe_data

FOCUS = ["KNN", "RandomForest", "LogisticRegression"]
DISPLAY = {"KNN": "KNN", "RandomForest": "Random Forest",
           "LogisticRegression": "Regresión logística"}


def model_details(results, ranking):
    x_train, x_test = results["x_train"], results["x_test"]
    y_train, y_test = results["y_train"], results["y_test"]
    answer = {}
    for name in FOCUS:
        clf = results["grids"][name].best_estimator_
        prob = clf.predict_proba(x_test)[:, 1]
        tr_prob = clf.predict_proba(x_train)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, prob)
        row = ranking.loc[ranking["modelo"] == name].iloc[0]
        matrix = confusion_matrix(y_test, (prob >= 0.5).astype(int),
                                  labels=[0, 1])
        answer[name] = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "confusion": matrix.tolist(),
            "auc_cv": float(row["AUC_CV"]),
            "auc_test": float(row["AUC_test"]),
            "accuracy": float(row["Accuracy_test"]),
            "auc_entrenamiento": float(roc_auc_score(y_train, tr_prob)),
            "parametros": str(results["grids"][name].best_params_),
        }
    return answer


def render_dashboard(frame, ranking, results, report):
    root = Path("docs")
    root.mkdir(exist_ok=True)
    src = Path("dashboard/assets/heart-hero.svg")
    dst = root / "assets/heart-hero.svg"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    details = model_details(results, ranking)
    (Path("reportes") / "modelos_dashboard.json").write_text(
        json.dumps(details, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (root / "modelos_dashboard.json").write_text(
        json.dumps(details, ensure_ascii=False), encoding="utf-8"
    )
    d = describe_data(frame)
    figures = descriptive_figures(frame)
    figures["comparacion"] = comparison_figure(ranking)
    chart_json = {name: json.loads(fig.to_json()) for name, fig in figures.items()}
    focused = ranking[ranking["modelo"].isin(FOCUS)]
    rows = "".join(
        "<tr><td>{}</td><td>{:.3f}</td><td>{:.3f}</td><td>{:.3f}</td>"
        "<td>{:+.3f}</td></tr>".format(
            DISPLAY[row.modelo], row.AUC_CV, row.AUC_test, row.Accuracy_test,
            details[row.modelo]["auc_entrenamiento"] - row.AUC_CV
        ) for row in focused.itertuples()
    )
    html_page = '''<!doctype html>
<html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="description" content="Dashboard de Machine Learning MLOps, Edwin Yunis y Jairo Serrano, dataset original de Kaggle.">
<title>Cardio ML · Dashboard — Edwin Yunis y Jairo Serrano</title>
<link rel="stylesheet" href="assets/style.css">
<script src="https://cdn.plot.ly/plotly-%%PLOTLY_VERSION%%.min.js"></script>
</head><body>
<div class="page">
<aside class="sidebar"><div>
<div class="logo"><span class="logo-pulse">♡</span><span>CARDIO<br>ML · MLOps</span></div>
<p class="side-label">Proyecto integrador</p>
<p class="side-meta"><b>Autores</b><br>Edwin Yunis<br>Jairo Serrano<br><br>
<b>Asignatura</b><br>Machine Learning<br><br>
<b>Fuente</b><br>Kaggle · Heart Failure Prediction</p>
</div><div class="side-footer">Maestría en Ingeniería Industrial<br>
Análisis retrospectivo · no diagnóstico clínico</div></aside>
<main class="content">
<header class="hero"><div>
<p class="eyebrow">PROYECTO INTEGRADOR · MACHINE LEARNING</p>
<h1>Análisis de enfermedad cardíaca</h1>
<p>Exploración de datos clínicos, comparación de modelos y flujo MLOps reproducible.</p>
<strong>Edwin Yunis · Jairo Serrano</strong>
</div><img src="assets/heart-hero.svg" alt="Ilustración cardiovascular con electrocardiograma"></header>
<div class="nav-tabs" role="tablist" aria-label="Secciones del dashboard">
<button type="button" class="tab-button tab--active" role="tab" aria-selected="true" data-tab="contexto">
01 · Contexto<span>Problema, datos y metodología</span></button>
<button type="button" class="tab-button" role="tab" aria-selected="false" data-tab="eda">
02 · EDA<span>Análisis exploratorio</span></button>
<button type="button" class="tab-button" role="tab" aria-selected="false" data-tab="models">
03 · Models<span>Comparación de clasificadores</span></button></div>

<section id="contexto" class="tab-panel active" role="tabpanel">
<h2>Contexto del problema</h2>
<div class="card"><h3>Objetivo y alcance</h3>
<p>Este trabajo utiliza la base <em>Heart Failure Prediction</em> que indicó el docente,
publicada por fedesoriano en Kaggle. El objetivo es clasificar
<code>HeartDisease</code> (0 = ausencia; 1 = presencia observada de enfermedad cardíaca).
El dataset es histórico y <strong>no permite predecir directamente eventos futuros de
insuficiencia cardíaca</strong>.</p>
<span class="mini-pill">Clasificación binaria</span>
<span class="mini-pill">Dataset original</span>
<span class="mini-pill">MLOps reproducible</span></div>
<div class="kpi-grid">
<div class="kpi"><b>%%REGISTROS%%</b><span>Registros clínicos</span></div>
<div class="kpi"><b>%%PREDICTORES%%</b><span>Variables predictoras</span></div>
<div class="kpi"><b>%%POSITIVA%%</b><span>Clase positiva</span></div>
<div class="kpi"><b>%%DUP%%</b><span>Registros duplicados</span></div></div>
<div class="card"><h3>Etapas del proyecto MLOps</h3>
<div class="flow">
<article><b>Etapa 0 · Estructura</b><p>Organización modular de notebooks y código.</p></article>
<article><b>Etapa 1 · EDA</b><p>Calidad, distribuciones y data leakage.</p></article>
<article><b>Etapa 2 · Modelos</b><p>Pipeline y GridSearchCV de cinco pliegues.</p></article>
<article><b>Etapa 3 · API</b><p>Servicio FastAPI dentro de Docker.</p></article>
<article><b>Etapa 4 · Kubernetes</b><p>Manifiestos para orquestación con Minikube.</p></article>
<article><b>Etapas 5 y 6 · CI/CD y deriva</b><p>GitHub Actions y Evidently.</p></article>
</div></div>
<div class="card"><h3>Rigor metodológico</h3>
<p>Las métricas se calculan exclusivamente sobre los %%REGISTROS%% casos originales.
El conjunto de prueba se reserva antes del ajuste de transformaciones.
No se generan pacientes, observaciones ni desenlaces artificiales.</p>
<div class="warning">Dashboard de investigación: los resultados no representan
diagnóstico, probabilidad clínica calibrada ni validación externa.</div>
<p>Documentación complementaria: <a href="informe_tecnico.html">Informe técnico y etapas detalladas ↗</a> ·
<a href="drift_report.html">Reporte de Evidently ↗</a></p></div>
</section>

<section id="eda" class="tab-panel" role="tabpanel" hidden>
<h2>Exploración de datos (EDA)</h2>
<p class="description">Gráficos interactivos generados directamente desde el CSV oficial;
puedes situar el cursor sobre las marcas para ver sus valores.</p>
<div class="kpi-grid">
<div class="kpi"><b>%%CLASE0%%</b><span>Sin enfermedad (0)</span></div>
<div class="kpi"><b>%%CLASE1%%</b><span>Con enfermedad (1)</span></div>
<div class="kpi"><b>%%BPZERO%%</b><span>RestingBP = 0</span></div>
<div class="kpi"><b>%%CHOLZERO%%</b><span>Cholesterol = 0</span></div></div>
<div class="grid-2">
<div class="card chart-card"><div id="plot-clases"></div></div>
<div class="card chart-card"><div id="plot-edad"></div></div>
<div class="card chart-card"><div id="plot-colesterol"></div></div>
<div class="card chart-card"><div id="plot-dolor"></div></div>
</div><div class="card chart-card"><div id="plot-correlaciones"></div></div>
<div class="card"><h3>Lectura del análisis</h3>
<p>Se revisan el balance de clases, la distribución de edades,
el tipo de dolor torácico y la relación entre colesterol y la clase.
Los ceros de colesterol y de presión arterial requieren atención como
valores de calidad de datos; en el modelado se tratan mediante
imputación ajustada solo con los registros de entrenamiento.</p>
<div class="warning">Las asociaciones gráficas no demuestran causalidad.</div></div>
</section>

<section id="models" class="tab-panel" role="tabpanel" hidden>
<h2>Modelos de Machine Learning</h2>
<div class="card"><h3>Protección frente a fuga y sobreajuste</h3>
<p>Se separa una prueba estratificada del 20 % antes de imputar, escalar o
codificar. Cada transformación forma parte de <code>Pipeline</code>.
<code>GridSearchCV</code> utiliza cinco pliegues estratificados y selecciona
hiperparámetros por AUC media de validación cruzada, sin usar la prueba
para seleccionar el modelo.</p>
<p>KNN ajusta el número de vecinos; Random Forest limita profundidad y
tamaño de hojas; la regresión logística utiliza regularización L2.
Estos controles <strong>reducen el riesgo de sobreajuste, pero no garantizan
eliminarlo</strong>.</p>
<div class="warning">Regresión logística es la técnica de regresión adecuada
para esta tarea de clasificación binaria.</div></div>
<div class="card chart-card"><div id="plot-comparacion"></div></div>
<div class="card"><h3>Comparación de los tres modelos solicitados</h3>
<div class="table-scroller"><table><thead><tr>
<th>Clasificador</th><th>AUC CV</th><th>AUC prueba</th>
<th>Accuracy</th><th>Brecha train−CV</th>
</tr></thead><tbody>%%RANKING%%</tbody></table></div>
<p class="description">La brecha de AUC entrenamiento − CV ayuda a
evaluar sobreajuste; una brecha pequeña no lo descarta.</p></div>
<div class="card"><label for="model-select">Explora un clasificador</label>
<select id="model-select"><option value="KNN">KNN</option>
<option value="RandomForest" selected>Random Forest</option>
<option value="LogisticRegression">Regresión logística</option></select>
<div class="metric-strip">
<article><strong id="score-cv">—</strong><span>AUC CV</span></article>
<article><strong id="score-test">—</strong><span>AUC prueba</span></article>
<article><strong id="score-accuracy">—</strong><span>Accuracy prueba</span></article>
<article><strong id="score-gap">—</strong><span>Brecha train−CV</span></article>
</div>
<div class="grid-2"><div class="chart-card"><div id="plot-roc"></div></div>
<div class="chart-card"><div id="plot-confusion"></div></div></div>
<p class="description" id="model-params"></p></div>
<div class="card"><h3>Interpretación y alcance</h3>
<p>Las AUC resumen discriminación y Accuracy refleja aciertos con un umbral de 0,5.
El conjunto de prueba se usa exclusivamente para evaluación final. El
dashboard se enfoca en tres modelos, aunque el trabajo original también
incluye SVC y Gradient Boosting.</p>
<a href="informe_tecnico.html">Ver reporte técnico completo ↗</a></div>
</section>
<footer>Edwin Yunis · Jairo Serrano · Machine Learning · Dashboard Plotly</footer>
</main></div>
<script>
const FIGS = %%CHARTS%%;
const DETAILS = %%MODELS%%;
const CONFIG = {responsive:true,displayModeBar:false};
let drawnEDA=false, drawnModels=false;
function chart(name,id){
 const f=FIGS[name]; return Plotly.newPlot(id,f.data,f.layout,CONFIG);
}
function drawEDA(){
 if(drawnEDA) return;
 for(const key of ["clases","edad","colesterol","dolor","correlaciones"]){
   chart(key,"plot-"+key);
 }
 drawnEDA=true;
}
function drawModel(){
 const id=document.getElementById("model-select").value;
 const m=DETAILS[id];
 document.getElementById("score-cv").textContent=m.auc_cv.toFixed(3);
 document.getElementById("score-test").textContent=m.auc_test.toFixed(3);
 document.getElementById("score-accuracy").textContent=m.accuracy.toFixed(3);
 document.getElementById("score-gap").textContent=(m.auc_entrenamiento-m.auc_cv).toFixed(3);
 document.getElementById("model-params").textContent="Parámetros seleccionados en CV: "+m.parametros;
 const roc=[
  {x:[0,1],y:[0,1],name:"Referencia",mode:"lines",
   line:{dash:"dash",color:"#98aab3"}},
  {x:m.fpr,y:m.tpr,name:"AUC "+m.auc_test.toFixed(3),mode:"lines",
   line:{color:"#1ba4c9",width:3}}
 ];
 Plotly.react("plot-roc",roc,{title:"Curva ROC · Prueba reservada",
  paper_bgcolor:"#fff",plot_bgcolor:"#fff",
  xaxis:{title:"Falsos positivos",range:[0,1]},
  yaxis:{title:"Verdaderos positivos",range:[0,1]},
  margin:{t:55,l:56,r:20,b:55}},CONFIG);
 Plotly.react("plot-confusion",[{
  z:m.confusion,x:["Predicho 0","Predicho 1"],
  y:["Real 0","Real 1"],type:"heatmap",showscale:false,
  colorscale:[[0,"#edf8fa"],[1,"#187598"]],
  text:m.confusion,texttemplate:"%{text}",textfont:{size:20},
  hovertemplate:"%{y} / %{x}: %{z}<extra></extra>"
 }],{title:"Matriz de confusión · Prueba reservada",
  yaxis:{autorange:"reversed"},margin:{t:55,l:75,r:20,b:55}},CONFIG);
}
function drawModels(){
 if(!drawnModels){chart("comparacion","plot-comparacion");drawnModels=true;}
 drawModel();
}
function showTab(name){
 for(const node of document.querySelectorAll(".tab-panel")){
  const selected=node.id===name;
  node.hidden=!selected;node.classList.toggle("active",selected);
 }
 for(const btn of document.querySelectorAll(".tab-button")){
  const on=btn.dataset.tab===name;
  btn.classList.toggle("tab--active",on);
  btn.setAttribute("aria-selected",on?"true":"false");
 }
 if(name==="eda")drawEDA();
 if(name==="models")drawModels();
}
document.querySelectorAll(".tab-button").forEach(btn=>
 btn.addEventListener("click",()=>showTab(btn.dataset.tab)));
document.getElementById("model-select").addEventListener("change",drawModel);
showTab("contexto");
</script>
</body></html>'''
    replacements = {
        "PLOTLY_VERSION": get_plotlyjs_version(),
        "REGISTROS": d["n_registros"],
        "PREDICTORES": d["n_variables"],
        "POSITIVA": f'{d["prevalencia_observada"]:.1%}',
        "DUP": d["duplicados_completos"],
        "CLASE0": int((frame["HeartDisease"] == 0).sum()),
        "CLASE1": int((frame["HeartDisease"] == 1).sum()),
        "BPZERO": d["restingbp_cero"],
        "CHOLZERO": d["cholesterol_cero"],
        "CHARTS": json.dumps(chart_json, ensure_ascii=False).replace("</", "<\\/"),
        "MODELS": json.dumps(details, ensure_ascii=False),
        "RANKING": rows,
    }
    for key, value in replacements.items():
        html_page = html_page.replace("%%" + key + "%%", str(value))
    if "%%" in html_page:
        raise ValueError("Marcadores HTML no resueltos")
    css = Path("dashboard/assets/style.css").read_text(encoding="utf-8")
    (root / "assets/style.css").write_text(css, encoding="utf-8")
    (root / "index.html").write_text(html_page, encoding="utf-8")
    print("Dashboard Plotly publicado con tres vistas y métricas reales.")
    return details
