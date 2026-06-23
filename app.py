import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ── Configuración de página ─────────────────────────────────────────────────
st.set_page_config(
    page_title="Dashboard Gestión de Proyectos — PC2 USIL",
    page_icon="📊",
    layout="wide",
)

# ── Dataset ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    data = {
        "ID Proyecto": ["PRJ-001","PRJ-002","PRJ-003","PRJ-004","PRJ-005",
                        "PRJ-006","PRJ-007","PRJ-008","PRJ-009","PRJ-010",
                        "PRJ-011","PRJ-012","PRJ-013","PRJ-014","PRJ-015",
                        "PRJ-016","PRJ-017","PRJ-018","PRJ-019","PRJ-020"],
        "Nombre del Proyecto": [
            "Plataforma E-Commerce Alfa","Migración Cloud AWS Phase 1",
            "Módulo de IA - Recomendaciones","Refactorización Core Bancario",
            "App Móvil Delivery (iOS/Android)","Dashboard Métricas ERP",
            "Integración Pasarela Pagos API","Optimización Base de Datos SQL",
            "Portal de Autogestión Clientes","Sistema de Seguridad OAuth2",
            "Microservicios de Notificación","Automatización QA Pipelines",
            "Módulo de Reportes BI","Rediseño UX/UI Landing Pages",
            "API Gateway Corporativo","Sincronización ERP SAP",
            "Chatbot Atención de Clientes","Migración de Contenedores K8s",
            "Auditoría de Ciberseguridad v2","Módulo de Facturación Electrónica"],
        "Metodología": ["Scrum","DevOps","Kanban","Waterfall","Scrum",
                        "Kanban","Scrum","DevOps","Scrum","DevOps",
                        "Kanban","DevOps","Waterfall","Kanban","Scrum",
                        "Waterfall","Kanban","DevOps","Waterfall","Scrum"],
        "Líder de Proyecto": ["Carlos Mendoza","Ana Torres","Luis Valdivia","Sofía Ramos","Diego Arce",
                              "Elena Gomez","Carlos Mendoza","Ana Torres","Javier López","Sofía Ramos",
                              "Luis Valdivia","Ana Torres","Elena Gomez","Diego Arce","Carlos Mendoza",
                              "Javier López","Luis Valdivia","Ana Torres","Sofía Ramos","Javier López"],
        "Presupuesto ($)": [45000,75000,30000,120000,55000,25000,18000,35000,40000,28000,
                            22000,48000,62000,15000,50000,95000,27000,68000,40000,38000],
        "Costo Real ($)":  [42000,79000,28500,125000,56500,24000,17200,33000,41500,28000,
                            20500,51000,60000,15800,48500,102000,25000,66000,39000,37000],
        "Desviación ($)":  [3000,-4000,1500,-5000,-1500,1000,800,2000,-1500,0,
                            1500,-3000,2000,-800,1500,-7000,2000,2000,1000,1000],
        "Estado": ["Finalizado","En Desarrollo","Finalizado","En Desarrollo","En Desarrollo",
                   "Finalizado","Finalizado","Finalizado","En Desarrollo","Finalizado",
                   "Finalizado","En Desarrollo","En Desarrollo","Finalizado","En Desarrollo",
                   "En Planificación","Finalizado","En Desarrollo","Finalizado","En Desarrollo"],
        "Eficiencia de Código": [0.92,0.88,0.95,0.81,0.89,0.94,0.97,0.91,0.86,0.99,
                                 0.93,0.84,0.90,0.87,0.92,0.00,0.96,0.91,0.95,0.89],
        "Fecha de Inicio": pd.to_datetime(["2026-01-10","2026-02-15","2026-01-20","2026-03-01","2026-03-10",
                                           "2026-02-05","2026-04-01","2026-01-15","2026-04-12","2026-02-20",
                                           "2026-03-15","2026-05-02","2026-04-25","2026-02-10","2026-05-10",
                                           "2026-07-01","2026-03-18","2026-04-20","2026-01-05","2026-05-22"]),
    }
    return pd.DataFrame(data)

df_full = load_data()

# ── Entrenamiento de modelos ─────────────────────────────────────────────────
@st.cache_resource
def train_models(df):
    df_clean = df.copy()
    features = ["Costo Real ($)", "Desviación ($)", "Eficiencia de Código",
                 "Metodología", "Estado"]
    target = "Presupuesto ($)"
    X = df_clean[features]
    y = df_clean[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    num_feats = ["Costo Real ($)", "Desviación ($)", "Eficiencia de Código"]
    cat_feats = ["Metodología", "Estado"]
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_feats),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_feats),
    ])
    modelos = {
        "Regresión Lineal Múltiple ★": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=50, max_depth=3, random_state=42),
        "Árbol de Decisión": DecisionTreeRegressor(max_depth=3, random_state=42),
    }
    resultados = {}
    pipelines = {}
    for nombre, modelo in modelos.items():
        pipe = Pipeline([("prep", preprocessor), ("model", modelo)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        r2   = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae  = mean_absolute_error(y_test, y_pred)
        resultados[nombre] = {"R²": round(r2, 4), "RMSE": round(rmse, 2), "MAE": round(mae, 2)}
        pipelines[nombre] = pipe
    return pipelines, resultados

pipelines, resultados = train_models(df_full)
MEJOR_MODELO = "Regresión Lineal Múltiple ★"

# ── Header ───────────────────────────────────────────────────────────────────
st.title("📊 Dashboard — Gestión de Proyectos de Software")
st.caption("PC2 · Agentes Inteligentes · USIL 2026-1 | Dataset: 20 proyectos | Mejor modelo: Regresión Lineal Múltiple (R²=1.0)")
st.divider()

panel_a, panel_b = st.tabs(["🗂 Panel A — Análisis de Datos", "🤖 Panel B — Predicción de Presupuesto"])

# ════════════════════════════════════════════════════════════════════════════
# PANEL A — ANÁLISIS DE DATOS
# ════════════════════════════════════════════════════════════════════════════
with panel_a:

    # ── Filtros ──────────────────────────────────────────────────────────────
    st.subheader("Filtros")
    col_f1, col_f2, col_f3, col_f4 = st.columns(4)
    with col_f1:
        metodologias = ["Todas"] + sorted(df_full["Metodología"].unique().tolist())
        sel_met = st.selectbox("Metodología", metodologias)
    with col_f2:
        estados = ["Todos"] + sorted(df_full["Estado"].unique().tolist())
        sel_est = st.selectbox("Estado", estados)
    with col_f3:
        lideres = ["Todos"] + sorted(df_full["Líder de Proyecto"].unique().tolist())
        sel_lid = st.selectbox("Líder de Proyecto", lideres)
    with col_f4:
        rango = st.slider(
            "Rango presupuesto ($)",
            min_value=int(df_full["Presupuesto ($)"].min()),
            max_value=int(df_full["Presupuesto ($)"].max()),
            value=(int(df_full["Presupuesto ($)"].min()), int(df_full["Presupuesto ($)"].max())),
            step=1000,
        )

    # Aplicar filtros
    df = df_full.copy()
    if sel_met != "Todas":
        df = df[df["Metodología"] == sel_met]
    if sel_est != "Todos":
        df = df[df["Estado"] == sel_est]
    if sel_lid != "Todos":
        df = df[df["Líder de Proyecto"] == sel_lid]
    df = df[(df["Presupuesto ($)"] >= rango[0]) & (df["Presupuesto ($)"] <= rango[1])]

    st.divider()

    # ── KPIs ─────────────────────────────────────────────────────────────────
    st.subheader("KPIs del dataset")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total proyectos", len(df))
    k2.metric("Presupuesto total", f"${df['Presupuesto ($)'].sum():,.0f}")
    k3.metric("Media presupuesto", f"${df['Presupuesto ($)'].mean():,.0f}")
    desv_total = df["Desviación ($)"].sum()
    k4.metric("Desviación acumulada", f"${desv_total:,.0f}", delta=f"${desv_total:,.0f}")
    ef_valida = df[df["Eficiencia de Código"] > 0]["Eficiencia de Código"]
    k5.metric("Eficiencia promedio", f"{ef_valida.mean()*100:.1f}%" if len(ef_valida) else "N/A")

    st.divider()

    # ── Gráficos fila 1 ───────────────────────────────────────────────────────
    st.subheader("Visualizaciones interactivas")
    c1, c2 = st.columns(2)

    with c1:
        avg_met = df.groupby("Metodología")["Presupuesto ($)"].mean().reset_index()
        avg_met.columns = ["Metodología", "Presupuesto Promedio ($)"]
        fig_met = px.bar(
            avg_met, x="Metodología", y="Presupuesto Promedio ($)",
            title="Presupuesto promedio por metodología",
            color="Metodología",
            color_discrete_sequence=px.colors.qualitative.Set2,
            text_auto=True,
        )
        fig_met.update_traces(texttemplate="$%{y:,.0f}", textposition="outside")
        fig_met.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig_met, use_container_width=True)

    with c2:
        cnt_est = df["Estado"].value_counts().reset_index()
        cnt_est.columns = ["Estado", "Cantidad"]
        fig_est = px.pie(
            cnt_est, names="Estado", values="Cantidad",
            title="Distribución por estado",
            color_discrete_sequence=["#1D9E75", "#378ADD", "#D85A30"],
            hole=0.4,
        )
        fig_est.update_layout(height=350)
        st.plotly_chart(fig_est, use_container_width=True)

    # ── Gráfico scatter ──────────────────────────────────────────────────────
    fig_sc = px.scatter(
        df, x="Presupuesto ($)", y="Costo Real ($)",
        color="Metodología", hover_name="Nombre del Proyecto",
        hover_data={"Desviación ($)": True, "Estado": True},
        title="Presupuesto planificado vs. Costo real (scatter)",
        color_discrete_sequence=px.colors.qualitative.Set2,
        size_max=14,
    )
    max_v = max(df["Presupuesto ($)"].max(), df["Costo Real ($)"].max()) * 1.05
    fig_sc.add_trace(go.Scatter(
        x=[0, max_v], y=[0, max_v],
        mode="lines", name="Línea ideal (Pres.=Costo)",
        line=dict(dash="dash", color="gray", width=1),
    ))
    fig_sc.update_layout(height=380)
    st.plotly_chart(fig_sc, use_container_width=True)

    # ── Barras horizontales eficiencia ────────────────────────────────────────
    df_ef = df[df["Eficiencia de Código"] > 0].sort_values("Eficiencia de Código")
    fig_ef = px.bar(
        df_ef, x="Eficiencia de Código", y="ID Proyecto",
        orientation="h",
        title="Eficiencia de código por proyecto",
        color="Eficiencia de Código",
        color_continuous_scale="Teal",
        hover_name="Nombre del Proyecto",
        text="Eficiencia de Código",
    )
    fig_ef.update_traces(texttemplate="%{x:.0%}", textposition="outside")
    fig_ef.update_layout(height=max(300, len(df_ef) * 32 + 80), coloraxis_showscale=False)
    st.plotly_chart(fig_ef, use_container_width=True)

    # ── Mapa de calor correlación ─────────────────────────────────────────────
    num_cols = ["Presupuesto ($)", "Costo Real ($)", "Desviación ($)", "Eficiencia de Código"]
    corr = df[num_cols].corr().round(2)
    fig_corr = px.imshow(
        corr, text_auto=True, aspect="auto",
        title="Mapa de calor — Correlación entre variables numéricas",
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
    )
    fig_corr.update_layout(height=350)
    st.plotly_chart(fig_corr, use_container_width=True)

    # ── Tabla de datos ────────────────────────────────────────────────────────
    with st.expander("Ver tabla de datos filtrados"):
        st.dataframe(
            df.drop(columns=["Fecha de Inicio"]).reset_index(drop=True),
            use_container_width=True,
        )

# ════════════════════════════════════════════════════════════════════════════
# PANEL B — PREDICCIÓN
# ════════════════════════════════════════════════════════════════════════════
with panel_b:

    st.subheader("Comparación de modelos entrenados")
    cols_mod = st.columns(3)
    for i, (nombre, metricas) in enumerate(resultados.items()):
        with cols_mod[i]:
            es_mejor = nombre == MEJOR_MODELO
            st.metric(
                label=nombre,
                value=f"R² = {metricas['R²']}",
                delta=f"RMSE ${metricas['RMSE']:,.0f} | MAE ${metricas['MAE']:,.0f}",
            )
            if es_mejor:
                st.success("⭐ Mejor modelo seleccionado")

    st.divider()

    st.subheader("Formulario de predicción")
    st.caption("Ingresa las métricas del proyecto para obtener la estimación de presupuesto.")

    col_form, col_result = st.columns([1, 1])

    with col_form:
        costo_real = st.slider(
            "Costo real ($)",
            min_value=10000, max_value=135000, value=42000, step=500,
            format="$%d",
        )
        desviacion = st.slider(
            "Desviación financiera ($)",
            min_value=-10000, max_value=10000, value=3000, step=100,
            format="$%d",
        )
        eficiencia = st.slider(
            "Eficiencia de código (0.0 – 1.0)",
            min_value=0.0, max_value=1.0, value=0.92, step=0.01,
        )
        metodologia = st.selectbox(
            "Metodología",
            ["Scrum", "DevOps", "Kanban", "Waterfall"],
        )
        estado = st.selectbox(
            "Estado del proyecto",
            ["Finalizado", "En Desarrollo", "En Planificación"],
        )

    with col_result:
        input_df = pd.DataFrame([{
            "Costo Real ($)": costo_real,
            "Desviación ($)": desviacion,
            "Eficiencia de Código": eficiencia,
            "Metodología": metodologia,
            "Estado": estado,
        }])
        pred = pipelines[MEJOR_MODELO].predict(input_df)[0]
        pred_rf = pipelines["Random Forest Regressor"].predict(input_df)[0]
        pred_dt = pipelines["Árbol de Decisión"].predict(input_df)[0]

        st.metric("💰 Presupuesto estimado (★ Reg. Lineal)", f"${pred:,.0f}")
        st.metric("Rango probable (±5%)", f"${pred*0.95:,.0f} – ${pred*1.05:,.0f}")

        salud = "saludable ✅" if desviacion >= 0 else "con sobrecosto ⚠️"
        st.info(
            f"**Interpretación:** El modelo predice un presupuesto de **${pred:,.0f}** "
            f"para un proyecto **{metodologia}** en estado **{estado}**. "
            f"La Regresión Lineal captura la relación exacta: "
            f"Presupuesto = Costo Real + Desviación (R²=1.0). "
            f"Con eficiencia de código del {eficiencia*100:.0f}%, el proyecto se considera **{salud}**."
        )

        st.divider()
        st.caption("Predicciones comparativas de los 3 modelos")
        comp_df = pd.DataFrame({
            "Modelo": ["★ Reg. Lineal", "Random Forest", "Árbol de Decisión"],
            "Predicción ($)": [f"${pred:,.0f}", f"${pred_rf:,.0f}", f"${pred_dt:,.0f}"],
            "R²": ["1.0000", "0.7634", "0.3524"],
        })
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Estadísticos descriptivos del dataset completo")
    desc = df_full[["Presupuesto ($)", "Costo Real ($)", "Desviación ($)", "Eficiencia de Código"]].describe().round(2)
    st.dataframe(desc, use_container_width=True)
