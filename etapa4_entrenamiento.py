"""
etapa4_entrenamiento.py - Entrenamiento y evaluación de modelos

Responsabilidades:
- Lee plan_experimental.json (entrada ejecutable del Agente 3), propuesta_modelos.json,
  metadata.json y estadisticas.json de las etapas anteriores.
- Ejecuta el plan: transformación del objetivo, encoding, escalado, modelos.
- Calcula métricas reales con Python (RMSE, MAE, R2, MAPE, MedAE).
- Compara con la línea base calculada en la Etapa 3.
- Pide al LLM interpretación de mercado, limitaciones y recomendaciones.
- Genera metricas.json, comparacion_modelos.csv, reporte_resultados.md y modelos.joblib.

Principios de diseño:
- Python ejecuta el plan y calcula TODAS las cifras; el LLM solo interpreta.
- Si el plan propone capacidades no implementadas, se aplica fallback y se registra.
- El porcentaje de error se calcula con Python para evitar alucinaciones del LLM.
- Los modelos entrenados se guardan para reproducibilidad.
- El reporte Markdown es narrativa de ciencia de datos;
  las trazabilidades internas viven solo en metricas.json.

Uso:
    python etapa4_entrenamiento.py
    python etapa4_entrenamiento.py --modelo qwen3.5:9b
    python etapa4_entrenamiento.py --revision data/revision_etapa4_entrenamiento.json
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.linear_model import Ridge, Lasso, LinearRegression, ElasticNet
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor,
)
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import boxcox
from scipy.special import inv_boxcox

from comun import (PRINCIPIOS_DE_TRABAJO, REGLAS_DE_EVIDENCIA, USO_DE_ANTECEDENTES,
                   ahora, asegurar_directorios, bloque_antecedentes,
                   consultar_llm_json, crear_llm, crear_parser, guardar_json,
                   hay_error, leer_json, leer_revision, normalizar_lista_texto,
                   redondear_estructura, validar_opcion, verificar_cifras)

# ============================================================================
# CONFIGURACIÓN
# ============================================================================
DATASET_EXPERIMENTAL = Path("data/dataset_experimental.csv")
METADATA_FILE = Path("data/metadata.json")
ESTADISTICAS_FILE = Path("data/estadisticas.json")
PROPUESTA_MODELOS = Path("data/propuesta_modelos.json")
PLAN_EXPERIMENTAL = Path("data/plan_experimental.json")
METRICAS_FILE = Path("data/metricas.json")
COMPARACION_FILE = Path("data/comparacion_modelos.csv")
REPORTE_RESULTADOS = Path("data/reports/reporte_resultados.md")
PLOTS_DIR = Path("data/plots")
MODELS_DIR = Path("data/models")

# ============================================================================
# REGISTRO DE CAPACIDADES
# ============================================================================
MODELOS_SKLEARN = {
    "DummyRegressor": DummyRegressor,
    "LinearRegression": LinearRegression,
    "Ridge": Ridge,
    "Lasso": Lasso,
    "ElasticNet": ElasticNet,
    "RandomForestRegressor": RandomForestRegressor,
    "GradientBoostingRegressor": GradientBoostingRegressor,
    "HistGradientBoostingRegressor": HistGradientBoostingRegressor,
}

TRANSFORMACIONES_OBJETIVO = {"ninguna", "log", "log1p", "sqrt", "box-cox"}
ENCODINGS = {"one_hot", "frequency", "target", "top_n"}
ESCALADOS = {"standard": StandardScaler, "minmax": MinMaxScaler,
             "robust": RobustScaler, "none": None}

FALLBACK_TRANSFORMACION = "ninguna"
FALLBACK_ENCODING = "frequency"
FALLBACK_ESCALADO = "standard"


# ============================================================================
# TRANSFORMACIONES DEL OBJETIVO
# ============================================================================
def transformar_objetivo(y, metodo: str):
    """Aplica transformación al objetivo. Devuelve (y_transformado, lambda_boxcox)."""
    if metodo == "ninguna":
        return y, None
    if metodo == "log":
        if (y <= 0).any():
            raise ValueError(f"log requiere valores positivos; hay {(y <= 0).sum()} no positivos")
        return np.log(y), None
    if metodo == "log1p":
        if (y < 0).any():
            raise ValueError(f"log1p requiere valores >= 0; hay {(y < 0).sum()} negativos")
        return np.log1p(y), None
    if metodo == "sqrt":
        if (y < 0).any():
            raise ValueError(f"sqrt requiere valores >= 0")
        return np.sqrt(y), None
    if metodo == "box-cox":
        if (y <= 0).any():
            raise ValueError(f"box-cox requiere valores estrictamente positivos")
        y_t, lmbda = boxcox(y)
        return y_t, lmbda
    raise ValueError(f"Transformación desconocida: {metodo}")


def revertir_objetivo(y_t, metodo: str, lmbda=None):
    """Revierte la transformación del objetivo."""
    if metodo == "ninguna":
        return y_t
    if metodo == "log":
        return np.exp(y_t)
    if metodo == "log1p":
        return np.expm1(y_t)
    if metodo == "sqrt":
        return y_t ** 2
    if metodo == "box-cox":
        return inv_boxcox(y_t, lmbda)
    raise ValueError(f"Transformación desconocida: {metodo}")


# ============================================================================
# ENCODING MANUAL (sin transformers personalizados)
# ============================================================================
def aplicar_encoding(df_train: pd.DataFrame, df_test: pd.DataFrame,
                    columnas_cat: list, encoding: str, y_train=None) -> tuple:
    """Aplica encoding a columnas categóricas. Devuelve (df_train_proc, df_test_proc).
    El encoding se calcula SOLO en train y se aplica a test para evitar data leakage."""
    df_train_proc = df_train.copy()
    df_test_proc = df_test.copy()

    if encoding == "one_hot":
        df_train_proc = pd.get_dummies(df_train_proc, columns=columnas_cat,
                                       drop_first=False, dtype=float)
        df_test_proc = pd.get_dummies(df_test_proc, columns=columnas_cat,
                                      drop_first=False, dtype=float)
        df_test_proc = df_test_proc.reindex(columns=df_train_proc.columns, fill_value=0.0)

    elif encoding == "frequency":
        for col in columnas_cat:
            freqs = df_train_proc[col].value_counts(normalize=True)
            min_freq = freqs.min() if len(freqs) > 0 else 0.0
            df_train_proc[col] = df_train_proc[col].map(freqs).fillna(min_freq)
            df_test_proc[col] = df_test_proc[col].map(freqs).fillna(min_freq)

    elif encoding == "target":
        if y_train is None:
            raise ValueError("Target encoding requiere y_train")
        for col in columnas_cat:
            global_mean = y_train.mean()
            grouped = pd.DataFrame({"cat": df_train_proc[col], "target": y_train}).groupby("cat")["target"]
            counts = grouped.count()
            means = grouped.mean()
            smoothing = 10.0
            smoothed = (counts * means + smoothing * global_mean) / (counts + smoothing)
            df_train_proc[col] = df_train_proc[col].map(smoothed).fillna(global_mean)
            df_test_proc[col] = df_test_proc[col].map(smoothed).fillna(global_mean)

    elif encoding == "top_n":
        n = 15
        for col in columnas_cat:
            top = df_train_proc[col].value_counts().head(n).index.tolist()
            df_train_proc[col] = df_train_proc[col].apply(lambda v: v if v in top else "__OTHER__")
            df_test_proc[col] = df_test_proc[col].apply(lambda v: v if v in top else "__OTHER__")
        df_train_proc = pd.get_dummies(df_train_proc, columns=columnas_cat,
                                       drop_first=False, dtype=float)
        df_test_proc = pd.get_dummies(df_test_proc, columns=columnas_cat,
                                      drop_first=False, dtype=float)
        df_test_proc = df_test_proc.reindex(columns=df_train_proc.columns, fill_value=0.0)

    return df_train_proc, df_test_proc


# ============================================================================
# MÉTRICAS
# ============================================================================
def calcular_metricas(y_real, y_pred) -> dict:
    """Calcula todas las métricas disponibles."""
    mae = float(mean_absolute_error(y_real, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_real, y_pred)))
    r2 = float(r2_score(y_real, y_pred))

    mask = y_real != 0
    if mask.sum() > 0:
        mape = float(np.mean(np.abs((y_real[mask] - y_pred[mask]) / y_real[mask])) * 100)
    else:
        mape = None

    medae = float(np.median(np.abs(y_real - y_pred)))

    return {"RMSE": rmse, "MAE": mae, "R2": r2, "MAPE": mape, "MedAE": medae}


# ============================================================================
# PREPARACIÓN DE DATOS
# ============================================================================
def preparar_datos(df: pd.DataFrame, objetivo: str, plan: dict):
    """Divide datos, aplica encoding y transforma objetivo."""
    X = df.drop(columns=[objetivo])
    y = df[objetivo].values.astype(float)

    columnas_num = X.select_dtypes(include=[np.number]).columns.tolist()
    columnas_cat = X.select_dtypes(
        exclude=[np.number, "datetime", "timedelta", "bool"]).columns.tolist()

    # Transformación del objetivo
    transf = plan.get("transformacion_objetivo", "ninguna")
    if not validar_opcion(transf, TRANSFORMACIONES_OBJETIVO):
        transf = FALLBACK_TRANSFORMACION

    y_t, lmbda = transformar_objetivo(y, transf)

    # Split
    test_size = plan.get("estrategia_validacion", {}).get("test_size", 0.2)
    X_train, X_test, y_train_t, y_test_t, y_train_orig, y_test_orig = train_test_split(
        X, y_t, y, test_size=test_size, random_state=42
    )

    # Preprocesamiento
    preproc_plan = plan.get("preprocesamiento", {})
    encoding = preproc_plan.get("encoding_categoricas", FALLBACK_ENCODING)
    escalado = preproc_plan.get("escalado_numericas", FALLBACK_ESCALADO)
    manejo_nulos = preproc_plan.get("manejo_nulos", "imputar_mediana")

    if not validar_opcion(encoding, ENCODINGS):
        encoding = FALLBACK_ENCODING
    if not validar_opcion(escalado, ESCALADOS.keys()):
        escalado = FALLBACK_ESCALADO

    # Aplicar encoding (usa solo y_train → sin leakage)
    X_train_proc, X_test_proc = aplicar_encoding(
        X_train, X_test, columnas_cat, encoding, y_train_orig
    )

    # Imputar nulos en numéricas
    estrategia_nulos = "median" if manejo_nulos == "imputar_mediana" else "mean"
    imputer_num = SimpleImputer(strategy=estrategia_nulos)
    if columnas_num:
        X_train_proc[columnas_num] = imputer_num.fit_transform(X_train_proc[columnas_num])
        X_test_proc[columnas_num] = imputer_num.transform(X_test_proc[columnas_num])

    # Escalar numéricas
    if escalado != "none" and columnas_num and ESCALADOS.get(escalado) is not None:
        scaler = ESCALADOS[escalado]()
        X_train_proc[columnas_num] = scaler.fit_transform(X_train_proc[columnas_num])
        X_test_proc[columnas_num] = scaler.transform(X_test_proc[columnas_num])

    X_train_array = X_train_proc.values.astype(float)
    X_test_array = X_test_proc.values.astype(float)

    return (X_train_array, X_test_array, y_train_t, y_test_t, y_train_orig, y_test_orig,
            transf, lmbda, {"encoding": encoding, "escalado": escalado,
                           "manejo_nulos": manejo_nulos, "transformacion": transf})


# ============================================================================
# ENTRENAMIENTO
# ============================================================================
def entrenar_modelo(nombre: str, clase_nombre: str, hiperparams: dict,
                    X_train, X_test, y_train_t, y_test_t, y_train_orig, y_test_orig,
                    transf: str, lmbda) -> dict:
    """Entrena un modelo y devuelve métricas en escala original."""
    if clase_nombre not in MODELOS_SKLEARN:
        return {"error": f"Modelo '{clase_nombre}' no disponible"}

    clase = MODELOS_SKLEARN[clase_nombre]
    modelo = clase(**hiperparams)

    modelo.fit(X_train, y_train_t)

    y_pred_train_t = modelo.predict(X_train)
    y_pred_test_t = modelo.predict(X_test)

    y_pred_train = revertir_objetivo(y_pred_train_t, transf, lmbda)
    y_pred_test = revertir_objetivo(y_pred_test_t, transf, lmbda)

    metricas_train = calcular_metricas(y_train_orig, y_pred_train)
    metricas_test = calcular_metricas(y_test_orig, y_pred_test)

    diff_r2 = metricas_train["R2"] - metricas_test["R2"]
    sobreajuste = diff_r2 > 0.2

    try:
        n_splits_cv = min(3, len(X_train))
        cv_scores = cross_val_score(clase(**hiperparams), X_train, y_train_t,
                                    cv=n_splits_cv, scoring="neg_mean_squared_error")
        cv_rmse = float(np.sqrt(-cv_scores.mean()))
        cv_std = float(np.sqrt(-cv_scores).std())
    except Exception:
        cv_rmse = None
        cv_std = None

    return {
        "modelo": modelo,
        "metricas_train": metricas_train,
        "metricas_test": metricas_test,
        "sobreajuste": sobreajuste,
        "diferencia_r2": float(diff_r2),
        "cv_rmse": cv_rmse,
        "cv_std": cv_std,
    }


# ============================================================================
# GRÁFICAS
# ============================================================================
def generar_graficas(resultados: list, objetivo: str):
    """Gráficas comparativas de modelos."""
    plt.style.use("seaborn-v0_8-whitegrid")

    nombres = [r["nombre"] for r in resultados]
    test_rmse = [r["metricas_test"]["RMSE"] for r in resultados]
    test_r2 = [r["metricas_test"]["R2"] for r in resultados]
    train_rmse = [r["metricas_train"]["RMSE"] for r in resultados]

    # Comparación RMSE
    fig, ax = plt.subplots(figsize=(10, 6))
    colores = ["steelblue" if rmse == min(test_rmse) else "lightblue" for rmse in test_rmse]
    ax.barh(nombres, test_rmse, color=colores, edgecolor="black")
    ax.set_xlabel("RMSE (test)")
    ax.set_title("Comparación de modelos - RMSE en test")
    ax.axvline(min(test_rmse), color="red", linestyle="--", label=f"Mejor: {min(test_rmse):.3f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "comparacion_rmse.png", dpi=100)
    plt.close(fig)

    # Comparación R²
    fig, ax = plt.subplots(figsize=(10, 6))
    colores = ["green" if r2 == max(test_r2) else "lightgreen" for r2 in test_r2]
    ax.barh(nombres, test_r2, color=colores, edgecolor="black")
    ax.set_xlabel("R² (test)")
    ax.set_title("Comparación de modelos - R² en test")
    ax.axvline(max(test_r2), color="red", linestyle="--", label=f"Mejor: {max(test_r2):.3f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "comparacion_r2.png", dpi=100)
    plt.close(fig)

    # Sobreajuste
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(nombres))
    width = 0.35
    ax.bar(x - width / 2, train_rmse, width, label="Train RMSE", color="orange")
    ax.bar(x + width / 2, test_rmse, width, label="Test RMSE", color="steelblue")
    ax.set_xticks(x)
    ax.set_xticklabels(nombres, rotation=45, ha="right")
    ax.set_ylabel("RMSE")
    ax.set_title("Detección de sobreajuste: Train vs Test")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "sobreajuste.png", dpi=100)
    plt.close(fig)


# ============================================================================
# INTERPRETACIÓN CON LLM
# ============================================================================
SYSTEM_INTERPRETACION = (
    PRINCIPIOS_DE_TRABAJO + USO_DE_ANTECEDENTES + REGLAS_DE_EVIDENCIA + """

Eres un científico de datos senior. Recibes hechos calculados sobre el rendimiento de
modelos (métricas, comparación con línea base, sobreajuste) y el contexto previo.

DECIDES TÚ
- Qué modelo es el ganador y por qué.
- Qué significan las métricas en el contexto real del problema.
- Qué limitaciones tienen los resultados.
- Qué recomendaciones harías para mejorar.

Cada conclusión debe citar evidencia concreta (métrica, valor). Sé CONCISO: máximo
2 oraciones por campo de texto.

Responde SOLO con JSON:
{
  "modelo_ganador": "nombre del modelo",
  "justificacion_ganador": "por qué es el mejor (máx 2 oraciones)",
  "analisis_sobreajuste": "evaluación train/test (máx 2 oraciones)",
  "comparacion_linea_base": "comparación con línea base (máx 2 oraciones)",
  "interpretacion_mercado": "significado en contexto real (máx 3 oraciones)",
  "limitaciones": ["limitación 1", "limitación 2"],
  "recomendaciones_futuras": ["recomendación 1", "recomendación 2"],
  "conclusion_general": "resumen en 2 oraciones",
  "confianza_general": "alta | media | baja"
}"""
)


def interpretar_resultados(llm, hechos: dict, antecedentes: dict, revision: str,
                          agente: str) -> tuple:
    """Devuelve (respuesta_llm, validaciones, cifras_dudosas)."""
    print(f"[{agente}] Generando interpretación...")
    user_prompt = (
        bloque_antecedentes(hechos, antecedentes) +
        f"\n\nInterpreta estos resultados de entrenamiento.\n{revision}"
    )
    respuesta = consultar_llm_json(
        llm, SYSTEM_INTERPRETACION, user_prompt, agente=agente,
        claves_requeridas=["modelo_ganador", "interpretacion_mercado"]
    )

    if hay_error(respuesta):
        return respuesta, [], []

    limitaciones = normalizar_lista_texto(respuesta.get("limitaciones"))
    recomendaciones = normalizar_lista_texto(respuesta.get("recomendaciones_futuras"))

    validaciones = []
    cifras_dudosas = verificar_cifras(respuesta, hechos)

    respuesta_normalizada = {
        "modelo_ganador": str(respuesta.get("modelo_ganador", "")),
        "justificacion_ganador": str(respuesta.get("justificacion_ganador", "")),
        "analisis_sobreajuste": str(respuesta.get("analisis_sobreajuste", "")),
        "comparacion_linea_base": str(respuesta.get("comparacion_linea_base", "")),
        "interpretacion_mercado": str(respuesta.get("interpretacion_mercado", "")),
        "limitaciones": limitaciones,
        "recomendaciones_futuras": recomendaciones,
        "conclusion_general": str(respuesta.get("conclusion_general", "")),
        "confianza_general": str(respuesta.get("confianza_general", "")),
    }

    return respuesta_normalizada, validaciones, cifras_dudosas


# ============================================================================
# REPORTE (narrativa limpia, sin trazabilidades internas)
# ============================================================================
def generar_reporte(resultados: list, interpretacion: dict, objetivo: str,
                   linea_base: dict, plan_ejecutado: dict, desviaciones: list):
    """Genera el informe Markdown de la Etapa 4."""
    ganador = next((r for r in resultados if r["nombre"] == interpretacion.get("modelo_ganador")),
                   resultados[0] if resultados else None)

    # CORREGIDO: valores calculados antes de la f-string (evita error de sintaxis)
    nombre_ganador = ganador["nombre"] if ganador else "N/A"
    rmse_ganador = ganador["metricas_test"]["RMSE"] if ganador else 0.0
    mae_ganador = ganador["metricas_test"]["MAE"] if ganador else 0.0
    r2_ganador = ganador["metricas_test"]["R2"] if ganador else 0.0

    c = f"""# Reporte de Resultados - Etapa 4

**Fecha:** {ahora()}
**Variable objetivo:** {objetivo}

## 1. Resumen Ejecutivo

{interpretacion.get('conclusion_general', 'No disponible')}

**Modelo ganador:** {interpretacion.get('modelo_ganador', 'N/A')}

## 2. Contexto del Pipeline

### Plan Ejecutado
- **Transformación del objetivo:** {plan_ejecutado.get('transformacion')}
- **Encoding categóricas:** {plan_ejecutado.get('encoding')}
- **Escalado numéricas:** {plan_ejecutado.get('escalado')}
- **Manejo de nulos:** {plan_ejecutado.get('manejo_nulos')}

### Comparación con Línea Base
{interpretacion.get('comparacion_linea_base', 'No disponible')}

| Modelo | RMSE | MAE | R² |
|--------|------|-----|-----|
| Línea base (DummyRegressor) | {linea_base.get('rmse', 0):.4f} | {linea_base.get('mae', 0):.4f} | {linea_base.get('r2', 0):.4f} |
| {nombre_ganador} | {rmse_ganador:.4f} | {mae_ganador:.4f} | {r2_ganador:.4f} |

## 3. Comparación de Modelos

| Modelo | Train RMSE | Test RMSE | Train MAE | Test MAE | Train R² | Test R² | Sobreajuste |
|--------|-----------|-----------|-----------|----------|----------|---------|-------------|
"""
    for r in resultados:
        mt = r["metricas_train"]
        me = r["metricas_test"]
        sobreajuste = "Sí" if r["sobreajuste"] else "No"
        c += (f"| {r['nombre']} | {mt['RMSE']:.4f} | {me['RMSE']:.4f} | "
              f"{mt['MAE']:.4f} | {me['MAE']:.4f} | {mt['R2']:.4f} | "
              f"{me['R2']:.4f} | {sobreajuste} |\n")

    c += f"""
## 4. Análisis de Sobreajuste

{interpretacion.get('analisis_sobreajuste', 'No disponible')}

![Sobreajuste](../plots/sobreajuste.png)

## 5. Comparación Visual

![RMSE](../plots/comparacion_rmse.png)

![R²](../plots/comparacion_r2.png)

## 6. Interpretación de Mercado

{interpretacion.get('interpretacion_mercado', 'No disponible')}

## 7. Limitaciones

"""
    limitaciones = interpretacion.get("limitaciones", [])
    if limitaciones:
        for lim in limitaciones:
            c += f"- {lim}\n"
    else:
        c += "No se identificaron limitaciones.\n"

    c += "\n## 8. Recomendaciones Futuras\n\n"
    recomendaciones = interpretacion.get("recomendaciones_futuras", [])
    if recomendaciones:
        for rec in recomendaciones:
            c += f"- {rec}\n"
    else:
        c += "No se generaron recomendaciones.\n"

    c += "\n## 9. Desviaciones de Ejecución\n\n"
    if desviaciones:
        for d in desviaciones:
            c += f"- **{d['campo']}**: {d['razon']}\n"
    else:
        c += "Sin desviaciones: todo el plan se ejecutó tal como fue propuesto.\n"

    REPORTE_RESULTADOS.write_text(c, encoding="utf-8")


# ============================================================================
# PUNTO DE ENTRADA
# ============================================================================
def main():
    args = crear_parser("qwen3.5:9b").parse_args()
    agente = "Etapa 4"

    print(f"\n[{agente}] Iniciando entrenamiento con modelo: {args.modelo}")
    asegurar_directorios()

    # Verificar dependencias
    for archivo in [DATASET_EXPERIMENTAL, PLAN_EXPERIMENTAL, PROPUESTA_MODELOS,
                    METADATA_FILE, ESTADISTICAS_FILE]:
        if not archivo.exists():
            print(f"[{agente}] ERROR: No se encuentra {archivo}.")
            sys.exit(1)

    plan = leer_json(PLAN_EXPERIMENTAL)
    propuesta = leer_json(PROPUESTA_MODELOS)
    metadata = leer_json(METADATA_FILE)
    estadisticas = leer_json(ESTADISTICAS_FILE)

    objetivo = plan.get("variable_objetivo")
    print(f"[{agente}] Variable objetivo: {objetivo}")

    # Cargar dataset
    df = pd.read_csv(DATASET_EXPERIMENTAL, encoding="utf-8")
    print(f"[{agente}] Dataset cargado: {df.shape}")

    revision = leer_revision(args.revision)

    # Preparar datos
    print(f"[{agente}] Preparando datos y preprocesamiento...")
    (X_train, X_test, y_train_t, y_test_t, y_train_orig, y_test_orig,
     transf, lmbda, plan_ejecutado) = preparar_datos(df, objetivo, plan)
    print(f"  -> Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"  -> Transformación objetivo: {transf}")
    print(f"  -> Encoding: {plan_ejecutado['encoding']}, Escalado: {plan_ejecutado['escalado']}")

    # Entrenar modelos
    resultados = []
    desviaciones = []
    modelos_a_entrenar = plan.get("modelos_a_entrenar", [])

    for modelo_info in modelos_a_entrenar:
        nombre = modelo_info["nombre"]
        clase_nombre = modelo_info["clase_sklearn"]
        hiperparams = modelo_info.get("hiperparametros", {})

        print(f"[{agente}] Entrenando {nombre} ({clase_nombre})...")

        try:
            resultado = entrenar_modelo(
                nombre, clase_nombre, hiperparams,
                X_train, X_test, y_train_t, y_test_t, y_train_orig, y_test_orig,
                transf, lmbda
            )

            if "error" in resultado:
                desviaciones.append({"campo": f"modelo '{nombre}'", "razon": resultado["error"]})
                print(f"  [!] Error: {resultado['error']}")
                continue

            resultados.append({
                "nombre": nombre,
                "clase_sklearn": clase_nombre,
                "hiperparametros": hiperparams,
                "rol": modelo_info.get("rol", "modelo_comparacion"),
                "metricas_train": resultado["metricas_train"],
                "metricas_test": resultado["metricas_test"],
                "sobreajuste": resultado["sobreajuste"],
                "diferencia_r2": resultado["diferencia_r2"],
                "cv_rmse": resultado["cv_rmse"],
                "cv_std": resultado["cv_std"],
            })

            print(f"  -> Test RMSE: {resultado['metricas_test']['RMSE']:.4f}, "
                  f"R²: {resultado['metricas_test']['R2']:.4f}")

            joblib.dump(resultado["modelo"], MODELS_DIR / f"{nombre}.joblib")

        except Exception as e:
            desviaciones.append({"campo": f"modelo '{nombre}'",
                                 "razon": f"Error inesperado: {str(e)}"})
            print(f"  [!] Error inesperado: {e}")

    if not resultados:
        print(f"[{agente}] ERROR: Ningún modelo se entrenó correctamente.")
        sys.exit(1)

    # Generar gráficas
    print(f"[{agente}] Generando gráficas comparativas...")
    generar_graficas(resultados, objetivo)

    # Calcular hechos para el LLM (porcentaje de error calculado por Python)
    precio_promedio = float(df[objetivo].mean())
    precio_mediana = float(df[objetivo].median())

    hechos = redondear_estructura({
        "resultados_entrenamiento": [
            {
                "nombre": r["nombre"],
                "rol": r["rol"],
                "metricas_train": r["metricas_train"],
                "metricas_test": r["metricas_test"],
                "sobreajuste": r["sobreajuste"],
                "diferencia_r2": r["diferencia_r2"],
                "cv_rmse": r["cv_rmse"],
            }
            for r in resultados
        ],
        "linea_base": propuesta.get("linea_base", {}),
        "precio_promedio": precio_promedio,
        "precio_mediana": precio_mediana,
        "transformacion_objetivo": transf,
        "preprocesamiento_ejecutado": plan_ejecutado,
    })

    # CORREGIDO: antecedentes leen de la estructura correcta
    antecedentes = {
        "descripcion_dataset": metadata.get("descripcion_semantica", ""),
        "variable_objetivo": objetivo,
        "tarea_ml": metadata.get("definicion_problema", {}).get("tarea_ml", ""),
        "advertencias_previas": metadata.get("advertencias", []),
        "puntos_clave_etapa_2": estadisticas.get("interpretacion_llm", {}).get(
            "puntos_clave_para_modelado", []),
        "limitaciones_etapa_2": estadisticas.get("interpretacion_llm", {}).get(
            "limitaciones_estadisticas", []),
        "puntos_clave_etapa_3": propuesta.get("plan_normalizado", {}).get(
            "puntos_clave_para_modelado", []),
        "limitaciones_etapa_3": propuesta.get("plan_normalizado", {}).get(
            "limitaciones_identificadas", []),
        # CORREGIDO: el ruido ahora vive en estadisticas.json (Etapa 2)
        "ruido_artificial": estadisticas.get("ruido_artificial", {}),
    }

    # Interpretar con LLM
    print(f"[{agente}] Interpretando resultados con LLM...")
    llm = crear_llm(args.modelo, temperatura=0.2, num_predict=8192)
    interpretacion, validaciones, cifras_dudosas = interpretar_resultados(
        llm, hechos, antecedentes, revision, agente
    )

    if hay_error(interpretacion):
        print(f"[{agente}] ERROR: la interpretación falló: {interpretacion.get('error')}")
        sys.exit(1)

    for v in validaciones:
        print(f"  [VALIDACIÓN] {v}")
    if cifras_dudosas:
        print(f"  [CIFRAS NO RESPALDADAS] {cifras_dudosas}")

    # Guardar metricas.json (incluye trazabilidades internas)
    salida_metricas = {
        "etapa": "4_entrenamiento",
        "estado": "EXITO",
        "timestamp": ahora(),
        "variable_objetivo": objetivo,
        "modelo_utilizado": args.modelo,
        "revision_recibida": bool(revision),
        "transformacion_objetivo": transf,
        "preprocesamiento_ejecutado": plan_ejecutado,
        "resultados_por_modelo": {
            r["nombre"]: {
                "clase_sklearn": r["clase_sklearn"],
                "hiperparametros": r["hiperparametros"],
                "rol": r["rol"],
                "metricas_train": r["metricas_train"],
                "metricas_test": r["metricas_test"],
                "sobreajuste": r["sobreajuste"],
                "diferencia_r2": r["diferencia_r2"],
                "cv_rmse": r["cv_rmse"],
                "cv_std": r["cv_std"],
            }
            for r in resultados
        },
        "modelo_ganador": interpretacion.get("modelo_ganador"),
        "interpretacion_llm": interpretacion,
        "desviaciones_ejecucion": desviaciones,
        "validaciones_sistema": validaciones,
        "cifras_no_verificadas": cifras_dudosas,
    }
    guardar_json(METRICAS_FILE, salida_metricas)

    # Guardar comparacion_modelos.csv
    filas_csv = []
    for r in resultados:
        fila = {"modelo": r["nombre"], "clase": r["clase_sklearn"], "rol": r["rol"]}
        for metrica, valor in r["metricas_train"].items():
            fila[f"train_{metrica}"] = valor
        for metrica, valor in r["metricas_test"].items():
            fila[f"test_{metrica}"] = valor
        fila["sobreajuste"] = r["sobreajuste"]
        fila["diferencia_r2"] = r["diferencia_r2"]
        filas_csv.append(fila)

    df_comparacion = pd.DataFrame(filas_csv)
    df_comparacion.to_csv(COMPARACION_FILE, index=False, encoding="utf-8")

    # Generar reporte (narrativa limpia)
    generar_reporte(resultados, interpretacion, objetivo, propuesta.get("linea_base", {}),
                   plan_ejecutado, desviaciones)

    print(f"\n[{agente}] Completada. Archivos generados:")
    print(f"  -> {METRICAS_FILE}")
    print(f"  -> {COMPARACION_FILE}")
    print(f"  -> {REPORTE_RESULTADOS}")
    print(f"  -> {PLOTS_DIR}/")
    print(f"  -> {MODELS_DIR}/")


if __name__ == "__main__":
    main()