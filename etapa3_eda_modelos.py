"""
etapa3_eda_modelos.py - EDA profundo y diseño de estrategia experimental

Responsabilidades:
- Lee metadata.json, estadisticas.json y dataset_experimental.csv de las etapas anteriores.
- Calcula hechos sobre cardinalidad, distribución del objetivo, posibles fugas y línea base.
- Pide al LLM que diseñe la estrategia experimental completa: transformación del objetivo,
  encoding, escalado, modelos, métricas, validación y preprocesamiento.
- Valida que el plan sea ejecutable (las capacidades existen en el registro del sistema).
- Si el LLM propone algo no ejecutable o incompleto, se aplica un fallback conservador
  y la desviación queda REGISTRADA.
- Genera propuesta_modelos.json, plan_experimental.json y reporte_eda.md.

Principios de diseño:
- El LLM decide la estrategia; el código valida legalidad y ejecuta.
- Si el plan es incompleto (campos vacíos o N/A), la etapa falla y el supervisor decide.
- Si el plan propone capacidades inexistentes, se registra la desviación y se usa fallback.
- La línea base (DummyRegressor) se calcula con Python, no la decide el LLM.
- El reporte Markdown es narrativa de ciencia de datos;
  las trazabilidades internas (validaciones, cifras dudosas) viven solo en los JSON.

Uso:
    python etapa3_eda_modelos.py
    python etapa3_eda_modelos.py --modelo qwen3.5:9b
    python etapa3_eda_modelos.py --revision data/revision_etapa3_eda_modelos.json
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.dummy import DummyRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from comun import (PRINCIPIOS_DE_TRABAJO, REGLAS_DE_EVIDENCIA, USO_DE_ANTECEDENTES,
                   a_json, ahora, asegurar_directorios, bloque_antecedentes,
                   consultar_llm_json, crear_llm, crear_parser, guardar_json,
                   hay_error, leer_json, leer_revision,
                   normalizar_lista_texto, redondear_estructura, validar_opcion,
                   verificar_cifras)

# ============================================================================
# CONFIGURACIÓN
# ============================================================================
DATASET_EXPERIMENTAL = Path("data/dataset_experimental.csv")
METADATA_FILE = Path("data/metadata.json")
ESTADISTICAS_FILE = Path("data/estadisticas.json")
PROPUESTA_MODELOS = Path("data/propuesta_modelos.json")
PLAN_EXPERIMENTAL = Path("data/plan_experimental.json")
REPORTE_EDA = Path("data/reports/reporte_eda.md")
PLOTS_DIR = Path("data/plots")

SEMILLA = 42

# ============================================================================
# REGISTRO DE CAPACIDADES DEL SISTEMA
# Lo que el LLM puede proponer. Si propone algo fuera de este registro,
# se aplica fallback y se registra la desviación.
# ============================================================================
CAPACIDADES_TRANSFORMACION = {"ninguna", "log", "log1p", "sqrt", "box-cox"}
CAPACIDADES_ENCODING = {"one_hot", "frequency", "target", "top_n"}
CAPACIDADES_ESCALADO = {"standard", "minmax", "robust", "none"}
CAPACIDADES_MANEJO_NULOS = {"imputar_media", "imputar_mediana", "eliminar"}
CAPACIDADES_VALIDACION = {"train_test_split", "cross_validation"}
CAPACIDADES_ROL_MODELO = {"linea_base", "modelo_principal", "modelo_comparacion"}

MODELOS_DISPONIBLES = {
    "DummyRegressor": "sklearn.dummy.DummyRegressor",
    "LinearRegression": "sklearn.linear_model.LinearRegression",
    "Ridge": "sklearn.linear_model.Ridge",
    "Lasso": "sklearn.linear_model.Lasso",
    "ElasticNet": "sklearn.linear_model.ElasticNet",
    "RandomForestRegressor": "sklearn.ensemble.RandomForestRegressor",
    "GradientBoostingRegressor": "sklearn.ensemble.GradientBoostingRegressor",
    "HistGradientBoostingRegressor": "sklearn.ensemble.HistGradientBoostingRegressor",
}

FALLBACK_TRANSFORMACION = "ninguna"
FALLBACK_ENCODING = "frequency"
FALLBACK_ESCALADO = "standard"
FALLBACK_MANEJO_NULOS = "imputar_mediana"
FALLBACK_VALIDACION = {"tipo": "train_test_split", "test_size": 0.2, "n_splits": 5,
                       "justificacion": "Fallback conservador: split simple para dataset pequeño."}


# ============================================================================
# HECHOS CALCULADOS (Python, no LLM)
# ============================================================================
def analizar_cardinalidad(df: pd.DataFrame) -> dict:
    """Cardinalidad de columnas no numéricas."""
    resultado = {}
    columnas_cat = df.select_dtypes(exclude=[np.number, "datetime", "timedelta", "bool"]).columns
    for col in columnas_cat:
        n_unicos = int(df[col].nunique())
        total = len(df)
        vc = df[col].value_counts(dropna=False)
        resultado[col] = {
            "n_unicos": n_unicos,
            "porcentaje_unicos": round(100 * n_unicos / total, 2) if total else 0.0,
            "categoria_mas_frecuente": str(vc.index[0]) if len(vc) > 0 else None,
            "frecuencia_mas_frecuente": int(vc.iloc[0]) if len(vc) > 0 else 0,
            "nivel_cardinalidad": (
                "baja" if n_unicos <= 10 else
                "media" if n_unicos <= 50 else
                "alta"
            ),
        }
    return resultado


def analizar_distribucion_objetivo(df: pd.DataFrame, objetivo: str) -> dict:
    """Distribución de la variable objetivo."""
    valores = df[objetivo].dropna()
    if len(valores) == 0:
        return {}
    skew = float(valores.skew())
    kurt = float(valores.kurtosis())
    media = float(valores.mean())
    return {
        "n": int(len(valores)),
        "media": media,
        "mediana": float(valores.median()),
        "std": float(valores.std()),
        "min": float(valores.min()),
        "max": float(valores.max()),
        "q05": float(valores.quantile(0.05)),
        "q95": float(valores.quantile(0.95)),
        "skewness": skew,
        "kurtosis": kurt,
        "es_simetrica": abs(skew) < 0.5,
        "tiene_cola_pesada": kurt > 3,
        "rango": float(valores.max() - valores.min()),
        "coeficiente_variacion": float(valores.std() / media) if media != 0 else None,
    }


def detectar_posibles_fugas(df: pd.DataFrame, objetivo: str) -> list:
    """Correlaciones extremas con el objetivo que podrían indicar fuga."""
    fugas = []
    columnas_numericas = df.select_dtypes(include=[np.number]).columns
    for col in columnas_numericas:
        if col == objetivo:
            continue
        r = df[col].corr(df[objetivo])
        if np.isnan(r):
            continue
        if abs(r) > 0.95:
            fugas.append({
                "columna": col,
                "correlacion": float(r),
                "riesgo": "alto",
                "advertencia": "Correlación extremadamente alta. Verificar si es variable causal o derivada del objetivo.",
            })
        elif abs(r) > 0.8:
            fugas.append({
                "columna": col,
                "correlacion": float(r),
                "riesgo": "medio",
                "advertencia": "Correlación muy alta. Revisar si es aceptable.",
            })
    return fugas


def calcular_linea_base(df: pd.DataFrame, objetivo: str) -> dict:
    """Línea base con DummyRegressor (mediana)."""
    X = df.drop(columns=[objetivo])
    y = df[objetivo]
    X_num = X.select_dtypes(include=[np.number]).fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(
        X_num, y, test_size=0.2, random_state=SEMILLA
    )
    modelo = DummyRegressor(strategy="median")
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    return {
        "modelo": "DummyRegressor",
        "estrategia": "median",
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "r2": float(r2_score(y_test, y_pred)),
    }


# ============================================================================
# GRÁFICAS EDA
# ============================================================================
def generar_graficas_eda(df: pd.DataFrame, objetivo: str):
    """Gráficas específicas del EDA."""
    plt.style.use("seaborn-v0_8-whitegrid")

    # Distribución con KDE
    fig, ax = plt.subplots(figsize=(8, 5))
    df[objetivo].hist(bins=30, ax=ax, color="steelblue", edgecolor="black",
                      alpha=0.7, density=True)
    df[objetivo].plot(kind="kde", ax=ax, color="red", linewidth=2)
    ax.set_xlabel(objetivo)
    ax.set_ylabel("Densidad")
    ax.set_title(f"Distribución de {objetivo} con KDE")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "eda_distribucion_kde.png", dpi=100)
    plt.close(fig)

    # Boxplot por primera columna binaria/categórica de baja cardinalidad
    columnas_cat = [col for col in df.select_dtypes(exclude=[np.number]).columns
                    if df[col].nunique() <= 10]
    if columnas_cat and objetivo in df.columns:
        col_grupo = columnas_cat[0]
        fig, ax = plt.subplots(figsize=(8, 5))
        df.boxplot(column=objetivo, by=col_grupo, ax=ax)
        ax.set_title(f"{objetivo} por {col_grupo}")
        ax.set_xlabel(col_grupo)
        ax.set_ylabel(objetivo)
        fig.suptitle("")
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"eda_{objetivo}_por_{col_grupo}.png", dpi=100)
        plt.close(fig)

    # Top 10 de la columna categórica con mayor cardinalidad
    columnas_alta = [col for col in df.select_dtypes(exclude=[np.number]).columns
                     if df[col].nunique() > 10]
    if columnas_alta and objetivo in df.columns:
        col_top = columnas_alta[0]
        top = df.groupby(col_top)[objetivo].mean().nlargest(10)
        fig, ax = plt.subplots(figsize=(10, 6))
        top.plot(kind="barh", ax=ax, color="steelblue")
        ax.set_xlabel(f"Promedio de {objetivo}")
        ax.set_ylabel(col_top)
        ax.set_title(f"Top 10 {col_top} por {objetivo} promedio")
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"eda_top_{col_top}.png", dpi=100)
        plt.close(fig)


# ============================================================================
# PROMPT DE PLANIFICACIÓN
# ============================================================================
SYSTEM_PLAN = (
    PRINCIPIOS_DE_TRABAJO + USO_DE_ANTECEDENTES + REGLAS_DE_EVIDENCIA + """

Eres un científico de datos senior. Diseñas una estrategia experimental completa que un
executor de Python ejecutará sin ambigüedad. Las capacidades disponibles son:

- Transformación del objetivo: ninguna, log, log1p, sqrt, box-cox
- Encoding: one_hot, frequency, target, top_n
- Escalado: standard, minmax, robust, none
- Manejo de nulos: imputar_media, imputar_mediana, eliminar
- Validación: train_test_split, cross_validation
- Modelos: DummyRegressor, LinearRegression, Ridge, Lasso, ElasticNet,
  RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
- Métricas: RMSE, MAE, R2, MAPE, MedAE

REGLA CRÍTICA DE ENCODING: Si (filas / columnas_resultantes) < 5, el modelo sobreajustará.
Para columnas con >20 únicos, prefiere frequency, target o top_n sobre one_hot.

DECIDES TÚ (basándote en los hechos que recibes):
- Transformación del objetivo según skewness/kurtosis
- Encoding según cardinalidad de cada columna
- Modelos con hiperparámetros concretos (números, no rangos)
- Métricas más informativas para ESTE problema
- Estrategia de validación que minimice fuga de información

Cada decisión debe citar hechos concretos (skewness, cardinalidad, etc.).

Responde SOLO con JSON:
{
  "transformacion_objetivo": "...",
  "justificacion_transformacion": "...",
  "modelos_propuestos": [
    {
      "nombre": "...",
      "clase_sklearn": "...",
      "justificacion": "...",
      "hiperparametros_iniciales": {"param": valor},
      "rol": "linea_base | modelo_principal | modelo_comparacion"
    }
  ],
  "metricas_evaluacion": [{"nombre": "...", "justificacion": "..."}],
  "estrategia_validacion": {
    "tipo": "...",
    "test_size": 0.2,
    "n_splits": 5,
    "justificacion": "..."
  },
  "preprocesamiento_recomendado": {
    "encoding_categoricas": "...",
    "escalado_numericas": "...",
    "manejo_nulos": "...",
    "justificacion": "..."
  },
  "puntos_clave_para_modelado": ["..."],
  "limitaciones_identificadas": ["..."],
  "confianza_general": "alta | media | baja"
}"""
)


# ============================================================================
# VALIDACIÓN DEL PLAN
# ============================================================================
def _numero(valor, defecto, minimo, maximo, tipo=float):
    """Convierte valor a tipo numérico con clamp entre minimo y maximo."""
    try:
        v = tipo(valor)
    except (TypeError, ValueError):
        return defecto
    return min(max(v, minimo), maximo)


def validar_plan(plan: dict) -> tuple:
    """Valida que el plan sea ejecutable. Devuelve (plan_normalizado, desviaciones)."""
    desviaciones = []

    # Transformación
    transf = plan.get("transformacion_objetivo")
    if not validar_opcion(transf, CAPACIDADES_TRANSFORMACION):
        desviaciones.append({
            "campo": "transformacion_objetivo",
            "propuesto": transf,
            "ejecutado": FALLBACK_TRANSFORMACION,
            "razon": f"'{transf}' no está en capacidades disponibles",
        })
        transf = FALLBACK_TRANSFORMACION

    # Preprocesamiento
    preproc = plan.get("preprocesamiento_recomendado") or {}

    encoding = preproc.get("encoding_categoricas")
    if not validar_opcion(encoding, CAPACIDADES_ENCODING):
        desviaciones.append({
            "campo": "preprocesamiento.encoding_categoricas",
            "propuesto": encoding,
            "ejecutado": FALLBACK_ENCODING,
            "razon": f"'{encoding}' no está en capacidades disponibles",
        })
        encoding = FALLBACK_ENCODING

    escalado = preproc.get("escalado_numericas")
    if not validar_opcion(escalado, CAPACIDADES_ESCALADO):
        desviaciones.append({
            "campo": "preprocesamiento.escalado_numericas",
            "propuesto": escalado,
            "ejecutado": FALLBACK_ESCALADO,
            "razon": f"'{escalado}' no está en capacidades disponibles",
        })
        escalado = FALLBACK_ESCALADO

    manejo_nulos = preproc.get("manejo_nulos")
    if not validar_opcion(manejo_nulos, CAPACIDADES_MANEJO_NULOS):
        desviaciones.append({
            "campo": "preprocesamiento.manejo_nulos",
            "propuesto": manejo_nulos,
            "ejecutado": FALLBACK_MANEJO_NULOS,
            "razon": f"'{manejo_nulos}' no está en capacidades disponibles",
        })
        manejo_nulos = FALLBACK_MANEJO_NULOS

    # Estrategia de validación
    val = plan.get("estrategia_validacion") or {}
    if not validar_opcion(val.get("tipo"), CAPACIDADES_VALIDACION):
        desviaciones.append({
            "campo": "estrategia_validacion.tipo",
            "propuesto": val.get("tipo"),
            "ejecutado": FALLBACK_VALIDACION["tipo"],
            "razon": f"'{val.get('tipo')}' no está en capacidades disponibles",
        })
        val = dict(FALLBACK_VALIDACION)
    else:
        val = {
            "tipo": val.get("tipo"),
            "test_size": _numero(val.get("test_size"), 0.2, 0.0, 0.5),
            "n_splits": _numero(val.get("n_splits"), 5, 2, 10, int),
            "justificacion": str(val.get("justificacion", "")),
        }

    # Modelos
    modelos_raw = plan.get("modelos_propuestos") or []
    if not isinstance(modelos_raw, list):
        modelos_raw = [modelos_raw]

    modelos_validos = []
    for m in modelos_raw:
        if not isinstance(m, dict):
            continue
        clase = m.get("clase_sklearn")
        nombre = m.get("nombre", clase)
        rol = m.get("rol")
        hiperparams = m.get("hiperparametros_iniciales") or {}

        if clase not in MODELOS_DISPONIBLES:
            desviaciones.append({
                "campo": f"modelo '{nombre}'.clase_sklearn",
                "propuesto": clase,
                "ejecutado": None,
                "razon": f"'{clase}' no está en capacidades disponibles; modelo descartado",
            })
            continue

        if not validar_opcion(rol, CAPACIDADES_ROL_MODELO):
            desviaciones.append({
                "campo": f"modelo '{nombre}'.rol",
                "propuesto": rol,
                "ejecutado": "modelo_comparacion",
                "razon": f"rol inválido, se asigna 'modelo_comparacion'",
            })
            rol = "modelo_comparacion"

        # Asegurar que hiperparámetros sean valores simples
        hiperparams_limpios = {}
        for k, v in hiperparams.items():
            if isinstance(v, (int, float, np.integer, np.floating)):
                hiperparams_limpios[str(k)] = float(v) if isinstance(v, (np.floating, float)) else int(v)
            else:
                hiperparams_limpios[str(k)] = v

        modelos_validos.append({
            "nombre": str(nombre),
            "clase_sklearn": clase,
            "justificacion": str(m.get("justificacion", "")),
            "hiperparametros_iniciales": hiperparams_limpios,
            "rol": rol,
        })

    # Métricas
    metricas_raw = plan.get("metricas_evaluacion") or []
    if not isinstance(metricas_raw, list):
        metricas_raw = [metricas_raw]
    metricas_validas = []
    for met in metricas_raw:
        if isinstance(met, dict):
            metricas_validas.append({
                "nombre": str(met.get("nombre", "")),
                "justificacion": str(met.get("justificacion", "")),
            })
        elif isinstance(met, str):
            metricas_validas.append({"nombre": met, "justificacion": ""})

    plan_normalizado = {
        "transformacion_objetivo": transf,
        "justificacion_transformacion": str(plan.get("justificacion_transformacion", "")),
        "modelos_propuestos": modelos_validos,
        "metricas_evaluacion": metricas_validas,
        "estrategia_validacion": val,
        "preprocesamiento_recomendado": {
            "encoding_categoricas": encoding,
            "escalado_numericas": escalado,
            "manejo_nulos": manejo_nulos,
            "justificacion": str(preproc.get("justificacion", "")),
        },
        "puntos_clave_para_modelado": normalizar_lista_texto(
            plan.get("puntos_clave_para_modelado")),
        "limitaciones_identificadas": normalizar_lista_texto(
            plan.get("limitaciones_identificadas")),
        "confianza_general": str(plan.get("confianza_general", "")),
    }
    return plan_normalizado, desviaciones


# ============================================================================
# PLANIFICACIÓN CON LLM
# ============================================================================
def planificar_estrategia(llm, hechos: dict, antecedentes: dict, revision: str,
                          agente: str) -> tuple:
    """Devuelve (respuesta_llm, plan_normalizado, desviaciones, validaciones, cifras_dudosas)."""
    print(f"[{agente}] Generando plan experimental...")
    user_prompt = (
        bloque_antecedentes(hechos, antecedentes) +
        f"\n\nDiseña la estrategia experimental completa.\n{revision}"
    )
    respuesta = consultar_llm_json(
        llm, SYSTEM_PLAN, user_prompt, agente=agente,
        claves_requeridas=["transformacion_objetivo", "modelos_propuestos",
                           "preprocesamiento_recomendado"]
    )

    if hay_error(respuesta):
        return respuesta, {}, [], [], []

    plan_normalizado, desviaciones = validar_plan(respuesta)
    cifras_dudosas = verificar_cifras(respuesta, hechos)

    validaciones = []
    if not plan_normalizado["modelos_propuestos"]:
        validaciones.append("El plan no contiene modelos ejecutables tras la validación")
    if not plan_normalizado["metricas_evaluacion"]:
        validaciones.append("El plan no contiene métricas de evaluación")
    if desviaciones:
        validaciones.append(
            f"{len(desviaciones)} desviaciones respecto a lo propuesto por el LLM"
        )

    return respuesta, plan_normalizado, desviaciones, validaciones, cifras_dudosas


# ============================================================================
# REPORTE (narrativa limpia, sin trazabilidades internas)
# ============================================================================
def generar_reporte(hechos: dict, plan: dict, desviaciones: list, objetivo: str):
    """Genera el informe Markdown de la Etapa 3."""
    cardinalidad = hechos.get("cardinalidad", {})
    distribucion = hechos.get("distribucion_objetivo", {})
    fugas = hechos.get("fugas_detectadas", [])
    linea_base = hechos.get("linea_base", {})

    cv = distribucion.get("coeficiente_variacion")
    cv_str = f"{cv:.3f}" if cv is not None else "N/A"

    c = f"""# Reporte EDA y Selección de Modelos - Etapa 3

**Fecha:** {ahora()}
**Variable objetivo:** {objetivo}

## 1. Análisis de Cardinalidad

| Columna | Únicos | % Únicos | Nivel | Categoría más frecuente |
|---------|--------|----------|-------|-------------------------|
"""
    for col, info in cardinalidad.items():
        c += (f"| {col} | {info['n_unicos']} | {info['porcentaje_unicos']:.1f}% | "
              f"{info['nivel_cardinalidad']} | {info['categoria_mas_frecuente']} |\n")

    c += f"""
## 2. Distribución de la Variable Objetivo

- **Media:** {distribucion.get('media', 0):.2f}
- **Mediana:** {distribucion.get('mediana', 0):.2f}
- **Desv. Est.:** {distribucion.get('std', 0):.2f}
- **Mín / Máx:** {distribucion.get('min', 0):.2f} / {distribucion.get('max', 0):.2f}
- **Skewness:** {distribucion.get('skewness', 0):.3f} (simétrica: {'Sí' if distribucion.get('es_simetrica') else 'No'})
- **Kurtosis:** {distribucion.get('kurtosis', 0):.3f} (cola pesada: {'Sí' if distribucion.get('tiene_cola_pesada') else 'No'})
- **Coeficiente de variación:** {cv_str}

![Distribución KDE](../plots/eda_distribucion_kde.png)

## 3. Detección de Fugas de Información

"""
    if fugas:
        for fuga in fugas:
            c += (f"- **{fuga['columna']}** (r={fuga['correlacion']:.3f}, "
                  f"riesgo {fuga['riesgo']}): {fuga['advertencia']}\n")
    else:
        c += "No se detectaron fugas de información evidentes.\n"

    c += f"""
## 4. Línea Base (calculada por Python)

- **Modelo:** {linea_base.get('modelo')} (estrategia: {linea_base.get('estrategia')})
- **RMSE:** {linea_base.get('rmse', 0):.4f}
- **MAE:** {linea_base.get('mae', 0):.4f}
- **R²:** {linea_base.get('r2', 0):.4f}

## 5. Estrategia Experimental Propuesta

### Transformación del Objetivo
- **Transformación:** {plan.get('transformacion_objetivo')}
- **Justificación:** {plan.get('justificacion_transformacion', 'No disponible')}

### Modelos Propuestos

"""
    for modelo in plan.get("modelos_propuestos", []):
        c += f"**{modelo['nombre']}** ({modelo['rol']})\n"
        c += f"- Clase sklearn: `{modelo['clase_sklearn']}`\n"
        c += f"- Justificación: {modelo['justificacion']}\n"
        c += f"- Hiperparámetros: {modelo['hiperparametros_iniciales']}\n\n"

    c += "### Métricas de Evaluación\n\n"
    for met in plan.get("metricas_evaluacion", []):
        c += f"- **{met['nombre']}**: {met['justificacion']}\n"

    val = plan.get("estrategia_validacion", {})
    c += f"""
### Estrategia de Validación

- **Tipo:** {val.get('tipo')}
- **Test size:** {val.get('test_size')}
- **N splits:** {val.get('n_splits')}
- **Justificación:** {val.get('justificacion')}

### Preprocesamiento Recomendado

"""
    preproc = plan.get("preprocesamiento_recomendado", {})
    c += f"- **Encoding categóricas:** {preproc.get('encoding_categoricas')}\n"
    c += f"- **Escalado numéricas:** {preproc.get('escalado_numericas')}\n"
    c += f"- **Manejo de nulos:** {preproc.get('manejo_nulos')}\n"
    c += f"- **Justificación:** {preproc.get('justificacion')}\n"

    c += "\n## 6. Puntos Clave para Modelado\n\n"
    puntos = plan.get("puntos_clave_para_modelado", [])
    if puntos:
        for punto in puntos:
            c += f"- {punto}\n"
    else:
        c += "No se identificaron puntos clave.\n"

    c += "\n## 7. Limitaciones Identificadas\n\n"
    limitaciones = plan.get("limitaciones_identificadas", [])
    if limitaciones:
        for lim in limitaciones:
            c += f"- {lim}\n"
    else:
        c += "No se identificaron limitaciones.\n"

    c += "\n## 8. Desviaciones de Validación\n\n"
    if desviaciones:
        for d in desviaciones:
            c += (f"- **{d['campo']}**: propuesto `{d['propuesto']}` → "
                  f"ejecutado `{d['ejecutado']}`. Razón: {d['razon']}\n")
    else:
        c += "Sin desviaciones: todo lo propuesto por el LLM es ejecutable.\n"

    c += "\n## 9. Gráficas EDA\n\n"
    c += "- [Distribución KDE](../plots/eda_distribucion_kde.png)\n"
    for f_grafica in PLOTS_DIR.glob("eda_*.png"):
        if f_grafica.name != "eda_distribucion_kde.png":
            c += f"- [{f_grafica.stem}](../plots/{f_grafica.name})\n"

    REPORTE_EDA.write_text(c, encoding="utf-8")


# ============================================================================
# PUNTO DE ENTRADA
# ============================================================================
def main():
    args = crear_parser("qwen3.5:9b").parse_args()
    agente = "Etapa 3"

    print(f"\n[{agente}] Iniciando EDA y selección de modelos con modelo: {args.modelo}")
    asegurar_directorios()

    # Verificar dependencias
    for archivo in [METADATA_FILE, ESTADISTICAS_FILE, DATASET_EXPERIMENTAL]:
        if not archivo.exists():
            print(f"[{agente}] ERROR: No se encuentra {archivo}. Ejecuta etapas anteriores.")
            sys.exit(1)

    metadata = leer_json(METADATA_FILE)
    estadisticas = leer_json(ESTADISTICAS_FILE)

    # CORREGIDO: leer de definicion_problema (nueva estructura de Etapa 1)
    objetivo = metadata.get("definicion_problema", {}).get("variable_objetivo")
    if not objetivo:
        print(f"[{agente}] ERROR: metadata.json no contiene 'variable_objetivo'.")
        sys.exit(1)

    print(f"[{agente}] Variable objetivo: {objetivo}")

    # Cargar dataset
    df = pd.read_csv(DATASET_EXPERIMENTAL, encoding="utf-8")
    print(f"[{agente}] Dataset cargado: {df.shape}")

    revision = leer_revision(args.revision)

    # Calcular hechos
    print(f"[{agente}] Analizando cardinalidad...")
    cardinalidad = analizar_cardinalidad(df)

    print(f"[{agente}] Analizando distribución del objetivo...")
    distribucion = analizar_distribucion_objetivo(df, objetivo)

    print(f"[{agente}] Detectando posibles fugas...")
    fugas = detectar_posibles_fugas(df, objetivo)

    print(f"[{agente}] Calculando línea base...")
    linea_base = calcular_linea_base(df, objetivo)
    print(f"  -> RMSE línea base: {linea_base['rmse']:.4f}")

    # Hechos para el LLM (redondeados y limitados para no saturar el contexto)
    hechos = redondear_estructura({
        "columnas_dataset": list(df.columns),
        "n_filas": len(df),
        "cardinalidad": cardinalidad,
        "distribucion_objetivo": distribucion,
        "fugas_detectadas": fugas,
        "linea_base": linea_base,
        "estadisticas_previas": {
            "descriptivas": estadisticas.get("estadisticas_descriptivas", {}),
            "outliers": estadisticas.get("outliers", {}),
            "pares_correlacion_interesantes": estadisticas.get(
                "correlaciones", {}).get("pares_interesantes", [])[:10],
        },
    })

    # Generar gráficas
    print(f"[{agente}] Generando gráficas EDA...")
    generar_graficas_eda(df, objetivo)

    # Planificar con LLM
    print(f"[{agente}] Diseñando estrategia experimental con LLM...")
    llm = crear_llm(args.modelo, temperatura=0.2)

    # CORREGIDO: antecedentes leen de la estructura correcta
    antecedentes = {
        "descripcion_dataset": metadata.get("descripcion_semantica", ""),
        "variable_objetivo": objetivo,
        "tarea_ml_recomendada": metadata.get("definicion_problema", {}).get("tarea_ml", ""),
        "lineas_analisis_sugeridas": metadata.get("lineas_de_analisis_sugeridas", []),
        "advertencias_previas": metadata.get("advertencias", []),
        # CORREGIDO: el ruido ahora vive en estadisticas.json (Etapa 2)
        "ruido_artificial": estadisticas.get("ruido_artificial", {}),
        "puntos_clave_etapa_2": estadisticas.get("interpretacion_llm", {}).get(
            "puntos_clave_para_modelado", []),
        "limitaciones_etapa_2": estadisticas.get("interpretacion_llm", {}).get(
            "limitaciones_estadisticas", []),
    }

    respuesta, plan, desviaciones, validaciones, cifras_dudosas = planificar_estrategia(
        llm, hechos, antecedentes, revision, agente
    )

    if hay_error(respuesta):
        print(f"[{agente}] ERROR: la planificación falló: {respuesta.get('error')}")
        sys.exit(1)

    for v in validaciones:
        print(f"  [VALIDACIÓN] {v}")
    if desviaciones:
        print(f"  [DESVIACIONES] {len(desviaciones)} ajustes respecto a lo propuesto por el LLM")
    if cifras_dudosas:
        print(f"  [CIFRAS NO RESPALDADAS] {cifras_dudosas}")

    # Guardar propuesta_modelos.json (incluye trazabilidades internas)
    propuesta_salida = {
        "etapa": "3_eda_modelos",
        "estado": "EXITO",
        "timestamp": ahora(),
        "variable_objetivo": objetivo,
        "modelo_utilizado": args.modelo,
        "revision_recibida": bool(revision),
        "respuesta_llm_cruda": respuesta,
        "plan_normalizado": plan,
        "desviaciones_validacion": desviaciones,
        "linea_base": linea_base,
        "validaciones_sistema": validaciones,
        "cifras_no_verificadas": cifras_dudosas,
    }
    guardar_json(PROPUESTA_MODELOS, propuesta_salida)

    # Guardar plan_experimental.json (lo que ejecutará la Etapa 4)
    plan_exp = {
        "etapa": "3_plan_experimental",
        "estado": "EXITO",
        "timestamp": ahora(),
        "variable_objetivo": objetivo,
        "semilla": SEMILLA,
        "transformacion_objetivo": plan["transformacion_objetivo"],
        "preprocesamiento": plan["preprocesamiento_recomendado"],
        "estrategia_validacion": plan["estrategia_validacion"],
        "metricas_evaluacion": plan["metricas_evaluacion"],
        "modelos_a_entrenar": [
            {"nombre": m["nombre"], "clase_sklearn": m["clase_sklearn"],
             "hiperparametros": m["hiperparametros_iniciales"], "rol": m["rol"]}
            for m in plan["modelos_propuestos"]
        ],
    }
    guardar_json(PLAN_EXPERIMENTAL, plan_exp)

    # Generar reporte (narrativa limpia)
    generar_reporte(hechos, plan, desviaciones, objetivo)

    print(f"\n[{agente}] Completada. Archivos generados:")
    print(f"  -> {PROPUESTA_MODELOS}")
    print(f"  -> {PLAN_EXPERIMENTAL}")
    print(f"  -> {REPORTE_EDA}")
    print(f"  -> {PLOTS_DIR}/")


if __name__ == "__main__":
    main()