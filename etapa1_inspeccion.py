"""
etapa1_inspeccion.py - Análisis y definición del problema

Responsabilidades:
- Inspecciona el dataset original (dimensiones, tipos, faltantes, duplicados).
- Pide al LLM que interprete el dataset y defina el problema de ML.
- Propone tratamiento para nulos, duplicados y valores corruptos.
- Propone qué columnas son elegibles para ruido artificial.
- Genera metadata.json (con propuestas) y reporte_inicial.md.

Principios de diseño:
- Esta etapa SOLO ANALIZA Y PROPONE. No modifica ningún dato.
- El LLM decide; el código valida coherencia y registra.
- La definición del problema es explícita, con candidatos evaluados.
- La procedencia del dataset se entrega como hecho, no como instrucción.
- El reporte Markdown es narrativa de ciencia de datos;
  las trazabilidades internas viven solo en metadata.json.

Uso:
    python etapa1_inspeccion.py
    python etapa1_inspeccion.py --modelo qwen2.5:7b
    python etapa1_inspeccion.py --revision data/revision_etapa1_inspeccion.json
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from comun import (PRINCIPIOS_DE_TRABAJO, REGLAS_DE_EVIDENCIA, _sin_acentos,
                   a_json, ahora, asegurar_directorios, consultar_llm_json,
                   crear_llm, crear_parser, guardar_json, hay_error,
                   leer_revision, normalizar_lista_texto, validar_opcion,
                   verificar_cifras)


# ============================================================================
# CONFIGURACIÓN
# ============================================================================
DATASET_ORIGINAL = Path("data/data.csv")
METADATA_FILE = Path("data/metadata.json")
REPORTE_INICIAL = Path("data/reports/reporte_inicial.md")

PROCEDENCIA = {
    "repositorio": "https://github.com/ethanjaredlee/ticketmaster-price-ml",
    "archivo": "data.csv",
    "licencia": "MIT",
    "autor": "ethanjaredlee",
}

TIPOS_SEMANTICOS = [
    "identificador", "categorica_nominal", "categorica_ordinal",
    "categorica_codificada_numerica", "binaria", "continua", "conteo",
    "fecha_o_componente_temporal", "texto_libre", "incierto",
]

ACCIONES_NULOS = {"imputar_mediana", "imputar_media", "imputar_moda",
                  "eliminar_filas", "eliminar_columnas", "ninguna"}
ACCIONES_DUPLICADOS = {"eliminar_duplicados", "conservar_todos"}
ACCIONES_CORRUPCION = {
    "imputar_mediana_global", "imputar_mediana_grupo", "eliminar_filas",
    "convertir_a_cero", "marcar_como_no_numerica", "ninguna",
}

CLAVES_SEMANTICA = [
    "descripcion_semantica", "significado_columnas", "tipos_semanticos",
    "razonamiento_objetivo", "variable_objetivo", "tarea_ml_recomendada",
]

UMBRAL_CASI_NUMERICA = 0.80


# ============================================================================
# HECHOS CALCULADOS POR PYTHON (específicos de esta etapa)
# ============================================================================
def es_columna_texto(serie: pd.Series) -> bool:
    return not (
        pd.api.types.is_numeric_dtype(serie)
        or pd.api.types.is_bool_dtype(serie)
        or pd.api.types.is_datetime64_any_dtype(serie)
    )

def _redondear(x, n=4):
    try:
        return None if pd.isna(x) else round(float(x), n)
    except (TypeError, ValueError):
        return None

def inspeccion_tecnica(df: pd.DataFrame) -> dict:
    return {
        "filas": int(len(df)),
        "columnas": int(len(df.columns)),
        "valores_nulos": int(df.isnull().sum().sum()),
        "filas_duplicadas": int(df.duplicated().sum()),
    }

def construir_perfil(df: pd.DataFrame) -> dict:
    perfil = {}
    for col in df.columns:
        s = df[col]
        n_nulos = int(s.isna().sum())
        info = {
            "tipo_dato": str(s.dtype),
            "n_nulos": n_nulos,
            "pct_nulos": round(100 * float(s.isna().mean()), 2),
            "valores_unicos": int(s.nunique()),
        }
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            info.update({
                "min": _redondear(s.min()),
                "max": _redondear(s.max()),
                "media": _redondear(s.mean()),
                "mediana": _redondear(s.median()),
                "std": _redondear(s.std()),
            })
        else:
            n_no_nulos = int(s.notna().sum())
            if n_no_nulos > 0:
                convertidos = pd.to_numeric(s, errors="coerce")
                frac = float(convertidos.notna().sum()) / n_no_nulos
                if frac >= 0.5:
                    raros = s[convertidos.isna() & s.notna()].astype(str)
                    info["fraccion_convertible_a_numero"] = round(frac, 4)
                    info["valores_no_numericos_ejemplo"] = raros.value_counts().head(5).to_dict()
            info["categorias_mas_frecuentes"] = s.value_counts(dropna=False).head(5).to_dict()
        perfil[col] = info
    return perfil

def perfilar_candidatas(df: pd.DataFrame) -> list:
    """Hechos sobre columnas que podrían ser variable objetivo. Sin recomendación."""
    candidatas = []
    for col in df.columns:
        s = df[col]
        n_unicos = s.nunique()
        es_id = (n_unicos == len(df)) or (n_unicos / len(df) > 0.95 and len(df) > 10)
        info = {
            "columna": col,
            "tipo_dato": str(s.dtype),
            "n_unicos": int(n_unicos),
            "pct_unicos": round(100 * n_unicos / len(df), 2),
            "pct_nulos": round(100 * s.isna().sum() / len(df), 2),
            "parece_identificador": es_id,
            "es_constante": n_unicos <= 1,
        }
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            info.update({
                "min": _redondear(s.min()),
                "max": _redondear(s.max()),
                "mediana": _redondear(s.median()),
                "std": _redondear(s.std()),
            })
        else:
            info["categorias_mas_frecuentes"] = s.value_counts(dropna=False).head(5).to_dict()
        candidatas.append(info)
    return candidatas

def detectar_duplicados(df: pd.DataFrame) -> dict:
    n_duplicados = int(df.duplicated().sum())
    if n_duplicados == 0:
        return {"duplicados": 0, "porcentaje": 0.0, "ejemplos": []}
    return {
        "duplicados": n_duplicados,
        "porcentaje": round(100 * n_duplicados / len(df), 2),
        "ejemplos": df[df.duplicated(keep=False)].head(3).to_dict(orient="records"),
    }

def detectar_columnas_casi_numericas(df: pd.DataFrame) -> list:
    sospechosas = []
    for col in df.columns:
        s = df[col]
        if not es_columna_texto(s):
            continue
        n_no_nulos = int(s.notna().sum())
        if n_no_nulos == 0:
            continue
        frac = float(pd.to_numeric(s, errors="coerce").notna().sum()) / n_no_nulos
        if UMBRAL_CASI_NUMERICA <= frac < 1.0:
            sospechosas.append(col)
    return sospechosas

def diagnosticar_columna_corrupta(df: pd.DataFrame, col: str) -> dict:
    serie_num = pd.to_numeric(df[col], errors="coerce")
    mascara = serie_num.isna() & df[col].notna()
    n_afectadas = int(mascara.sum())
    if n_afectadas == 0:
        return None
    validos = serie_num.dropna()
    return {
        "columna": col,
        "valores_no_numericos_y_conteo": df[col][mascara].astype(str).value_counts().head(10).to_dict(),
        "filas_afectadas": n_afectadas,
        "pct_filas_afectadas": round(100 * n_afectadas / len(df), 2),
        "estadisticas_valores_validos": {
            "n_validos": int(len(validos)),
            "min": _redondear(validos.min()),
            "max": _redondear(validos.max()),
            "mediana": _redondear(validos.median()),
        },
    }

def detectar_valores_extranos(df: pd.DataFrame) -> dict:
    resultados = {}
    patrones = ["???", "N/A", "NA", "?", "null", "None"]
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        valores_str = df[col].astype(str)
        encontrados = {}
        for patron in patrones:
            mascara = valores_str.str.strip() == patron
            if mascara.any():
                encontrados[patron] = int(mascara.sum())
        if encontrados:
            resultados[col] = {
                "valores_extranos": encontrados,
                "total_extranos": sum(encontrados.values()),
                "porcentaje": round(100 * sum(encontrados.values()) / len(df), 2),
            }
    return resultados


# ============================================================================
# PROMPTS
# ============================================================================
SYSTEM_SEMANTICA = PRINCIPIOS_DE_TRABAJO + REGLAS_DE_EVIDENCIA + """

TU FORMACIÓN: especialista en calidad y comprensión de datos. Examinas un dataset que no
conoces: entiendes qué representa, qué problema admite plantearse con él y qué cuidados exige.

DECIDES TÚ
- Qué representa cada columna y qué tipo de variable es REALMENTE, más allá de su tipo de dato.
  PRESTA ATENCIÓN A:
  * Columnas con solo 2 valores (0/1): son BINARIAS, no continuas.
  * Columnas con valores como "???", "N/A": son texto con problemas de calidad.
  * Meses (1-12), días (1-7): son ordinales temporales.
- Si existe un problema de predicción natural y cuál es la variable objetivo.
- Qué análisis estadísticos merecen hacerse según LO QUE VES en estos datos.

UN BUEN RESULTADO
- Específico a ESTE dataset: cada línea de análisis menciona columnas reales.
- Cada advertencia cita la columna y la evidencia (cifra).
- Las líneas de análisis sugeridas son DIFERENTES entre sí.

Los valores permitidos para "tipos_semanticos" son: __TIPOS__.

Responde SOLO con JSON:
{
  "descripcion_semantica": "2-3 oraciones sobre qué representa el dataset",
  "significado_columnas": {"<columna>": "significado breve"},
  "tipos_semanticos": {"<columna>": "<tipo>"},
  "razonamiento_objetivo": "candidatos considerados y por qué eliges uno (o ninguno)",
  "variable_objetivo": "<columna> o null",
  "tarea_ml_recomendada": "regresion, clasificacion, clustering u otra",
  "lineas_de_analisis_sugeridas": ["análisis 1", "análisis 2", "análisis 3"],
  "advertencias": ["columna: evidencia y riesgo"],
  "confianza_general": "alta | media | baja, con motivo"
}""".replace("__TIPOS__", ", ".join(TIPOS_SEMANTICOS))


SYSTEM_LIMPIEZA = PRINCIPIOS_DE_TRABAJO + REGLAS_DE_EVIDENCIA + """

TU FORMACIÓN: especialista en calidad de datos. Propón cómo tratar los problemas detectados.
NO ejecutas nada: solo propones.

REGLAS IMPORTANTES:
1. NO elimines columnas con alta cardinalidad (>100 únicos) solo por tener pocos nulos.
   Prefiere imputar en esos casos.
2. Si una columna tiene >30% de nulos O >30% de corruptos, SÍ considera eliminarla.
3. Para columnas BINARIAS (0/1), usa imputar_moda.
4. Si detectas "???", "N/A" en columnas que deberían ser numéricas, trátalos como corruptos.

ACCIONES DISPONIBLES PARA NULOS:
imputar_mediana, imputar_media, imputar_moda, eliminar_filas, eliminar_columnas, ninguna

ACCIONES DISPONIBLES PARA DUPLICADOS:
eliminar_duplicados, conservar_todos

ACCIONES DISPONIBLES PARA CORRUPTOS:
imputar_mediana_global, imputar_mediana_grupo, eliminar_filas, convertir_a_cero,
marcar_como_no_numerica, ninguna

Responde SOLO con JSON:
{
  "propuesta_nulos": {"accion": "...", "columnas_afectadas": [...], "justificacion": "..."},
  "propuesta_duplicados": {"accion": "...", "justificacion": "..."},
  "propuesta_corruptos": [
    {"columna": "...", "accion": "...", "columna_grupo": "o null", "justificacion": "..."}
  ],
  "confianza": "alta | media | baja"
}"""


SYSTEM_RUIDO = PRINCIPIOS_DE_TRABAJO + REGLAS_DE_EVIDENCIA + """

TU FORMACIÓN: especialista en calidad de datos. Se añadirá ruido gaussiano pequeño a columnas
numéricas. Decide cuáles tienen sentido perturbar y cuáles no.

REGLAS IMPORTANTES:
1. NO incluyas columnas BINARIAS (0/1) en "columnas_perturbar".
2. NO incluyas columnas ORDINALES discretas (mes 1-12) en "columnas_perturbar".
3. SÍ incluye columnas CONTINUAS que representan mediciones.
4. La variable objetivo está excluida automáticamente.

Responde SOLO con JSON:
{
  "columnas_perturbar": ["columnas CONTINUAS que SÍ deben recibir ruido"],
  "columnas_excluir": {"<columna>": "por qué no tiene sentido perturbarla"},
  "justificacion_general": "criterio usado",
  "confianza": "alta | media | baja"
}"""


# ============================================================================
# CONSULTAS AL LLM
# ============================================================================
def consultar_semantica(llm, df, revision, agente):
    hechos = {
        "procedencia": PROCEDENCIA,
        "dataset": {**inspeccion_tecnica(df), "nombres_columnas": list(df.columns)},
        "perfil_por_columna": construir_perfil(df),
        "candidatas_a_objetivo": perfilar_candidatas(df),
        "duplicados": detectar_duplicados(df),
        "muestra_aleatoria": df.sample(min(5, len(df)), random_state=42).to_dict(orient="records"),
    }
    user_prompt = (
        "DATASET (hechos calculados):\n" + a_json(hechos["dataset"]) +
        "\n\nPROCEDENCIA:\n" + a_json(hechos["procedencia"]) +
        "\n\nPERFIL POR COLUMNA:\n" + a_json(hechos["perfil_por_columna"]) +
        "\n\nCANDIDATAS A OBJETIVO:\n" + a_json(hechos["candidatas_a_objetivo"]) +
        "\n\nDUPLICADOS:\n" + a_json(hechos["duplicados"]) +
        "\n\nMUESTRA (5 filas):\n" + a_json(hechos["muestra_aleatoria"]) +
        "\n\nDescribe el dataset, define el problema y toma tus decisiones." + revision
    )
    return consultar_llm_json(llm, SYSTEM_SEMANTICA, user_prompt, agente=agente,
                              claves_requeridas=CLAVES_SEMANTICA), hechos

def consultar_limpieza(llm, df, analisis, revision, agente):
    casi_numericas = detectar_columnas_casi_numericas(df)
    corruptas = {}
    for col in casi_numericas:
        diag = diagnosticar_columna_corrupta(df, col)
        if diag:
            corruptas[col] = diag

    hechos = {
        "nulos_por_columna": {c: int(df[c].isna().sum()) for c in df.columns if df[c].isna().sum() > 0},
        "duplicados": detectar_duplicados(df),
        "columnas_casi_numericas_con_corrupcion": corruptas,
        "valores_extranos_detectados": detectar_valores_extranos(df),
    }
    notas = {
        "descripcion_semantica": analisis.get("descripcion_semantica", ""),
        "variable_objetivo": analisis.get("variable_objetivo"),
        "tipos_semanticos": analisis.get("tipos_semanticos", {}),
    }
    user_prompt = (
        "HECHOS SOBRE CALIDAD:\n" + a_json(hechos) +
        "\n\nCONTEXTO SEMÁNTICO:\n" + a_json(notas) +
        "\n\nPropón cómo tratar cada problema." + revision
    )
    return consultar_llm_json(llm, SYSTEM_LIMPIEZA, user_prompt, agente=agente,
                              claves_requeridas=["propuesta_nulos", "propuesta_duplicados"]), hechos

def consultar_ruido(llm, df, analisis, revision, agente):
    columnas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
    objetivo = analisis.get("variable_objetivo")
    hechos = {
        "columnas_numericas": columnas_numericas,
        "variable_objetivo": objetivo,
        "tipos_semanticos": analisis.get("tipos_semanticos", {}),
        "perfil_numericas": {
            col: {"n_unicos": int(df[col].nunique()), "std": _redondear(df[col].std()),
                  "min": _redondear(df[col].min()), "max": _redondear(df[col].max())}
            for col in columnas_numericas
        },
    }
    notas = {
        "descripcion_semantica": analisis.get("descripcion_semantica", ""),
        "significado_columnas": analisis.get("significado_columnas", {}),
    }
    user_prompt = (
        "HECHOS NUMÉRICOS:\n" + a_json(hechos) +
        "\n\nCONTEXTO SEMÁNTICO:\n" + a_json(notas) +
        "\n\nDecide qué columnas deben recibir ruido." + revision
    )
    return consultar_llm_json(llm, SYSTEM_RUIDO, user_prompt, agente=agente,
                              claves_requeridas=["columnas_perturbar", "columnas_excluir"]), hechos


# ============================================================================
# VALIDACIÓN Y NORMALIZACIÓN (específicas de esta etapa)
# ============================================================================
def validar_definicion(analisis: dict, df: pd.DataFrame) -> list:
    errores = []
    col = analisis.get("variable_objetivo")
    tarea = analisis.get("tarea_ml_recomendada")
    if tarea == "ninguna_viable":
        return []
    if col not in df.columns:
        errores.append(f"El objetivo '{col}' no existe en el dataset.")
        return errores
    serie = df[col]
    if tarea == "regresion" and not pd.api.types.is_numeric_dtype(serie):
        errores.append(f"'{col}' no es numérica: no es ejecutable como regresión.")
    if serie.nunique() <= 1:
        errores.append(f"'{col}' es constante.")
    if serie.nunique() == len(df) and pd.api.types.is_numeric_dtype(serie):
        errores.append(f"'{col}' parece identificador.")
    if tarea == "clasificacion" and not 2 <= serie.nunique() <= 50:
        errores.append(f"'{col}' tiene {serie.nunique()} categorías.")
    return errores

def normalizar_analisis(respuesta: dict, df: pd.DataFrame) -> tuple:
    validaciones = []
    columnas = list(df.columns)

    sig_raw = respuesta.get("significado_columnas", {})
    sig_raw = sig_raw if isinstance(sig_raw, dict) else {}
    significado, sin_describir = {}, []
    for c in columnas:
        if sig_raw.get(c):
            significado[c] = str(sig_raw[c])
        else:
            significado[c] = "no descrito por el modelo"
            sin_describir.append(c)
    if sin_describir:
        validaciones.append(f"Columnas sin describir: {sin_describir}")

    tip_raw = respuesta.get("tipos_semanticos", {})
    tip_raw = tip_raw if isinstance(tip_raw, dict) else {}
    tipos, tipos_invalidos = {}, []
    for c in columnas:
        t = tip_raw.get(c)
        if validar_opcion(t, TIPOS_SEMANTICOS):
            tipos[c] = t
        else:
            tipos[c] = "incierto"
            tipos_invalidos.append(c)
    if tipos_invalidos:
        validaciones.append(f"Tipos semánticos inválidos (asignado 'incierto'): {tipos_invalidos}")

    objetivo = respuesta.get("variable_objetivo")
    if not isinstance(objetivo, str) or objetivo.strip().lower() in ("", "null", "none"):
        objetivo = None
    elif objetivo not in columnas:
        validaciones.append(f"variable_objetivo '{objetivo}' no existe; descartada.")
        objetivo = None

    tarea_raw = str(respuesta.get("tarea_ml_recomendada") or "")
    tarea_txt = _sin_acentos(tarea_raw.lower())
    tarea = next((t for t in ("regresion", "clasificacion", "clustering") if t in tarea_txt), "otra")

    return {
        "descripcion_semantica": str(respuesta.get("descripcion_semantica", "")),
        "significado_columnas": significado,
        "tipos_semanticos": tipos,
        "razonamiento_objetivo": str(respuesta.get("razonamiento_objetivo", "")),
        "variable_objetivo": objetivo,
        "tarea_ml_recomendada": tarea,
        "tarea_ml_detalle": tarea_raw,
        "lineas_de_analisis_sugeridas": normalizar_lista_texto(respuesta.get("lineas_de_analisis_sugeridas")),
        "advertencias": normalizar_lista_texto(respuesta.get("advertencias")),
        "confianza_general": str(respuesta.get("confianza_general", "")),
    }, validaciones

def validar_propuesta_limpieza(respuesta: dict) -> tuple:
    desviaciones = []
    nulos = respuesta.get("propuesta_nulos", {})
    if not validar_opcion(nulos.get("accion", "ninguna"), ACCIONES_NULOS):
        desviaciones.append(f"accion nulos '{nulos.get('accion')}' no válida → 'ninguna'")
        nulos["accion"] = "ninguna"

    dups = respuesta.get("propuesta_duplicados", {})
    if not validar_opcion(dups.get("accion", "conservar_todos"), ACCIONES_DUPLICADOS):
        desviaciones.append(f"accion duplicados '{dups.get('accion')}' no válida → 'conservar_todos'")
        dups["accion"] = "conservar_todos"

    corruptos = respuesta.get("propuesta_corruptos", [])
    if not isinstance(corruptos, list):
        corruptos = [corruptos]
    for item in corruptos:
        if isinstance(item, dict) and not validar_opcion(item.get("accion", "ninguna"), ACCIONES_CORRUPCION):
            desviaciones.append(f"accion corruptos [{item.get('columna')}] no válida → 'ninguna'")
            item["accion"] = "ninguna"

    return {"propuesta_nulos": nulos, "propuesta_duplicados": dups, "propuesta_corruptos": corruptos}, desviaciones

def validar_propuesta_ruido(respuesta: dict, df: pd.DataFrame, objetivo) -> tuple:
    desviaciones = []
    columnas_numericas = set(df.select_dtypes(include=[np.number]).columns)
    perturbar = respuesta.get("columnas_perturbar", [])
    if not isinstance(perturbar, list):
        perturbar = [perturbar]
    perturbar_validas = []
    for col in perturbar:
        if col == objetivo:
            desviaciones.append(f"'{col}' es el objetivo; excluida automáticamente.")
        elif col not in columnas_numericas:
            desviaciones.append(f"'{col}' no es numérica; excluida.")
        else:
            perturbar_validas.append(col)
    excluir = respuesta.get("columnas_excluir", {})
    if not isinstance(excluir, dict):
        excluir = {}
    excluir_validas = {k: str(v) for k, v in excluir.items() if k in columnas_numericas}
    return {"columnas_perturbar": perturbar_validas, "columnas_excluir": excluir_validas,
            "justificacion_general": str(respuesta.get("justificacion_general", ""))}, desviaciones


# ============================================================================
# REPORTE MARKDOWN (narrativa limpia, sin trazabilidades internas)
# ============================================================================
def generar_reporte(df, analisis, propuesta_limpieza, propuesta_ruido, hechos_limpieza):
    tec = inspeccion_tecnica(df)
    tipos = analisis.get("tipos_semanticos", {})

    c = f"""# Reporte Inicial - Análisis y Definición del Problema

**Fecha:** {ahora()}
**Rol de esta etapa:** Análisis y propuesta (no ejecuta modificaciones)

## 1. Definición del Problema

**Variable objetivo:** {analisis.get('variable_objetivo', 'Ninguna')}
**Tarea recomendada:** {analisis.get('tarea_ml_recomendada', 'Ninguna')}
**Confianza:** {analisis.get('confianza_general', 'No indicada')}

**Razonamiento:**
{analisis.get('razonamiento_objetivo', 'No disponible')}

## 2. Descripción del Dataset

{analisis.get('descripcion_semantica', 'No disponible')}

**Procedencia:**
- Repositorio: {PROCEDENCIA['repositorio']}
- Licencia: {PROCEDENCIA['licencia']}

## 3. Inspección Técnica

| Métrica | Valor |
|---------|-------|
| Filas | {tec['filas']} |
| Columnas | {tec['columnas']} |
| Valores nulos | {tec['valores_nulos']} |
| Filas duplicadas | {tec['filas_duplicadas']} |

## 4. Significado y Tipo de las Columnas

"""
    for col, significado in analisis.get("significado_columnas", {}).items():
        c += f"- **{col}** ({tipos.get(col, 'incierto')}): {significado}\n"

    c += "\n## 5. Hallazgos de Calidad\n\n"
    nulos = hechos_limpieza.get("nulos_por_columna", {})
    if nulos:
        c += "### Valores Nulos\n\n| Columna | Nulos | % |\n|---------|-------|---|\n"
        for col, n in nulos.items():
            c += f"| {col} | {n} | {round(100*n/tec['filas'], 2)}% |\n"
        c += "\n"
    dups = hechos_limpieza.get("duplicados", {})
    c += f"### Filas Duplicadas\n\n- **Total:** {dups.get('duplicados', 0)} ({dups.get('porcentaje', 0)}%)\n\n"
    corruptas = hechos_limpieza.get("columnas_casi_numericas_con_corrupcion", {})
    if corruptas:
        c += "### Valores Corruptos\n\n"
        for col, diag in corruptas.items():
            c += f"**{col}**: {diag['filas_afectadas']} filas ({diag['pct_filas_afectadas']}%)\n"
            c += f"- Valores: {list(diag['valores_no_numericos_y_conteo'].keys())}\n\n"
    valores_ext = hechos_limpieza.get("valores_extranos_detectados", {})
    if valores_ext:
        c += "### Valores Extraños\n\n"
        for col, info in valores_ext.items():
            c += f"**{col}**: {info['total_extranos']} valores ({info['porcentaje']}%)\n"
            c += f"- Tipos: {info['valores_extranos']}\n\n"

    c += "\n## 6. Propuestas de Limpieza (para Etapa 2)\n\n"
    pn = propuesta_limpieza.get("propuesta_nulos", {})
    c += f"### Nulos\n- **Acción:** {pn.get('accion', 'ninguna')}\n- **Justificación:** {pn.get('justificacion', 'N/A')}\n\n"
    pd_ = propuesta_limpieza.get("propuesta_duplicados", {})
    c += f"### Duplicados\n- **Acción:** {pd_.get('accion', 'conservar_todos')}\n- **Justificación:** {pd_.get('justificacion', 'N/A')}\n\n"
    pc = propuesta_limpieza.get("propuesta_corruptos", [])
    if pc:
        c += "### Corruptos\n\n"
        for item in pc:
            if isinstance(item, dict):
                c += f"**{item.get('columna', 'N/A')}**\n- Acción: {item.get('accion', 'ninguna')}\n- Justificación: {item.get('justificacion', 'N/A')}\n\n"

    c += "\n## 7. Propuesta de Ruido Artificial (para Etapa 2)\n\n"
    pr = propuesta_ruido
    c += f"- **Columnas a perturbar:** {', '.join(pr.get('columnas_perturbar', [])) or 'ninguna'}\n"
    exc = pr.get("columnas_excluir", {})
    c += f"- **Columnas a excluir:**\n"
    if exc:
        for col, motivo in exc.items():
            c += f"  - {col}: {motivo}\n"
    else:
        c += "  - ninguna\n"
    c += f"- **Criterio:** {pr.get('justificacion_general', 'N/A')}\n"

    c += "\n## 8. Líneas de Análisis Sugeridas\n\n"
    for rec in analisis.get("lineas_de_analisis_sugeridas", []):
        c += f"- {rec}\n"

    c += "\n## 9. Advertencias\n\n"
    for adv in analisis.get("advertencias", []):
        c += f"- [ADVERTENCIA] {adv}\n"

    REPORTE_INICIAL.write_text(c, encoding="utf-8")


# ============================================================================
# PUNTO DE ENTRADA
# ============================================================================
def cargar_csv(ruta: Path) -> tuple:
    for cod in ["utf-8-sig", "latin-1", "cp1252"]:
        try:
            return pd.read_csv(ruta, encoding=cod), cod
        except UnicodeDecodeError:
            continue
    return None, None

def main():
    args = crear_parser("qwen2.5:7b").parse_args()
    agente = "Etapa 1"

    print(f"\n[{agente}] Iniciando análisis con modelo: {args.modelo}")
    asegurar_directorios()

    if not DATASET_ORIGINAL.exists():
        print(f"[{agente}] ERROR: No se encuentra {DATASET_ORIGINAL}")
        sys.exit(1)

    df, codificacion = cargar_csv(DATASET_ORIGINAL)
    if df is None:
        print(f"[{agente}] ERROR: No se pudo leer el CSV")
        sys.exit(1)
    print(f"[{agente}] Dataset cargado ({codificacion}): {df.shape}")

    revision = leer_revision(args.revision)
    if revision:
        print(f"[{agente}] Se recibieron observaciones de revisión.")

    llm_analisis = crear_llm(args.modelo, temperatura=0.1)
    llm_limpieza = crear_llm(args.modelo, temperatura=0.1)
    llm_ruido = crear_llm(args.modelo, temperatura=0.0)

    cifras_dudosas, validaciones = [], []

    # Consulta 1: Semántica + definición del problema
    print(f"[{agente}] Analizando semántica y definiendo el problema...")
    respuesta_sem, hechos_sem = consultar_semantica(llm_analisis, df, revision, agente)
    if hay_error(respuesta_sem):
        print(f"[{agente}] ERROR semántico: {respuesta_sem.get('error')}")
        sys.exit(1)
    dudosas = verificar_cifras(respuesta_sem, hechos_sem)
    if dudosas:
        cifras_dudosas.append(("analisis_semantico", dudosas))

    analisis, valid_sem = normalizar_analisis(respuesta_sem, df)
    validaciones.extend(valid_sem)

    valid_def = validar_definicion(analisis, df)
    validaciones.extend(valid_def)
    if valid_def:
        print(f"[{agente}] Definición del problema BLOQUEADA:")
        for v in valid_def:
            print(f"  [BLOQUEANTE] {v}")
        guardar_json(METADATA_FILE, {"etapa": "1_inspeccion", "estado": "BLOQUEADO",
                                     "timestamp": ahora(), "errores_definicion": valid_def,
                                     "analisis_parcial": analisis})
        sys.exit(1)

    print(f"  -> Tarea: {analisis['tarea_ml_recomendada']}")
    print(f"  -> Objetivo: {analisis['variable_objetivo']}")

    # Consulta 2: Propuesta de limpieza
    print(f"[{agente}] Generando propuesta de limpieza...")
    respuesta_lim, hechos_lim = consultar_limpieza(llm_limpieza, df, analisis, revision, agente)
    if hay_error(respuesta_lim):
        print(f"[{agente}] ERROR limpieza: {respuesta_lim.get('error')}")
        sys.exit(1)
    dudosas = verificar_cifras(respuesta_lim, hechos_lim)
    if dudosas:
        cifras_dudosas.append(("propuesta_limpieza", dudosas))
    propuesta_limpieza, desv_lim = validar_propuesta_limpieza(respuesta_lim)
    validaciones.extend(desv_lim)
    for d in desv_lim:
        print(f"  [DESVIACIÓN] {d}")

    # Consulta 3: Propuesta de ruido
    print(f"[{agente}] Generando propuesta de ruido...")
    respuesta_ruido, hechos_ruido = consultar_ruido(llm_ruido, df, analisis, revision, agente)
    if hay_error(respuesta_ruido):
        print(f"[{agente}] ERROR ruido: {respuesta_ruido.get('error')}")
        sys.exit(1)
    dudosas = verificar_cifras(respuesta_ruido, hechos_ruido)
    if dudosas:
        cifras_dudosas.append(("propuesta_ruido", dudosas))
    propuesta_ruido, desv_ruido = validar_propuesta_ruido(
        respuesta_ruido, df, analisis.get("variable_objetivo"))
    validaciones.extend(desv_ruido)
    for d in desv_ruido:
        print(f"  [DESVIACIÓN] {d}")

    # Guardar metadata.json (con trazabilidades internas)
    metadata = {
        "etapa": "1_inspeccion", "estado": "EXITO", "rol": "analisis_y_propuesta",
        "timestamp": ahora(), "dataset_original": str(DATASET_ORIGINAL),
        "codificacion_detectada": codificacion, "modelo_utilizado": args.modelo,
        "revision_recibida": bool(revision), "procedencia": PROCEDENCIA,
        "inspeccion_tecnica": inspeccion_tecnica(df),
        "definicion_problema": {
            "candidatos_evaluados": hechos_sem.get("candidatas_a_objetivo", []),
            "razonamiento_objetivo": analisis["razonamiento_objetivo"],
            "variable_objetivo": analisis["variable_objetivo"],
            "tarea_ml": analisis["tarea_ml_recomendada"],
            "tarea_ml_detalle": analisis["tarea_ml_detalle"],
            "confianza": analisis["confianza_general"],
        },
        "descripcion_semantica": analisis["descripcion_semantica"],
        "significado_columnas": analisis["significado_columnas"],
        "tipos_semanticos": analisis["tipos_semanticos"],
        "hallazgos_calidad": {
            "nulos_por_columna": hechos_lim.get("nulos_por_columna", {}),
            "duplicados": hechos_lim.get("duplicados", {}),
            "corruptos": hechos_lim.get("columnas_casi_numericas_con_corrupcion", {}),
            "valores_extranos": hechos_lim.get("valores_extranos_detectados", {}),
        },
        "propuestas_limpieza": propuesta_limpieza,
        "propuesta_ruido": propuesta_ruido,
        "lineas_de_analisis_sugeridas": analisis["lineas_de_analisis_sugeridas"],
        "advertencias": analisis["advertencias"],
        "validaciones_sistema": validaciones,
        "cifras_no_verificadas": [{"paso": p, "cifras": c} for p, c in cifras_dudosas],
    }
    guardar_json(METADATA_FILE, metadata)
    generar_reporte(df, analisis, propuesta_limpieza, propuesta_ruido, hechos_lim)

    print(f"\n[{agente}] Completada. Archivos generados:")
    print(f"  -> {METADATA_FILE}")
    print(f"  -> {REPORTE_INICIAL}")

if __name__ == "__main__":
    main()