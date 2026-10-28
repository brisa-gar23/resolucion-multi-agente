"""
etapa2_estadistica.py - Ejecución de limpieza, ruido y análisis estadístico

Responsabilidades:
- Lee metadata.json de la Etapa 1 (propuestas de limpieza y ruido).
- EJECUTA las propuestas: tratamiento de nulos, duplicados y valores corruptos.
- APLICA ruido artificial del 1% sobre las columnas propuestas.
- Genera dataset_experimental.csv, registro_limpieza.json y registro_ruido.json.
- Calcula estadísticas descriptivas, outliers, correlaciones y análisis por grupo.
- Pide al LLM que interprete los resultados.
- Genera estadisticas.json, reporte_estadistico.md y gráficas.

Principios de diseño:
- Esta etapa EJECUTA. Las decisiones fueron tomadas por la Etapa 1.
- Si una propuesta no es ejecutable, se aplica fallback y se registra la desviación.
- Python calcula TODAS las cifras; el LLM solo interpreta.
- La variable objetivo NUNCA recibe ruido ni se elimina.

Uso:
    python etapa2_estadistica.py
    python etapa2_estadistica.py --modelo qwen2.5:7b
    python etapa2_estadistica.py --revision data/revision_etapa2_estadistica.json
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from comun import (PRINCIPIOS_DE_TRABAJO, REGLAS_DE_EVIDENCIA, USO_DE_ANTECEDENTES,
                   a_json, ahora, asegurar_directorios, bloque_antecedentes,
                   consultar_llm_json, crear_llm, crear_parser, guardar_json,
                   hay_error, leer_json, leer_revision, normalizar_afirmaciones,
                   normalizar_lista_texto, redondear_estructura, validar_opcion, verificar_cifras)

# ============================================================================
# CONFIGURACIÓN
# ============================================================================
DATASET_ORIGINAL = Path("data/data.csv")
DATASET_EXPERIMENTAL = Path("data/dataset_experimental.csv")
METADATA_FILE = Path("data/metadata.json")
REGISTRO_LIMPIEZA = Path("data/registro_limpieza.json")
REGISTRO_RUIDO = Path("data/registro_ruido.json")
ESTADISTICAS_FILE = Path("data/estadisticas.json")
REPORTE_ESTADISTICO = Path("data/reports/reporte_estadistico.md")
PLOTS_DIR = Path("data/plots")

# Parámetros del ruido artificial
SEMILLA_RUIDO = 42
PORCENTAJE_RUIDO = 1.0  # % de celdas sobre las que se añade ruido
FACTOR_SIGMA_RUIDO = 0.05  # sigma = 0.05 * desviación estándar de la columna

# Límite de seguridad: no se ejecutan eliminaciones masivas
MAX_FRACCION_ELIMINABLE = 0.30

# Acciones ejecutables
ACCIONES_NULOS = {"imputar_mediana", "imputar_media", "imputar_moda",
                  "eliminar_filas", "eliminar_columnas", "ninguna"}
ACCIONES_DUPLICADOS = {"eliminar_duplicados", "conservar_todos"}
ACCIONES_CORRUPCION = {
    "imputar_mediana_global", "imputar_mediana_grupo", "eliminar_filas",
    "convertir_a_cero", "marcar_como_no_numerica", "ninguna",
}

# Fallbacks conservadores
FALLBACK_NULOS_NUMERICA = "imputar_mediana"
FALLBACK_NULOS_CATEGORICA = "imputar_moda"
FALLBACK_DUPLICADOS = "conservar_todos"
FALLBACK_CORRUPCION = "imputar_mediana_global"


# ============================================================================
# CARGA DEL DATASET
# ============================================================================
def cargar_csv(ruta: Path) -> tuple:
    """Prueba varias codificaciones."""
    for cod in ["utf-8-sig", "latin-1", "cp1252"]:
        try:
            return pd.read_csv(ruta, encoding=cod), cod
        except UnicodeDecodeError:
            continue
    return None, None


# ============================================================================
# 1. EJECUCIÓN DE LIMPIEZA
# ============================================================================
def ejecutar_limpieza_nulos(df: pd.DataFrame, propuesta: dict, columnas_protegidas: set) -> tuple:
    """Ejecuta la propuesta de tratamiento de nulos. Devuelve (df, registro)."""
    accion = propuesta.get("accion", "ninguna")
    columnas_afectadas = propuesta.get("columnas_afectadas", [])
    justificacion = propuesta.get("justificacion", "")

    registro = {
        "accion_propuesta": accion,
        "accion_ejecutada": accion,
        "fuente": "llm",
        "desviacion": None,
        "columnas_afectadas": columnas_afectadas,
        "justificacion_llm": justificacion,
        "detalle": {},
    }

    # Validar acción
    if not validar_opcion(accion, ACCIONES_NULOS):
        registro["accion_ejecutada"] = "ninguna"
        registro["fuente"] = "fallback"
        registro["desviacion"] = f"Acción '{accion}' no válida; se usó 'ninguna'"
        return df, registro

    if accion == "ninguna":
        return df, registro

    # Identificar columnas con nulos (solo las propuestas, o todas si no se especificaron)
    columnas_con_nulos = [c for c in df.columns if df[c].isna().sum() > 0]
    if columnas_afectadas:
        columnas_a_tratar = [c for c in columnas_afectadas if c in columnas_con_nulos]
    else:
        columnas_a_tratar = columnas_con_nulos

    if not columnas_a_tratar:
        registro["detalle"] = {"nota": "No se encontraron columnas con nulos"}
        return df, registro

    df_limpio = df.copy()
    nulos_antes = int(df[columnas_a_tratar].isna().sum().sum())

    if accion == "eliminar_filas":
        n_antes = len(df_limpio)
        df_limpio = df_limpio.dropna(subset=columnas_a_tratar)
        n_perdidas = n_antes - len(df_limpio)

        # Verificar que no se elimina demasiado
        if n_perdidas / n_antes > MAX_FRACCION_ELIMINABLE:
            # Fallback: imputar en lugar de eliminar
            df_limpio = df.copy()
            accion_fallback = FALLBACK_NULOS_NUMERICA
            for col in columnas_a_tratar:
                if pd.api.types.is_numeric_dtype(df_limpio[col]):
                    df_limpio[col] = df_limpio[col].fillna(df_limpio[col].median())
                else:
                    df_limpio[col] = df_limpio[col].fillna(df_limpio[col].mode()[0])
            registro["accion_ejecutada"] = f"imputar ({accion_fallback})"
            registro["fuente"] = "fallback"
            registro["desviacion"] = (
                f"eliminar_filas afectaría al {100*n_perdidas/n_antes:.1f}% "
                f"(límite {int(MAX_FRACCION_ELIMINABLE*100)}%); se imputó en su lugar"
            )
            registro["detalle"] = {
                "nulos_antes": nulos_antes,
                "nulos_despues": int(df_limpio[columnas_a_tratar].isna().sum().sum()),
                "filas": len(df_limpio),
            }
            return df_limpio, registro

        registro["detalle"] = {
            "nulos_antes": nulos_antes,
            "filas_antes": n_antes,
            "filas_despues": len(df_limpio),
            "filas_eliminadas": n_perdidas,
        }
        return df_limpio, registro

    elif accion == "eliminar_columnas":
        columnas_a_eliminar = [c for c in columnas_a_tratar if c not in columnas_protegidas]
        protegidas_encontradas = [c for c in columnas_a_tratar if c in columnas_protegidas]
        if protegidas_encontradas:
            registro["desviacion"] = (
                f"Se excluyeron columnas protegidas: {protegidas_encontradas}"
            )
            registro["fuente"] = "fallback"

        df_limpio = df_limpio.drop(columns=columnas_a_eliminar, errors="ignore")
        registro["accion_ejecutada"] = accion
        registro["detalle"] = {
            "columnas_eliminadas": columnas_a_eliminar,
            "columnas_protegidas": protegidas_encontradas,
        }
        return df_limpio, registro

    else:
        # Imputación
        detalle = {}
        for col in columnas_a_tratar:
            if accion == "imputar_mediana":
                valor = df_limpio[col].median() if pd.api.types.is_numeric_dtype(df_limpio[col]) else df_limpio[col].mode()[0]
            elif accion == "imputar_media":
                valor = df_limpio[col].mean() if pd.api.types.is_numeric_dtype(df_limpio[col]) else df_limpio[col].mode()[0]
            elif accion == "imputar_moda":
                valor = df_limpio[col].mode()[0] if len(df_limpio[col].mode()) > 0 else None
            else:
                continue

            if valor is not None:
                n_imputados = int(df_limpio[col].isna().sum())
                df_limpio[col] = df_limpio[col].fillna(valor)
                detalle[col] = {"valor_imputado": float(valor) if pd.api.types.is_numeric_dtype(df_limpio[col]) else str(valor),
                                "celdas_imputadas": n_imputados}

        registro["accion_ejecutada"] = accion
        registro["detalle"] = {
            "columnas_imputadas": columnas_a_tratar,
            "detalle_por_columna": detalle,
            "nulos_antes": nulos_antes,
            "nulos_despues": int(df_limpio[columnas_a_tratar].isna().sum().sum()),
        }
        return df_limpio, registro


def ejecutar_limpieza_duplicados(df: pd.DataFrame, propuesta: dict) -> tuple:
    """Ejecuta la propuesta de tratamiento de duplicados. Devuelve (df, registro)."""
    accion = propuesta.get("accion", "conservar_todos")
    justificacion = propuesta.get("justificacion", "")

    registro = {
        "accion_propuesta": accion,
        "accion_ejecutada": accion,
        "fuente": "llm",
        "desviacion": None,
        "justificacion_llm": justificacion,
        "detalle": {},
    }

    if not validar_opcion(accion, ACCIONES_DUPLICADOS):
        registro["accion_ejecutada"] = FALLBACK_DUPLICADOS
        registro["fuente"] = "fallback"
        registro["desviacion"] = f"Acción '{accion}' no válida; se usó '{FALLBACK_DUPLICADOS}'"
        return df, registro

    if accion == "conservar_todos":
        registro["detalle"] = {"duplicados_detectados": int(df.duplicated().sum())}
        return df, registro

    # Eliminar duplicados
    n_antes = len(df)
    df_limpio = df.drop_duplicates()
    n_eliminados = n_antes - len(df_limpio)

    registro["accion_ejecutada"] = accion
    registro["detalle"] = {
        "filas_antes": n_antes,
        "filas_despues": len(df_limpio),
        "duplicados_eliminados": n_eliminados,
    }
    return df_limpio, registro


def ejecutar_limpieza_corruptos(df: pd.DataFrame, propuestas: list) -> tuple:
    """Ejecuta las propuestas de tratamiento de valores corruptos por columna."""
    df_limpio = df.copy()
    registros = []

    for item in propuestas:
        col = item.get("columna")
        accion = item.get("accion", "ninguna")
        col_grupo = item.get("columna_grupo")
        justificacion = item.get("justificacion", "")

        registro = {
            "columna": col,
            "accion_propuesta": accion,
            "accion_ejecutada": accion,
            "fuente": "llm",
            "desviacion": None,
            "justificacion_llm": justificacion,
            "detalle": {},
        }

        if col not in df_limpio.columns:
            registro["accion_ejecutada"] = "ninguna"
            registro["fuente"] = "fallback"
            registro["desviacion"] = f"Columna '{col}' no existe en el dataset"
            registros.append(registro)
            continue

        if not validar_opcion(accion, ACCIONES_CORRUPCION):
            registro["accion_ejecutada"] = FALLBACK_CORRUPCION
            registro["fuente"] = "fallback"
            registro["desviacion"] = f"Acción '{accion}' no válida; se usó '{FALLBACK_CORRUPCION}'"
            accion = FALLBACK_CORRUPCION

        # Identificar valores corruptos
        serie_num = pd.to_numeric(df_limpio[col], errors="coerce")
        mascara = serie_num.isna() & df_limpio[col].notna()
        n_afectadas = int(mascara.sum())

        if n_afectadas == 0:
            registro["accion_ejecutada"] = "ninguna"
            registro["detalle"] = {"nota": "No se encontraron valores corruptos"}
            registros.append(registro)
            continue

        if accion == "ninguna":
            registro["detalle"] = {"valores_corruptos": n_afectadas, "nota": "No se trató"}
            registros.append(registro)
            continue

        if accion == "marcar_como_no_numerica":
            registro["accion_ejecutada"] = accion
            registro["detalle"] = {"valores_corruptos": n_afectadas,
                                   "nota": "Columna conservada como categórica"}
            registros.append(registro)
            continue

        if accion == "eliminar_filas":
            n_antes = len(df_limpio)
            if n_afectadas / n_antes > MAX_FRACCION_ELIMINABLE:
                registro["accion_ejecutada"] = FALLBACK_CORRUPCION
                registro["fuente"] = "fallback"
                registro["desviacion"] = (
                    f"eliminar_filas afectaría al {100*n_afectadas/n_antes:.1f}% "
                    f"(límite {int(MAX_FRACCION_ELIMINABLE*100)}%); se imputó en su lugar"
                )
                accion = FALLBACK_CORRUPCION
            else:
                df_limpio = df_limpio[~mascara].reset_index(drop=True)
                df_limpio[col] = pd.to_numeric(df_limpio[col], errors="coerce")
                registro["accion_ejecutada"] = accion
                registro["detalle"] = {
                    "filas_antes": n_antes,
                    "filas_despues": len(df_limpio),
                    "filas_eliminadas": n_afectadas,
                }
                registros.append(registro)
                continue

        # Imputaciones
        mediana_global = serie_num.median()
        if accion == "imputar_mediana_global":
            serie_num[mascara] = mediana_global
            df_limpio[col] = serie_num
            registro["accion_ejecutada"] = accion
            registro["detalle"] = {
                "valores_corruptos": n_afectadas,
                "valor_imputado": float(mediana_global),
            }

        elif accion == "imputar_mediana_grupo":
            if col_grupo and col_grupo in df_limpio.columns:
                mediana_grupo = serie_num.groupby(df_limpio[col_grupo]).transform("median")
                serie_num[mascara] = mediana_grupo[mascara]
                sin_grupo = mascara & serie_num.isna()
                if sin_grupo.any():
                    serie_num[sin_grupo] = mediana_global
                df_limpio[col] = serie_num
                registro["accion_ejecutada"] = accion
                registro["detalle"] = {
                    "valores_corruptos": n_afectadas,
                    "columna_grupo": col_grupo,
                    "imputados_con_mediana_grupo": int(mascara.sum() - sin_grupo.sum()),
                    "imputados_con_mediana_global": int(sin_grupo.sum()),
                }
            else:
                # Fallback: mediana global
                serie_num[mascara] = mediana_global
                df_limpio[col] = serie_num
                registro["accion_ejecutada"] = FALLBACK_CORRUPCION
                registro["fuente"] = "fallback"
                registro["desviacion"] = (
                    f"columna_grupo '{col_grupo}' no válida; se usó mediana global"
                )
                registro["detalle"] = {
                    "valores_corruptos": n_afectadas,
                    "valor_imputado": float(mediana_global),
                }

        elif accion == "convertir_a_cero":
            serie_num[mascara] = 0.0
            df_limpio[col] = serie_num
            registro["accion_ejecutada"] = accion
            registro["detalle"] = {"valores_corruptos": n_afectadas, "valor_asignado": 0.0}

        registros.append(registro)

    return df_limpio, registros


# ============================================================================
# 2. RUIDO ARTIFICIAL
# ============================================================================
def aplicar_ruido(df: pd.DataFrame, propuesta_ruido: dict, objetivo: str) -> tuple:
    """Aplica ruido gaussiano aditivo según la propuesta de la Etapa 1.
    Devuelve (df_con_ruido, registro_ruido)."""
    rng = np.random.default_rng(SEMILLA_RUIDO)

    columnas_perturbar = propuesta_ruido.get("columnas_perturbar", [])
    columnas_excluir = propuesta_ruido.get("columnas_excluir", {})
    justificacion_general = propuesta_ruido.get("justificacion_general", "")

    # Validar que las columnas a perturbar existen y son numéricas
    columnas_numericas = set(df.select_dtypes(include=[np.number]).columns)
    elegibles = []
    exclusiones = {}

    # La variable objetivo siempre se excluye
    if objetivo:
        exclusiones[objetivo] = "variable objetivo: no se perturba la etiqueta a predecir"

    for col in columnas_perturbar:
        if col == objetivo:
            exclusiones[col] = "variable objetivo: no se perturba la etiqueta a predecir"
        elif col not in columnas_numericas:
            exclusiones[col] = f"no es columna numérica; se excluyó"
        else:
            elegibles.append(col)

    # Añadir exclusiones propuestas por el LLM
    for col, motivo in columnas_excluir.items():
        if col in columnas_numericas and col not in exclusiones:
            exclusiones[col] = f"decisión del LLM: {motivo}"

    nota = (
        "El ruido artificial simula errores de medición o pequeñas perturbaciones en los datos. "
        "Es distinto de la aleatoriedad del LLM: el ruido es determinista (semilla fija) y "
        "modifica los datos de entrada, mientras que la aleatoriedad del LLM afecta las "
        "decisiones de los agentes pero no los datos."
    )

    if not elegibles:
        return df, {
            "aplicado": False,
            "razon": "No hay columnas numéricas elegibles",
            "columnas_afectadas": [],
            "exclusiones": exclusiones,
            "justificacion_general": justificacion_general,
            "nota_didactica": nota,
        }

    df_ruido = df.copy()
    detalle, afectadas, total = {}, [], 0

    for col in elegibles:
        valores = df_ruido[col].to_numpy(dtype=float, copy=True)
        posiciones = np.flatnonzero(~np.isnan(valores))
        if len(posiciones) == 0:
            exclusiones[col] = "sin valores válidos"
            continue
        std_col = float(np.nanstd(valores))
        if not std_col > 0:
            exclusiones[col] = "desviación estándar nula: no hay escala para el ruido"
            continue
        n_modificar = max(1, int(len(posiciones) * PORCENTAJE_RUIDO / 100))
        indices = rng.choice(posiciones, size=n_modificar, replace=False)
        sigma = FACTOR_SIGMA_RUIDO * std_col
        valores[indices] += rng.normal(0, sigma, size=n_modificar)
        df_ruido[col] = valores
        detalle[col] = {"celdas_modificadas": int(n_modificar), "sigma": sigma}
        afectadas.append(col)
        total += n_modificar

    registro = {
        "aplicado": bool(afectadas),
        "porcentaje_celdas_modificadas": PORCENTAJE_RUIDO,
        "semilla": SEMILLA_RUIDO,
        "metodo": f"Ruido gaussiano aditivo (sigma = {FACTOR_SIGMA_RUIDO} * desviacion_estandar)",
        "columnas_afectadas": afectadas,
        "detalle_por_columna": detalle,
        "total_celdas_modificadas": int(total),
        "exclusiones": exclusiones,
        "justificacion_general": justificacion_general,
        "nota_didactica": nota,
    }
    return df_ruido, registro


# ============================================================================
# 3. CÁLCULOS ESTADÍSTICOS (Python, no LLM)
# ============================================================================
def calcular_estadisticas_descriptivas(df: pd.DataFrame) -> dict:
    """Estadísticas descriptivas para columnas numéricas."""
    resultado = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        valores = df[col].dropna()
        if len(valores) == 0:
            continue
        resultado[col] = {
            "count": int(len(valores)),
            "mean": float(valores.mean()),
            "median": float(valores.median()),
            "std": float(valores.std()),
            "min": float(valores.min()),
            "max": float(valores.max()),
            "q25": float(valores.quantile(0.25)),
            "q75": float(valores.quantile(0.75)),
            "skewness": float(stats.skew(valores)),
            "kurtosis": float(stats.kurtosis(valores)),
        }
    return resultado


def detectar_outliers_iqr(df: pd.DataFrame) -> dict:
    """Detecta valores atípicos usando el método IQR."""
    resultado = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        valores = df[col].dropna()
        if len(valores) == 0:
            continue
        q1 = valores.quantile(0.25)
        q3 = valores.quantile(0.75)
        iqr = q3 - q1
        limite_inferior = q1 - 1.5 * iqr
        limite_superior = q3 + 1.5 * iqr
        outliers = valores[(valores < limite_inferior) | (valores > limite_superior)]
        resultado[col] = {
            "n_outliers": int(len(outliers)),
            "porcentaje_outliers": float(len(outliers) / len(valores) * 100),
            "limite_inferior": float(limite_inferior),
            "limite_superior": float(limite_superior),
        }
    return resultado


def calcular_correlaciones(df: pd.DataFrame) -> dict:
    """Matriz de correlación de Pearson y Spearman, pares interesantes."""
    columnas_numericas = df.select_dtypes(include=[np.number]).columns
    if len(columnas_numericas) < 2:
        return {"pearson": {}, "spearman": {}, "pares_interesantes": []}

    pearson = df[columnas_numericas].corr(method="pearson")
    spearman = df[columnas_numericas].corr(method="spearman")

    pares_interesantes = []
    for i, col1 in enumerate(columnas_numericas):
        for col2 in columnas_numericas[i + 1:]:
            r_pearson = pearson.loc[col1, col2]
            r_spearman = spearman.loc[col1, col2]
            if (abs(r_pearson) > 0.1 or abs(r_spearman) > 0.1) and not np.isnan(r_pearson):
                pares_interesantes.append({
                    "par": [col1, col2],
                    "pearson": float(r_pearson),
                    "spearman": float(r_spearman),
                    "diferencia": float(abs(r_pearson - r_spearman)),
                })

    pearson_dict = {col: {col2: float(pearson.loc[col, col2]) for col2 in columnas_numericas}
                    for col in columnas_numericas}
    spearman_dict = {col: {col2: float(spearman.loc[col, col2]) for col2 in columnas_numericas}
                     for col in columnas_numericas}

    return {"pearson": pearson_dict, "spearman": spearman_dict, "pares_interesantes": pares_interesantes}


def calcular_estadisticas_por_grupo(df: pd.DataFrame, objetivo: str) -> dict:
    """Estadísticas de la variable objetivo agrupada por categorías."""
    resultado = {}
    columnas_categoricas = [
        col for col in df.select_dtypes(exclude=[np.number]).columns
        if df[col].nunique() <= 20 and col != objetivo
    ]

    for col_grupo in columnas_categoricas:
        if objetivo not in df.columns:
            continue
        agrupado = df.groupby(col_grupo)[objetivo].agg(
            ["count", "mean", "median", "std", "min", "max"]
        ).reset_index()
        resultado[col_grupo] = agrupado.to_dict(orient="records")
    return resultado


# ============================================================================
# 4. GRÁFICAS
# ============================================================================
def generar_graficas(df: pd.DataFrame, objetivo: str):
    """Genera gráficas estadísticas básicas."""
    plt.style.use("seaborn-v0_8-whitegrid")

    # Histograma del objetivo
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df[objetivo].dropna(), bins=30, color="steelblue", edgecolor="black", alpha=0.7)
    ax.set_xlabel(objetivo)
    ax.set_ylabel("Frecuencia")
    ax.set_title(f"Distribución de {objetivo}")
    ax.axvline(df[objetivo].mean(), color="red", linestyle="--", label=f"Media: {df[objetivo].mean():.2f}")
    ax.axvline(df[objetivo].median(), color="green", linestyle="--", label=f"Mediana: {df[objetivo].median():.2f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "distribucion_objetivo.png", dpi=100)
    plt.close(fig)

    # Boxplot por primera columna categórica disponible
    columnas_cat = [col for col in df.select_dtypes(exclude=[np.number]).columns if df[col].nunique() <= 10]
    if columnas_cat and objetivo in df.columns:
        col_grupo = columnas_cat[0]
        fig, ax = plt.subplots(figsize=(10, 6))
        df.boxplot(column=objetivo, by=col_grupo, ax=ax, rot=45)
        ax.set_title(f"{objetivo} por {col_grupo}")
        ax.set_xlabel(col_grupo)
        ax.set_ylabel(objetivo)
        fig.suptitle("")
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "boxplot_por_grupo.png", dpi=100)
        plt.close(fig)

    # Heatmap de correlaciones
    columnas_numericas = df.select_dtypes(include=[np.number]).columns
    if len(columnas_numericas) >= 2:
        matriz = df[columnas_numericas].corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(matriz, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(columnas_numericas)))
        ax.set_yticks(range(len(columnas_numericas)))
        ax.set_xticklabels(columnas_numericas, rotation=45, ha="right")
        ax.set_yticklabels(columnas_numericas)
        for i in range(len(columnas_numericas)):
            for j in range(len(columnas_numericas)):
                ax.text(j, i, f"{matriz.iloc[i, j]:.2f}", ha="center", va="center",
                        color="black" if abs(matriz.iloc[i, j]) < 0.5 else "white", fontsize=8)
        fig.colorbar(im)
        ax.set_title("Matriz de correlación")
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "correlaciones.png", dpi=100)
        plt.close(fig)


# ============================================================================
# 5. INTERPRETACIÓN CON LLM
# ============================================================================
SYSTEM_INTERPRETACION = (
    PRINCIPIOS_DE_TRABAJO + USO_DE_ANTECEDENTES + REGLAS_DE_EVIDENCIA + """

TU FORMACIÓN: estadístico aplicado. Evalúas qué dicen los datos y con qué solidez lo dicen:
distribuciones, atípicos, relaciones, diferencias entre grupos, y cuándo una técnica deja de ser
fiable (muestras pequeñas, colas pesadas, no linealidad, comparaciones múltiples). Los cálculos ya
están hechos y son correctos; tú los interpretas y no calculas nada nuevo.

DECIDES TÚ
- Qué hallazgos importan para predecir la variable objetivo y cuáles son ruido (entre 3 y 6).
- Si los atípicos son errores, casos legítimos o parte natural de la distribución. Se inyectó ruido artificial en ciertas columnas (ver
  "ruido_artificial"): no lo confundas con fenómenos reales.
- Qué tan fiables son las relaciones dadas la forma de las distribuciones y el tamaño de la muestra.
- Qué notas previas se confirman, se matizan o se descartan con estos datos.
- Qué debería tener en cuenta quien construya modelos predictivos con estos datos.

CITA la evidencia exacta (columna, estadístico, valor) en cada hallazgo. Los puntos clave
deben ser específicos de ESTOS datos, no consejos genéricos.

Responde SOLO con JSON, (en este orden):
{
  "hallazgos": [{"tipo": "DATO|INFERENCIA|HIPOTESIS", "afirmacion": "...", "evidencia": "clave=valor"}], (entre 3 y 6 hallazgos)
  "distribucion_objetivo": "forma, asimetría, colas y qué implica",
  "outliers": "qué son probablemente y qué merecen",
  "puntos_clave_para_modelado": ["..."], (máximo 4)
  "limitaciones_estadisticas": ["..."], (máximo 3)
  "confianza_general": "alta | media | baja"
}"""
)


def interpretar_resultados(llm, hechos_llm: dict, hechos_completos: dict,
                           antecedentes: dict, revision: str, agente: str) -> tuple:
    """hechos_llm: versión compacta para el prompt.
       hechos_completos: versión completa para verificar cifras sin falsos positivos."""
    print(f"[{agente}] Generando interpretación...")
    user_prompt = (
        bloque_antecedentes(hechos_llm, antecedentes) +
        f"\n\nInterpreta estos resultados estadísticos.\n{revision}"
    )
    respuesta = consultar_llm_json(
        llm, SYSTEM_INTERPRETACION, user_prompt, agente=agente,
        claves_requeridas=["hallazgos", "distribucion_objetivo"]
    )

    if hay_error(respuesta):
        return respuesta, [], []

    hallazgos, stats_hallazgos = normalizar_afirmaciones(respuesta.get("hallazgos"))
    puntos_clave = normalizar_lista_texto(respuesta.get("puntos_clave_para_modelado"))
    limitaciones = normalizar_lista_texto(respuesta.get("limitaciones_estadisticas"))

    validaciones = []
    if stats_hallazgos["sin_clasificar"] > 0:
        validaciones.append(
            f"{stats_hallazgos['sin_clasificar']} hallazgos sin clasificar")
    if stats_hallazgos["dato_sin_evidencia"] > 0:
        validaciones.append(
            f"{stats_hallazgos['dato_sin_evidencia']} DATOS sin evidencia citada")

    # Verificar contra los hechos COMPLETOS, no contra el resumen
    cifras_dudosas = verificar_cifras(respuesta, hechos_completos)

    respuesta_normalizada = {
        "hallazgos": hallazgos,
        "distribucion_objetivo": str(respuesta.get("distribucion_objetivo", "")),
        "outliers": str(respuesta.get("outliers", "")),
        "puntos_clave_para_modelado": puntos_clave,
        "limitaciones_estadisticas": limitaciones,
        "confianza_general": str(respuesta.get("confianza_general", "")),
        "estadisticas_normalizacion": stats_hallazgos,
    }
    return respuesta_normalizada, validaciones, cifras_dudosas


# ============================================================================
# 6. REPORTE
# ============================================================================
def generar_reporte(df_original, df_experimental, registro_limpieza, registro_ruido,
                    estadisticas, interpretacion, objetivo):
    """Genera el informe Markdown de la Etapa 2."""
    tec_o = {"filas": len(df_original), "columnas": len(df_original.columns),
             "nulos": int(df_original.isna().sum().sum()),
             "duplicados": int(df_original.duplicated().sum())}
    tec_e = {"filas": len(df_experimental), "columnas": len(df_experimental.columns),
             "nulos": int(df_experimental.isna().sum().sum()),
             "duplicados": int(df_experimental.duplicated().sum())}

    c = f"""# Reporte Estadístico - Etapa 2: Ejecución de Limpieza y Análisis

**Fecha:** {ahora()}
**Variable objetivo:** {objetivo}
**Rol de esta etapa:** Ejecución de limpieza + ruido + análisis estadístico

## 1. Ejecución de Limpieza (propuestas de Etapa 1)

### Nulos
- **Acción propuesta:** {registro_limpieza['nulos'].get('accion_propuesta', 'N/A')}
- **Acción ejecutada:** {registro_limpieza['nulos'].get('accion_ejecutada', 'N/A')}
- **Fuente:** {registro_limpieza['nulos'].get('fuente', 'N/A')}
- **Justificación:** {registro_limpieza['nulos'].get('justificacion_llm', 'N/A')}
"""
    if registro_limpieza['nulos'].get('desviacion'):
        c += f"- **Desviación:** {registro_limpieza['nulos']['desviacion']}\n"

    c += f"""
### Duplicados
- **Acción propuesta:** {registro_limpieza['duplicados'].get('accion_propuesta', 'N/A')}
- **Acción ejecutada:** {registro_limpieza['duplicados'].get('accion_ejecutada', 'N/A')}
- **Fuente:** {registro_limpieza['duplicados'].get('fuente', 'N/A')}
- **Justificación:** {registro_limpieza['duplicados'].get('justificacion_llm', 'N/A')}
"""
    if registro_limpieza['duplicados'].get('desviacion'):
        c += f"- **Desviación:** {registro_limpieza['duplicados']['desviacion']}\n"

    if registro_limpieza['corruptos']:
        c += "\n### Valores Corruptos\n\n"
        for item in registro_limpieza['corruptos']:
            c += f"**{item['columna']}**\n"
            c += f"- Acción propuesta: {item.get('accion_propuesta', 'N/A')}\n"
            c += f"- Acción ejecutada: {item.get('accion_ejecutada', 'N/A')}\n"
            c += f"- Fuente: {item.get('fuente', 'N/A')}\n"
            if item.get('desviacion'):
                c += f"- Desviación: {item['desviacion']}\n"
            c += "\n"

    c += f"""
## 2. Ruido Artificial Aplicado

- **Aplicado:** {registro_ruido.get('aplicado', False)}
- **Porcentaje de celdas modificadas:** {registro_ruido.get('porcentaje_celdas_modificadas', 0)}%
- **Semilla:** {registro_ruido.get('semilla')}
- **Método:** {registro_ruido.get('metodo')}
- **Columnas afectadas:** {', '.join(registro_ruido.get('columnas_afectadas', [])) or 'ninguna'}
- **Total de celdas modificadas:** {registro_ruido.get('total_celdas_modificadas', 0)}

**Columnas excluidas y motivo:**

"""
    exclusiones = registro_ruido.get("exclusiones", {})
    if exclusiones:
        for col, motivo in exclusiones.items():
            c += f"- {col}: {motivo}\n"
    else:
        c += "- ninguna\n"

    c += f"""
## 3. Comparación Antes/Después de Limpieza

| Métrica | Original | Experimental |
|---------|----------|--------------|
| Filas | {tec_o['filas']} | {tec_e['filas']} |
| Columnas | {tec_o['columnas']} | {tec_e['columnas']} |
| Valores nulos | {tec_o['nulos']} | {tec_e['nulos']} |
| Duplicados | {tec_o['duplicados']} | {tec_e['duplicados']} |

## 4. Hallazgos Principales

"""
    for h in interpretacion.get("hallazgos", []):
        c += f"- **[{h['tipo']}]** {h['afirmacion']}\n"
        if h.get("evidencia"):
            c += f"  - *Evidencia:* {h['evidencia']}\n"

    c += f"""
## 5. Distribución de la Variable Objetivo

{interpretacion.get('distribucion_objetivo', 'No disponible')}

![Distribución](../plots/distribucion_objetivo.png)

## 6. Estadísticas Descriptivas

| Columna | Media | Mediana | Desv. Est. | Min | Max | Skewness |
|---------|-------|---------|------------|-----|-----|----------|
"""
    for col, stats in estadisticas["descriptivas"].items():
        c += (f"| {col} | {stats['mean']:.2f} | {stats['median']:.2f} | "
              f"{stats['std']:.2f} | {stats['min']:.2f} | {stats['max']:.2f} | "
              f"{stats['skewness']:.2f} |\n")

    c += f"""
## 7. Valores Atípicos (IQR)

{interpretacion.get('outliers', 'No disponible')}

| Columna | N° Outliers | % Outliers |
|---------|-------------|------------|
"""
    for col, info in estadisticas["outliers"].items():
        c += f"| {col} | {info['n_outliers']} | {info['porcentaje_outliers']:.2f}% |\n"

    c += f"""

## 8. Correlaciones Relevantes

{interpretacion.get('relaciones_relevantes', 'No disponible')}

![Correlaciones](../plots/correlaciones.png)

| Par | Pearson | Spearman | Diferencia |
|-----|---------|----------|------------|
"""
    pares = estadisticas["correlaciones"]["pares_interesantes"]
    if pares:
        for par in pares:
            c += (f"| {' vs '.join(par['par'])} | {par['pearson']:.3f} | "
                  f"{par['spearman']:.3f} | {par['diferencia']:.3f} |\n")
    else:
        c += "| — | — | — | — |\n"
        c += "\n*No se detectaron correlaciones fuertes (|r| > 0.1) entre variables numéricas.*\n"

    REPORTE_ESTADISTICO.write_text(c, encoding="utf-8")


# ============================================================================
# PUNTO DE ENTRADA
# ============================================================================
def main():
    args = crear_parser("qwen2.5:7b").parse_args()
    agente = "Etapa 2"

    print(f"\n[{agente}] Iniciando ejecución de limpieza y análisis con modelo: {args.modelo}")
    asegurar_directorios()

    # Verificar dependencias
    for archivo in [DATASET_ORIGINAL, METADATA_FILE]:
        if not archivo.exists():
            print(f"[{agente}] ERROR: No se encuentra {archivo}. Ejecuta Etapa 1 primero.")
            sys.exit(1)

    metadata = leer_json(METADATA_FILE)
    if metadata.get("estado") != "EXITO":
        print(f"[{agente}] ERROR: La Etapa 1 no completó con éxito (estado: {metadata.get('estado')})")
        sys.exit(1)

    objetivo = metadata.get("definicion_problema", {}).get("variable_objetivo")
    if not objetivo:
        print(f"[{agente}] ERROR: metadata.json no contiene 'variable_objetivo'.")
        sys.exit(1)

    print(f"[{agente}] Variable objetivo: {objetivo}")

    # Cargar dataset original
    df_original, codificacion = cargar_csv(DATASET_ORIGINAL)
    if df_original is None:
        print(f"[{agente}] ERROR: No se pudo leer el CSV con ninguna codificación común")
        sys.exit(1)
    print(f"[{agente}] Dataset cargado ({codificacion}): {df_original.shape}")

    revision = leer_revision(args.revision)

    # --- EJECUCIÓN DE LIMPIEZA ---
    print(f"[{agente}] Ejecutando limpieza según propuestas de Etapa 1...")
    propuestas = metadata.get("propuestas_limpieza", {})
    columnas_protegidas = {objetivo}

    df_limpio, reg_nulos = ejecutar_limpieza_nulos(
        df_original, propuestas.get("propuesta_nulos", {}), columnas_protegidas)
    print(f"  -> Nulos: {reg_nulos['accion_ejecutada']} (fuente: {reg_nulos['fuente']})")

    df_limpio, reg_dups = ejecutar_limpieza_duplicados(
        df_limpio, propuestas.get("propuesta_duplicados", {}))
    print(f"  -> Duplicados: {reg_dups['accion_ejecutada']} (fuente: {reg_dups['fuente']})")

    df_limpio, reg_corruptos = ejecutar_limpieza_corruptos(
        df_limpio, propuestas.get("propuesta_corruptos", []))
    for item in reg_corruptos:
        print(f"  -> Corruptos [{item['columna']}]: {item['accion_ejecutada']} "
              f"(fuente: {item['fuente']})")

    registro_limpieza = {
        "timestamp": ahora(),
        "nulos": reg_nulos,
        "duplicados": reg_dups,
        "corruptos": reg_corruptos,
        "filas_original": len(df_original),
        "filas_despues_limpieza": len(df_limpio),
        "columnas_original": len(df_original.columns),
        "columnas_despues_limpieza": len(df_limpio.columns),
    }
    guardar_json(REGISTRO_LIMPIEZA, registro_limpieza)

    # --- APLICAR RUIDO ---
    print(f"[{agente}] Aplicando ruido artificial del {PORCENTAJE_RUIDO}%...")
    propuesta_ruido = metadata.get("propuesta_ruido", {})
    df_experimental, registro_ruido = aplicar_ruido(df_limpio, propuesta_ruido, objetivo)
    if registro_ruido["aplicado"]:
        print(f"  -> Columnas con ruido: {registro_ruido['columnas_afectadas']}")
    else:
        print(f"  -> No se aplicó ruido: {registro_ruido.get('razon', 'N/A')}")

    guardar_json(REGISTRO_RUIDO, registro_ruido)

    # Guardar dataset experimental
    df_experimental.to_csv(DATASET_EXPERIMENTAL, index=False, encoding="utf-8")
    print(f"[{agente}] Dataset experimental guardado: {DATASET_EXPERIMENTAL}")

    # Verificar que la variable objetivo sigue siendo válida
    if objetivo not in df_experimental.columns:
        print(f"[{agente}] ERROR: La variable objetivo '{objetivo}' se perdió durante la limpieza.")
        sys.exit(1)
    if not pd.api.types.is_numeric_dtype(df_experimental[objetivo]):
        print(f"[{agente}] ERROR: La variable objetivo '{objetivo}' ya no es numérica.")
        sys.exit(1)

    # --- CÁLCULOS ESTADÍSTICOS ---
    print(f"[{agente}] Calculando estadísticas descriptivas...")
    descriptivas = calcular_estadisticas_descriptivas(df_experimental)

    print(f"[{agente}] Detectando valores atípicos (IQR)...")
    outliers = detectar_outliers_iqr(df_experimental)

    print(f"[{agente}] Calculando correlaciones (Pearson y Spearman)...")
    correlaciones = calcular_correlaciones(df_experimental)

    print(f"[{agente}] Calculando estadísticas por grupo...")
    por_grupo = calcular_estadisticas_por_grupo(df_experimental, objetivo)

    estadisticas = {
        "descriptivas": descriptivas,
        "outliers": outliers,
        "correlaciones": correlaciones,
        "por_grupo": por_grupo,
    }

    # Generar gráficas
    print(f"[{agente}] Generando gráficas...")
    generar_graficas(df_experimental, objetivo)

    # --- INTERPRETACIÓN CON LLM ---
    print(f"[{agente}] Interpretando resultados con LLM...")
    llm = crear_llm(args.modelo, temperatura=0.2)

    hechos_llm = redondear_estructura({
        "descriptivas": descriptivas,
        "outliers": outliers,
        "pares_correlacion_interesantes": correlaciones["pares_interesantes"][:10],
        "por_grupo_resumen": {
            col: registros[:5] for col, registros in por_grupo.items()
        },
    })
    antecedentes = {
        "descripcion_dataset": metadata.get("descripcion_semantica", ""),
        "variable_objetivo": objetivo,
        "tarea_ml": metadata.get("definicion_problema", {}).get("tarea_ml", ""),
        "lineas_analisis_sugeridas": metadata.get("lineas_de_analisis_sugeridas", []),
        "advertencias_previas": metadata.get("advertencias", []),
        "ruido_artificial": {"aplicado": registro_ruido.get("aplicado"),
             "columnas_afectadas": registro_ruido.get("columnas_afectadas")},
        "limpieza_ejecutada": {
            "nulos": reg_nulos.get("accion_ejecutada"),
            "duplicados": reg_dups.get("accion_ejecutada"),
            "corruptos": [r.get("accion_ejecutada") for r in reg_corruptos],
        },
    }

    interpretacion, validaciones, cifras_dudas = interpretar_resultados(
        llm, hechos_llm, estadisticas, antecedentes, revision, agente
    )

    if hay_error(interpretacion):
        print(f"[{agente}] ERROR: la interpretación falló: {interpretacion.get('error')}")
        sys.exit(1)

    for v in validaciones:
        print(f"  [VALIDACIÓN] {v}")
    if cifras_dudas:
        print(f"  [CIFRAS NO RESPALDADAS] {cifras_dudas}")

    # --- GUARDAR ARTEFACTOS ---
    salida = {
        "etapa": "2_estadistica",
        "estado": "EXITO",
        "timestamp": ahora(),
        "variable_objetivo": objetivo,
        "modelo_utilizado": args.modelo,
        "revision_recibida": bool(revision),
        "dataset_experimental": str(DATASET_EXPERIMENTAL),
        "limpieza_ejecutada": registro_limpieza,
        "ruido_artificial": registro_ruido,
        "estadisticas_descriptivas": descriptivas,
        "outliers": outliers,
        "correlaciones": correlaciones,
        "por_grupo": por_grupo,
        "interpretacion_llm": interpretacion,
        "graficas_generadas": [
            "plots/distribucion_objetivo.png",
            "plots/boxplot_por_grupo.png",
            "plots/correlaciones.png",
        ],
        "validaciones_sistema": validaciones,
        "cifras_no_verificadas": cifras_dudas,
    }
    guardar_json(ESTADISTICAS_FILE, salida)

    generar_reporte(df_original, df_experimental, registro_limpieza, registro_ruido,
                    estadisticas, interpretacion, objetivo)

    print(f"\n[{agente}] Completada. Archivos generados:")
    print(f"  -> {DATASET_EXPERIMENTAL}")
    print(f"  -> {REGISTRO_LIMPIEZA}")
    print(f"  -> {REGISTRO_RUIDO}")
    print(f"  -> {ESTADISTICAS_FILE}")
    print(f"  -> {REPORTE_ESTADISTICO}")
    print(f"  -> {PLOTS_DIR}/")


if __name__ == "__main__":
    main()