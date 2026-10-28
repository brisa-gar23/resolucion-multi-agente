"""
comun.py - Utilidades y bloques de prompt compartidos por todas las etapas.

Contenido:
  1. Configuración común (semilla, contexto del LLM, directorios).
  2. Utilidades de E/S (JSON, sanitización, timestamps).
  3. Cliente LLM y consulta con reintentos y validación de claves.
  4. Verificador de cifras: detecta números del LLM que no existen en los hechos.
  5. Bloques de prompt reutilizables (principios, uso de antecedentes, reglas de evidencia).
  6. Constructores de bloques de contexto y de observaciones de revisión.

Cada etapa lo importa con:  from comun import ...
"""

import argparse
import json
import re
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

# 1. CONFIGURACIÓN COMÚN
DATA_DIR = Path("data")
SUBDIRECTORIOS = ["reports", "plots", "models"]

SEMILLA_GLOBAL = 42 # reproducibilidad

# Se fuerza un contexto amplio para que no se pierda información en prompts grandes.
NUM_CTX_DEFECTO = 16384

def asegurar_directorios():
    """Crea la estructura de subcarpetas."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for sub in SUBDIRECTORIOS:
        (DATA_DIR / sub).mkdir(parents=True, exist_ok=True)

def ahora() -> str:
    return datetime.now().isoformat(timespec="seconds")

def crear_parser(modelo_defecto: str) -> argparse.ArgumentParser:
    """Parser común: --modelo y --revision (observaciones del supervisor)."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelo", default=modelo_defecto)
    parser.add_argument(
        "--revision",
        default=None,
        help="Ruta a un JSON con observaciones de revisión sobre un intento anterior.",
    )
    return parser


# 2. LIMPIEZA DE FORMATO Y UTILIDADES DE JSON
def limpiar_json(texto: str) -> str:
    """Quita los bloques de código markdown que algunos modelos añaden."""
    texto = texto.strip()
    if texto.startswith("```"):
        lineas = texto.split("\n")
        if lineas[0].startswith("```"):
            lineas = lineas[1:]
        if lineas and lineas[-1].strip() == "```":
            lineas = lineas[:-1]
        texto = "\n".join(lineas)
    return texto.strip()

def sanitizar_para_json(obj):
    """Convierte recursivamente tipos numpy/pandas/Path a tipos nativos serializables.
    NaN e infinitos se convierten en None."""
    if isinstance(obj, dict):
        return {str(k): sanitizar_para_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [sanitizar_para_json(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return None if (np.isnan(obj) or np.isinf(obj)) else float(obj)
    if isinstance(obj, np.ndarray):
        return sanitizar_para_json(obj.tolist())
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, float):
        return None if (np.isnan(obj) or np.isinf(obj)) else obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (pd.Timestamp, datetime, date)):
        return obj.isoformat()
    try:
        if pd.isna(obj):
            return None
    except (TypeError, ValueError):
        pass
    return obj

def a_json(obj, indent: int = 2) -> str:
    """Serializa a texto JSON legible para incluir en un prompt."""
    return json.dumps(sanitizar_para_json(obj), indent=indent, ensure_ascii=False)

def leer_json(ruta) -> dict:
    with open(ruta, "r", encoding="utf-8") as f:
        return json.load(f)

def guardar_json(ruta, obj):
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(sanitizar_para_json(obj), f, indent=2, ensure_ascii=False)

def validar_opcion(valor, permitidas) -> bool:
    """True si el valor propuesto por el LLM es una de las opciones permitidas."""
    return isinstance(valor, str) and valor in set(permitidas)


# 3. CLIENTE LLM
def crear_llm(modelo: str, temperatura: float = 0.2, formato="json",
              num_ctx: int = NUM_CTX_DEFECTO, seed: int = SEMILLA_GLOBAL, num_predict: int = 4096) -> ChatOllama:
    """Crea el cliente de Ollama.
    - formato="json" fuerza JSON válido; también acepta un JSON Schema (dict).
    - seed fija la aleatoriedad del muestreo para que las corridas sean comparables.
    - num_ctx amplía la ventana de contexto para que no se trunque el prompt.
    """
    return ChatOllama(
        model=modelo,
        temperature=temperatura,
        format=formato,
        num_ctx=num_ctx,
        num_predict=num_predict,
        seed=seed,
    )

def hay_error(respuesta) -> bool:
    """True si consultar_llm_json no pudo obtener una respuesta válida."""
    return (not isinstance(respuesta, dict)) or (
        "error" in respuesta and len(respuesta) == 1
    )

def consultar_llm_json(llm, system_prompt: str, user_prompt: str, agente: str = "Agente", 
                       claves_requeridas=None, max_intentos: int = 2) -> dict:
    """Consulta al LLM y devuelve un diccionario.

    - Reintenta si el JSON es inválido o faltan claves requeridas, informando al modelo de qué falló en el intento anterior.
    - Si todos los intentos fallan devuelve {"error": "..."}.
    """
    # Aviso si el prompt probablemente excede el contexto (aprox. 3.5 caracteres/token)
    num_ctx = getattr(llm, "num_ctx", None)
    if num_ctx:
        tokens_aprox = (len(system_prompt) + len(user_prompt)) / 3.5
        if tokens_aprox > num_ctx * 0.85:
            print(f"[{agente}] AVISO: el prompt (~{int(tokens_aprox)} tokens) "
                  f"se acerca al contexto ({num_ctx}); Ollama podría truncarlo.")

    ultimo_error = "desconocido"
    prompt_actual = user_prompt

    for intento in range(1, max_intentos + 1):
        try:
            respuesta = llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=prompt_actual),
            ])
            datos = json.loads(limpiar_json(respuesta.content))
            if not isinstance(datos, dict):
                raise ValueError("la respuesta no es un objeto JSON")
            faltantes = [c for c in (claves_requeridas or []) if c not in datos]
            if faltantes:
                raise ValueError(f"faltan claves requeridas: {faltantes}")
            return datos
        except Exception as e:
            ultimo_error = f"{type(e).__name__}: {e}"
            if intento < max_intentos:
                print(f"[{agente}] Respuesta inválida, reintentando: {ultimo_error}")
                prompt_actual = (
                    user_prompt
                    + f"\n\nAVISO: tu respuesta anterior no fue válida ({ultimo_error}). "
                    "Responde únicamente con un objeto JSON que siga exactamente el formato indicado, sin texto adicional."
                )
            else:
                print(f"[{agente}] Error final de respuesta: {ultimo_error}")

    return {"error": ultimo_error}


# 4. VERIFICADOR DE CIFRAS
# Un número "suelto": no pegado a letras/dígitos (evita capturar el 25 de "q25")
_PATRON_NUMERO = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w])")

def _numeros_en_texto(texto: str) -> list:
    texto = re.sub(r"(?<=\d),(?=\d{3}\b)", "", str(texto))  # 1,200 -> 1200
    return _PATRON_NUMERO.findall(texto)

def numeros_de(obj, acc=None) -> set:
    """Recolecta todos los números de una estructura (valores numéricos y
    números presentes dentro de textos)."""
    acc = set() if acc is None else acc
    if isinstance(obj, dict):
        for v in obj.values():
            numeros_de(v, acc)
    elif isinstance(obj, (list, tuple, set)):
        for v in obj:
            numeros_de(v, acc)
    elif isinstance(obj, (bool, np.bool_)):
        pass
    elif isinstance(obj, (int, float, np.integer, np.floating)):
        x = float(obj)
        if not (np.isnan(x) or np.isinf(x)):
            acc.add(x)
    elif isinstance(obj, str):
        for m in _numeros_en_texto(obj):
            acc.add(float(m))
    return acc

def textos_de(obj, acc=None) -> list:
    """Recolecta todos los textos (valores, no claves) de una estructura."""
    acc = [] if acc is None else acc
    if isinstance(obj, dict):
        for v in obj.values():
            textos_de(v, acc)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            textos_de(v, acc)
    elif isinstance(obj, str):
        acc.append(obj)
    return acc

def cifras_no_respaldadas(texto: str, hechos: dict, tol: float = 0.02) -> list:
    """Devuelve las cifras del texto que no aparecen en los hechos (con tolerancia).
    Se aceptan como respaldadas: el valor, su valor absoluto y, si es una
    proporción, su equivalente en porcentaje. Se ignoran enteros pequeños (0-10), que suelen ser conteos de la propia redacción.
    """
    validas = set()
    for v in numeros_de(hechos):
        validas.add(v)
        validas.add(abs(v))
        if abs(v) <= 1.5:
            validas.add(abs(v) * 100)
    dudosas = []
    for m in _numeros_en_texto(texto):
        x = float(m)
        es_entero_pequeno = ("." not in m) and abs(x) <= 10
        if es_entero_pequeno:
            continue
        if not any(abs(abs(x) - v) <= max(tol * abs(v), 0.01) for v in validas):
            dudosas.append(m)
    return dudosas

def verificar_cifras(respuesta: dict, hechos: dict, tol: float = 0.02) -> list:
    """Revisa todos los textos de la respuesta del LLM y devuelve las cifras
    sin respaldo en los hechos (sin duplicados, en orden de aparición).
    Se registra en el JSON de salida."""
    vistas, resultado = set(), []
    for texto in textos_de(respuesta):
        for cifra in cifras_no_respaldadas(texto, hechos, tol):
            if cifra not in vistas:
                vistas.add(cifra)
                resultado.append(cifra)
    return resultado


# 5. BLOQUES DE PROMPT REUTILIZABLES
# Sin mencionar etapas anteriores ni agentes, para que puedan ser usados en cualquier etapa.

PRINCIPIOS_DE_TRABAJO = """
Eres un especialista en el área de Análisis y Ciencia de Datos que trabaja de forma autónoma: se te entrega el material y decides tú
cómo abordarlo. No habrá quien te corrija a tiempo ni a quien preguntar.
Tu informe se archiva en el expediente del proyecto y lo consultará quien continúe el trabajo,
por lo que debe ser específico, verificable y comprensible sin contexto adicional.

- Razona con la evidencia antes de concluir.
- Prefiere decisiones que conserven información y sean reversibles sobre las destructivas.
- Si una decisión es incierta, tómala igualmente, declarando el supuesto y tu confianza (no olvides indicar que es una hipótesis).
- Cita nombres exactos de columnas y valores exactos.
- Escribe en español.
- Respeta el formato de salida exactamente: lo procesa un programa."""

USO_DE_ANTECEDENTES = """
SOBRE LOS ANTECEDENTES
Recibirás HECHOS VERIFICADOS y NOTAS DE TRABAJO PREVIAS.
- Los hechos son tu referencia. Si una nota los contradice, prevalecen los hechos y lo señalas.
- Las notas son hipótesis de partida: puedes adoptarlas, ajustarlas o descartarlas según lo que veas.
  No necesitas comentar todas, solo las que influyan en tus decisiones.
- No des por cierto algo solo porque aparece en las notas."""

REGLAS_DE_EVIDENCIA = """
CÓMO AFIRMAS COSAS:
Cada afirmación relevante lleva uno de estos tipos:
- DATO: aparece en los datos que se te entregaron. Cita la clave y el valor exacto.
- INFERENCIA: la deduces de esos datos. Indica de cuáles.
- HIPOTESIS: conocimiento del dominio o suposición que los datos no pueden confirmar.
  Es valiosa y está permitida, pero se etiqueta como hipótesis y nunca se presenta como resultado.
Reglas:
- No escribas ninguna cifra que no esté en los datos entregados (salvo un cálculo trivial sobre ellos, mostrando la cuenta).
- Si los datos no bastan para concluir, escribe "no concluyente" y di qué información faltaría.
- No nombres columnas, modelos ni opciones que no aparezcan en lo que se te entregó."""


# 6. CONSTRUCTORES DE CONTEXTO
def bloque_antecedentes(hechos: dict, notas: dict) -> str:
    """Arma el bloque de contexto previo separando por fiabilidad:
    - hechos: cifras calculadas por código (prevalecen siempre).
    - notas: texto interpretativo anterior (hipótesis de partida, pueden tener errores).
    """
    return (
        "HECHOS VERIFICADOS (calculados por código sobre los datos; son la referencia):\n"
        f"{a_json(hechos)}\n\n"
        "NOTAS DE TRABAJO PREVIAS (interpretaciones anteriores a tu intervención; son hipótesis "
        "de partida, pueden contener errores y no te obligan a nada):\n"
        f"{a_json(notas)}"
    )

def leer_revision(ruta) -> str:
    """Lee las observaciones de revisión (si existen) y devuelve el texto a
    inyectar en el prompt de usuario. Devuelve "" si no hay revisión."""
    if not ruta:
        return ""
    ruta = Path(ruta)
    if not ruta.exists():
        return ""
    try:
        datos = leer_json(ruta)
    except Exception:
        return ""
    observaciones = datos.get("instrucciones_para_reintento")
    if not observaciones:
        return ""
    return (
        "\n\nOBSERVACIONES DE REVISIÓN SOBRE UN INTENTO ANTERIOR DE ESTE TRABAJO\n"
        "Un revisor detectó el siguiente problema. Tenlo en cuenta, pero decide tú cómo "
        "resolverlo y verifica con los datos si la observación es correcta:\n"
        f"{observaciones}"
    )


# 7. NORMALIZACIÓN DE RESPUESTAS DEL LLM
import unicodedata

TIPOS_AFIRMACION = ("DATO", "INFERENCIA", "HIPOTESIS")

def _sin_acentos(texto: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", texto) if unicodedata.category(c) != "Mn"
    )

def normalizar_lista_texto(valor) -> list:
    """Garantiza una lista de textos (los modelos pequeños a veces devuelven dicts)."""
    if valor is None:
        return []
    if not isinstance(valor, (list, tuple)):
        valor = [valor]
    resultado = []
    for item in valor:
        if isinstance(item, dict):
            resultado.append("; ".join(f"{k}: {v}" for k, v in item.items()))
        else:
            resultado.append(str(item))
    return resultado

def normalizar_afirmaciones(valor) -> tuple:
    """Normaliza una lista de afirmaciones {tipo, afirmacion, evidencia}.
    Devuelve (lista_normalizada, estadisticas). Las estadísticas permiten medir
    la disciplina del agente: afirmaciones sin clasificar y datos sin evidencia."""
    if valor is None:
        items = []
    elif isinstance(valor, (list, tuple)):
        items = list(valor)
    else:
        items = [valor]

    resultado, sin_clasificar, dato_sin_evidencia = [], 0, 0
    por_tipo = {t: 0 for t in TIPOS_AFIRMACION}
    por_tipo["SIN_CLASIFICAR"] = 0

    for it in items:
        if isinstance(it, dict):
            tipo = _sin_acentos(str(it.get("tipo", "")).strip().upper())
            afirmacion = str(it.get("afirmacion", it.get("afirmación", "")))
            evidencia = str(it.get("evidencia", "") or "")
        else:
            tipo, afirmacion, evidencia = "", str(it), ""
        if tipo not in TIPOS_AFIRMACION:
            tipo = "SIN_CLASIFICAR"
            sin_clasificar += 1
        if tipo == "DATO" and not evidencia.strip():
            dato_sin_evidencia += 1
        por_tipo[tipo] += 1
        resultado.append({"tipo": tipo, "afirmacion": afirmacion, "evidencia": evidencia})

    estadisticas = {
        "total": len(resultado),
        "por_tipo": por_tipo,
        "sin_clasificar": sin_clasificar,
        "dato_sin_evidencia": dato_sin_evidencia,
    }
    return resultado, estadisticas

def numeros_numericos_de(obj, acc=None) -> list:
    """Solo valores numéricos (no los números escritos dentro de textos). Sirve para
    registrar como 'legítimas' las decisiones numéricas propias del LLM
    (hiperparámetros, test_size...) al verificar sus cifras."""
    acc = [] if acc is None else acc
    if isinstance(obj, dict):
        for v in obj.values():
            numeros_numericos_de(v, acc)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            numeros_numericos_de(v, acc)
    elif isinstance(obj, (bool, np.bool_)):
        pass
    elif isinstance(obj, (int, float, np.integer, np.floating)):
        x = float(obj)
        if not (np.isnan(x) or np.isinf(x)):
            acc.append(x)
    return acc

def redondear_estructura(obj, cifras: int = 5):
    """Sanitiza y redondea floats a cifras significativas (conserva p-valores pequeños).
    Reduce el tamaño de los prompts sin perder información útil."""
    def _r(o):
        if isinstance(o, dict):
            return {k: _r(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_r(v) for v in o]
        if isinstance(o, float):
            return float(f"{o:.{cifras}g}")
        return o
    return _r(sanitizar_para_json(obj))
