# Reporte Inicial - Análisis y Definición del Problema

**Fecha:** 2026-10-28T09:16:12
**Rol de esta etapa:** Análisis y propuesta (no ejecuta modificaciones)

## 1. Definición del Problema

**Variable objetivo:** score
**Tarea recomendada:** regresion
**Confianza:** media, con motivo de la presencia de valores no numéricos y nulos, y la posible confusión en la codificación de 'weekend'

**Razonamiento:**
El dataset podría admitir varios problemas de predicción, pero el más natural parece ser predecir la puntuación del concierto ('score') basándose en las demás características.

## 2. Descripción del Dataset

El dataset contiene información sobre conciertos, incluyendo la ciudad, el artista, el lugar del concierto, si es un fin de semana, el mes, la puntuación del concierto, el género musical y el precio mínimo de las entradas. El objetivo es analizar y posiblemente predecir características relacionadas con estos conciertos.

**Procedencia:**
- Repositorio: https://github.com/ethanjaredlee/ticketmaster-price-ml
- Licencia: MIT

## 3. Inspección Técnica

| Métrica | Valor |
|---------|-------|
| Filas | 1198 |
| Columnas | 9 |
| Valores nulos | 17 |
| Filas duplicadas | 28 |

## 4. Significado y Tipo de las Columnas

- **city** (categorica_nominal): Ciudad donde se realiza el concierto
- **artist** (categorica_nominal): Artista que realiza el concierto
- **venue** (categorica_nominal): Lugar donde se realiza el concierto
- **weekend** (binaria): Indica si el concierto se realiza en fin de semana (0) o no (1)
- **pop** (conteo): Población de la ciudad donde se realiza el concierto, representada como un número
- **month** (categorica_ordinal): Mes en que se realiza el concierto
- **score** (continua): Puntuación del concierto, un valor numérico entre 65 y 100
- **genre** (categorica_nominal): Género musical del concierto
- **minprice** (continua): Precio mínimo de las entradas para el concierto

## 5. Hallazgos de Calidad

### Valores Nulos

| Columna | Nulos | % |
|---------|-------|---|
| venue | 17 | 1.42% |

### Filas Duplicadas

- **Total:** 28 (2.34%)

### Valores Extraños

**pop**: 267 valores (22.29%)
- Tipos: {'???': 267}


## 6. Propuestas de Limpieza (para Etapa 2)

### Nulos
- **Acción:** imputar_mediana
- **Justificación:** La columna 'venue' tiene 17 nulos, que representan un pequeño porcentaje del total. Imputar la mediana es una buena opción para mantener la distribución de los datos.

### Duplicados
- **Acción:** eliminar_duplicados
- **Justificación:** Se detectan 28 duplicados, que representan un 2.34% del total. Eliminar estos duplicados no afectará significativamente el conjunto de datos y mejorará la calidad.

### Corruptos

**pop**
- Acción: convertir_a_cero
- Justificación: Los valores '???' en la columna 'pop' son considerados corruptos y deben ser manejados. Convertirlos a cero es una opción viable, ya que estos valores no aportan información relevante y podrían distorsionar el análisis.


## 7. Propuesta de Ruido Artificial (para Etapa 2)

- **Columnas a perturbar:** minprice
- **Columnas a excluir:**
  - weekend: Es una columna binaria y no se deben perturbar.
  - month: Es una columna ordinal discreta y no se deben perturbar.
- **Criterio:** Se incluye 'minprice' ya que es una columna continua que representa una medición (precio mínimo de las entradas). Se excluyen 'weekend' y 'month' según las reglas especificadas.

## 8. Líneas de Análisis Sugeridas

- Análisis de correlación entre 'pop' y 'score'
- Análisis de correlación entre 'month' y 'score'
- Análisis de correlación entre 'genre' y 'score'

## 9. Advertencias

- [ADVERTENCIA] pop: DATO: 'pop' contiene valores no numéricos como '???'. Se deben limpiar estos valores para que la columna sea numérica.
- [ADVERTENCIA] venue: DATO: 'venue' tiene valores nulos (17). Se deben manejar estos valores nulos.
- [ADVERTENCIA] weekend: INFERENCIA: 'weekend' es binaria, pero su media (0.4324) sugiere que hay más conciertos en fin de semana que no. Esto podría ser una confusión en la codificación.
- [ADVERTENCIA] genre: INFERENCIA: 'genre' tiene una categoría 'Undefined', lo cual podría ser un error de etiquetado o una categoría rara.
