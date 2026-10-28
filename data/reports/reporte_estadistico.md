# Reporte Estadístico - Etapa 2: Ejecución de Limpieza y Análisis

**Fecha:** 2026-10-28T09:20:27
**Variable objetivo:** score
**Rol de esta etapa:** Ejecución de limpieza + ruido + análisis estadístico

## 1. Ejecución de Limpieza (propuestas de Etapa 1)

### Nulos
- **Acción propuesta:** imputar_mediana
- **Acción ejecutada:** imputar_mediana
- **Fuente:** llm
- **Justificación:** La columna 'venue' tiene 17 nulos, que representan un pequeño porcentaje del total. Imputar la mediana es una buena opción para mantener la distribución de los datos.

### Duplicados
- **Acción propuesta:** eliminar_duplicados
- **Acción ejecutada:** eliminar_duplicados
- **Fuente:** llm
- **Justificación:** Se detectan 28 duplicados, que representan un 2.34% del total. Eliminar estos duplicados no afectará significativamente el conjunto de datos y mejorará la calidad.

### Valores Corruptos

**pop**
- Acción propuesta: convertir_a_cero
- Acción ejecutada: convertir_a_cero
- Fuente: llm


## 2. Ruido Artificial Aplicado

- **Aplicado:** True
- **Porcentaje de celdas modificadas:** 1.0%
- **Semilla:** 42
- **Método:** Ruido gaussiano aditivo (sigma = 0.05 * desviacion_estandar)
- **Columnas afectadas:** minprice
- **Total de celdas modificadas:** 11

**Columnas excluidas y motivo:**

- score: variable objetivo: no se perturba la etiqueta a predecir
- weekend: decisión del LLM: Es una columna binaria y no se deben perturbar.
- month: decisión del LLM: Es una columna ordinal discreta y no se deben perturbar.

## 3. Comparación Antes/Después de Limpieza

| Métrica | Original | Experimental |
|---------|----------|--------------|
| Filas | 1198 | 1170 |
| Columnas | 9 | 9 |
| Valores nulos | 17 | 0 |
| Duplicados | 28 | 0 |

## 4. Hallazgos Principales

- **[INFERENCIA]** La variable 'weekend' no parece estar correctamente codificada, ya que su media es 0.43077, lo que sugiere que hay más conciertos en días laborables que en fines de semana.
  - *Evidencia:* weekend.mean=0.43077
- **[INFERENCIA]** La variable 'pop' tiene una distribución altamente sesgada y colas pesadas, lo que podría afectar la interpretación de las correlaciones y las relaciones lineales.
  - *Evidencia:* pop.skewness=3.216, pop.kurtosis=12.392
- **[INFERENCIA]** La variable 'minprice' presenta una correlación significativa con 'month', lo que sugiere que el precio de las entradas puede variar según el mes del año.
  - *Evidencia:* month.minprice.pearson=0.083795, month.minprice.spearman=0.11695
- **[HIPOTESIS]** La variable 'genre' podría contener errores de etiquetado, ya que hay una categoría 'Undefined' que podría ser un error o una categoría rara.
  - *Evidencia:* genre.genre='Undefined'

## 5. Distribución de la Variable Objetivo

La variable 'score' tiene una distribución ligeramente sesgada a la derecha con colas pesadas, lo que podría indicar que hay una concentración de puntuaciones altas y pocas bajas.

![Distribución](../plots/distribucion_objetivo.png)

## 6. Estadísticas Descriptivas

| Columna | Media | Mediana | Desv. Est. | Min | Max | Skewness |
|---------|-------|---------|------------|-----|-----|----------|
| weekend | 0.43 | 0.00 | 0.50 | 0.00 | 1.00 | 0.28 |
| pop | 370313.78 | 125017.00 | 610596.73 | 0.00 | 3884307.00 | 3.22 |
| month | 7.69 | 8.00 | 1.68 | 1.00 | 12.00 | -0.60 |
| score | 83.38 | 85.00 | 7.67 | 65.00 | 100.00 | -0.21 |
| minprice | 54.89 | 35.25 | 146.10 | 1.00 | 2999.00 | 13.61 |

## 7. Valores Atípicos (IQR)

Los outliers en 'pop' y 'minprice' son probablemente errores o casos legítimos, ya que su porcentaje es significativo. En 'weekend' y 'score' no se observan outliers, lo que sugiere que los datos son más limpios en estas variables.

| Columna | N° Outliers | % Outliers |
|---------|-------------|------------|
| weekend | 0 | 0.00% |
| pop | 109 | 9.32% |
| month | 26 | 2.22% |
| score | 0 | 0.00% |
| minprice | 75 | 6.41% |


## 8. Correlaciones Relevantes

No disponible

![Correlaciones](../plots/correlaciones.png)

| Par | Pearson | Spearman | Diferencia |
|-----|---------|----------|------------|
| pop vs month | 0.003 | 0.102 | 0.099 |
| month vs minprice | 0.084 | 0.117 | 0.033 |
| score vs minprice | 0.138 | 0.203 | 0.065 |
