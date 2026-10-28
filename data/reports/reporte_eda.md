# Reporte EDA y Selección de Modelos - Etapa 3

**Fecha:** 2026-10-28T09:35:10
**Variable objetivo:** score

## 1. Análisis de Cardinalidad

| Columna | Únicos | % Únicos | Nivel | Categoría más frecuente |
|---------|--------|----------|-------|-------------------------|
| city | 203 | 17.4% | alta | Atlanta |
| artist | 87 | 7.4% | alta | Ed Sheeran |
| venue | 335 | 28.6% | alta | Ruoff Home Mortgage Music Center |
| genre | 18 | 1.5% | media | Rock |

## 2. Distribución de la Variable Objetivo

- **Media:** 83.38
- **Mediana:** 85.00
- **Desv. Est.:** 7.67
- **Mín / Máx:** 65.00 / 100.00
- **Skewness:** -0.209 (simétrica: Sí)
- **Kurtosis:** -0.538 (cola pesada: No)
- **Coeficiente de variación:** 0.092

![Distribución KDE](../plots/eda_distribucion_kde.png)

## 3. Detección de Fugas de Información

No se detectaron fugas de información evidentes.

## 4. Línea Base (calculada por Python)

- **Modelo:** DummyRegressor (estrategia: median)
- **RMSE:** 8.4416
- **MAE:** 6.8333
- **R²:** -0.1340

## 5. Estrategia Experimental Propuesta

### Transformación del Objetivo
- **Transformación:** ninguna
- **Justificación:** DATO: score tiene skewness=-0.20937, kurtosis=-0.53829, es_simetrica=true. La distribución ya está cerca de normal sin necesidad de transformación.

### Modelos Propuestos

**RandomForestRegressor** (modelo_principal)
- Clase sklearn: `RandomForestRegressor`
- Justificación: DATO: score tiene rango=35.0 y std=7.6662, lo que indica variabilidad moderada. Random Forest maneja bien no linealidades y es robusto a outliers en features como pop (9.3% outliers).
- Hiperparámetros: {'n_estimators': 100, 'max_depth': 5, 'min_samples_split': 5, 'min_samples_leaf': 2}

**Ridge** (modelo_comparacion)
- Clase sklearn: `Ridge`
- Justificación: DATO: lineal base tiene r2=-0.13399. Ridge regularización L2 ayuda con multicolinealidad potencial entre month-minprice (pearson=0.083795).
- Hiperparámetros: {'alpha': 1.0, 'max_iter': 1000}

**DummyRegressor** (linea_base)
- Clase sklearn: `DummyRegressor`
- Justificación: DATO: línea base con rmse=8.4416, mae=6.8333, r2=-0.13399. Necesario para establecer baseline.
- Hiperparámetros: {'strategy': 'median'}

### Métricas de Evaluación

- **RMSE**: DATO: score tiene std=7.6662, rango=35.0. RMSE mide error en misma escala que el objetivo.
- **MAE**: DATO: línea base mae=6.8333. MAE es robusto a outliers como los de pop (9.3%).
- **R2**: DATO: línea base r2=-0.13399. R2 indica si el modelo supera la media.

### Estrategia de Validación

- **Tipo:** train_test_split
- **Test size:** 0.2
- **N splits:** 5
- **Justificación:** DATO: n_filas=1170. Con test_size=0.2, train=936 filas, test=234 filas. Suficiente para evaluación sin cross-validation complejo.

### Preprocesamiento Recomendado

- **Encoding categóricas:** frequency
- **Escalado numéricas:** robust
- **Manejo de nulos:** imputar_media
- **Justificación:** DATO: city(203), artist(87), venue(335) tienen cardinalidad alta (>20). Regla crítica: si filas/columnas_resultantes<5, sobreajuste. frequency evita crear muchas columnas. DATO: venue tiene 17 nulos, imputar_media es reversible. pop necesita limpieza antes de encoding.

## 6. Puntos Clave para Modelado

- DATO: pop contiene valores no numéricos como '???' que deben limpiarse antes del modelado
- DATO: venue tiene 17 nulos que requieren imputación
- DATO: minprice tiene ruido artificial (11 celdas modificadas con sigma=7.306)
- DATO: genre tiene categoría 'Undefined' que podría ser error de etiquetado
- DATO: weekend media=0.43077 sugiere posible confusión en codificación binaria

## 7. Limitaciones Identificadas

- DATO: city, artist, venue tienen cardinalidad alta (>20 únicos). one_hot crearía >5 columnas por feature, riesgo de sobreajuste con n_filas=1170
- DATO: pop tiene skewness=3.216 y 9.3% outliers, requiere transformación o robust scaling
- DATO: minprice tiene skewness=13.613 y 6.4% outliers, ruido artificial añadido (11 celdas)
- NOTA: genre podría tener 'Undefined' que no se puede verificar sin ver datos crudos

## 8. Desviaciones de Validación

- **transformacion_objetivo**: propuesto `none` → ejecutado `ninguna`. Razón: 'none' no está en capacidades disponibles

## 9. Gráficas EDA

- [Distribución KDE](../plots/eda_distribucion_kde.png)
- [eda_top_city](../plots/eda_top_city.png)
