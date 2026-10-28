# Reporte de Resultados - Etapa 4

**Fecha:** 2026-10-28T09:49:06
**Variable objetivo:** score

## 1. Resumen Ejecutivo

RandomForestRegressor es el modelo ganador debido a su superior rendimiento en métricas clave sin sobreajuste, aunque persisten desafíos de calidad de datos que requieren atención para optimizar resultados futuros.

**Modelo ganador:** RandomForestRegressor

## 2. Contexto del Pipeline

### Plan Ejecutado
- **Transformación del objetivo:** ninguna
- **Encoding categóricas:** frequency
- **Escalado numéricas:** robust
- **Manejo de nulos:** imputar_media

### Comparación con Línea Base
RandomForestRegressor logra un RMSE de 4.9689 en test, frente a 8.4416 del modelo DummyRegressor (línea base), lo que indica una mejora sustancial en la precisión predictiva.

| Modelo | RMSE | MAE | R² |
|--------|------|-----|-----|
| Línea base (DummyRegressor) | 8.4416 | 6.8333 | -0.1340 |
| RandomForestRegressor | 4.9689 | 3.7889 | 0.6071 |

## 3. Comparación de Modelos

| Modelo | Train RMSE | Test RMSE | Train MAE | Test MAE | Train R² | Test R² | Sobreajuste |
|--------|-----------|-----------|-----------|----------|----------|---------|-------------|
| RandomForestRegressor | 4.0172 | 4.9689 | 3.0156 | 3.7889 | 0.7178 | 0.6071 | No |
| Ridge | 7.4658 | 8.0007 | 6.2892 | 6.6464 | 0.0252 | -0.0186 | No |
| DummyRegressor | 7.6728 | 8.4416 | 6.3248 | 6.8333 | -0.0296 | -0.1340 | No |

## 4. Análisis de Sobreajuste

La diferencia entre R² en train (0.71777) y test (0.60711) es de 0.11066, indicando una caída moderada pero aceptable sin sobreajuste detectado (sobreajuste: false).

![Sobreajuste](../plots/sobreajuste.png)

## 5. Comparación Visual

![RMSE](../plots/comparacion_rmse.png)

![R²](../plots/comparacion_r2.png)

## 6. Interpretación de Mercado

En el contexto real, predecir con precisión la puntuación del concierto ('score') permite a organizadores y artistas anticipar la recepción del público, optimizando estrategias de precios y promoción basadas en variables como género, precio y ubicación. Sin embargo, la calidad de los datos actuales (por ejemplo, valores nulos o categorías no limpias) podría afectar la fiabilidad de estas predicciones en escenarios con información incompleta.

## 7. Limitaciones

- La columna 'pop' contiene valores no numéricos ('???') que deben limpiarse antes del modelado
- Existen 17 valores nulos en 'venue' que requieren imputación adecuada
- La categoría 'Undefined' en 'genre' podría indicar errores de etiquetado que afectan la interpretación

## 8. Recomendaciones Futuras

- Limpiar los valores no numéricos en 'pop' para asegurar consistencia en las características
- Imputar los nulos de 'venue' con técnicas robustas como media o KNN
- Investigar y corregir posibles errores de etiquetado en la categoría 'genre'

## 9. Desviaciones de Ejecución

Sin desviaciones: todo el plan se ejecutó tal como fue propuesto.
