# T06 · Aprendizaje No Supervisado

## Resumen corto
Sin etiquetas. Se busca estructura en los datos: **clustering** (agrupar ejemplos similares) y **reducción de dimensionalidad** (PCA). Tienes una práctica resuelta de K-Means y jerárquico.

## Resumen completo
### K-Means
- Agrupa en **k** clusters. Algoritmo: inicializar k centroides → asignar cada punto al centroide más cercano → recalcular centroides → repetir hasta converger.
- Hay que elegir **k** (método del codo / elbow: graficar inercia vs k).
- Sensible a escala e inicialización (`k-means++`).

### Clustering jerárquico
- Construye un **dendrograma** uniendo (aglomerativo) o dividiendo clusters por similitud. No requiere fijar k de antemano; se corta el dendrograma a la altura deseada.

### PCA (reducción de dimensionalidad)
- Proyecta los datos a menos dimensiones conservando la máxima **varianza**. Útil para **visualizar** (2D) y para acelerar/limpiar antes de otros modelos. (Tu `data_mining.py` lo usa para el scatter PCA.)

## Conceptos clave
- Cluster, centroide, inercia, método del codo, dendrograma, PCA, componentes principales, varianza explicada.

## Preguntas frecuentes
- *"Datos sin etiquetar, ¿cómo afronto el problema?"* → no supervisado: **K-Means/jerárquico** para agrupar, **PCA** para reducir/visualizar. Si surgen etiquetas, pasar a supervisado.

## Errores habituales
- No escalar antes de clustering/PCA.
- Elegir k arbitrario sin justificar (usar el codo).

## Material disponible
- `Practicas resueltas/Clustering/Clustering - K_Means Jerárquico.ipynb` (+ `shopping_data.csv`).
