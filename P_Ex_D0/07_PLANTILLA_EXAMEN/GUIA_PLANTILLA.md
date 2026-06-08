# 07 · ⭐ Plantilla de Examen (notebook + módulo)

Herramienta final para el examen práctico. **Carpeta autocontenida**: cópiala/zipéala tal cual.

## Contenido
| Archivo | Qué es |
|---|---|
| `PLANTILLA_EXAMEN.ipynb` | Notebook ejecutable de principio a fin. **No define el MLP**: lo importa. |
| `practica5_mlp.py` | Módulo reutilizable con `MLP`, `MLPRelu`, `to_onehot` (basado en tu práctica 5). |
| `GUIA_PLANTILLA.md` | Este archivo. |

## Cómo se usa (importación, sin duplicar código)
El notebook hace:
```python
from practica5_mlp import MLP, MLPRelu, to_onehot
```
→ el código del perceptrón vive **solo** en `practica5_mlp.py`. Si corriges el MLP, lo corriges en un único sitio y el notebook lo hereda. `MLPRelu` hereda de `MLP` (no reescribe lo común).

## Flujo del día del examen
1. Copia este folder y mete dentro el **CSV** del examen.
2. Abre `PLANTILLA_EXAMEN.ipynb` y edita solo la **CELDA CONFIG**:
   ```python
   FILE = "Customer.csv"      # el CSV que te den
   TARGET = "Segmentation"    # columna objetivo
   COLS_DROP = ["ID"]         # IDs/columnas a eliminar
   RANDOM_STATE = 13          # el que fije el enunciado
   TEST_SIZE = 0.20           # el que fije el enunciado
   ```
3. `Kernel → Restart & Run All`. Debe ejecutar sin errores.
4. Ajusta hiperparámetros si no llegas al accuracy mínimo (ver `../03_PRACTICA/`).

## Orden de celdas del notebook
Config → Imports (importa el módulo) → Carga/exploración → Limpieza+imputación (Ej1) → Encoding OHE+LabelEncoder → Split+escalado → Scatter PCA (Ej2) → **MLP propio** (Ej3, prueba ≥3 capas + final + confusión) → MLPClassifier (Ej4) → KNN (Ej5) → RandomForest (Ej5) → DecisionTree → Comparación (análisis) → **MLPRelu relu+softmax** (Ej6) → MLP binario (opcional).

## Entrega (requisito del examen)
Incluye en el zip: el `.ipynb`, **`practica5_mlp.py`** (tu librería, exigida) y el CSV. Rutas **relativas**. Tu nombre en la 1ª celda y en el nombre del zip.

> La librería se basa en tu práctica 5 (mismo algoritmo: thetas con bias, feedforward, backprop, coste + L2). No es código ajeno → no es copia.

## Detalle de cada paso
Para entender/ampliar cada bloque, ver las guías de `../03_PRACTICA/`:
`01_limpieza_datos.md`, `03_usar_mi_MLP.md`, `04_modelos_sklearn.md`, `05_MLPRelu_softmax.md`, `07_errores_comunes.md`.
