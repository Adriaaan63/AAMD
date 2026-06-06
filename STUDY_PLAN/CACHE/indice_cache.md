# Índice de CACHE — AAMD Preparación

> Registro persistente de archivos procesados para evitar relecturas costosas.
> Si un archivo NO ha cambiado (mismo hash), reutilizar el resumen de aquí en vez de releer el PDF/documento.
> Última actualización: 2026-06-06

## Convenciones
- **Estado**: `PROCESADO` (resumido) · `NO_LEIDO` (solo metadatos, lectura diferida) · `DUPLICADO` (idéntico a otro)
- Los PDF de teoría son **slides costosos** → solo se procesan bajo demanda (ver política de budget).

---

## Documentos PROCESADOS (resumen disponible, NO releer si hash igual)

| Hash (MD5) | Fecha mod | Archivo | Resumen |
|---|---|---|---|
| `5fa044c9…e4c5` | 2026-01-09 | `Examenes/Ordinaria/Enunciado.pdf` | Examen ordinaria enero 2026 (el que suspendió). Dataset **Customer.csv** (segmentación de clientes, multiclase). Teórico 2pts (8 preguntas diagnóstico/elección modelo) + Práctico 8pts (limpieza+imputación, scatter, MLP propio, MLPClassifier, KNN, RandomForest, análisis, MLPRelu con relu+softmax). |
| `c291fdd1…fb2c` | 2026-06-06 | `Examenes/Teorico/ExamenesTeoriaResueltos.pdf` | Teóricos de Enero y Junio **CON respuestas modelo**. Cubre: extractor de características (CNN/autoencoder) antes de MLP; linealidad de modelos; diagnóstico baseline/overfit; cálculo neuronas con OHE; autoencoder codificador/decodificador; refuerzo (Q-Learning, no entró este año). |
| `5142414f…c8b3` | 2024-01-18 | `Examenes/.../ExamenEnero20024.pdf` | Examen enero 2024. Dataset **dementia** (Demented/Nondemented/Converted). Práctico: limpieza, scatter, MLP propio (≥1 capa oculta, ≥65%), DecisionTree (≥85%, rs=42), comparación, MLP binario (Converted→Demented). Anexo con API pandas/numpy útil. |
| `0a5541b9…f7ac` | 2025-12-08 | `Examenes/.../ExamenErasmus/Enunciado.pdf` | **En realidad es examen enero 2025**. Dataset **heart.csv** (cardiopatía, binaria). Mismo patrón: limpieza, scatter, MLP propio (rs=0, test 25%, ≥84%), MLPClassifier, KNN, RandomForest, comparación, MLPRelu softmax. |
| `e123e6da…f04e8` | 2025-11-28 | `Practicas resueltas/Practica5/LearningPy/models/MLP.py` | **Librería MLP propia de Marcos** (núcleo del examen). Clase MLP genérica multicapa con sigmoid, cross-entropy + L2, feedforward/backprop. Núcleo CORRECTO. Funciones helper finales ROTAS (ver diagnóstico). |
| `e123e6da…f04e8` | 2026-01-07 | `Examenes/Ordinaria/Practico/mlp.py` | **DUPLICADO** (hash idéntico al MLP.py de Práctica 5). Es el que entregó en el examen. |

## Documentos NO_LEIDO (solo metadatos — lectura diferida según budget_control)

> Decisión: los 9 PDF de teoría son slide decks grandes (Tema01=3.3MB … total ≈10MB) y **redundantes** con lo que los exámenes ya revelan que se pregunta. No se leen en profundidad para no malgastar presupuesto. Los resúmenes de `02_RESUMENES` se construyen desde los exámenes + respuestas modelo + el código. Si se necesita más profundidad en un tema concreto, procesar solo ese PDF.

| Archivo | Tamaño | Tema | ¿Cuándo leer? |
|---|---|---|---|
| `Teoria/Tema01 Introducción…pdf` | 3.3 MB | Intro ML | Solo si falta concepto base |
| `Teoria/Tema02 Regresion…pdf` (+Multivariable, +Clasificación) | ~2 MB | Regresión/Clasificación/Logística | Baja prioridad examen |
| `Teoria/Tema03 Redes Neuronas…pdf` (+Aprendizaje, +Ejemplos) | ~2 MB | Redes neuronales / MLP | Cubierto por el código MLP.py |
| `Teoria/Tema04 Diseño de sistemas…pdf` | 1.8 MB | **Diseño ML (diagnóstico)** | ALTA prioridad teórica — cubierto por exámenes resueltos |
| `Teoria/Tema05 Otras técnicas…pdf` | 1.3 MB | **KNN, árboles, RandomForest, SVM** | ALTA prioridad práctica |
| `Teoria/Tema06 Aprendizaje no supervisado.pdf` | 886 KB | Clustering / K-Means | Media |
| `Teoria/Tema07 Deep Learning.pdf` | 831 KB | **CNN, autoencoders** | Media-alta (teoría) |
| `Teoria/Tema08…Refuerzo.pdf` | 294 KB | Aprendizaje por refuerzo | BAJA (profesor dijo que no entró) |
| `Teoria/Tema09-IAGenerativa.pdf` | 597 KB | IA generativa (GAN) | Baja |
| `Teoria/videos_apoyo.md` | <1 KB | 4 vídeos YouTube (3Blue1Brown NN + otros) | PROCESADO (links de apoyo) |

## Carpetas redundantes detectadas (NO reprocesar)
- `Examenes/Ordinaria/EntregaExamen_Marcos_PerezMartinez/Teoria/` = **copia idéntica** de `Teoria/` (mismos tamaños). Procesar solo `Teoria/`.
- `Examenes/Ordinaria/EntregaExamen…/Python/` ≈ copia de `Practicas resueltas/Practica5/LearningPy/` (proyecto Unity tanques). Material del proyecto, no del examen tabular.
- `raw_data_sets/` (40+ CSVs TankTraining) = datos del proyecto Unity. **Irrelevantes** para el examen.

## Reestructuración (plantilla de examen)
- La plantilla del examen vive en `07_PLANTILLA_EXAMEN/` (no en `03_PRACTICA/`).
- El MLP/MLPRelu **no se duplican** en el notebook: se centralizan en el módulo `07_PLANTILLA_EXAMEN/practica5_mlp.py` y el notebook hace `from practica5_mlp import MLP, MLPRelu, to_onehot`.
- `practica5_mlp.py` está basado en el MLP de la Práctica 5 (versión corregida; `MLPRelu` hereda de `MLP`). Notebook validado: JSON correcto, sin clases inline, módulo entrena al 100% en datos sintéticos.
- Generador del notebook: `CACHE/_build_nb.py` (ejecutar para regenerar tras editar defaults).

## Notas de proceso
- Asignatura: **AAMD** (Aprendizaje Automático y Minería de Datos), Grado en Desarrollo de Videojuegos, UCM.
- Estructura examen **invariante** 2024→2026: Teórico 2pts + Práctico 8pts (pipeline de clasificación con CSV).
- Convocatoria objetivo: **EXTRAORDINARIA** (la ordinaria de enero 2026 está suspendida: 1,25/8 práctico + 0,65/2 teórico).
