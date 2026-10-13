# Laboratorio IA 8: Redes con memoria sensorial para romper ciclos

**Materia:** Inteligencia Artificial

## 1. Qué se conservó y qué se cambió (respecto al Lab 7)
| Elemento | Lab 7 | Lab 8 |
|---|---|---|
| Mundo (cuadrícula 8x8, obstáculos, sensores U D L R) | sí | **conservado** |
| Decisión por `argmax` de 4 logits | sí | **conservado** |
| Algoritmo genético (40 individuos, 6 élites, 16 padres, cruce en un punto, mutación gaussiana) | sí | **conservado** |
| Entrada de la red | 4 (`s_t`) | **12** (`s_t`, `s_{t-1}`, `a_{t-1}` one-hot) |
| Capa oculta | 8 neuronas | **16 neuronas** |
| Fitness | `500 - 35d + 4p - 30c - 3r + 2000m` | se agrega **`- 10b`** (bucles sensoriomotores) |
| Medida de ciclo | posiciones repetidas (`r`) | `r` **y** `b`: estado completo `(posición, obs. anterior, acción anterior)` repetido |
| Mapas de prueba | 1 modificado | `modificado`, `complejo` (20 obstáculos) e `inicio_alterno` |
| Red de control | no | `--sin-memoria`: misma red con las 8 entradas de memoria en 0 |

`b` detecta un ciclo real: como la política es determinista, si el estado completo que ve la
red se repite, el robot repetirá esa secuencia para siempre.

## 2. Cómo reproducir
```bash
python experimento_memoria.py                 # 3 semillas (7, 21, 42), con y sin memoria, 4 mapas
python robot_red_neuronal_memoria.py --cargar-red mejor_red_mem_s7.pt --mapa complejo --sin-pausa
```
Las redes se entrenan **solo** en el mapa `original` y se prueban sin cambiar sus pesos.

## 3. Tabla de resultados
<!-- Pega aquí la tabla completa y el resumen de resultados_lab8.md -->

## 4. Análisis de los ciclos
<!-- Usa la sección "Analisis de ciclos" y las trayectorias de resultados_lab8.md.
     Describe: ¿en qué celdas se atascan?, ¿qué periodo tiene el ciclo?, ¿la red con memoria
     cicla menos que la de control?, ¿en qué mapas? -->

## 5. Respuestas a las preguntas del laboratorio
- ¿La memoria sensorial reduce la cantidad de ciclos? (*)
- ¿La red aprende a escapar de patrones repetitivos? (*)
- ¿La red con memoria llega a la meta en mapas más complejos? (*)
- ¿Qué pasa si el fitness penaliza repetición de posiciones y secuencias? (*)

## 6. Conclusión
<!-- Borrador: ajusta lo marcado con (*) con TUS resultados antes de entregar. -->
La memoria temporal amplía lo que la red puede distinguir: sin ella solo ve 16 situaciones
locales, y con ella el estado de entrada incluye lo que se vio y se hizo un paso antes (*).
Esto permite romper ciclos de periodo corto, porque la misma observación ya no obliga siempre
a la misma acción (*). Sin embargo, una memoria de un solo paso es de corto alcance: no
reconoce ciclos largos ni conoce la dirección de la meta, y una red entrenada en un único
mapa puede memorizar una ruta en vez de aprender una estrategia general (*). La penalización
`-10b` empuja a la evolución hacia políticas que no se repiten, pero no sustituye a la
información que falta. Para mejorar la generalización haría falta entrenar en varios mapas,
ampliar la ventana de memoria y dar a la red la dirección relativa de la meta.
