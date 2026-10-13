# Laboratorio IA 7: Red neuronal pequeña y evolución por algoritmo genético

**Materia:** Inteligencia Artificial

## 1. Objetivo
Controlar al robot de percepción local con una red neuronal pequeña (PyTorch, 4-8-4 con ReLU)
cuyos pesos se evolucionan con un algoritmo genético (40 individuos, 6 élites, cruce en un
punto y mutación gaussiana). El mejor individuo se guarda en disco y se prueba en un mapa
modificado sin cambiar sus pesos.

## 2. Mapas
- **Original:** `OBSTACULOS` (10 obstáculos), inicio (0,0), meta (7,7).
- **Modificado:** `OBSTACULOS_MODIFICADO` (10 obstáculos distintos), mismo tamaño, inicio y meta.
  Inicio y meta están libres y el camino mínimo es de 14 pasos, igual que en el original.

## 3. Cómo reproducir
```bash
python experimento.py
```
Genera `mejor_red_s7.pt`, `mejor_red_s21.pt`, `mejor_red_s42.pt`, `resultados.csv` y `resultados.md`.

Prueba individual de un modelo guardado:
```bash
python robot_red_neuronal_evolutiva.py --cargar-red mejor_red_s7.pt --mapa original --sin-pausa
python robot_red_neuronal_evolutiva.py --cargar-red mejor_red_s7.pt --mapa modificado --sin-pausa
```

## 4. Tabla de resultados
<!-- Pega aquí la tabla generada en resultados.md -->

| Semilla | Mapa | Llegó | Generación | Fitness | Choques | Pasos útiles |
|---|---|---|---|---|---|---|
| 7 | Original | | | | | |
| 7 | Modificado | | | | | |
| 21 | Original | | | | | |
| 21 | Modificado | | | | | |
| 42 | Original | | | | | |
| 42 | Modificado | | | | | |

## 5. Trayectorias
Los dibujos de cada trayectoria (original y modificado) están en `resultados.md`.
<!-- Describe aquí 1 o 2 trayectorias representativas: por dónde avanza, dónde se atasca o cicla. -->

## 6. Conclusión (8 a 10 líneas)
<!-- Borrador: ajusta las frases marcadas con (*) según TUS resultados antes de entregar. -->
La evolución encontró, por selección de fitness y sin ninguna regla escrita a mano, pesos que
llevan al robot a la meta en el mapa de entrenamiento (*). Cambió la representación del
individuo, no el mecanismo de aprendizaje: antes se evolucionaban reglas y ahora pesos. La red
solo recibe cuatro bits de percepción local, es decir, solo puede distinguir 16 situaciones, y
con `argmax` su política es determinista: ante la misma observación siempre hace lo mismo.
Un peso ya no codifica una regla concreta; contribuye a varias respuestas a la vez. Como la red
no sabe dónde está ni dónde queda la meta, dos celdas distintas con los mismos vecinos reciben
la misma acción, y eso permite ciclos (*). En el mapa modificado, los resultados (*) indican
que la política aprendida no generaliza de forma general: reutiliza respuestas locales que
sirven en el mapa original y falla cuando la geometría cambia. Para que generalice mejor
haría falta entrenar en varios mapas y dar a la red la dirección de la meta o la posición
relativa (por ejemplo, dos entradas extra con el signo de la diferencia en fila y columna).
