"""Experimento del Laboratorio IA 7.

Entrena tres redes (semillas 7, 21 y 42), guarda cada mejor individuo en
mejor_red_s<semilla>.pt y prueba cada una, SIN cambiar sus pesos, en el mapa
original y en el mapa modificado. Genera:

  - resultados.csv : tabla de resultados en formato CSV
  - resultados.md  : tabla en Markdown + trayectorias dibujadas en texto

Uso (desde la carpeta del repositorio):
    python experimento.py
"""
import csv
import sys

import robot_red_neuronal_evolutiva as lab

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SEMILLAS = (7, 21, 42)
GENERACIONES = 40
MAPAS = (
    ("Original", lab.OBSTACULOS),
    ("Modificado", lab.OBSTACULOS_MODIFICADO),
)


def dibujar_texto(trayectoria, obstaculos):
    """Dibuja el mapa en ASCII: S inicio, G meta, # obstaculo, * recorrido, R final."""
    visitadas = set(trayectoria)
    final = trayectoria[-1]
    lineas = []
    for f in range(lab.TAMANO):
        fila = ""
        for c in range(lab.TAMANO):
            celda = (f, c)
            if celda == final and final != lab.META:
                fila += "R "
            elif celda == lab.INICIO:
                fila += "S "
            elif celda == lab.META:
                fila += "G "
            elif celda in obstaculos:
                fila += "# "
            elif celda in visitadas:
                fila += "* "
            else:
                fila += ". "
        lineas.append(fila.rstrip())
    return "\n".join(lineas)


def probar(red, obstaculos):
    trayectoria, choques, pasos_utiles, _, acciones = lab.recorrer(red, obstaculos=obstaculos)
    llego = trayectoria[-1] == lab.META
    repetidas = len(trayectoria) - len(set(trayectoria))
    return {
        "llego": llego,
        "fitness": lab.evaluar(red, obstaculos=obstaculos),
        "choques": choques,
        "pasos_utiles": pasos_utiles,
        "pasos_totales": len(trayectoria) - 1,
        # La politica es determinista y solo depende de la posicion, asi que
        # repetir una celda sin haber llegado a la meta implica un ciclo.
        "ciclo": repetidas > 0 and not llego,
        "trayectoria": trayectoria,
        "acciones": "".join(acciones),
    }


def main():
    filas = []
    detalles = []
    for semilla in SEMILLAS:
        print(f"== Entrenando semilla {semilla} ({GENERACIONES} generaciones) ==")
        evo = lab.evolucionar(semilla=semilla, maximo_generaciones=GENERACIONES)
        ruta = f"mejor_red_s{semilla}.pt"
        lab.guardar_red(evo["red"], ruta)
        print(f"   guardado: {ruta} | llego={evo['llego']} | generacion={evo['generacion']}"
              f" | fitness={evo['puntaje']}")

        # Se recarga desde disco para probar exactamente lo que se guardo.
        red = lab.cargar_red(ruta)
        for nombre_mapa, obstaculos in MAPAS:
            r = probar(red, obstaculos)
            filas.append({
                "Semilla": semilla,
                "Mapa": nombre_mapa,
                "Llego": "Si" if r["llego"] else "No",
                "Generacion": evo["generacion"],
                "Fitness": r["fitness"],
                "Choques": r["choques"],
                "Pasos utiles": r["pasos_utiles"],
                "Ciclo": "Si" if r["ciclo"] else "No",
            })
            detalles.append((semilla, nombre_mapa, r, obstaculos))
            print(f"   {nombre_mapa:<10} llego={r['llego']!s:<5} fitness={r['fitness']:>6}"
                  f" choques={r['choques']:>2} pasos_utiles={r['pasos_utiles']:>2} ciclo={r['ciclo']}")

    campos = list(filas[0].keys())
    with open("resultados.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        w.writerows(filas)

    with open("resultados.md", "w", encoding="utf-8") as f:
        f.write("# Resultados del experimento\n\n")
        f.write("Generacion = generacion en la que termino la evolucion en el mapa original "
                "(es la misma para ambas filas de una semilla). Las redes se probaron sin "
                "cambiar sus pesos.\n\n")
        f.write("| " + " | ".join(campos) + " |\n")
        f.write("|" + "|".join(["---"] * len(campos)) + "|\n")
        for fila in filas:
            f.write("| " + " | ".join(str(fila[c]) for c in campos) + " |\n")
        f.write("\n## Trayectorias\n\n")
        f.write("Leyenda: `S` inicio, `G` meta, `#` obstaculo, `*` celda visitada, "
                "`R` posicion final si no llego.\n")
        for semilla, nombre_mapa, r, obstaculos in detalles:
            f.write(f"\n### Semilla {semilla} - mapa {nombre_mapa}\n\n")
            f.write(f"Acciones: `{r['acciones']}`\n\n")
            f.write("```\n" + dibujar_texto(r["trayectoria"], obstaculos) + "\n```\n")

    print("\nListo: resultados.csv y resultados.md generados.")


if __name__ == "__main__":
    main()
