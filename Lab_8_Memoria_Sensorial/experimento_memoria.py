"""Experimento del Laboratorio IA 8: efecto de la memoria sensorial sobre los ciclos.

Para cada semilla entrena dos redes con el MISMO algoritmo, fitness y semilla:
  - "Con memoria": entrada [s_t, s_(t-1), a_(t-1)]  (12 valores)
  - "Sin memoria": las 8 entradas de memoria valen 0 (red de control)
Ambas se entrenan SOLO en el mapa original, se guardan en .pt y se prueban despues,
sin cambiar sus pesos, en los cuatro mapas de MAPAS.

Genera:
  - resultados_lab8.csv : tabla completa
  - resultados_lab8.md  : tabla, resumen, analisis de ciclos y trayectorias

Uso:
    python experimento_memoria.py
    python experimento_memoria.py --semillas 7 21 42 100 200   (mas semillas)
"""
import argparse
import csv
import sys

import robot_red_neuronal_memoria as lab

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

MODELOS = (("Con memoria", True, "mem"), ("Sin memoria", False, "sinmem"))


def dibujar_texto(trayectoria, obstaculos, inicio, meta):
    """S inicio, G meta, # obstaculo, * recorrido, R posicion final si no llego."""
    visitadas = set(trayectoria)
    final = trayectoria[-1]
    lineas = []
    for f in range(lab.TAMANO):
        fila = ""
        for c in range(lab.TAMANO):
            celda = (f, c)
            if celda == final and final != meta:
                fila += "R "
            elif celda == inicio:
                fila += "S "
            elif celda == meta:
                fila += "G "
            elif celda in obstaculos:
                fila += "# "
            elif celda in visitadas:
                fila += "* "
            else:
                fila += ". "
        lineas.append(fila.rstrip())
    return "\n".join(lineas)


def describir_ciclo(red, obstaculos, inicio, meta):
    """Repite el recorrido y, si el estado completo (posicion, obs. anterior, accion
    anterior) se repite, devuelve (periodo, celdas del ciclo). Si no hay ciclo, None."""
    posicion = inicio
    obs_ant, acc_ant = None, None
    vistos = {}
    posiciones = []
    for paso in range(lab.PASOS_MAXIMOS):
        estado = (posicion, obs_ant, acc_ant)
        if estado in vistos:
            ini = vistos[estado]
            celdas = posiciones[ini:]
            return len(celdas), celdas
        vistos[estado] = paso
        posiciones.append(posicion)
        comando, obs = lab.decidir(red, posicion, obstaculos=obstaculos,
                                   observacion_anterior=obs_ant, accion_anterior=acc_ant)
        posicion, _ = lab.mover(posicion, comando, obstaculos=obstaculos)
        if posicion == meta:
            return None
        obs_ant, acc_ant = obs, comando
    return None


def probar(red, nombre_mapa):
    obstaculos, inicio, meta = lab.MAPAS[nombre_mapa]
    tray, choques, utiles, _, acciones, rep, bucles = lab.recorrer(
        red, obstaculos=obstaculos, inicio=inicio, meta=meta)
    llego = tray[-1] == meta
    return {
        "llego": llego,
        "fitness": lab.evaluar(red, obstaculos=obstaculos, inicio=inicio, meta=meta),
        "choques": choques,
        "pasos_utiles": utiles,
        "pasos_totales": len(tray) - 1,
        "repeticiones": rep,
        "bucles": bucles,
        "ciclo": bucles > 0,
        "trayectoria": tray,
        "acciones": "".join(acciones),
        "ciclo_info": describir_ciclo(red, obstaculos, inicio, meta),
        "obstaculos": obstaculos, "inicio": inicio, "meta": meta,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--semillas", type=int, nargs="+", default=[7, 21, 42])
    ap.add_argument("--generaciones", type=int, default=40)
    args = ap.parse_args()

    filas, detalles = [], []
    for semilla in args.semillas:
        for nombre_modelo, usar_memoria, sufijo in MODELOS:
            print(f"== Semilla {semilla} | {nombre_modelo} ==")
            historial = []
            ruta = f"mejor_red_{sufijo}_s{semilla}.pt"
            red = lab.evolucionar(generaciones=args.generaciones, semilla=semilla,
                                  sin_pausa=True, guardar_mejor=ruta,
                                  usar_memoria=usar_memoria, historial=historial)
            primera = next((g for g, _, llego in historial if llego), None)
            gen_txt = primera if primera is not None else "no llego"
            red = lab.cargar_red(ruta)  # se prueba exactamente lo guardado
            for nombre_mapa in lab.MAPAS:
                r = probar(red, nombre_mapa)
                filas.append({
                    "Semilla": semilla, "Modelo": nombre_modelo, "Mapa": nombre_mapa,
                    "Llego": "Si" if r["llego"] else "No",
                    "Gen. 1a llegada (entrenamiento)": gen_txt,
                    "Fitness": r["fitness"], "Choques": r["choques"],
                    "Pasos utiles": r["pasos_utiles"], "Pasos totales": r["pasos_totales"],
                    "r (repetidas)": r["repeticiones"], "b (bucles)": r["bucles"],
                    "Ciclo": "Si" if r["ciclo"] else "No",
                })
                detalles.append((semilla, nombre_modelo, nombre_mapa, r))
                print(f"   {nombre_mapa:<15} llego={r['llego']!s:<5} fit={r['fitness']:>6} "
                      f"choques={r['choques']:>2} utiles={r['pasos_utiles']:>2} "
                      f"r={r['repeticiones']:>2} b={r['bucles']:>2}")

    campos = list(filas[0].keys())
    with open("resultados_lab8.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        w.writerows(filas)

    with open("resultados_lab8.md", "w", encoding="utf-8") as f:
        f.write("# Resultados Laboratorio 8\n\n")
        f.write("Las redes se entrenaron solo en el mapa `original` y se probaron sin "
                "cambiar sus pesos en los demas mapas. r = posiciones repetidas, "
                "b = bucles sensoriomotores (estado completo repetido).\n\n")
        f.write("## Tabla completa\n\n")
        f.write("| " + " | ".join(campos) + " |\n|" + "|".join(["---"] * len(campos)) + "|\n")
        for fila in filas:
            f.write("| " + " | ".join(str(fila[c]) for c in campos) + " |\n")

        f.write("\n## Resumen por modelo y mapa\n\n")
        f.write("| Modelo | Mapa | Llegadas | Con ciclo | b promedio | Choques promedio |\n|---|---|---|---|---|---|\n")
        for nombre_modelo, _, _ in MODELOS:
            for nombre_mapa in lab.MAPAS:
                sub = [d for d in detalles if d[1] == nombre_modelo and d[2] == nombre_mapa]
                n = len(sub)
                f.write(f"| {nombre_modelo} | {nombre_mapa} | {sum(d[3]['llego'] for d in sub)}/{n} "
                        f"| {sum(d[3]['ciclo'] for d in sub)}/{n} "
                        f"| {sum(d[3]['bucles'] for d in sub) / n:.1f} "
                        f"| {sum(d[3]['choques'] for d in sub) / n:.1f} |\n")

        f.write("\n## Analisis de ciclos\n\n")
        hubo = False
        for semilla, modelo, mapa, r in detalles:
            if r["ciclo_info"]:
                hubo = True
                periodo, celdas = r["ciclo_info"]
                f.write(f"- Semilla {semilla}, {modelo}, mapa {mapa}: ciclo de periodo {periodo} "
                        f"por las celdas {celdas}; b = {r['bucles']}.\n")
        if not hubo:
            f.write("No se detectaron ciclos en ninguna prueba.\n")

        f.write("\n## Trayectorias\n\nLeyenda: `S` inicio, `G` meta, `#` obstaculo, "
                "`*` visitada, `R` posicion final si no llego.\n")
        for semilla, modelo, mapa, r in detalles:
            f.write(f"\n### Semilla {semilla} - {modelo} - mapa {mapa}\n\n")
            f.write(f"Acciones: `{r['acciones']}`\n\n```\n")
            f.write(dibujar_texto(r["trayectoria"], r["obstaculos"], r["inicio"], r["meta"]))
            f.write("\n```\n")

    print("\nListo: resultados_lab8.csv y resultados_lab8.md generados.")


if __name__ == "__main__":
    main()
