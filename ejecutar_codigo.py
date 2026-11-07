#!/usr/bin/env python3
# Lanzador para ejecutar el código original sin modificar sus cálculos.
import argparse
from codigo_tesis_original import ejecutar_revision_terna

def main():
    p = argparse.ArgumentParser(description='Ejecuta el código original de tesis (modelo histórico en niveles)')
    p.add_argument('--archivo', required=True, help='Ruta a Excel con BASE_CORREGIDA')
    p.add_argument('--guardar', action='store_true', help='Guardar salidas junto al Excel')
    p.add_argument('--sin-graficas', action='store_true', help='No generar figuras')
    p.add_argument('--sin-toda-yamamoto', action='store_true', help='Omitir prueba Toda-Yamamoto')
    p.add_argument('--sin-mostrar', action='store_true', help='No imprimir tablas')
    a = p.parse_args()
    ejecutar_revision_terna(
        archivo=a.archivo,
        mostrar=not a.sin_mostrar,
        guardar=a.guardar,
        graficas=not a.sin_graficas,
        toda_yamamoto=not a.sin_toda_yamamoto,
        precios_wti=False,
        bootstrap_q0=False,
    )

if __name__ == '__main__':
    main()
