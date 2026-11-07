# Volatilidad de precios internacionales y balanza comercial de Guatemala (2010–2024)

**Autor:** José Andrés Estrada Morales  
**Universidad:** Universidad Rafael Landívar, Guatemala (2026)  
**Investigación:** *Volatilidad condicional de los precios internacionales de productos agrícolas de exportación y del petróleo WTI y su asociación dinámica con el saldo de la balanza comercial de Guatemala (2010–2024): un análisis mediante modelos GARCH y ARDL*.

## Archivos del repositorio

- `codigo_tesis_original.py`: código Python extraído del HTML facilitado por el autor, sin modificar los cálculos.
- `Base_datos_tesis_corregida_2010_2024.xlsx`: base Excel subida al repositorio por el propietario. Su publicación no supone verificación independiente de permisos de redistribución.
- `ejecutar_codigo.py`: lanzador para ejecutar el código original desde terminal.
- `requirements.txt`: bibliotecas Python.
- `NOTA_REPRODUCIBILIDAD.md`: alcance y diferencias frente a las estimaciones de la versión final.

## Advertencia de reproducibilidad

**El código de este repositorio es una versión histórica del análisis y no reproduce íntegramente las tablas definitivas de la tesis.** El ARDL principal del código usa el saldo de la balanza comercial **en niveles**. La versión final de la tesis reporta un modelo de la **primera diferencia mensual del saldo (ΔSC)**, con 2 rezagos y 169 observaciones. No se incluye una implementación completa y verificada del contraste Wald-HAC de ese modelo final ni del bootstrap de dos etapas que reestime las cinco volatilidades. Revisar la [nota de reproducibilidad](NOTA_REPRODUCIBILIDAD.md) antes de citar resultados.

## Ejecución

Python 3.11 o posterior es recomendable. Con el Excel en el mismo directorio:

```bash
python -m pip install -r requirements.txt
python ejecutar_codigo.py --archivo Base_datos_tesis_corregida_2010_2024.xlsx
```

Para guardar salidas:

```bash
python ejecutar_codigo.py --archivo Base_datos_tesis_corregida_2010_2024.xlsx --guardar
```

El código no fue reestimado y validado de principio a fin en el proceso de publicación del repositorio. Algunas pruebas opcionales dependen del entorno y de los paquetes instalados.

## Cita y uso

Puede citarse el enlace a este repositorio junto con la fecha de consulta. Para un identificador persistente, el autor puede crear un *release* y archivarlo en Zenodo. **No se ha seleccionado aún una licencia de reutilización**, por lo que la consulta pública no implica autorización general de reproducción o modificación.
