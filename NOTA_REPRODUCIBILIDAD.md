# Nota de reproducibilidad para evaluadores

## Procedencia del código

El archivo `codigo_tesis_original.py` conserva el contenido de código extraído de `Codigo_Tesis(2).html`, suministrado por el autor. No se modificaron los algoritmos, los estimadores ni las salidas calculadas. Se verificó que el código tiene sintaxis Python válida, **pero no se realizó una ejecución independiente completa**.

## Diferencias respecto a la tesis final

- **En este repositorio:** el ARDL original modela `balanza`, correspondiente al saldo de la balanza comercial **en niveles**.
- **En la tesis definitiva:** el modelo principal tiene como dependiente `ΔSC` (primera diferencia mensual del saldo), 2 rezagos de `ΔSC`, 5 volatilidades contemporáneas, diferencias de controles, 11 efectos mensuales y 169 observaciones.
- El código publicado **no incluye una rutina independiente completa y verificada** para replicar exactamente la inferencia Wald-HAC del modelo definitivo.
- El bootstrap del código original mantiene fija la matriz de regresores: **no equivale a un procedimiento completo que reestime las cinco etapas GARCH**.

En consecuencia, **no debe presentarse el código actual como una replicación exacta de las tablas econométricas finales**. Para completar esa reproducción es necesario depositar el código de la especificación final, las decisiones de limpieza y transformación y las salidas contrastadas contra los cuadros de la tesis.

## Datos y derechos

El propietario del repositorio cargó `Base_datos_tesis_corregida_2010_2024.xlsx`. La presencia pública de ese archivo **no equivale a verificar el derecho de redistribución** de cada serie. Conviene comprobar las condiciones de las fuentes originales antes de mantenerlo públicamente disponible.

## Estado de validación

La publicación facilita revisión académica y transparencia, pero no certifica por sí misma la validez de todas las inferencias, del panel de Phillips–Perron ni de los cálculos que todavía no puedan reproducirse con este depósito.

**Estado:** código histórico disponible; reproducción exacta del modelo final pendiente.
