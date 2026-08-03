# ASUNCIONES de la implementación (revisión pendiente del usuario)
1. **Mini-evals por bloque**: sustituidos en esta pasada por asserts integrados (conservación, columnas exhaustivas, guardarraíles) + la ejecución end-to-end como smoke. El harness de 3 casos/bloque queda pendiente de la sesión de obra formal.
2. **L2 con hermanas singleton**: L2 junta hermanas pequeñas entre sí; si tras anular la dim muda la hermana queda sola (p. ej. tele: el web grande conserva su id por tener soporte), el nivel no gana n — lo resuelve la credibilidad (etapa 3), visible en support_chain. Coherente con P5/P6.
3. **Contrafactual Simpson**: walk-forward sobre los últimos 6 meses con verdad (train+test), método segmentado = tasas por serie ≤t−1 × pesos reales de t. El mes en curso no se excluye aquí por ser diagnóstico (sí se excluiría del backtest de selección real).
4. **k de credibilidad**: fijadas pragmáticas (tasa k=60, uplift k=24) — calibración varianza-dentro/entre pendiente con datos reales.
5. **Backtest**: 1 origen por horizonte (h=1..4) por presupuesto; el diseño real usa múltiples orígenes. Series elegibles: ≥10 meses y n≥suelo/2 al grano L2.
6. **Uplift**: definido como AUV_renovado/AUV_pipeline por fila; vara s/√meses. Intervalos de descuento: el sintético ya trae niveles (d0/d40); el mapeo continuo→intervalos queda como config futura.
7. **Padres de tasa**: la credibilidad encoge hacia la celda mandatory (no hacia la cadena completa de padres por levels) — suficiente para la demo; la jerarquía multinivel queda para la obra.
8. **SQL**: main corre sobre sintético; config_sqlserver de la cantera se enchufa sobrescribiendo Config.write (una clase).
