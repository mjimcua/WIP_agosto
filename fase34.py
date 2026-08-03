"""fase34.py — Ensamblaje (fila futura × dos lookups) + backtest por horizonte con catálogo de técnicas."""
import numpy as np, pandas as pd

def f3_ensamblaje(fina, v2, est, gu, cfg):
    """FASE 3: esperado = pipeline × tasa(serie) × uplift(comb) fila a fila; etiquetas semánticas; guardarraíles."""
    fut = fina[fina[cfg.role_col] == "projection"].copy()
    fut["fs_id"] = fut[cfg.grano_tasa].astype(str).agg("|".join, axis=1)
    tasa = est.set_index("fs_id")["tasa_final"].clip(upper=cfg.rate_cap)
    fut["tasa"] = fut["fs_id"].map(tasa)
    fut["tasa"] = fut["tasa"].fillna(fut[cfg.mandatory].astype(str).agg("|".join, axis=1).map(
        est.assign(cel=est["fs_id"].str.split("|").str[0]).groupby("cel")["tasa_final"].mean()))
    fut["gu"] = fut[cfg.grano_uplift].astype(str).agg("|".join, axis=1)
    upl = gu.set_index(["gu", "comb_id"])["uplift_final"]
    fut["uplift"] = [upl.get((a, b), 1.0) for a, b in zip(fut["gu"], fut["comb_id"])]
    fut["esperado_usd"] = fut[cfg.pipeline_usd_col] * fut["tasa"] * fut["uplift"]
    fut["etiqueta"] = ""
    for col, val, lab in cfg.semantic_labels:
        fut.loc[fut[col] == val, "etiqueta"] = lab
    cfg.write(fut[["fu_key", "comb_key", "fs_id", cfg.period_col, cfg.pipeline_usd_col, "tasa", "uplift",
                   "esperado_usd", "etiqueta"]].assign(**{cfg.period_col: fut[cfg.period_col].astype(str)}), "forecast_detail")
    res = fut.groupby("etiqueta")["esperado_usd"].sum()
    print(f"[f3] forecast total ${fut['esperado_usd'].sum():,.0f} · por etiqueta: { {k or 'genuino': f'${x:,.0f}' for k,x in res.items()} }")
    return fut

# ── técnicas (catálogo v1) ──────────────────────────────────────────
def _tecnicas(s, h, cfg):
    """s: Serie period→tasa (historia ≤ origen). Devuelve dict tecnica→pred para horizonte h."""
    out = {}
    x = s.dropna()
    if not len(x): return out
    out["T0_naive"] = x.iloc[-1]
    out["T2_promedio"] = x.mean()
    w = np.exp(-np.arange(len(x))[::-1] / 3.0); out["T4_ewma"] = float(np.average(x, weights=w))
    tgt = x.index[-1] + h
    mm = x[x.index.month == tgt.month]
    if len(mm): out["T1_naive_estacional"] = mm.iloc[-1]
    if len(x) >= 13:
        idx = x.groupby(x.index.month).mean(); nivel = x.iloc[-6:].mean()
        out["T7_indice_estacional"] = nivel * idx.get(tgt.month, idx.mean()) / idx.mean()
    if len(x) >= 12:
        t = np.arange(len(x)); b, a = np.polyfit(t, x.values, 1)
        out["T8_tendencia_sat"] = min(cfg.rate_cap, max(0.01, a + b * (len(x) - 1 + h)))
    rec = x.iloc[-6:].mean(); nmes = 6; z = nmes / (nmes + 6)
    out["T14_credibilidad_t"] = z * rec + (1 - z) * x.mean()
    return out

def f4_backtest(v2, est, cfg, horizontes=(1, 2, 3, 4)):
    """FASE 4: walk-forward por horizonte; formato largo; retadores T0/T2; campeón por serie (pool L2)."""
    hist = v2[(v2["universo"] == "normal") & v2["tasa"].notna()]
    series = {}
    for l2, sub in hist.groupby("fs_id_L2"):
        agg = sub.groupby(cfg.period_col).apply(lambda x: (x[cfg.renewed_units_col].sum(), x[cfg.pipeline_units_col].sum()), include_groups=False)
        s = pd.Series({m: r / max(p, 1) for m, (r, p) in agg.items()}).sort_index()
        n = sub.groupby(cfg.period_col)[cfg.pipeline_units_col].sum().median()
        if len(s) >= 10 and n >= cfg.support_floor / 2: series[l2] = s
    filas = []
    for l2, s in series.items():
        for h in horizontes:
            origen = s.index[-1] - h
            base, tgt = s[s.index <= origen], s.index[-1] - h + h
            real = s.get(s.index[-1] - h + h)
            if base is None or len(base) < 8 or real is None: continue
            for tec, pred in _tecnicas(base, h, cfg).items():
                filas.append(dict(fs_id_L2=l2, tecnica=tec, h=h, pred=round(float(pred), 4),
                                  real=round(float(real), 4), abs_err_pp=100 * abs(pred - real)))
    bt = pd.DataFrame(filas); cfg.write(bt, "backtest_predictions")
    lb = bt.groupby("tecnica")["abs_err_pp"].mean().sort_values()
    print("[f4] leaderboard (|err| medio pp):"); [print(f"   {t:22s} {e:5.2f}") for t, e in lb.items()]
    # campeón por serie: mejor media, debe batir a T2 y T0 por >0.1pp
    camp = {}
    for l2, sub in bt.groupby("fs_id_L2"):
        m = sub.groupby("tecnica")["abs_err_pp"].mean()
        ret = min(m.get("T2_promedio", 99), m.get("T0_naive", 99))
        best = m.idxmin()
        camp[l2] = best if m[best] < ret - 0.1 else "T2_promedio"
    fin = pd.DataFrame({"fs_id_L2": list(camp), "tecnica_elegida": list(camp.values())})
    fin = fin.merge(bt.groupby(["fs_id_L2", "tecnica"])["abs_err_pp"].mean().rename("err_bt").reset_index(),
                    left_on=["fs_id_L2", "tecnica_elegida"], right_on=["fs_id_L2", "tecnica"], how="left")
    cfg.write(fin[["fs_id_L2", "tecnica_elegida", "err_bt"]], "forecast_final_seleccion")
    print(f"[f4] campeones: {fin['tecnica_elegida'].value_counts().to_dict()}")
    return bt, fin, series

DIM_TECNICA = [
    ("T0_naive", "naive", "último mes observado", "1 mes"),
    ("T1_naive_estacional", "estacional", "mismo mes del año anterior", "13 meses"),
    ("T2_promedio", "promedio", "media de toda la historia", "3 meses"),
    ("T4_ewma", "recencia", "media con pesos decrecientes (hl=3)", "3 meses"),
    ("T7_indice_estacional", "estacional", "nivel reciente × índice mensual", "13 meses"),
    ("T8_tendencia_sat", "tendencia", "regresión temporal saturada al techo", "12 meses"),
    ("T14_credibilidad_t", "credibilidad", "z·reciente + (1−z)·histórico", "6 meses"),
]

def f4_tablas_fu(v2, fina, est, gu, bt, series, cfg):
    """dim_tecnica (maestra) + estampado por FU: backtest_predictions_fu (con error, meses reservados)
    y forecast_predictions_fu (projection × técnica: banda, no error). P5: cálculo en el pool, resultado por miembro."""
    import pandas as pd, numpy as np
    dim = pd.DataFrame(DIM_TECNICA, columns=["tecnica_id", "familia", "descripcion", "elegibilidad_min"])
    cfg.write(dim, "dim_tecnica")
    # backtest por FU: la pred del pool estampada en cada FU miembro del mes objetivo, error vs SU real
    filas = []
    hist = v2[(v2["universo"] == "normal") & v2["tasa"].notna()]
    for l2, s in series.items():
        objetivo = s.index[-1]
        miembros = hist[(hist["fs_id_L2"] == l2) & (hist[cfg.period_col] == objetivo)]
        for _, r in bt[bt["fs_id_L2"] == l2].iterrows():
            for _, m in miembros.iterrows():
                filas.append(dict(fu_key=m["fu_key"], fs_id=m["fs_id"], fs_id_L2=l2,
                    period=str(objetivo), h=r["h"], tecnica_id=r["tecnica"],
                    tasa_pred=r["pred"], tasa_real_fu=round(float(m["tasa"]), 4),
                    abs_err_pp=round(100 * abs(r["pred"] - m["tasa"]), 2),
                    err_usd=round(abs(r["pred"] - m["tasa"]) * m[cfg.pipeline_usd_col], 2)))
    btfu = pd.DataFrame(filas); cfg.write(btfu, "backtest_predictions_fu")
    # forecast por FU de projection: cada técnica aplicada a su horizonte + uplift de su comb → $
    upl = gu.set_index(["gu", "comb_id"])["uplift_final"]
    fut = fina[fina[cfg.role_col] == "projection"].copy()
    fut["fs_id"] = fut[cfg.grano_tasa].astype(str).agg("|".join, axis=1)
    fut["gu"] = fut[cfg.grano_uplift].astype(str).agg("|".join, axis=1)
    l2map = est.set_index("fs_id")["fs_id_L2"]
    filas = []
    for _, r in fut.iterrows():
        l2 = l2map.get(r["fs_id"]);  s = series.get(l2)
        if s is None: continue
        h = (r[cfg.period_col] - s.index[-1]).n
        u = float(upl.get((r["gu"], r["comb_id"]), 1.0))
        for tec, pred in _tecnicas(s, h, cfg).items():
            pred = min(pred, cfg.rate_cap)
            filas.append(dict(fu_key=r["fu_key"], fs_id=r["fs_id"], fs_id_L2=l2,
                period=str(r[cfg.period_col]), h=h, tecnica_id=tec,
                tasa_pred=round(float(pred), 4), uplift_pred=round(u, 3),
                forecast_usd=round(float(r[cfg.pipeline_usd_col] * pred * u), 2)))
    ffu = pd.DataFrame(filas); cfg.write(ffu, "forecast_predictions_fu")
    print(f"[f4+] dim_tecnica {len(dim)} · backtest_fu {len(btfu)} filas · forecast_fu {len(ffu)} filas ({ffu['fu_key'].nunique()} FUs × técnicas × h)")
    return dim, btfu, ffu
