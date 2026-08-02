"""fase2.py — Rama revalorización (§7): uplift condicionado a renovar, vara s/√n, mejoras, uplift_chain."""
import numpy as np, pandas as pd

def f2_uplift_fine(fina, cfg):
    """2.1 grano uplift × comb, SOLO renovadores; uplift ponderado por $; n = renovadores."""
    r = fina[(fina[cfg.ren_u].fillna(0) > 0)].copy()
    r["auv_p"] = r[cfg.pipe_d] / r[cfg.pipe_u].clip(lower=1)
    r["upl"] = (r[cfg.ren_d] / r[cfg.ren_u]) / r["auv_p"]
    r["gu"] = r[cfg.grano_uplift].astype(str).agg("|".join, axis=1)
    g = r.groupby(["gu", "comb_id"], as_index=False).agg(
        n_ren=(cfg.ren_u, "sum"), meses=(cfg.period_col, "nunique"),
        upl_w=(cfg.ren_d, "sum"), base=(cfg.ren_u, lambda s: 1))
    den = r.groupby(["gu", "comb_id"]).apply(lambda x: (x[cfg.ren_u] * x["auv_p"]).sum(), include_groups=False).rename("den").reset_index()
    g = g.merge(den, on=["gu", "comb_id"]); g["uplift"] = g["upl_w"] / g["den"].clip(lower=1e-9)
    s = r.groupby(["gu", "comb_id"])["upl"].std().rename("s").reset_index()
    g = g.merge(s, on=["gu", "comb_id"]).fillna({"s": 0})
    g["se"] = g["s"] / np.sqrt(g["meses"].clip(lower=1))
    print(f"[f2.1] {len(g)} celdas de uplift · rango [{g['uplift'].min():.2f}, {g['uplift'].max():.2f}]")
    return g, r

def f2_diagnose(g, r, cfg):
    """2.2 vara continua + η²-uplift por eje (extra_revalorizacion), pesos = $ renovado."""
    from fase1 import _eta2_w
    base = r.groupby(["gu", "comb_id"] + cfg.extra_revalorizacion, as_index=False).agg(
        u=("upl", "mean"), w=(cfg.ren_d, "sum"))
    e2 = {d: _eta2_w(base, d, "u", "w") for d in cfg.extra_revalorizacion}
    print(f"[f2.2] η²-uplift: { {k: round(x,3) for k,x in e2.items()} } · celdas con s alta y n decente = heterogeneidad interna")
    return e2

def f2_improve(g, e2, cfg):
    """2.3 ejes mudos → '*' · credibilidad al padre del MISMO punto de partida (newcust se conserva, resto '*')."""
    anular = [d for d in cfg.extra_revalorizacion if e2.get(d, 0) < 0.05]
    padre_ejes = [d for d in cfg.extra_revalorizacion if d == "newcust"]
    def padre(comb):
        p = dict(zip(cfg.extra_revalorizacion, comb.split("|")))
        return "|".join(p[d] if d in padre_ejes else "*" for d in cfg.extra_revalorizacion)
    g = g.copy(); g["padre"] = g["comb_id"].map(padre)
    pg = g.groupby(["gu"]).apply(lambda x: np.average(x["uplift"], weights=x["n_ren"]), include_groups=False)
    pp = g.groupby(["gu", "padre"]).apply(lambda x: np.average(x["uplift"], weights=x["n_ren"]), include_groups=False)
    filas = []
    for _, row in g.iterrows():
        n = row["n_ren"] / max(row["meses"], 1); z = n / (n + cfg.k_uplift)
        up = pp.get((row["gu"], row["padre"]), pg.get(row["gu"], row["uplift"]))
        uf = z * row["uplift"] + (1 - z) * up
        for et, u, se in [("0_fino", row["uplift"], row["se"]),
                          ("1_padre", up, row["se"]), ("2_shrink", uf, row["se"] * z)]:
            filas.append(dict(gu=row["gu"], comb_id=row["comb_id"], etapa=et,
                              uplift=round(float(u), 3), se=round(float(se), 4), n_ren=row["n_ren"]))
        g.loc[row.name, "uplift_final"] = max(uf, 0.01)
    chain = pd.DataFrame(filas); cfg.write(chain, "uplift_chain")
    print(f"[f2.3] ejes mudos anulados: {anular or 'ninguno'} · credibilidad aplicada (k={cfg.k_uplift})")
    return g, chain
