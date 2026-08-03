"""fase0.py — Contrato, carga, acondicionado y referencia (DISEÑO_V2 §5). Raw inmutable: etiquetar, no amputar."""
import numpy as np, pandas as pd
from config import hkey

def f0_load_and_validate(raw, cfg):
    """0.1 ENTRADA raw+config · SALIDA raw con period normalizado · REGLAS: config exhaustivo o excepción;
    dinero>=0; uplift!=0 en renovadores; roles censados; mes en curso declarado · BORDES: para con lista."""
    cfg.validar_columnas(raw)
    df = raw.copy(); df[cfg.period_col] = pd.PeriodIndex(df[cfg.period_col].astype(str), freq="M")
    prob = []
    for c in (cfg.pipeline_units_col, cfg.pipeline_usd_col): 
        if (df[c] < 0).any(): prob.append(f"{c} negativo")
    ren = df[cfg.renewed_units_col].fillna(0) > 0
    if ((df.loc[ren, cfg.renewed_usd_col].fillna(0) <= 0)).any(): prob.append("renovación con USD<=0 (uplift 0 prohibido)")
    if prob: raise ValueError("f0_load: " + "; ".join(prob))
    roles = df[cfg.role_col].value_counts().to_dict()
    for r in ("train", "test", "projection"):
        if roles.get(r, 0) == 0: print(f"⚠ rol '{r}' VACÍO")
    cur = sorted(df.loc[df[cfg.cur_col] == 1, cfg.period_col].astype(str).unique())
    print(f"[f0.1] {len(df):,} filas · roles {roles} · mes en curso {cur} (test incompleto: fuera del backtest)")
    return df

def f0_split_and_key(df, cfg):
    """0.2 SALIDA (vista_tasa, fina, lookups) · REGLAS: ratios recalculados del agregado; conservación exacta o parar;
    fu_id=grano_tasa|mes; comb_id=extra_revalorizacion; keys hash."""
    g = cfg.grano_tasa
    fina = df.copy()
    fina["fu_id"] = fina[g].astype(str).agg("|".join, axis=1) + "|" + fina[cfg.period_col].astype(str)
    fina["comb_id"] = fina[cfg.extra_revalorizacion].astype(str).agg("|".join, axis=1) if cfg.extra_revalorizacion else "na"
    fina["fu_key"] = fina["fu_id"].map(hkey); fina["comb_key"] = fina["comb_id"].map(hkey)
    keys = g + [cfg.period_col, cfg.role_col, cfg.cur_col, cfg.flag_ts]
    vista = fina.groupby(keys, as_index=False, observed=True)[cfg.medidas].sum(min_count=1)
    vista["fu_id"] = vista[g].astype(str).agg("|".join, axis=1) + "|" + vista[cfg.period_col].astype(str)
    vista["fu_key"] = vista["fu_id"].map(hkey)
    assert abs(vista[cfg.pipeline_usd_col].sum() - fina[cfg.pipeline_usd_col].sum()) < 1e-6, "conservación rota"
    lk_fu = vista[["fu_id", "fu_key"]].drop_duplicates()
    lk_comb = fina[["comb_id", "comb_key"]].drop_duplicates()
    print(f"[f0.2] fina {len(fina):,} → vista {len(vista):,} · conservación ✓ · lookups {len(lk_fu)}/{len(lk_comb)}")
    return vista, fina, lk_fu, lk_comb

def f0_universe_routes(vista, cfg):
    """0.3 etiquetas: universo, patrón de cobertura, ruta — sin borrar nada (P4)."""
    v = vista.copy()
    v["universo"] = np.where(v[cfg.flag_ts] == 1, "time_series", "normal")
    v["fs_id"] = v[cfg.grano_tasa].astype(str).agg("|".join, axis=1)
    pat = v.groupby("fs_id")[cfg.role_col].agg(lambda s: "_".join(sorted(set(s)))).rename("cobertura")
    v = v.merge(pat, on="fs_id")
    def ruta(p):
        if "projection" not in p: return "no_impact"
        return "trainable" if "train" in p else "heuristic"
    v["ruta"] = v["cobertura"].map(ruta)
    print(f"[f0.3] rutas: {v.drop_duplicates('fs_id')['ruta'].value_counts().to_dict()}")
    return v

def f0_fu_summary(v, cfg):
    """0.4 la referencia inmutable: por fu_key, soporte y error binomial esperado (p=0.5, cota conservadora)."""
    s = v[["fu_key", "fu_id", cfg.period_col, cfg.role_col, cfg.cur_col, "universo", "ruta",
           cfg.pipeline_units_col, cfg.pipeline_usd_col]].copy()
    n = s[cfg.pipeline_units_col].clip(lower=1)
    s["se_pp_max"] = 100 * np.sqrt(.25 / n); s["moe_pp_max"] = cfg.z * s["se_pp_max"]
    s["moe_usd_max"] = s["moe_pp_max"] / 100 * s[cfg.pipeline_usd_col]
    s[cfg.period_col] = s[cfg.period_col].astype(str)
    return cfg.write(s, "forecast_units_raw_summary")
