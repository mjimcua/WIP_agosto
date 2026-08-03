"""fase1.py — Rama renovación (DISEÑO_V2 §6): series+huecos, diagnóstico t1, tres mejoras, support_chain, diagnóstico t2."""
import numpy as np, pandas as pd
from config import hkey

def _act(v, cfg): return v.isin(cfg.tv_positive_values)

def f1_series_and_gaps(v, cfg):
    """1.1 fs_id/fs_key, padres, huecos→sintéticas (0 legítimo, rol heredado, solo trainable, dentro de historia)."""
    v = v.copy(); v["fs_key"] = v["fs_id"].map(hkey); v["sintetica"] = 0
    filas = []
    for fs, sub in v[v["universo"] == "normal"].groupby("fs_id"):
        if sub["ruta"].iat[0] != "trainable": continue
        hist = sub[sub[cfg.role_col] != "projection"]
        rango = pd.period_range(hist[cfg.period_col].min(), hist[cfg.period_col].max(), freq="M")
        falta = rango.difference(pd.PeriodIndex(hist[cfg.period_col]))
        for m in falta:
            f = sub.iloc[0].copy(); f[cfg.period_col] = m
            prev = hist[hist[cfg.period_col] < m]
            f[cfg.role_col] = prev[cfg.role_col].iloc[-1] if len(prev) else "train"
            for c in cfg.medidas: f[c] = 0.0
            f[cfg.cur_col] = 0; f["sintetica"] = 1
            f["fu_id"] = fs + "|" + str(m); f["fu_key"] = hkey(f["fu_id"])
            filas.append(f)
    if filas: v = pd.concat([v, pd.DataFrame(filas)], ignore_index=True)
    v["tasa"] = np.where(v["sintetica"] == 1, 0.0,
                np.where(v[cfg.pipeline_units_col] > 0, v[cfg.renewed_units_col] / v[cfg.pipeline_units_col], np.nan))
    v.loc[v[cfg.role_col] == "projection", "tasa"] = np.nan
    # summary por serie
    hist = v[(v["universo"] == "normal") & (v[cfg.role_col] != "projection")]
    g = hist.groupby(["fs_id", "fs_key"], as_index=False).agg(
        n_avg=(cfg.pipeline_units_col, lambda s: s[s > 0].median() if (s > 0).any() else 0),
        meses=("sintetica", "size"), huecos=("sintetica", "sum"),
        ren=(cfg.renewed_units_col, "sum"), pipe=(cfg.pipeline_units_col, "sum"))
    g["tasa_hist"] = g["ren"] / g["pipe"].clip(lower=1)
    g["se_pp"] = 100 * np.sqrt((g["tasa_hist"] * (1 - g["tasa_hist"])).clip(lower=.0025) / g["n_avg"].clip(lower=1))
    proj = v[v[cfg.role_col] == "projection"].groupby("fs_key")[cfg.pipeline_usd_col].sum().rename("usd_proj")
    g = g.merge(proj, on="fs_key", how="left").fillna({"usd_proj": 0})
    g["moe_usd"] = cfg.z * g["se_pp"] / 100 * g["usd_proj"]
    cfg.write(g.assign(), "forecast_series_raw_summary")
    print(f"[f1.1] {len(g)} series · sintéticas añadidas {int(v['sintetica'].sum())}")
    return v, g

def _eta2_w(df, dim, val, w):
    m = np.average(df[val], weights=df[w]); sst = np.sum(df[w] * (df[val] - m) ** 2)
    if sst <= 0: return 0.0
    ssb = sum(np.sum(sub[w]) * (np.average(sub[val], weights=sub[w]) - m) ** 2
              for _, sub in df.groupby(dim, observed=True))
    return float(ssb / sst)

def f1_diagnose_round1(v, g, cfg):
    """1.2 foto binomial · Simpson (descomposición + contrafactual $ walk-forward) · ANOVA sin timevarying · combinaciones."""
    bajo = g[g["n_avg"] < cfg.support_floor]
    print(f"[f1.2] foto binomial: {len(bajo)}/{len(g)} series bajo suelo · ${bajo['usd_proj'].sum():,.0f} de ${g['usd_proj'].sum():,.0f} sin derecho a hablar solas")
    hist = v[(v["universo"] == "normal") & v["tasa"].notna() & (v["sintetica"] == 0)].copy()
    hist["celda"] = hist[cfg.mandatory].astype(str).agg("|".join, axis=1)
    # contrafactual walk-forward (últimos 6 meses con verdad)
    meses = sorted(hist[cfg.period_col].unique())[-6:]
    filas = []
    for celda, sub in hist.groupby("celda"):
        for t in meses:
            pas, act = sub[sub[cfg.period_col] < t], sub[sub[cfg.period_col] == t]
            if len(pas) < 6 or not len(act): continue
            real = act[cfg.renewed_units_col].sum() / act[cfg.pipeline_units_col].sum()
            plano = pas[cfg.renewed_units_col].sum() / pas[cfg.pipeline_units_col].sum()
            th = pas.groupby("fs_id").apply(lambda s: s[cfg.renewed_units_col].sum() / max(s[cfg.pipeline_units_col].sum(), 1), include_groups=False)
            w = act.groupby("fs_id")[cfg.pipeline_units_col].sum()
            seg = float(np.average(th.reindex(w.index).fillna(plano), weights=w))
            pipe_d = act[cfg.pipeline_usd_col].sum()
            filas.append(dict(celda=celda, mes=str(t), err_plano_pp=100*(plano-real), err_seg_pp=100*(seg-real),
                              ahorro_usd=(abs(plano-real)-abs(seg-real))*pipe_d))
    cf = pd.DataFrame(filas); cfg.write(cf, "simpson_contrafactual")
    print(f"[f1.2] contrafactual Simpson: ahorro del segmentado = ${cf['ahorro_usd'].sum():,.0f} en {len(meses)} meses walk-forward")
    # ANOVA (sin timevarying) + combinaciones
    base = g.merge(v.drop_duplicates("fs_id")[["fs_id"] + cfg.mandatory + cfg.extra_renovacion +
                   list(cfg.timevarying)], on="fs_id")
    sin_tv = base[~base[list(cfg.timevarying)].apply(lambda r: _act(r, cfg).any(), axis=1)] if cfg.timevarying else base
    ejes = cfg.mandatory + cfg.extra_renovacion
    e2 = {d: _eta2_w(sin_tv, d, "tasa_hist", "n_avg") for d in ejes}
    pares = {f"{a}×{b}": _eta2_w(sin_tv.assign(_p=sin_tv[a].astype(str)+"|"+sin_tv[b].astype(str)), "_p", "tasa_hist", "n_avg")
             for a, b in [(x, y) for i, x in enumerate(ejes) for y in ejes[i+1:]]}
    print(f"[f1.2] η² (sin tv): { {k: round(x,3) for k,x in e2.items()} } · pares: { {k: round(x,3) for k,x in pares.items()} }")
    return e2, pares, cf

def f1_improve_support(v, g, e2, cfg):
    """1.3 L1 por SIGNO a través de columnas · L2 asterisco (η² mínimo, sin mandatory ni tv) · L3 credibilidad z=n/(n+k)."""
    tv = list(cfg.timevarying); info = v.drop_duplicates("fs_id").set_index("fs_id")
    est = g.set_index("fs_id").copy()
    act = info[tv].apply(lambda col: _act(col, cfg)) if tv else pd.DataFrame(index=info.index)
    def signo(fs):
        s = {cfg.timevarying[c] for c in tv if act.loc[fs, c]}
        return "" if not s else ("neg+pos" if len(s) == 2 else ("neg" if "negative" in s else "pos"))
    est["signo"] = [signo(fs) for fs in est.index]
    estable = [d for d in cfg.grano_tasa if d not in tv]
    def id_l1(fs):
        if est.loc[fs, "signo"] and est.loc[fs, "n_avg"] < cfg.support_floor:
            return "|".join(str(info.loc[fs, d]) for d in estable) + "|SIG=" + est.loc[fs, "signo"]
        return fs
    est["fs_id_L1"] = [id_l1(fs) for fs in est.index]
    # L2: sobre pools L1 aún bajo suelo y SIN señal; anula la dim extra de menor η²
    n1 = est.groupby("fs_id_L1")["n_avg"].sum()
    anulable = min([d for d in cfg.extra_renovacion], key=lambda d: e2.get(d, 1), default=None)
    pos = {d: i for i, d in enumerate(cfg.grano_tasa)}
    def id_l2(fs):
        l1 = est.loc[fs, "fs_id_L1"]
        if "SIG=" in l1 or n1[l1] >= cfg.support_floor or anulable is None: return l1
        p = l1.split("|"); p[pos[anulable]] = "*"; return "|".join(p)
    est["fs_id_L2"] = [id_l2(fs) for fs in est.index]
    # estimación por etapa (P5: cálculo con el pool, estampado por miembro)
    v2 = v.merge(est[["fs_id_L1", "fs_id_L2"]], on="fs_id", how="left")
    hist = v2[(v2["universo"] == "normal") & v2["tasa"].notna()]
    def pool(idcol):
        pm = hist.groupby([idcol, cfg.period_col]).agg(ren=(cfg.renewed_units_col, "sum"),
                                                       pipe=(cfg.pipeline_units_col, "sum")).reset_index()
        p = pm.groupby(idcol).agg(ren=("ren", "sum"), pipe=("pipe", "sum"),
                                  n=("pipe", lambda s: s[s > 0].median() if (s > 0).any() else 0))
        p["tasa"] = p["ren"] / p["pipe"].clip(lower=1); return p
    p1, p2 = pool("fs_id_L1"), pool("fs_id_L2")
    celda = hist.assign(_c=hist[cfg.mandatory].astype(str).agg("|".join, axis=1)).groupby("_c").agg(
        ren=(cfg.renewed_units_col, "sum"), pipe=(cfg.pipeline_units_col, "sum"))
    celda["tasa"] = celda["ren"] / celda["pipe"].clip(lower=1)
    def se(t, n):
        pq = max(t * (1 - t), .0025)
        return 100 * np.sqrt(pq / max(n, 1))
    filas = []
    for fs in est.index:
        r = est.loc[fs]; n0 = r["n_avg"]; t0 = r["tasa_hist"]
        etapas = [("0_raw", fs, n0, t0)]
        l1 = r["fs_id_L1"]; etapas.append(("1_L1", l1, p1.loc[l1, "n"] if l1 in p1.index else n0,
                                          p1.loc[l1, "tasa"] if l1 in p1.index else t0))
        l2 = r["fs_id_L2"]; etapas.append(("2_L2", l2, p2.loc[l2, "n"] if l2 in p2.index else n0,
                                          p2.loc[l2, "tasa"] if l2 in p2.index else t0))
        # L3 credibilidad hacia la celda mandatory
        nz = etapas[-1][2]; tz = etapas[-1][3]
        cel = "|".join(str(info.loc[fs, d]) for d in cfg.mandatory)
        tp = celda.loc[cel, "tasa"] if cel in celda.index else tz
        z = nz / (nz + cfg.k_cred)
        t3 = z * tz + (1 - z) * tp
        se3 = np.sqrt(z**2 * se(tz, nz)**2 + (1 - z)**2 * se(tp, celda.loc[cel, "pipe"] / 30 if cel in celda.index else nz)**2)
        n_impl = (t3 * (1 - t3)) / (se3 / 100) ** 2 if se3 > 0 else nz
        etapas.append(("3_shrink", f"z={z:.2f}→{cel}", n_impl, t3))
        for et, ide, n, t in etapas:
            filas.append(dict(fs_id=fs, fs_key=r["fs_key"], etapa=et, id_efectivo=ide,
                              n_efectivo=round(float(n), 1), tasa=round(float(t), 4),
                              se_pp=round(float(se(t, n)) if et != "3_shrink" else float(se3), 2),
                              usd_proj=r["usd_proj"]))
        est.loc[fs, "tasa_final"] = t3; est.loc[fs, "se_final"] = se3
    chain = pd.DataFrame(filas)
    chain["moe_usd"] = cfg.z * chain["se_pp"] / 100 * chain["usd_proj"]
    chain["actuo"] = chain.groupby("fs_id")["id_efectivo"].transform(lambda s: s != s.iloc[0])
    return v2, est.reset_index(), chain

def f1_support_chain(chain, cfg):
    """1.4 waterfall del dinero bajo el suelo + atribución + % que no necesitó nada."""
    cfg.write(chain, "support_chain")
    print("[f1.4] WATERFALL — $ a predecir en series bajo el suelo, por etapa:")
    for et, sub in chain.groupby("etapa"):
        bajo = sub[sub["n_efectivo"] < cfg.support_floor]
        print(f"   {et:9s} ${bajo['usd_proj'].sum():>12,.0f}  ({len(bajo)} series)")
    sano = chain[chain["etapa"] == "0_raw"]
    ok0 = sano[sano["n_efectivo"] >= cfg.support_floor]["usd_proj"].sum()
    print(f"   dinero que nunca necesitó reparación: {100*ok0/max(sano['usd_proj'].sum(),1):.0f}%")

def f1_diagnose_round2(v2, est, cfg):
    """1.5 historia y dinámica SOBRE POOLS consolidados; tendencia y estación vs cota propia; saturación al techo."""
    hist = v2[(v2["universo"] == "normal") & v2["tasa"].notna()]
    filas = []
    for l2, sub in hist.groupby("fs_id_L2"):
        s = sub.groupby(cfg.period_col).apply(lambda x: x[cfg.renewed_units_col].sum() / max(x[cfg.pipeline_units_col].sum(), 1), include_groups=False)
        n = sub.groupby(cfg.period_col)[cfg.pipeline_units_col].sum().median()
        cota = 100 * np.sqrt(.25 / max(n, 1)) * cfg.z
        gate, est_amp, pend = "apto_promedio", 0.0, 0.0
        if n < cfg.support_floor: gate = "soporte"
        elif len(s) < 13: gate = "temporal"
        else:
            mes = s.groupby(s.index.month).mean(); est_amp = 100 * (mes.max() - mes.min())
            x = np.arange(len(s)); pend = 100 * 12 * np.polyfit(x, s.values, 1)[0]
            if est_amp > 2 * cota: gate = "estacional"
            elif abs(pend) > 2 * cota: gate = "tendencia"
        filas.append(dict(fs_id_L2=l2, meses=len(s), n_pool=round(float(n), 1), cota_pp=round(cota, 2),
                          amp_estacional_pp=round(est_amp, 1), pendiente_pp_ano=round(pend, 1), gate=gate))
    d = pd.DataFrame(filas); cfg.write(d, "diagnostico_dinamica")
    print(f"[f1.5] gates: {d['gate'].value_counts().to_dict()}")
    return d
