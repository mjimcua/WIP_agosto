"""synthetic.py — dataset sintético v2. MATRIZ DE COBERTURA feature→escenario:
config exhaustivo→columna 'sobrante' opcional | conservación→combos de descuento por FU | universos/rutas→flag_ts, train_only, projection_only
huecos→NA|B meses ausentes | L1 signo→dormant/softcancel/no_instalado neg pequeñas (15/12/10: solo juntas cruzan 30) + autorenew pos minoritaria
L2 asterisco→product con η² alto vs channel mudo, hermanas pequeñas | credibilidad→series ínfimas con celda padre gorda
Simpson→NA: product A (.90, peso cae) vs B (.50, peso sube): agregado cae con partes estables | combinaciones→region×product interacción
uplift punto de partida→newcust+d40 aterrizaje 1.5 vs veteranos 1.03 | intervalos→discount niveles | dinámica→EU|A estacional, EU|B tendencia que satura
mes en curso→2026-02 etiquetado test | residuo→no_instalado=1 → etiqueta"""
import numpy as np, pandas as pd, itertools

MESES = pd.period_range("2024-01", "2026-06", freq="M")
def rol(m):
    s = str(m)
    return "projection" if s >= "2026-03" else ("test" if s >= "2025-11" else "train")

def build_raw(seed=7):
    rng = np.random.default_rng(seed)
    filas = []
    def serie(region, product, channel, dorm, soft, noin, auto, n0, tasa_f, combos, meses=None, ts=0):
        for i, m in enumerate(meses if meses is not None else MESES):
            r = rol(m)
            base = tasa_f(i, m)
            for (disc, newc, peso) in combos:
                n = max(1, int(round(n0 * peso)))
                auv = 30.0 * (0.6 if disc == "d40" else 1.0)
                pu, pd_ = n, n * auv
                if r == "projection":
                    ru = rd = np.nan
                else:
                    p = min(0.99, max(0.01, base + rng.normal(0, 0.015)))
                    ru = rng.binomial(n, p)
                    upl = 1.5 if (newc == 1 and disc == "d40") else (1.10 if disc == "d40" else 1.03)
                    rd = ru * auv * (upl + rng.normal(0, 0.02))
                filas.append(dict(period=str(m), dataset_role=r, pipe_units=pu, pipe_usd=pd_,
                    ren_units=ru, ren_usd=rd, is_current_month=int(str(m) == "2026-02"),
                    flag_time_series=ts, region=region, product=product, channel=channel,
                    dormant=dorm, softcancel=soft, no_instalado=noin, autorenew=auto,
                    discount=disc, newcust=newc))
    mix = [("d0", 0, .7), ("d40", 1, .3)]
    plano = [("d0", 0, 1.0)]
    # EU|A estacional grande (A4): base .80 + seno anual
    serie("EU","A","web",0,0,0,0, 600, lambda i,m: .80 + .05*np.sin(2*np.pi*(m.month-1)/12), mix)
    # EU|B tendencia decreciente .86→.70 (A1, satura en backtest)
    serie("EU","B","web",0,0,0,0, 300, lambda i,m: .86 - .006*i, mix)
    # canal mudo: mismas tasas en 'tele' (η² channel ≈ 0), pequeñas → L2 anula channel
    serie("EU","A","tele",0,0,0,0, 12, lambda i,m: .80 + .05*np.sin(2*np.pi*(m.month-1)/12), plano)
    serie("EU","B","tele",0,0,0,0, 10, lambda i,m: .86 - .006*i, plano)
    # NA Simpson: A .90 peso cae 1.0→0.4; B .50 peso sube
    for i, m in enumerate(MESES):
        wA = max(.4, 1.0 - .022*i)
        serie("NA","A","web",0,0,0,0, int(400*wA), lambda i2,m2: .90, plano, meses=[m])
        serie("NA","B","web",0,0,0,0, int(400*(1.6-wA)), lambda i2,m2: .50, plano, meses=[m])
    # timevarying pequeñas EU (L1 por signo: 15+12+10=37 cruza el suelo 30)
    serie("EU","A","web",1,0,0,0, 15, lambda i,m: .45, plano)
    serie("EU","A","web",0,1,0,0, 12, lambda i,m: .35, plano)
    serie("EU","A","web",0,0,1,0, 10, lambda i,m: .40, plano)   # no_instalado → residuo
    serie("EU","A","web",1,1,0,0,  5, lambda i,m: .30, plano)
    serie("EU","A","web",0,0,0,1,  6, lambda i,m: .93, plano)   # positiva minoritaria
    # huecos: NA|B tele con meses ausentes dentro de la historia
    conh = [m for j, m in enumerate(MESES) if j % 4 != 2]
    serie("NA","B","tele",0,0,0,0, 40, lambda i,m: .55, plano, meses=conh)
    # rutas: train_only (no_impact) y projection_only (heuristic)
    serie("NA","A","tele",0,0,0,0, 25, lambda i,m: .70, plano, meses=[m for m in MESES if rol(m)=="train"])
    serie("EU","B","tienda",0,0,0,0, 30, lambda i,m: .60, plano, meses=[m for m in MESES if rol(m)=="projection"])
    # universo time_series
    serie("EU","A","kiosk",0,0,0,0, 50, lambda i,m: .65, plano, ts=1)
    return pd.DataFrame(filas)
