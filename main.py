"""main.py — SFF v2 end-to-end sobre el sintético. python main.py [seed]"""
import sys
from config import Config
from synthetic import build_raw
import fase0 as f0, fase1 as f1, fase2 as f2, fase34 as f34

cfg = Config(mandatory=["region"],
             timevarying={"dormant": "negative", "softcancel": "negative",
                          "no_instalado": "negative", "autorenew": "positive"},
             extra_renovacion=["product", "channel"],
             extra_revalorizacion=["discount", "newcust"],
             semantic_labels=[("no_instalado", 1, "residuo")])
raw = build_raw(int(sys.argv[1]) if len(sys.argv) > 1 else 7)
print("═" * 70, "\nFASE 0 — contrato, carga, referencia")
df = f0.f0_load_and_validate(raw, cfg)
vista, fina, lk_fu, lk_comb = f0.f0_split_and_key(df, cfg)
cfg.write(lk_fu, "lookup_fu"); cfg.write(lk_comb, "lookup_comb")
v = f0.f0_universe_routes(vista, cfg)
f0.f0_fu_summary(v, cfg)
print("═" * 70, "\nFASE 1 — rama renovación")
v, g = f1.f1_series_and_gaps(v, cfg)
e2, pares, cf = f1.f1_diagnose_round1(v, g, cfg)
v2, est, chain = f1.f1_improve_support(v, g, e2, cfg)
f1.f1_support_chain(chain, cfg)
dyn = f1.f1_diagnose_round2(v2, est, cfg)
print("═" * 70, "\nFASE 2 — rama revalorización")
gu, r = f2.f2_uplift_fine(fina, cfg)
e2u = f2.f2_diagnose(gu, r, cfg)
gu, uchain = f2.f2_improve(gu, e2u, cfg)
print("═" * 70, "\nFASE 3 — ensamblaje")
fut = f34.f3_ensamblaje(fina, v2, est, gu, cfg)
print("═" * 70, "\nFASE 4 — backtest por horizonte")
bt, fin, series = f34.f4_backtest(v2, est, cfg)
dim, btfu, ffu = f34.f4_tablas_fu(v2, fina, est, gu, bt, series, cfg)
print("═" * 70, "\n✓ ejecución completa · tablas en", cfg.outdir)
