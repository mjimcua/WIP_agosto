"""config.py — SFF v2. Config exhaustivo (P3), taxonomía 4 grupos (§3), claves hash (P7), write con process_date+execution_id.
CONTRATO: toda columna del raw debe estar declarada o el programa para. Granos derivables:
grano_tasa = mandatory + timevarying + extra_renovacion · grano_uplift = mandatory + extra_revalorizacion."""
from dataclasses import dataclass, field
import hashlib, os, datetime, uuid
import pandas as pd

def hkey(s):  # clave hash determinista int64 del id (P7)
    return int(hashlib.md5(str(s).encode()).hexdigest()[:12], 16)

@dataclass
class Config:
    period_col: str = "period"; role_col: str = "dataset_role"
    pipe_u: str = "pipe_units"; pipe_d: str = "pipe_usd"
    ren_u: str = "ren_units"; ren_d: str = "ren_usd"
    cur_col: str = "is_current_month"; flag_ts: str = "flag_time_series"
    mandatory: list = field(default_factory=lambda: ["region"])
    timevarying: dict = field(default_factory=dict)          # {col: 'negative'|'positive'}
    tv_positive_values: list = field(default_factory=lambda: [1, "1", True])
    extra_renovacion: list = field(default_factory=list)
    extra_revalorizacion: list = field(default_factory=list)
    ignore_cols: list = field(default_factory=list)
    support_floor: float = 30.0; z: float = 1.645            # 90%
    rate_cap: float = 0.95; k_cred: float = 60.0; k_uplift: float = 24.0
    semantic_labels: list = field(default_factory=list)       # [(col, valor, etiqueta)]
    outdir: str = "./salida"
    def __post_init__(self):
        for d, s in self.timevarying.items():
            if s not in ("negative", "positive"):
                raise ValueError(f"timevarying['{d}']='{s}': debe ser 'negative' o 'positive'")
        m = set(self.mandatory)
        if m & set(self.timevarying) or m & set(self.extra_renovacion) or m & set(self.extra_revalorizacion):
            raise ValueError("mandatory es excluyente con timevarying y extras")
        if set(self.timevarying) & (set(self.extra_renovacion) | set(self.extra_revalorizacion)):
            raise ValueError("timevarying es excluyente con los grupos extra")
        self.execution_id = uuid.uuid4().hex[:10]; os.makedirs(self.outdir, exist_ok=True)
    @property
    def grano_tasa(self): return self.mandatory + list(self.timevarying) + self.extra_renovacion
    @property
    def grano_uplift(self): return self.mandatory + self.extra_revalorizacion
    @property
    def medidas(self): return [self.pipe_u, self.pipe_d, self.ren_u, self.ren_d]
    def validar_columnas(self, df):
        rol = set([self.period_col, self.role_col, self.cur_col, self.flag_ts] + self.medidas
                  + self.grano_tasa + self.extra_revalorizacion + self.ignore_cols)
        huerfanas = [c for c in df.columns if c not in rol]
        if huerfanas:
            raise ValueError(f"columnas sin adscripción en config: {huerfanas} — declara su grupo o ignore_cols")
    def write(self, df, nombre):
        out = df.copy(); out["process_date"] = datetime.datetime.now().isoformat(timespec="seconds")
        out["execution_id"] = self.execution_id
        p = os.path.join(self.outdir, f"{nombre}.csv"); out.to_csv(p, index=False)
        print(f"[write] {len(out):,} filas → {nombre}"); return out
