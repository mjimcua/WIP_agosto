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
    # ── nombres de columna del raw (defaults = producción, fotos del usuario) ──
    verbosity: str = "execution"
    period_col: str = "period"; role_col: str = "dataset_role"
    pipeline_units_col: str = "total_tr_units"
    pipeline_usd_col: str = "total_tr_usd"
    renewed_units_col: str = "total_renewed_units"
    renewed_usd_col: str = "total_renewed_usd"
    reacq_units_col: str = "total_reacquired_units"      # fuera del estudio de renovación; se declara
    reacq_usd_col: str = "total_reacquired_usd"
    auv_pipeline_col: str = "TR_AUV"                     # si vienen en el raw, se declaran como medida
    auv_renewed_col: str = "REN_AUV"
    auv_reacq_col: str = "ReAC_AUV"
    cur_col: str = "is_current_month"; flag_ts: str = "flag_time_series"
    mandatory: list = field(default_factory=lambda: [
        "tr_regional_level_1", "tr_regional_level_2", "tr_regional_level_3",
        "tr_product_level_1", "tr_product_level_2", "tr_origin_type_SKU_based",
        "tr_term_level_1", "tr_term_level_2", "tr_band_level_1", "tr_band_level_2"])
    timevarying: dict = field(default_factory=lambda: {"softcancel": "negative"})
    tv_positive_values: list = field(default_factory=lambda: [1, "1", True])
    extra_renovacion: list = field(default_factory=list)
    # antiguas covariate_cols de pricing → su nuevo hogar en la taxonomía:
    extra_revalorizacion: list = field(default_factory=lambda: [
        "price_cap", "msrp_increased", "discount_interval", "prev_OperationGroup"])
    ignore_cols: list = field(default_factory=lambda: ["dummy_field", "row_id", "_filter"])
    support_floor: float = 30.0; z: float = 1.645            # 90%
    rate_cap: float = 0.95; k_cred: float = 60.0; k_uplift: float = 24.0
    semantic_labels: list = field(default_factory=list)       # [(col, valor, etiqueta)]
    outdir: str = "./salida"
    # ── persistencia SQL (SQLAlchemy; adaptado de la cantera) ──
    sql_server: str = None; sql_database: str = None
    sql_driver: str = "ODBC Driver 17 for SQL Server"
    sql_username: str = None; sql_password: str = None; sql_trusted: bool = True
    sql_schema: str = "dbo"                     # el schema es un dato
    sql_write_mode: str = "replace"             # replace (DROP+CREATE) | truncate (conserva definición)
    sql_chunksize: int = 50_000
    sql_engine: object = None                           # o inyecta un engine ya creado
    sql_table_prefix: str = "sff_"
    sql_table_names: dict = field(default_factory=dict)   # overrides nombre_lógico→tabla física
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
    def medidas(self):  # núcleo de cálculo
        return [self.pipeline_units_col, self.pipeline_usd_col,
                self.renewed_units_col, self.renewed_usd_col]
    @property
    def medidas_declaradas(self):  # medidas del raw fuera del cálculo (reacq, AUVs)
        return [c for c in (self.reacq_units_col, self.reacq_usd_col, self.auv_pipeline_col,
                            self.auv_renewed_col, self.auv_reacq_col) if c]
    def validar_columnas(self, df):
        rol = set([self.period_col, self.role_col, self.cur_col, self.flag_ts] + self.medidas
                  + self.medidas_declaradas + self.grano_tasa + self.extra_revalorizacion
                  + self.ignore_cols)
        huerfanas = [c for c in df.columns if c not in rol]
        if huerfanas:
            raise ValueError(f"columnas sin adscripción en config: {huerfanas} — declara su grupo o ignore_cols")
    # ── nombres físicos predefinidos (editables vía sql_table_names) ──
    _NOMBRES = {"forecast_units_raw_summary": "fu_summary", "forecast_series_raw_summary": "fs_summary",
        "lookup_fu": "lookup_fu", "lookup_comb": "lookup_comb", "fact_fu": "fact_fu",
        "fact_fine": "fact_fine", "simpson_contrafactual": "simpson_contrafactual",
        "support_chain": "support_chain", "diagnostico_dinamica": "diag_dinamica",
        "uplift_chain": "uplift_chain", "forecast_detail": "forecast_detail",
        "backtest_predictions": "backtest_pred", "backtest_predictions_fu": "backtest_pred_fu",
        "forecast_predictions_fu": "forecast_pred_fu", "dim_tecnica": "dim_tecnica",
        "forecast_final_seleccion": "forecast_seleccion"}
    @property
    def engine(self):
        if self.sql_engine is None and self.sql_server:
            import urllib.parse
            from sqlalchemy import create_engine
            auth = ("Trusted_Connection=yes;" if self.sql_trusted
                    else f"UID={self.sql_username};PWD={self.sql_password};")
            odbc = (f"DRIVER={{{self.sql_driver}}};SERVER={self.sql_server};"
                    f"DATABASE={self.sql_database};{auth}")
            self.sql_engine = create_engine(
                "mssql+pyodbc:///?odbc_connect=" + urllib.parse.quote_plus(odbc),
                fast_executemany=True)
        return self.sql_engine
    @staticmethod
    def _sanitize(df):
        out = df.copy()
        for c in list(out.columns):
            dt = out[c].dtype
            if isinstance(dt, pd.PeriodDtype):
                if f"{c}_date" not in out.columns:
                    out[f"{c}_date"] = out[c].dt.to_timestamp()
                out[c] = out[c].astype(str)
            elif isinstance(dt, pd.IntervalDtype):
                out[c] = out[c].astype(str)
            else:
                ej = out[c].dropna().head(1)
                if len(ej) and isinstance(ej.iloc[0], pd.Period):
                    out[c] = out[c].astype(str)
                elif len(ej) and isinstance(ej.iloc[0], (pd.DataFrame, pd.Series, list, dict, set)):
                    raise ValueError(f"write: columna '{c}' con objetos anidados — no escribible")
        return out
    def _dtypes(self, out):
        from sqlalchemy.types import NVARCHAR
        m = {}
        for c in out.columns:
            if str(out[c].dtype) in ("object", "str", "string"):
                s = out[c].dropna().astype(str)
                n = int(s.str.len().max()) if len(s) else 1
                m[c] = NVARCHAR(min(max(int(n * 1.3) + 4, 8), 4000))
        return m
    def write(self, df, nombre):
        out = self._sanitize(df)
        out["process_date"] = datetime.datetime.now().isoformat(timespec="seconds")
        out["execution_id"] = self.execution_id
        fisico = self.sql_table_names.get(nombre, self.sql_table_prefix + self._NOMBRES.get(nombre, nombre))
        if self.engine is not None:
            qual = f"[{self.sql_schema}].[{fisico}]" if self.sql_schema else f"[{fisico}]"
            modo = self.sql_write_mode
            if modo == "truncate":
                from sqlalchemy import inspect, text
                if inspect(self.engine).has_table(fisico, schema=self.sql_schema):
                    with self.engine.begin() as con:
                        con.execute(text(f"DELETE FROM {qual}"))
                    out.to_sql(fisico, self.engine, schema=self.sql_schema, if_exists="append",
                               index=False, chunksize=self.sql_chunksize, dtype=self._dtypes(out))
                    print(f"[write] {len(out):,} filas → {qual} (truncate)"); return out
                modo = "replace"
            out.to_sql(fisico, self.engine, schema=self.sql_schema, if_exists="replace",
                       index=False, chunksize=self.sql_chunksize, dtype=self._dtypes(out))
            print(f"[write] {len(out):,} filas → {qual} (SQL, {modo})")
        else:
            out.to_csv(os.path.join(self.outdir, f"{fisico}.csv"), index=False)
            print(f"[write] {len(out):,} filas → {fisico}.csv (sin engine: CSV)")
        return out
