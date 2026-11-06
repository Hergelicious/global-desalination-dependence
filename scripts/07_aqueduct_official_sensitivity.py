"""Sensitivity of the main model (Table 1, model D) to the water-stress measure and to small island states
(Supplementary Table 8). Uses WRI's official demand-weighted Aqueduct 4.0 country scores
(country_baseline, indicator bws, weight Tot), extracted to inputs/aqueduct40_country_bws_official.csv from
Aqueduct40_rankings_download_Y2023M07D05.xlsx (https://www.wri.org/data/aqueduct-40-country-rankings).
WRI reports no score (-9999) for most small island states."""
import os, json, warnings
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); O = f"{ROOT}/results/"; os.makedirs(O, exist_ok=True)
w = pd.read_csv(f"{ROOT}/inputs/aqueduct40_country_bws_official.csv"); w.loc[w.aq_bws_official < 0, "aq_bws_official"] = np.nan
X = pd.read_csv(f"{ROOT}/data/panel_crosssection.csv").merge(w[["iso3", "aq_bws_official"]], on="iso3", how="left")
X["y"] = np.log1p(X.desal_m3_pc); X["log_gni"] = np.log(X.gni_pc_ppp)
X["sids"] = X.is_sids.astype(int); X["hydro"] = X.hydrocarbon.astype(int)
C = X[(X.landlocked != "landlocked") & X.gni_pc_ppp.notna()].dropna(subset=["aq_bws_score", "fossil_elec_pct", "oilgas_rents_pct"])
common = C.dropna(subset=["aq_bws_official"])
specs = [("Main model, area-weighted Aqueduct stress", "aq_bws_score", C, ""),
         ("Area-weighted Aqueduct stress, countries with an official WRI score", "aq_bws_score", common, ""),
         ("Official WRI demand-weighted Aqueduct stress", "aq_bws_official", common, ""),
         ("Area-weighted Aqueduct stress, small island states excluded", "aq_bws_score", C[~C.is_sids], ""),
         ("SDG 6.4.2 stress, small island states excluded", "water_stress_pct", C[~C.is_sids].dropna(subset=["water_stress_pct"]), ""),
         ("Main model with island and hydrocarbon indicators", "aq_bws_score", C, " + sids + hydro")]
rows = []
for lab, s, d, extra in specs:
    m = smf.ols(f"y ~ log_gni + {s} + fossil_elec_pct + oilgas_rents_pct{extra}", d).fit(cov_type="HC3"); ci = m.conf_int()
    row = {"model": lab, "n": int(m.nobs), "R2": float(m.rsquared)}
    for k, nm in [(s, "stress"), ("fossil_elec_pct", "fossil"), ("log_gni", "income")]:
        row.update({f"{nm}_coef": float(m.params[k]), f"{nm}_lo": float(ci.loc[k, 0]), f"{nm}_hi": float(ci.loc[k, 1]), f"{nm}_p": float(m.pvalues[k])})
    rows.append(row)
T = pd.DataFrame(rows); T.to_csv(f"{O}supp_table8_stress_measure_sensitivity.csv", index=False)
out = {"n_coastal_without_official_score": int(C.aq_bws_official.isna().sum()),
       "without_official_score": C[C.aq_bws_official.isna()].name.tolist(),
       "spearman_area_vs_official": float(common[["aq_bws_score", "aq_bws_official"]].corr(method="spearman").iloc[0, 1])}
json.dump(out, open(f"{O}results_stress_measure_sensitivity.json", "w"), indent=1)
pd.set_option("display.width", 250); print(T.round(4).to_string()); print(out)
