"""Change since 2000: desalination growth against the change in renewable electricity (Fig. 4, Supplementary Table 2)."""
import os, json, warnings
import numpy as np, pandas as pd
import statsmodels.formula.api as smf, statsmodels.api as sm
from scipy import stats
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); D = f"{ROOT}/data/"; O = f"{ROOT}/results/"
os.makedirs(O, exist_ok=True)
X = pd.read_csv(f"{D}panel_crosssection.csv")
X["log_desal_pc1"] = np.log1p(X.desal_m3_pc); X["log_gni"] = np.log(X.gni_pc_ppp); X["log_clim"] = np.log1p(X.clim_aff_per1000_yr)
X["log_drought"] = np.log1p(X.drought_aff_per1000_yr); X["log_floodstorm"] = np.log1p(X.floodstorm_aff_per1000_yr)
X["coastal"] = ~(X.landlocked == "landlocked"); X["has_series"] = X.desal_year.notna()
C = X[X.coastal & X.gni_pc_ppp.notna()].copy()
out = {}
X["log_gni"] = np.log(X.gni_pc_ppp)
T = X[(X.desal_km3_2000.notna()) & (X.desal_n_distinct_2000on >= 3)].copy()
T["desal_growth"] = np.log((T.desal_km3 + 1e-4) / (T.desal_km3_2000 + 1e-4)); T["d_renew_fe"] = T.renew_share_pct - T.renew_2000; T["d_renew_el"] = T.renew_elec_pct - T.renew_elec_2000; T["d_nonhydro"] = T.nonhydro_renew_elec_pct - T.nonhydro_renew_elec_2000
T["gcc"] = T.is_gcc; T["small_start"] = T.desal_km3_2000 < 0.01
def dyn(d, ycol, label, ctrl=""):
    d = d.dropna(subset=[ycol, "desal_growth"]); r, p = stats.spearmanr(d.desal_growth, d[ycol])
    f = f"{ycol} ~ desal_growth" + ctrl; m = smf.ols(f, d.dropna(subset=[v.strip() for v in f.replace("~", "+").split("+")])).fit(cov_type="HC3"); ci = m.conf_int()
    return {"test": label, "n": int(m.nobs), "spearman_rho": round(r, 3), "spearman_p": round(p, 4), "slope": round(m.params.desal_growth, 3), "ci_lo": round(ci.loc["desal_growth", 0], 3), "ci_hi": round(ci.loc["desal_growth", 1], 3), "p": round(m.pvalues.desal_growth, 4)}
DY = [dyn(T, "d_renew_el", "Renewable electricity share, all"), dyn(T[~T.gcc], "d_renew_el", "Renewable electricity, excluding GCC")]
DY += [dyn(T[~T.is_mena], "d_renew_el", "Renewable electricity, excluding MENA"), dyn(T[~T.small_start], "d_renew_el", "Renewable electricity, 2000 volume >= 0.01 km3"),
       dyn(T[~T.iso3.isin(["KAZ", "CHN"])], "d_renew_el", "Renewable electricity, excluding Kazakhstan"),
       dyn(T, "d_renew_el", "Renewable electricity + initial share, stress, income, rents", " + renew_elec_2000 + aq_bws_score + log_gni + oilgas_rents_pct"),
       dyn(T, "d_renew_el", "Renewable electricity + initial share + MENA", " + renew_elec_2000 + is_mena"),
       dyn(T, "d_nonhydro", "Non-hydro renewable electricity (solar, wind, other), all"), dyn(T[~T.gcc], "d_nonhydro", "Non-hydro renewable electricity, excluding GCC"),
       dyn(T, "d_renew_fe", "Renewable share of final energy (SDG 7.2.1), all"), dyn(T[~T.gcc], "d_renew_fe", "SDG 7.2.1, excluding GCC"),
       dyn(T, "d_renew_fe", "SDG 7.2.1 + initial share, stress, income, rents", " + renew_2000 + aq_bws_score + log_gni + oilgas_rents_pct")]
DY += [dyn(T[T.pathway != "Landlocked"], "d_renew_el", "Renewable electricity, coastal countries only"),
       dyn(T[(T.pathway != "Landlocked") & ~T.small_start], "d_renew_el", "Renewable electricity, coastal only, 2000 volume >= 0.01 km3")]
# volume-weighted
d = T.dropna(subset=["d_renew_el"]); w = np.sqrt(d.desal_km3 + d.desal_km3_2000); mw = smf.wls("d_renew_el ~ desal_growth", d, weights=w).fit(cov_type="HC3"); ciw = mw.conf_int()
DY.append({"test": "Renewable electricity, volume-weighted (sqrt)", "n": int(mw.nobs), "spearman_rho": None, "spearman_p": None, "slope": round(mw.params.desal_growth, 3), "ci_lo": round(ciw.loc["desal_growth", 0], 3), "ci_hi": round(ciw.loc["desal_growth", 1], 3), "p": round(mw.pvalues.desal_growth, 4)})
pd.DataFrame(DY).to_csv(f"{O}supp_table2_change_sensitivity.csv", index=False)
grew = T[T.desal_growth > np.log(2)]; shr = T[T.desal_growth <= 0]
out["dyn_medians_el"] = {"doublers": [float(grew.d_renew_el.median()), int(grew.d_renew_el.notna().sum())], "nongrowers": [float(shr.d_renew_el.median()), int(shr.d_renew_el.notna().sum())]}
out["dyn_medians_nonhydro"] = {"doublers": float(grew.d_nonhydro.median()), "nongrowers": float(shr.d_nonhydro.median())}
# included vs excluded countries in dynamic sample
ex = X[X.desaliner & ~X.index.isin(T.index)]; out["dyn_sample_vs_excluded"] = {"included": [int(len(T)), float(T.gni_pc_ppp.median()), float(T.desal_km3.median()), float(T.aq_bws_score.median())], "excluded_desaliners": [int(len(ex)), float(ex.gni_pc_ppp.median()), float(ex.desal_km3.median()), float(ex.aq_bws_score.median())]}
T[["name", "iso3", "pathway", "desal_km3_2000", "desal_km3", "desal_year", "desal_growth", "renew_elec_2000", "renew_elec_pct", "renew_elec_year", "d_renew_el", "nonhydro_renew_elec_2000", "nonhydro_renew_elec_pct", "d_nonhydro", "renew_2000", "renew_share_pct", "d_renew_fe", "desal_n_distinct_2000on"]].sort_values("desal_growth", ascending=False).to_csv(f"{O}change_2000_latest_by_country.csv", index=False)
out["dyn_year_alignment"] = {"desal_latest_year_median": float(T.desal_year.median()), "renew_elec_latest_year_median": float(T.renew_elec_year.median())}

# coastal countries only (the 39 include two landlocked brackish-water desalinators)
Tc = T[T.pathway != "Landlocked"].dropna(subset=["d_renew_el"]); r, p = stats.spearmanr(Tc.desal_growth, Tc.d_renew_el); out["coastal_only"] = [float(r), float(p), int(len(Tc))]
Tb = Tc[Tc.desal_km3_2000 >= 0.01]; r, p = stats.spearmanr(Tb.desal_growth, Tb.d_renew_el); out["coastal_only_start_ge_0.01km3"] = [float(r), float(p), int(len(Tb))]
json.dump(out, open(f"{O}results_change_since_2000.json", "w"), indent=1, default=float)
print(pd.DataFrame(DY).to_string()); print(json.dumps(out, indent=1, default=float))
