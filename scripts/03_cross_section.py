"""Cross-sectional analysis: dependence bounds, Table 1 regressions and robustness, two-part model, pathway groups,
leave-one-out residuals (Fig. 3b), standardised effects (Fig. 2b), Spearman table and the wastewater test."""
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
out = {"n_states": int(len(X)), "n_coastal": int(X.coastal.sum()), "n_desaliners": int(X.desaliner.sum()), "n_with_series": int(X.has_series.sum())}

# ---------- dependence bounds (Methods 4.3) ----------
hi = X[X.desal_share_lower_pct > 5].copy(); hi["gap"] = hi.desal_share_upper_pct - hi.desal_share_lower_pct
out["share_bounds_over5"] = hi[["name", "desal_share_lower_pct", "desal_share_upper_pct", "lower_bound_exact"]].round(1).values.tolist()
out["share_bounds_median_gap_over5"] = float(hi.gap.median()); out["share_bounds_max_gap_over5_not_exact"] = float(hi[~hi.lower_bound_exact].gap.max())
out["bounds_exact"] = X[X.lower_bound_exact & (X.desal_share_lower_pct > 5)].name.tolist()
Co = X[X.coastal]; out["n_over10_lower_upper"] = [int((Co.desal_share_lower_pct > 10).sum()), int((Co.desal_share_upper_pct > 10).sum())]
# municipal-scale comparison
mu = X[X.desaliner & X.desal_vs_municipal_pct.notna()].sort_values("desal_vs_municipal_pct", ascending=False)
out["desal_vs_municipal_top12"] = mu[["name", "desal_vs_municipal_pct", "desal_share_lower_pct"]].round(1).head(12).values.tolist()
out["n_desal_exceeds_half_municipal"] = int((mu.desal_vs_municipal_pct >= 50).sum())

def fit(f, d, label):
    vs = [v.strip() for v in f.replace("~", "+").split("+")]; d = d.dropna(subset=vs); d = d[np.isfinite(d[vs]).all(axis=1)]
    m = smf.ols(f, d).fit(cov_type="HC3"); ci = m.conf_int()
    return {"model": label, "n": int(m.nobs), "R2": float(round(m.rsquared, 3)), **{f"{k}": [float(round(m.params[k], 4)), float(round(ci.loc[k, 0], 4)), float(round(ci.loc[k, 1], 4)), float(round(m.pvalues[k], 4))] for k in m.params.index if k != "Intercept"}}

# ---------- Table 1 and robustness ----------
R = []
R.append(fit("log_desal_pc1 ~ log_gni", C, "A income"))
R.append(fit("log_desal_pc1 ~ log_gni + aq_bws_score", C, "B + Aqueduct stress"))
R.append(fit("log_desal_pc1 ~ log_gni + aq_bws_score + fossil_elec_pct", C, "C + fossil electricity"))
R.append(fit("log_desal_pc1 ~ log_gni + aq_bws_score + fossil_elec_pct + oilgas_rents_pct", C, "D + oil and gas rents"))
R.append(fit("log_desal_pc1 ~ log_gni + aq_bws_score + fossil_elec_pct + oilgas_rents_pct + is_mena", C, "D + MENA indicator"))
R.append(fit("log_desal_pc1 ~ log_gni + aq_bws_score + fossil_elec_pct + oilgas_rents_pct + log_clim", C, "E + hazard exposure"))
R.append(fit("log_desal_pc1 ~ log_gni + aq_bws_score + fossil_elec_pct + oilgas_rents_pct + log_drought", C, "E + drought exposure only"))
R.append(fit("log_desal_pc1 ~ log_gni + aq_bws_score + fossil_elec_pct + oilgas_rents_pct + log_floodstorm", C, "E + flood and storm exposure"))
R.append(fit("log_desal_pc1 ~ log_gni + aq_bws_score + fossil_elec_pct + oilgas_rents_pct + ndgain_2023", C, "E + ND-GAIN"))
R.append(fit("log_desal_pc1 ~ log_gni + aq_bws_score + renew_share_pct + oilgas_rents_pct", C, "C' renewable final energy instead"))
R.append(fit("log_desal_pc1 ~ log_gni + aq_bws_score + nonhydro_renew_elec_pct + oilgas_rents_pct", C, "C'' non-hydro renewable electricity"))
R.append(fit("log_desal_pc1 ~ log_gni + water_stress_pct + fossil_elec_pct + oilgas_rents_pct", C, "C with SDG 6.4.2 stress"))
R.append(fit("log_desal_pc1 ~ log_gni + aq_bws_score + fossil_elec_pct + oilgas_rents_pct", C[~C.is_gcc], "D excluding GCC"))
R.append(fit("log_desal_pc1 ~ log_gni + aq_bws_score + fossil_elec_pct + oilgas_rents_pct", C[~C.is_mena], "D excluding MENA"))
R.append(fit("log_desal_pc1 ~ log_gni + aq_bws_score + fossil_elec_pct + oilgas_rents_pct", C[C.has_series], "D reported series only (no imputed zeros)"))
R.append(fit("log_desal_pc1 ~ log_gni + aq_bws_score + fossil_elec_pct + oilgas_rents_pct", C[C.iso3 != "CHN"], "D excluding China"))
R.append(fit("log_desal_pc1 ~ log_gni + aq_bws_score + fossil_elec_pct + oilgas_rents_pct", C[(C.desal_km3 == 0) | (C.desal_last_update_year >= 2010)], "D desalination value last changed in 2010 or later (or zero)"))
# year-matched sensitivity: for values that last changed in 2010 or later, energy, income, rents and population are taken
# from that year (within two years) instead of the latest year; countries without desalination keep their latest values
YM = C.copy(); upd = YM.desal_last_update_year >= 2010
for a, b in [("fossil_elec_pct", "fossil_elec_at_desal_year"), ("gni_pc_ppp", "gni_pc_ppp_at_desal_year"), ("oilgas_rents_pct", "oilgas_rents_at_desal_year"), ("pop", "pop_at_desal_year")]:
    YM.loc[upd, a] = YM.loc[upd, b]
YM["log_gni"] = np.log(YM.gni_pc_ppp); YM["log_desal_pc1"] = np.log1p(YM.desal_km3 * 1e9 / YM["pop"])
R.append(fit("log_desal_pc1 ~ log_gni + aq_bws_score + fossil_elec_pct + oilgas_rents_pct", YM[(YM.desal_km3 == 0) | upd], "D year-matched, value last changed in 2010 or later (or zero)"))
R.append(fit("log_desal_pc1 ~ log_gni + aq_bws_score + fossil_elec_pct + oilgas_rents_pct", YM[upd], "D year-matched, reported values that changed in 2010 or later only"))
ym_ok = YM[(YM.desal_km3 == 0) | upd].dropna(subset=["log_gni", "fossil_elec_pct", "oilgas_rents_pct"]).index
R.append(fit("log_desal_pc1 ~ log_gni + aq_bws_score + fossil_elec_pct + oilgas_rents_pct", C.loc[ym_ok], "D latest values, same countries as the year-matched model"))
pd.DataFrame(R).to_csv(f"{O}table1_regressions.csv", index=False)
# reporting years of the cross-section inputs
yrs = C.dropna(subset=["aq_bws_score", "fossil_elec_pct"])
out["reporting_years_median"] = {c: float(yrs[c].median()) for c in ["desal_year", "desal_last_update_year", "eg_elc_fosl_zs_year", "ny_gnp_pcap_pp_kd_year", "er_h2o_fwst_zs_year", "oilgas_rents_pct_year"]}
out["desal_last_update_desaliners_median_p10_p90"] = [float(yrs[yrs.desaliner].desal_last_update_year.median()), float(yrs[yrs.desaliner].desal_last_update_year.quantile(0.1)), float(yrs[yrs.desaliner].desal_last_update_year.quantile(0.9))]
pd.DataFrame(R).to_csv(f"{O}table1_regressions.csv", index=False)
# two-part model
d2 = C.dropna(subset=["log_gni", "aq_bws_score", "fossil_elec_pct", "oilgas_rents_pct"]).copy()
lg = smf.logit("desaliner ~ log_gni + aq_bws_score + fossil_elec_pct + oilgas_rents_pct", d2.assign(desaliner=d2.desaliner.astype(int))).fit(disp=0)
out["twopart_logit"] = {k: [round(lg.params[k], 4), round(lg.pvalues[k], 4)] for k in lg.params.index}; out["twopart_logit_n"] = int(lg.nobs)
pos = d2[d2.desaliner]; ol = smf.ols("log_desal_pc1 ~ log_gni + aq_bws_score + fossil_elec_pct + oilgas_rents_pct", pos).fit(cov_type="HC3"); ci = ol.conf_int()
out["twopart_positive"] = {k: [round(ol.params[k], 4), round(ci.loc[k, 0], 4), round(ci.loc[k, 1], 4), round(ol.pvalues[k], 4)] for k in ol.params.index}; out["twopart_positive_n"] = int(ol.nobs)
# pathways
P = C[C.pathway != "Landlocked"]
out["pathway_medians"] = P.groupby("pathway").agg(n=("iso3", "size"), desal_share=("desal_share_lower_pct", "median"), desal_pc=("desal_m3_pc", "median"), fossil=("fossil_elec_pct", "median"), stress=("aq_bws_score", "median"), share_desalinating=("desaliner", "mean")).round(2).reset_index().values.tolist()
out["pathway_kruskal_desal_pc"] = list(map(float, stats.kruskal(*[g.log_desal_pc1.values for _, g in P.groupby("pathway")])))
R2 = fit("log_desal_pc1 ~ log_gni + aq_bws_score + fossil_elec_pct + is_sids + hydrocarbon", C, "pathways"); out["pathway_model"] = R2
top9 = X[X.desaliner].sort_values("desal_share_lower_pct", ascending=False).head(9); out["top9_pathways"] = top9[["name", "pathway", "desal_share_lower_pct"]].round(1).values.tolist()
# leave-one-out predictions from model D, back-transformed to m3 per person (Fig. 3b, Supplementary Table 3)
dD = C.dropna(subset=["log_gni", "aq_bws_score", "fossil_elec_pct", "oilgas_rents_pct"]).copy()
Xd = sm.add_constant(dD[["log_gni", "aq_bws_score", "fossil_elec_pct", "oilgas_rents_pct"]]); yd = dD.log_desal_pc1
pred = np.array([sm.OLS(yd.drop(i), Xd.drop(i)).fit().predict(Xd.loc[[i]]).iloc[0] for i in dD.index])
dD["pred_log1p"] = pred; dD["resid"] = yd - pred; dD["pred_m3"] = np.clip(np.expm1(pred), 0, None); dD["obs_m3"] = dD.desal_m3_pc
dD[["name", "iso3", "pathway", "obs_m3", "pred_m3", "log_desal_pc1", "pred_log1p", "resid", "has_series"]].sort_values("resid").to_csv(f"{O}supp_table3_residuals.csv", index=False)
out["loo_r2_modelD"] = float(1 - ((yd - pred) ** 2).sum() / ((yd - yd.mean()) ** 2).sum())

# standardised effects (Fig. 2b)
def z(s): return (s - s.mean()) / s.std()
Z = dD.copy()
for v in ["log_gni", "aq_bws_score", "fossil_elec_pct", "oilgas_rents_pct", "ndgain_2023", "aq_drought_score", "aq_ws_change_bau50_score"]: Z["z_" + v] = z(Z[v])
for v, src in [("clim", "clim_aff_per1000_yr"), ("dr", "drought_aff_per1000_yr"), ("fs", "floodstorm_aff_per1000_yr")]: Z["z_" + v] = z(np.log1p(Z[src]))
base = "log_desal_pc1 ~ z_aq_bws_score + z_fossil_elec_pct + z_log_gni + z_oilgas_rents_pct"
m = smf.ols(base, Z).fit(cov_type="HC3"); rows = []
for nm, v in [("Water stress (Aqueduct)", "z_aq_bws_score"), ("Fossil share of electricity", "z_fossil_elec_pct"), ("Income (log GNI per capita)", "z_log_gni"), ("Oil and gas rents", "z_oilgas_rents_pct")]:
    c = m.conf_int().loc[v]; rows.append((nm, m.params[v], c[0], c[1], "main model"))
for nm, v in [("All climate hazards (EM-DAT)", "z_clim"), ("Drought (EM-DAT)", "z_dr"), ("Floods and storms (EM-DAT)", "z_fs"), ("Drought risk (Aqueduct)", "z_aq_drought_score"),
              ("Projected stress change, 2050", "z_aq_ws_change_bau50_score"), ("ND-GAIN index", "z_ndgain_2023")]:
    mm = smf.ols(base + " + " + v, Z.dropna(subset=[v])).fit(cov_type="HC3"); c = mm.conf_int().loc[v]; rows.append((nm, mm.params[v], c[0], c[1], "added one at a time"))
pd.DataFrame(rows, columns=["term", "beta_sd", "ci_lo", "ci_hi", "block"]).to_csv(f"{O}fig2b_standardised_effects.csv", index=False)

# Spearman correlations among desalinating coastal countries (Supplementary Table 6)
Dd = X[X.coastal & X.desaliner]; sp = []
for v, lab in [("renew_share_pct", "Renewable share of final energy (SDG 7.2.1)"), ("fossil_elec_pct", "Fossil share of electricity"),
               ("water_stress_pct", "Water stress (SDG 6.4.2)"), ("aq_bws_score", "Aqueduct 4.0 baseline water stress score (0-5)"),
               ("aq_ws_change_bau50_score", "Aqueduct 4.0 change in water stress score to 2050, BAU"), ("aq_drought_score", "Aqueduct 4.0 drought risk score"), ("ndgain_2023", "ND-GAIN index 2023"),
               ("clim_aff_per1000_yr", "People affected by climate hazards, per 1000 per yr (EM-DAT 2000-2023)"),
               ("sdg631_wastewater_treated", "Domestic wastewater safely treated (SDG 6.3.1)"), ("undernourish_pct", "Undernourishment (SDG 2.1.1)"),
               ("wash_mort_per100k", "WASH mortality (SDG 3.9.2)"), ("gni_pc_ppp", "GNI per capita, PPP")]:
    dd = Dd.dropna(subset=[v, "desal_share_lower_pct"]); r, p = stats.spearmanr(dd.desal_share_lower_pct, dd[v]); sp.append({"variable": lab, "rho": r, "p": p, "n": len(dd)})
pd.DataFrame(sp).to_csv(f"{O}supp_table6_spearman.csv", index=False)

# ---------- wastewater (Supplementary Fig. 1) ----------
d = C.dropna(subset=["sdg631_wastewater_treated", "log_gni"]).copy()
m = smf.ols("sdg631_wastewater_treated ~ log_gni + log_desal_pc1", d).fit(cov_type="HC3"); ci = m.conf_int()
out["wastewater"] = {"coef": round(m.params.log_desal_pc1, 3), "ci": [round(ci.loc["log_desal_pc1", 0], 3), round(ci.loc["log_desal_pc1", 1], 3)], "p": round(m.pvalues.log_desal_pc1, 3), "n": int(m.nobs)}
m2 = smf.ols("sdg631_wastewater_treated ~ log_gni + log_desal_pc1", d[d.iso3 != "MLT"]).fit(cov_type="HC3"); out["wastewater_exMalta"] = [round(m2.params.log_desal_pc1, 3), round(m2.pvalues.log_desal_pc1, 3)]

json.dump(out, open(f"{O}results_cross_section.json", "w"), indent=1, default=float)
print(json.dumps(out, indent=1, default=float))
