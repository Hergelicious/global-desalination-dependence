"""Lock-in exposure index (Fig. 5c, Supplementary Table 5, screening tool data).
LEI = 100 x [(Aqueduct stress / 5) + (fossil share of electricity / 100)] / 2. The additive form is compared with the product,
a regression-weighted score and each input alone by the area under the curve for countries above 10% dependence."""
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
from sklearn.metrics import roc_auc_score
out = {}
C["y"] = C.log_desal_pc1
Q = C.dropna(subset=["aq_bws_score", "fossil_elec_pct"]).copy()
Q["mult"] = (Q.aq_bws_score / 5) * (Q.fossil_elec_pct / 100); Q["add"] = (Q.aq_bws_score / 5 + Q.fossil_elec_pct / 100) / 2
mreg = smf.ols("y ~ aq_bws_score + fossil_elec_pct", Q).fit(); Q["reg"] = mreg.fittedvalues
Q["hi"] = (Q.desal_share_lower_pct.fillna(0) > 10).astype(int)
idx = {}
for k in ["mult", "add", "reg", "aq_bws_score", "fossil_elec_pct"]:
    idx[k] = {"spearman_y": float(stats.spearmanr(Q[k], Q.y)[0]), "auc_share_gt10": float(roc_auc_score(Q.hi, Q[k])), "auc_any": float(roc_auc_score(Q.desaliner.astype(int), Q[k]))}
# bootstrap (country resampling) intervals for the AUCs; these comparisons are in-sample and descriptive
rng = np.random.default_rng(0); B = {k: [] for k in ["mult", "add", "reg", "aq_bws_score", "fossil_elec_pct"]}
for _ in range(2000):
    s_ = Q.sample(len(Q), replace=True, random_state=int(rng.integers(1e9)))
    if s_.hi.nunique() < 2: continue
    for k in B: B[k].append(roc_auc_score(s_.hi, s_[k]))
for k in B: idx[k]["auc_share_gt10_boot95"] = [float(np.quantile(B[k], 0.025)), float(np.quantile(B[k], 0.975))]
out["index_compare"] = idx; out["index_n"] = int(len(Q)); out["n_share_gt10_in_Q"] = int(Q.hi.sum())


Co = X[X.coastal].copy()
Co["LEI"] = 100 * (Co.aq_bws_score.clip(0, 5) / 5 + Co.fossil_elec_pct.clip(0, 100) / 100) / 2
Co["share"] = Co.desal_share_lower_pct.fillna(0)
T = Co[["name", "iso3", "pathway", "LEI", "aq_bws_score", "fossil_elec_pct", "renew_elec_pct", "share", "desal_m3_pc"]].sort_values("LEI", ascending=False)
T.round(2).to_csv(f"{O}supp_table5_exposure_index.csv", index=False)
d = T.dropna(subset=["LEI"]); r, p = stats.spearmanr(d.LEI, np.log1p(d.desal_m3_pc.fillna(0)))
out["lei_vs_desal_pc_spearman"] = [float(r), float(p), int(len(d))]
out["lei_median_share_gt10"] = [float(d[d.share > 10].LEI.median()), int((d.share > 10).sum())]; out["lei_median_other"] = float(d[d.share <= 10].LEI.median())
mult = d.aq_bws_score / 5 * d.fossil_elec_pct; out["additive_vs_product_spearman"] = float(stats.spearmanr(d.LEI, mult)[0])
out["top20_overlap_additive_product"] = len(set(d.nlargest(20, "LEI").iso3) & set(d.assign(m=mult).nlargest(20, "m").iso3))
d.rename(columns={"share": "desal_share"})[["name", "iso3", "LEI", "aq_bws_score", "fossil_elec_pct", "renew_elec_pct", "desal_share", "desal_m3_pc"]].round(2).to_json(f"{O}screener_countries.json", orient="records")
json.dump(out, open(f"{O}results_exposure_index.json", "w"), indent=1, default=float)
print(json.dumps(out, indent=1, default=float)); print(T.head(20).round(1).to_string())
