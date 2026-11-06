"""Staged machine-learning comparison (Fig. 5a,b; Supplementary Tables 1 and 7).
Ridge regression and histogram gradient boosting, five nested feature sets, 5-fold cross-validation repeated 20 times.
Imputation and scaling sit inside the pipeline, so they are fitted on each training fold only."""
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
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
out = {}
C["y"] = C.log_desal_pc1

# ---------- means, s.d. and regional transfer (Supplementary Table 1) ----------
stages = {"A income": ["log_gni"], "B + water stress": ["log_gni", "aq_bws_score"], "C + electricity mix": ["log_gni", "aq_bws_score", "fossil_elec_pct"],
          "D + rents, aridity, geography": ["log_gni", "aq_bws_score", "fossil_elec_pct", "oilgas_rents_pct", "aq_arid_lowuse_share", "is_sids", "is_mena"],
          "E + climate exposure": ["log_gni", "aq_bws_score", "fossil_elec_pct", "oilgas_rents_pct", "aq_arid_lowuse_share", "is_sids", "is_mena", "log_clim", "log_drought", "aq_drought_score", "ndgain_2023", "aq_ws_change_bau50_score"]}
M = C.dropna(subset=["log_desal_pc1", "log_gni", "aq_bws_score", "fossil_elec_pct"]).copy(); M["is_sids"] = M.is_sids.astype(float); M["is_mena"] = M.is_mena.astype(float)
y = M.log_desal_pc1; ML = []
def r2(yv, pv): return float(1 - ((yv - pv) ** 2).sum() / ((yv - yv.mean()) ** 2).sum())
rkf = RepeatedKFold(n_splits=5, n_repeats=20, random_state=0)
for st, fs in stages.items():
    Xs = M[fs]
    ridge = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0))
    gbm = make_pipeline(SimpleImputer(strategy="median"), HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=300, min_samples_leaf=8, l2_regularization=1.0, random_state=0))
    res = {"stage": st, "n_features": len(fs)}
    for nm, mdl in [("ridge", ridge), ("gbm", gbm)]:
        sc = []
        for tr, te in rkf.split(Xs):
            mdl.fit(Xs.iloc[tr], y.iloc[tr]); sc.append(r2(y.iloc[te], mdl.predict(Xs.iloc[te])))
        res[f"{nm}_cv_r2_mean"] = round(float(np.mean(sc)), 3); res[f"{nm}_cv_r2_sd"] = round(float(np.std(sc)), 3)
        # regional holdout: train outside MENA, test on MENA; and the reverse
        tr, te = M.is_mena == 0, M.is_mena == 1
        mdl.fit(Xs[tr], y[tr]); pm = mdl.predict(Xs[te]); res[f"{nm}_holdout_MENA_rmse"] = round(float(np.sqrt(((y[te] - pm) ** 2).mean())), 3); res[f"{nm}_holdout_MENA_r2"] = round(r2(y[te], pm), 3)
        tr2, te2 = ~M.is_gcc, M.is_gcc
        mdl.fit(Xs[tr2], y[tr2]); pg = mdl.predict(Xs[te2]); res[f"{nm}_holdout_GCC_rmse"] = round(float(np.sqrt(((y[te2] - pg) ** 2).mean())), 3); res[f"{nm}_holdout_GCC_meanresid"] = round(float((y[te2] - pg).mean()), 3)
    ML.append(res)
pd.DataFrame(ML).to_csv(f"{O}supp_table1_ml_stages.csv", index=False)
tr, te = M.is_mena == 0, M.is_mena == 1
out["mena_baseline_rmse_training_mean"] = float(np.sqrt(((y[te] - y[tr].mean()) ** 2).mean()))
out["mena_baseline_rmse_zero"] = float(np.sqrt((y[te] ** 2).mean()))
out["n_mena"] = int(te.sum())
# two-part classification stage
Mc = M.copy(); yc = Mc.desaliner.astype(int)
for st in ["B + water stress", "C + electricity mix", "D + rents, aridity, geography"]:
    clf = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))
    out[f"clf_auc_{st}"] = round(float(np.mean(cross_val_score(clf, Mc[stages[st]], yc, cv=RepeatedKFold(n_splits=5, n_repeats=20, random_state=0), scoring="roc_auc"))), 3)

# ---------- medians, percentiles and paired differences on identical splits (Fig. 5a,b; Supplementary Table 7) ----------
# paired differences (percentiles across overlapping splits describe cross-validation variability, not independent-sample confidence intervals)
stages = {"A": ["log_gni"], "B": ["log_gni", "aq_bws_score"], "C": ["log_gni", "aq_bws_score", "fossil_elec_pct"],
          "D": ["log_gni", "aq_bws_score", "fossil_elec_pct", "oilgas_rents_pct", "aq_arid_lowuse_share", "is_sids", "is_mena"],
          "E": ["log_gni", "aq_bws_score", "fossil_elec_pct", "oilgas_rents_pct", "aq_arid_lowuse_share", "is_sids", "is_mena", "clim_aff_per1000_yr", "drought_aff_per1000_yr", "aq_drought_score", "ndgain_2023", "aq_ws_change_bau50_score"]}
def mk(nm):
    if nm == "ridge": return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0))
    return make_pipeline(SimpleImputer(strategy="median"), HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=300, min_samples_leaf=8, l2_regularization=1.0, random_state=0))
def run(M, label):
    M = M.copy(); M["is_sids"] = M.is_sids.astype(float); M["is_mena"] = M.is_mena.astype(float)
    for c in ["clim_aff_per1000_yr", "drought_aff_per1000_yr"]: M[c] = np.log1p(M[c])
    y = M.y.values; res = {}
    splits = list(RepeatedKFold(n_splits=5, n_repeats=20, random_state=0).split(M))
    for nm in ["ridge", "gbm"]:
        for st, fs in stages.items():
            r = []
            for tr, te in splits:
                m = mk(nm).fit(M[fs].iloc[tr], y[tr]); p = m.predict(M[fs].iloc[te]); r.append(1 - ((y[te] - p) ** 2).sum() / ((y[te] - y[te].mean()) ** 2).sum())
            res[(nm, st)] = np.array(r)
    rows = []
    for nm in ["ridge", "gbm"]:
        for st in stages: v = res[(nm, st)]; rows.append({"sample": label, "model": nm, "stage": st, "r2_median": np.median(v), "r2_mean": v.mean(), "r2_q05": np.quantile(v, 0.05), "r2_q95": np.quantile(v, 0.95)})
    diffs = []
    for nm in ["ridge", "gbm"]:
        for a, b_ in [("B", "A"), ("C", "B"), ("D", "C"), ("E", "D"), ("E", "C")]:
            d = res[(nm, a)] - res[(nm, b_)]; diffs.append({"sample": label, "model": nm, "contrast": f"{a} − {b_}", "mean": d.mean(), "q025": np.quantile(d, 0.025), "q975": np.quantile(d, 0.975), "share_positive": (d > 0).mean()})
    d = res[("gbm", "C")] - res[("ridge", "C")]; diffs.append({"sample": label, "model": "gbm − ridge", "contrast": "C", "mean": d.mean(), "q025": np.quantile(d, 0.025), "q975": np.quantile(d, 0.975), "share_positive": (d > 0).mean()})
    return pd.DataFrame(rows), pd.DataFrame(diffs), len(M)
M_all = C.dropna(subset=["y", "log_gni", "aq_bws_score", "fossil_elec_pct"])
r1, d1, n1 = run(M_all, "all coastal (zeros imputed)")
M_rep = M_all[M_all.has_series]
r2, d2, n2 = run(M_rep, "reported series only")
pd.concat([r1, r2]).to_csv(f"{O}fig5a_ml_r2_percentiles.csv", index=False); pd.concat([d1, d2]).to_csv(f"{O}supp_table7_ml_paired_differences.csv", index=False)
out["ml_n"] = [n1, n2]

# interaction test
dD = C.dropna(subset=["log_gni", "aq_bws_score", "fossil_elec_pct", "oilgas_rents_pct"]).copy()
mi = smf.ols("y ~ log_gni + aq_bws_score * fossil_elec_pct + oilgas_rents_pct", dD).fit(cov_type="HC3"); ci = mi.conf_int()
out["interaction"] = [float(mi.params["aq_bws_score:fossil_elec_pct"]), float(ci.loc["aq_bws_score:fossil_elec_pct", 0]), float(ci.loc["aq_bws_score:fossil_elec_pct", 1]), float(mi.pvalues["aq_bws_score:fossil_elec_pct"]), int(mi.nobs)]

json.dump(out, open(f"{O}results_machine_learning.json", "w"), indent=1, default=float)
print(json.dumps(out, indent=1, default=float))
