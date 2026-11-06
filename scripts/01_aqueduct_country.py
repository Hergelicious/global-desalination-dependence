"""Aggregate WRI Aqueduct 4.0 sub-basin indicators to countries by area weighting.

Input : Aqueduct 4.0 annual baseline and future CSVs (Aqueduct40_waterrisk_download_Y2023M07D05/CVS/)
Output: data/aqueduct40_country_areaweighted.csv (states in country_list.csv)
Each row of the baseline table is a HydroBASINS level-6 sub-basin x admin-1 polygon with its area.
Polygons without data (score -9999) are dropped; 'arid and low water use' polygons carry a stress score of 5.
"""
import os, sys
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AQ = os.environ.get("AQUEDUCT_DIR", f"{ROOT}/inputs/Aqueduct40_waterrisk_download_Y2023M07D05/CVS")

b = pd.read_csv(f"{AQ}/Aqueduct40_baseline_annual_y2023m07d05.csv",
                usecols=["pfaf_id", "gid_0", "area_km2", "bws_raw", "bws_score", "bws_cat", "drr_score", "iav_score", "sev_score"])
f = pd.read_csv(f"{AQ}/Aqueduct40_future_annual_y2023m07d05.csv",
                usecols=["pfaf_id", "bau30_ws_x_r", "bau50_ws_x_r", "bau80_ws_x_r", "pes50_ws_x_r", "opt50_ws_x_r", "bau50_ws_x_s"])
b = b.dropna(subset=["gid_0"]).merge(f, on="pfaf_id", how="left")

def wmean(g, col):
    v = g[col]; ok = v.notna() & (v > -9999)
    return np.average(v[ok], weights=g.area_km2[ok]) if ok.any() else np.nan

rows = []
for iso, g in b.groupby("gid_0"):
    rows.append({
        "iso3": iso,
        "aq_bws_raw": wmean(g, "bws_raw"), "aq_bws_score": wmean(g, "bws_score"),
        "aq_drought_score": wmean(g, "drr_score"), "aq_interannual_var_score": wmean(g, "iav_score"),
        "aq_seasonal_var_score": wmean(g, "sev_score"),
        "aq_ws_bau30_raw": wmean(g, "bau30_ws_x_r"), "aq_ws_bau50_raw": wmean(g, "bau50_ws_x_r"), "aq_ws_bau80_raw": wmean(g, "bau80_ws_x_r"),
        "aq_ws_pes50_raw": wmean(g, "pes50_ws_x_r"), "aq_ws_opt50_raw": wmean(g, "opt50_ws_x_r"),
        "aq_ws_bau50_score": wmean(g, "bau50_ws_x_s"),
        "aq_arid_lowuse_share": g.loc[g.bws_cat == -1, "area_km2"].sum() / g.area_km2.sum(),
        "area_km2": g.area_km2.sum(),
    })
out = pd.DataFrame(rows)
out = out[out.iso3.isin(pd.read_csv(f"{ROOT}/country_list.csv").iso3)]
out["aq_ws_change_bau50"] = out.aq_ws_bau50_raw - out.aq_bws_raw
out["aq_ws_change_bau50_score"] = out.aq_ws_bau50_score - out.aq_bws_score
os.makedirs(f"{ROOT}/data", exist_ok=True)
out.to_csv(f"{ROOT}/data/aqueduct40_country_areaweighted.csv", index=False)
print(f"{len(out)} countries written")
