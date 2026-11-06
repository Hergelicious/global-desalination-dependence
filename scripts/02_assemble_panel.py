"""Assemble the country table used in every analysis.

Inputs (see README for download links):
  WDI_DIR     open-numbers mirror of the World Development Indicators (ddf--open_numbers--world_development_indicators)
  SG_DIR      Gapminder systema_globalis (AQUASTAT desalinated water produced; EM-DAT people affected)
  inputs/sdg631_domestic_wastewater_safely_treated.csv   SDG 6.3.1, UN-Water SDG 6 data portal
  inputs/ndgain_2023_index.csv                           ND-GAIN Country Index 2023
  data/aqueduct40_country_areaweighted.csv               from 01_aqueduct_country.py
  AQUASTAT_CSV (optional) AQUASTAT dissemination export, used only to cross-check SDG 6.4.2
  country_list.csv                                       the 194 states analysed
Output: data/panel_crosssection.csv, data/desalination_timeseries.csv
"""
import os, re, unicodedata
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W = os.environ.get("WDI_DIR", f"{ROOT}/inputs/ddf--open_numbers--world_development_indicators") + "/"
S = os.environ.get("SG_DIR", f"{ROOT}/inputs/ddf--gapminder--systema_globalis") + "/countries-etc-datapoints/"
AQUASTAT_CSV = os.environ.get("AQUASTAT_CSV", f"{ROOT}/inputs/aquastat.csv")
I, D = f"{ROOT}/inputs/", f"{ROOT}/data/"
os.makedirs(D, exist_ok=True)

def wdi(code):
    f = f"{W}datapoints/ddf--datapoints--{code}--by--geo--time.csv"
    d = pd.read_csv(f); d.columns = ["geo", "year", code]; return d

def sg(code):
    d = pd.read_csv(f"{S}ddf--datapoints--{code}--by--geo--time.csv"); d.columns = ["geo", "year", code]; return d

ent = pd.read_csv(f"{W}ddf--entities--geo--country.csv", dtype=str)
keep = set(pd.read_csv(f"{ROOT}/country_list.csv").iso3)
ent = ent[(ent["un_state"] == "TRUE") & ent["iso3166_1_alpha3"].isin(keep)][["country", "name", "iso3166_1_alpha3", "un_sdg_region", "income_groups", "landlocked", "world_6region"]]
ent.columns = ["geo", "name", "iso3", "sdg_region", "income_group", "landlocked", "region6"]

def norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"\(.*?\)", "", s); s = s.replace("&", "and")
    return re.sub(r"[^a-z]", "", s)

alias = {  # external name -> WDI name
 "czechia": "czechrepublic", "turkiye": "turkey", "republicofkorea": "southkorea", "vietnam": "vietnam",
 "unitedrepublicoftanzania": "tanzania", "unitedstatesofamerica": "unitedstates",
 "unitedkingdomofgreatbritainandnorthernireland": "unitedkingdom", "laopeoplesdemocraticrepublic": "lao",
 "russianfederation": "russia", "republicofmoldova": "moldova", "iranislamicrepublicof": "iran",
 "boliviaplurinationalstateof": "bolivia", "stateofpalestine": "palestine", "cotedivoire": "cotedivoire",
 "democraticrepublicofthecongo": "congodemrep", "congo": "congorep", "eswatini": "swaziland",
 "northmacedonia": "northmacedonia", "macedonia": "northmacedonia", "capeverde": "capeverde", "caboverde": "capeverde",
 "brunei darussalam": "brunei", "bruneidarussalam": "brunei", "syria": "syria", "laos": "lao",
 "dempeoplesrepofkorea": "northkorea", "demrepofthecongo": "congodemrep", "centralafricanrep": "centralafricanrepublic",
 "stvincentandgrenadines": "stvincentandthegrenadines", "saotomeandprincipe": "saotomeandprincipe",
 "micronesia": "micronesiafedsts", "kyrgyzstan": "kyrgyzrepublic", "slovakia": "slovakrepublic",
 "gambia": "gambia", "bahamas": "bahamas", "egypt": "egypt", "yemen": "yemen", "venezuela": "venezuela",
 "nauru": "nauru", "naoero": "nauru", "timorleste": "timorleste", "hongkong": "hongkongchina",
 "chinahongkongspecialadministrativeregion": "__skip__", "chinamacaospecialadministrativeregion": "__skip__",
 "unitedarabemirates": "uae", "unitedkingdom": "uk", "unitedstates": "usa", "unitedstatesofamerica": "usa",
 "unitedkingdomofgreatbritainandnorthernireland": "uk", "saintlucia": "stlucia", "swaziland": "eswatini", "eswatini": "eswatini",
}
wnames = {norm(n): g for n, g in zip(ent["name"], ent["geo"])}

def to_geo(name):
    k = norm(name)
    if k in wnames: return wnames[k]
    k2 = alias.get(k)
    if k2 == "__skip__": return None
    if k2 and k2 in wnames: return wnames[k2]
    return None

# ---- latest-value helper ----
def latest(d, code, since=2010):
    d = d[d.year >= since].dropna(subset=[code]).sort_values("year")
    return d.groupby("geo").tail(1).rename(columns={"year": f"{code}_year"})[["geo", code, f"{code}_year"]]

# ---- WDI series ----
ren = wdi("eg_fec_rnew_zs"); und = wdi("sn_itk_defc_zs"); wash = wdi("sh_sta_wash_p5")
gni = wdi("ny_gnp_pcap_pp_kd"); pop = wdi("sp_pop_totl"); stress = wdi("er_h2o_fwst_zs")
wdraw = wdi("er_h2o_fwtl_k3"); irwr = wdi("er_h2o_intr_k3"); mdat = wdi("en_clc_mdat_zs")
fosl = wdi("eg_elc_fosl_zs"); urb = wdi("sp_urb_totl_in_zs")

# ---- Gapminder/AQUASTAT desalination ----
des = sg("desalinated_water_produced_billion_cu_meters"); des.columns = ["geo", "year", "desal_km3"]
des = des.sort_values(["geo", "year"])
# flag carried-forward values: identical to previous year
des["changed"] = des.groupby("geo")["desal_km3"].diff().fillna(1) != 0
# number of distinct reported levels since 2000 (proxy for real updates)
nupd = des[des.year >= 2000].groupby("geo")["desal_km3"].nunique().rename("desal_n_distinct_2000on")
des_latest = des.sort_values("year").groupby("geo").tail(1)[["geo", "year", "desal_km3"]].rename(columns={"year": "desal_year"})
des_2000 = des[des.year == 2000][["geo", "desal_km3"]].rename(columns={"desal_km3": "desal_km3_2000"})
# first year the latest value appeared (end of carry-forward run start) = last real update
last_update = des[des.changed].sort_values("year").groupby("geo").tail(1)[["geo", "year"]].rename(columns={"year": "desal_last_update_year"})

# ---- EM-DAT via Gapminder: climate hazards affected and deaths ----
haz = None
for h in ["drought", "flood", "storm", "extreme_temperature"]:
    a = sg(f"{h}_affected_annual_number"); a.columns = ["geo", "year", f"{h}_aff"]
    haz = a if haz is None else haz.merge(a, on=["geo", "year"], how="outer")
haz = haz[(haz.year >= 2000) & (haz.year <= 2023)].fillna(0)
haz["clim_aff"] = haz[[c for c in haz.columns if c.endswith("_aff")]].sum(axis=1)
hz = haz.groupby("geo").agg(clim_aff_sum=("clim_aff", "sum"), hz_years=("year", "nunique")).reset_index()
popmean = pop[(pop.year >= 2000) & (pop.year <= 2023)].groupby("geo")["sp_pop_totl"].mean().rename("pop_mean_2000_23")
hz = hz.merge(popmean, on="geo"); hz["clim_aff_per1000_yr"] = hz.clim_aff_sum / hz.hz_years / hz.pop_mean_2000_23 * 1000

# ---- SDG 6.3.1 and ND-GAIN (transcribed) ----
s631 = pd.read_csv(f"{I}sdg631_domestic_wastewater_safely_treated.csv")
s631["geo"] = s631.country.map(to_geo); s631 = s631.dropna(subset=["geo"])[["geo", "value", "period"]].rename(columns={"value": "sdg631_wastewater_treated", "period": "sdg631_period"})
ndg = pd.read_csv(f"{I}ndgain_2023_index.csv"); ndg = ndg[ndg.score != "-"]
ndg["geo"] = ndg.country.map(to_geo); ndg = ndg.dropna(subset=["geo"])
ndg["ndgain_2023"] = ndg.score.astype(float); ndg = ndg[["geo", "ndgain_2023"]]

# ---- assemble cross-section ----
X = ent.copy()
for d, c in [(ren, "eg_fec_rnew_zs"), (und, "sn_itk_defc_zs"), (wash, "sh_sta_wash_p5"), (gni, "ny_gnp_pcap_pp_kd"),
             (pop, "sp_pop_totl"), (stress, "er_h2o_fwst_zs"), (wdraw, "er_h2o_fwtl_k3"), (irwr, "er_h2o_intr_k3"),
             (fosl, "eg_elc_fosl_zs"), (urb, "sp_urb_totl_in_zs")]:
    X = X.merge(latest(d, c), on="geo", how="left")
X = X.merge(latest(mdat, "en_clc_mdat_zs", since=1900), on="geo", how="left")
X = X.merge(des_latest, on="geo", how="left").merge(des_2000, on="geo", how="left").merge(nupd, on="geo", how="left").merge(last_update, on="geo", how="left")
X = X.merge(hz[["geo", "clim_aff_per1000_yr"]], on="geo", how="left").merge(s631, on="geo", how="left").merge(ndg, on="geo", how="left")
# 2000 renewable share for the lock-in test
r2000 = ren[ren.year == 2000][["geo", "eg_fec_rnew_zs"]].rename(columns={"eg_fec_rnew_zs": "renew_2000"})
X = X.merge(r2000, on="geo", how="left")
X["desal_km3"] = X["desal_km3"].fillna(0.0)
X["desal_m3_pc"] = X.desal_km3 * 1e9 / X.sp_pop_totl
X["desal_share_supply_pct"] = X.desal_km3 / (X.er_h2o_fwtl_k3 + X.desal_km3) * 100
X["desal_share_irwr_pct"] = X.desal_km3 / X.er_h2o_intr_k3 * 100
X["desaliner"] = X.desal_km3 > 0
X = X.rename(columns={"eg_fec_rnew_zs": "renew_share_pct", "sn_itk_defc_zs": "undernourish_pct", "sh_sta_wash_p5": "wash_mort_per100k",
                      "ny_gnp_pcap_pp_kd": "gni_pc_ppp", "sp_pop_totl": "pop", "er_h2o_fwst_zs": "water_stress_pct",
                      "er_h2o_fwtl_k3": "withdrawal_km3", "er_h2o_intr_k3": "irwr_km3", "en_clc_mdat_zs": "hazard_pop_share_1990_2009",
                      "eg_elc_fosl_zs": "fossil_elec_pct", "sp_urb_totl_in_zs": "urban_pct"})
# ---- Aqueduct 4.0 country aggregates (area-weighted scores) and AQUASTAT 2021-2023 SDG 6.4.2 export ----
aq = pd.read_csv(f"{D}aqueduct40_country_areaweighted.csv")[["iso3", "aq_bws_score", "aq_drought_score", "aq_interannual_var_score", "aq_ws_bau50_score", "aq_ws_change_bau50_score", "aq_arid_lowuse_share"]]
X = X.merge(aq, on="iso3", how="left")
if os.path.exists(AQUASTAT_CSV):
    aqs = pd.read_csv(AQUASTAT_CSV)
    aqs = aqs[(aqs.Variable == "SDG 6.4.2. Water Stress") & (~aqs.IsAggregate)].sort_values("Year").groupby("m49").tail(1)
    aqs["iso3n"] = aqs.m49.astype(int)
    ent_num = pd.read_csv(f"{W}ddf--entities--geo--country.csv", dtype=str)[["country", "iso3166_1_numeric"]].dropna()
    ent_num["iso3n"] = ent_num.iso3166_1_numeric.astype(int)
    aqs = aqs.merge(ent_num, on="iso3n")[["country", "Value", "Year"]].rename(columns={"country": "geo", "Value": "water_stress_aquastat_2023", "Year": "water_stress_aquastat_year"})
    X = X.merge(aqs, on="geo", how="left")
des.merge(ent[["geo", "iso3", "name"]], on="geo").to_csv(f"{D}desalination_timeseries.csv", index=False)

# ---- v2 additions: rents, electricity mix (latest and 2000), sector withdrawals, hazard split, groups, dependence bounds ----
petr = wdi("ny_gdp_petr_rt_zs"); ngas = wdi("ny_gdp_ngas_rt_zs")
rents = petr.merge(ngas, on=["geo", "year"], how="outer").fillna(0); rents["oilgas_rents_pct"] = rents.ny_gdp_petr_rt_zs + rents.ny_gdp_ngas_rt_zs
X = X.merge(latest(rents, "oilgas_rents_pct"), on="geo", how="left")
elr = wdi("eg_elc_rnew_zs"); X = X.merge(latest(elr, "eg_elc_rnew_zs").rename(columns={"eg_elc_rnew_zs": "renew_elec_pct", "eg_elc_rnew_zs_year": "renew_elec_year"}), on="geo", how="left")
X = X.merge(elr[elr.year == 2000][["geo", "eg_elc_rnew_zs"]].rename(columns={"eg_elc_rnew_zs": "renew_elec_2000"}), on="geo", how="left")
nhx = wdi("eg_elc_rnwx_zs"); X = X.merge(latest(nhx, "eg_elc_rnwx_zs").rename(columns={"eg_elc_rnwx_zs": "nonhydro_renew_elec_pct", "eg_elc_rnwx_zs_year": "nonhydro_year"}), on="geo", how="left")
X = X.merge(nhx[nhx.year == 2000][["geo", "eg_elc_rnwx_zs"]].rename(columns={"eg_elc_rnwx_zs": "nonhydro_renew_elec_2000"}), on="geo", how="left")
fdm = wdi("er_h2o_fwdm_zs"); X = X.merge(latest(fdm, "er_h2o_fwdm_zs")[["geo", "er_h2o_fwdm_zs"]].rename(columns={"er_h2o_fwdm_zs": "municipal_withdrawal_share_pct"}), on="geo", how="left")
X["municipal_withdrawal_km3"] = X.withdrawal_km3 * X.municipal_withdrawal_share_pct / 100
X["desal_vs_municipal_pct"] = (X.desal_km3 / X.municipal_withdrawal_km3 * 100).clip(upper=100)
# dependence bounds: WDI total withdrawal may or may not include desalinated water
X["desal_share_lower_pct"] = X.desal_km3 / (X.withdrawal_km3 + X.desal_km3) * 100          # assumes withdrawal excludes desalination
X["desal_share_upper_pct"] = (X.desal_km3 / X.withdrawal_km3 * 100).clip(upper=100)          # assumes withdrawal includes desalination
X["lower_bound_exact"] = X.desal_km3 > X.withdrawal_km3                                      # only where D > F can F not include D
# hazard split
hz2 = haz.groupby("geo").agg(dr=("drought_aff", "sum"), fl=("flood_aff", "sum"), st=("storm_aff", "sum"), yrs=("year", "nunique")).reset_index().merge(popmean, on="geo")
hz2["drought_aff_per1000_yr"] = hz2.dr / hz2.yrs / hz2.pop_mean_2000_23 * 1000; hz2["floodstorm_aff_per1000_yr"] = (hz2.fl + hz2.st) / hz2.yrs / hz2.pop_mean_2000_23 * 1000
X = X.merge(hz2[["geo", "drought_aff_per1000_yr", "floodstorm_aff_per1000_yr"]], on="geo", how="left")
# year-matched values: energy, income, rents and population in the year the desalination value last changed
def at_year(d, code, geo_year, tol=2):
    d = d.dropna(subset=[code]).merge(geo_year, on="geo"); d = d[(d.year - d.target).abs() <= tol]
    d = d.assign(dist=(d.year - d.target).abs()).sort_values(["geo", "dist", "year"]).groupby("geo").head(1)
    return d.set_index("geo")[code]
gy = X[["geo", "desal_last_update_year"]].dropna().rename(columns={"desal_last_update_year": "target"})
X = X.set_index("geo")
X["fossil_elec_at_desal_year"] = at_year(fosl, "eg_elc_fosl_zs", gy)
X["gni_pc_ppp_at_desal_year"] = at_year(gni, "ny_gnp_pcap_pp_kd", gy)
X["pop_at_desal_year"] = at_year(pop, "sp_pop_totl", gy)
X["oilgas_rents_at_desal_year"] = at_year(rents, "oilgas_rents_pct", gy)
X = X.reset_index()
# groups
SIDS = set("ATG BHS BHR BRB BLZ CPV COM CUB DMA DOM FJI GRD GNB GUY HTI JAM KIR MDV MHL FSM MUS NRU PLW PNG WSM STP SYC SGP SLB KNA LCA VCT SUR TLS TON TTO TUV VUT".split())
GCC = set("BHR KWT OMN QAT SAU ARE".split())
MENA = set("DZA BHR DJI EGY IRN IRQ JOR KWT LBN LBY MAR OMN PSE QAT SAU SYR TUN ARE YEM".split())
X["is_sids"] = X.iso3.isin(SIDS); X["is_gcc"] = X.iso3.isin(GCC); X["is_mena"] = X.iso3.isin(MENA)
X["hydrocarbon"] = X.oilgas_rents_pct >= 10
X["pathway"] = np.where(X.hydrocarbon, "Hydrocarbon economy", np.where(X.is_sids, "Small island state", "Other coastal"))
X.loc[X.landlocked == "landlocked", "pathway"] = "Landlocked"
X.to_csv(f"{D}panel_crosssection.csv", index=False)
print(X.shape, "states;", int((X.landlocked != "landlocked").sum()), "coastal;", int(X.desaliner.sum()), "with desalination")
