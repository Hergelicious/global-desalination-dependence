# Water scarcity and fossil electricity shape global desalination dependence

Code and derived data for the article of the same title (Hassan, submitted to *Desalination*).
The scripts assemble an open country-level dataset for 194 states (150 coastal) and reproduce every
number, table and supplementary table in the article.

**Archived version:** Zenodo DOI to be added on release.
**Contact:** Ahmed Hassona Hassan, CEITEC, Masaryk University, Brno (ahmed.hassan@ceitec.muni.cz)

## Contents

| Path | What it is |
| --- | --- |
| `scripts/01_aqueduct_country.py` | Aggregates WRI Aqueduct 4.0 sub-basin scores to countries by area weighting |
| `scripts/02_assemble_panel.py` | Builds the country table from WDI, AQUASTAT (via Gapminder), EM-DAT (via Gapminder), SDG 6.3.1, ND-GAIN and Aqueduct |
| `scripts/03_cross_section.py` | Dependence bounds, Table 1 and its robustness models, two-part model, data-timing checks (latest and year-matched covariates), pathway groups, leave-one-out predictions (Fig. 3b, Supplementary Table 3), standardised effects (Fig. 2b), Spearman table (Supplementary Table 6), wastewater test (Supplementary Fig. 1) |
| `scripts/04_change_since_2000.py` | Desalination growth against change in renewable electricity, 2000 to latest (Fig. 4, Supplementary Table 2) |
| `scripts/05_machine_learning.py` | Staged ridge and gradient-boosting comparison with repeated cross-validation, paired differences, regional transfer, interaction test (Fig. 5a,b, Supplementary Tables 1 and 7) |
| `scripts/06_exposure_index.py` | Lock-in exposure index and its comparison with alternative forms, with bootstrap ranges (Fig. 5c, Supplementary Table 5) |
| `scripts/07_aqueduct_official_sensitivity.py` | Main model with WRI's official demand-weighted Aqueduct country scores and without small island states (Supplementary Table 8) |
| `run_all.sh` | Runs 01 to 07 in order |
| `country_list.csv` | ISO3 codes and names of the 194 states analysed |
| `inputs/` | SDG 6.3.1 (UN-Water) and ND-GAIN 2023, transcribed from public web pages, and WRI's official Aqueduct 4.0 country water-stress scores (bws, all sectors) |
| `data/` | Assembled country table (`panel_crosssection.csv`), desalination time series and Aqueduct country aggregates |
| `results/` | Output tables and JSON summaries written by scripts 03 to 06 |
| `tool/index.html` | Stand-alone screening tool for the exposure index (open in any browser) |

## Quick start

The assembled country table is included, so the analyses run without downloading anything:

```bash
python -m pip install -r requirements.txt
python scripts/03_cross_section.py
python scripts/04_change_since_2000.py
python scripts/05_machine_learning.py      # about 3 minutes
python scripts/06_exposure_index.py
python scripts/07_aqueduct_official_sensitivity.py
```

Outputs are written to `results/`.

## Rebuilding the country table from the raw sources

1. Clone the two open-numbers mirrors into `inputs/` (versions used in the article in brackets):
   - World Development Indicators: <https://github.com/open-numbers/ddf--open_numbers--world_development_indicators> (commit `1bd7fac`, 18 October 2026)
   - Gapminder systema_globalis: <https://github.com/open-numbers/ddf--gapminder--systema_globalis> (commit `ee19060`, 7 May 2026); provides AQUASTAT *Desalinated water produced* and EM-DAT people affected by drought, flood, storm and extreme temperature
2. Download WRI Aqueduct 4.0 (`Aqueduct40_waterrisk_download_Y2023M07D05`) from <https://www.wri.org/data/aqueduct-global-maps-40-data> and unzip it into `inputs/`.
3. Optional: an AQUASTAT dissemination export (<https://data.apps.fao.org/aquastat/>) saved as `inputs/aquastat.csv`, used only to cross-check SDG 6.4.2.
4. Run `bash run_all.sh`. Paths can be changed with the environment variables `WDI_DIR`, `SG_DIR`, `AQUEDUCT_DIR` and `AQUASTAT_CSV`.

`inputs/aqueduct40_country_bws_official.csv` is extracted from `Aqueduct40_rankings_download_Y2023M07D05.xlsx` (sheet `country_baseline`, indicator `bws`, weight `Tot`) at <https://www.wri.org/data/aqueduct-40-country-rankings>; -9999 marks countries without a score. The transcribed inputs come from the UN-Water SDG 6 Data Portal (<https://sdg6data.org>, indicator 6.3.1, domestic wastewater safely treated) and the ND-GAIN Country Index ranking (<https://gain.nd.edu/our-work/country-index/rankings/>).

## Key definitions

- **Desalination per person**: AQUASTAT desalinated water produced (km³ per year, latest reported value) divided by total population; models use log(1 + m³ per person per year). Countries without an AQUASTAT series are set to zero in the main models and dropped in the sensitivity models.
- **Desalination share of supply**: lower bound D / (F + D) and upper bound D / F (capped at 100%), where F is total freshwater withdrawal (WDI ER.H2O.FWTL.K3). The lower bound is exact only where D > F.
- **Groups**: hydrocarbon economies (oil and gas rents at least 10% of GDP), UN small island developing states not in that group, other coastal countries.
- **Lock-in exposure index**: 100 × [(Aqueduct baseline water-stress score / 5) + (fossil share of electricity / 100)] / 2.

## Software

Tested with Python 3.13, pandas 3.0, NumPy 2.5, SciPy 1.18, statsmodels 0.15 and scikit-learn 1.9. Random seeds are fixed (`random_state=0`), so the cross-validation results are reproducible.

## Licence and citation

Code: MIT licence (see `LICENSE`). Derived data: CC BY 4.0, subject to the terms of the original sources (World Bank WDI CC BY 4.0; FAO AQUASTAT; Gapminder CC BY 4.0; WRI Aqueduct CC BY 4.0; EM-DAT, CRED/UCLouvain; ND-GAIN, University of Notre Dame; UN-Water SDG 6 Data Portal).
Please cite the article and the Zenodo record (see `CITATION.cff`).
