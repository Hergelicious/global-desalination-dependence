#!/usr/bin/env bash
# Rebuild the country table from raw sources and rerun every analysis.
set -euo pipefail
cd "$(dirname "$0")"
python scripts/01_aqueduct_country.py
python scripts/02_assemble_panel.py
python scripts/03_cross_section.py
python scripts/04_change_since_2000.py
python scripts/05_machine_learning.py
python scripts/06_exposure_index.py
python scripts/07_aqueduct_official_sensitivity.py
