# Hungary Election Model (refactored)

This is a refactor of the single-file `hungary_model.py` into an importable package (`hungary_model/`)
with enterprise-style separation of concerns:

- `turnout.py` — turnout models and station vote prediction
- `coefficients.py` — BLOCK coefficients and mapping into 2026 party space
- `vote_allocation.py` — IPF raking and baseline+marginal allocation
- `seats.py` — EVK winners, compensation votes, D'Hondt
- `polls.py` — poll loading, averaging, raw/decided conversions
- `pollster_analysis.py` — backtests (bias/MAE/RMSE) incl. UNDECIDED evaluation and aggregation
- `monte_carlo.py` — Monte Carlo wrapper (with optional multinomial national-share sampling)
- `data.py` — loading/normalization incl. 2026 EVK mapping and EVK name metadata

`app.py` is the Streamlit UI.

## Data layout

## Documentation

- Turnout models: `docs/turnout_models.md` (also shown in the app under Methodology).


By default the app expects:

- DB directory: `./Hungary Election Results/`
- Poll directory: `./`

Required DB files (unchanged from original):
- `national_result_2018.db`
- `national_result_2022.db`
- `single_constituency_result_2018.db`
- `single_constituency_result_2022.db`
- `ep_results_2019.db`
- `ep_results_2024.db`

Required poll CSVs:
- `2018_poll_hungary.csv`
- `2019_poll_hungary.csv`
- `2022_poll_hungary.csv`
- `2024_poll_hungary.csv`
- `2026_poll_hungary.csv`

### 2026 EVK mapping (optional but recommended)

If present, the loader will use the 2026 polling-station→EVK map for **all historical elections**
so coefficients and EVK aggregation are consistent with 2026 boundaries.

Expected file (auto-detected):
- `vtr_ogy2026_fffffff.db` in the DB directory

Expected columns (table: `polling_stations`):
- `maz`, `taz`, `sorszam` (or `sorsz`)
- `evk` (district number inside county)
- `evk_nev` (constituency name, optional)
- an optional registered voters column such as `num_voters`

## Running

```bash
streamlit run app.py
```

Use *Paths (auto-detected; override optional)* in the sidebar if your folders differ.

## Deploying on Streamlit Community Cloud (GitHub)

Streamlit Community Cloud copies your repository and runs `streamlit run` from the **repo root**.

### 1) Add dependencies
This repo includes a `requirements.txt` (and optional `.streamlit/config.toml`). Community Cloud will install Python dependencies from `requirements.txt`.

### 2) Put data in the repo (recommended) OR upload a zip in the UI

**Recommended repo layout**
```
your_repository/
├── app.py
├── requirements.txt
├── data/
│   ├── 2018_poll_hungary.csv
│   ├── 2019_poll_hungary.csv
│   ├── 2022_poll_hungary.csv
│   ├── 2024_poll_hungary.csv
│   ├── 2026_poll_hungary.csv
│   └── Hungary Election Results/
│       ├── national_result_2018.db
│       ├── national_result_2022.db
│       ├── single_constituency_result_2018.db
│       ├── single_constituency_result_2022.db
│       ├── ep_results_2019.db
│       └── ep_results_2024.db
```

The app auto-detects inputs in:
- `./Hungary Election Results/` + `./` (repo root), or
- `./data/Hungary Election Results/` + `./data/`

**If you don't want to commit data to GitHub**
The sidebar includes an **Upload data bundle (.zip)** option. Upload a zip containing:
- `Hungary Election Results/` with the required `*.db` files
- the required poll CSV files

Note: Community Cloud defaults to **200 MB** per uploaded file, but you can increase this via `.streamlit/config.toml` (`server.maxUploadSize`).

### 3) Select Python version in Cloud settings
Community Cloud lets you pick the Python version in the deployment **Advanced settings** UI.
