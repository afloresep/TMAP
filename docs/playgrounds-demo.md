# Roche demo playgrounds — runbook

## Build the static HTML and models

```bash
pip install -e ".[playground]"
pip install fair-esm  # for the proteins playground only
python scripts/build_word_playground.py
python scripts/build_chembl_playground.py
python scripts/build_spike_playground.py --n 4000
python scripts/build_endosymbiosis_playground.py
```

## Run the backend

```bash
python -m tmap.playgrounds.serve --port 8000
```

First start loads ESM-2 (proteins), ~20 s on MPS / ~60 s on CPU. The server registers any playground whose model file is present; missing models or missing dependencies are skipped gracefully.

## Run the site

```bash
cd ../tmap-website
npm install
npm run dev
```

Open <http://localhost:3000/playgrounds>.

## Scripted demo path (~20 minutes)

1. `/playgrounds/words` — type `jazz`, then `jazz -> engineering`. Show the path tracing semantic bridge.
2. `/playgrounds/chembl` — paste imatinib `CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5`; show 2D structure in detail pane; trace path to vemurafenib `CCCS(=O)(=O)NC1=CC=C(C(=C1F)C(=O)C2=CNC3=NC=C(C=C23)C4=CC=C(C=C4)Cl)F`.
3. `/playgrounds/spike` — search by clade or paste a recent variant; switch coloring to Date; trace `Wuhan-Hu-1 -> JN.1` (or other strain pair from the dataset).
4. `/playgrounds/proteins` — pick HSP60 from gallery; show AlphaFold structure in Mol*; trace cross-kingdom path from a bacterial GroEL to its mitochondrial homolog.

## Quick smoke check

```bash
./scripts/smoke_playgrounds.sh
```
