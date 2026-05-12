# Roche Demo Playgrounds — Design

**Date:** 2026-05-12
**Owner:** Albert Flores
**Purpose:** Build a portfolio of four interactive TMAP playgrounds for a live click-through demo at Roche. Audience: senior leadership, data scientists, and medicinal/computational chemists.

## Goal

Show what TMAP can do that UMAP/t-SNE cannot:

1. **Nearest-neighbor search** — paste a query, see the closest points highlighted on the map.
2. **Path tracing** — pick two points, see the MST path between them.
3. **Add-points** — drop a new item onto an existing map without refitting.
4. **Cross-domain semantics** — same function across kingdoms, scaffold hopping, evolutionary stepping stones.

The demo is a live click-through driven by the presenter. Each playground should have 1-3 scripted "wow" moments hittable in 3-7 minutes.

## Non-Goals

- Public deployment (the cloud-hosted version is a follow-up).
- Robustness against arbitrary input — the demo is presenter-driven.
- Mobile / small-screen support.
- A SCOPe fold-space playground — the static `protein-fold` example on the website already covers that story.

## The Four Playgrounds

| | PG1 Words | PG2 ChEMBL | PG3 Spike | PG4 Proteins |
|---|---|---|---|---|
| Status | already built | already built | new | new |
| Dataset | word50k_cache | chembl_full | sars_cov2_spike | endosymbiosis |
| Size | ~80k words | full ChEMBL 36 | Nextstrain Open subsample | 7,258 proteins |
| Encoder for base map | sentence-transformers | Morgan FP + MinHash | AA k-mer + MinHash | ESM-2 (650M) cosine |
| Paste encoder | sentence-transformers (cached) | RDKit Morgan | AA shingles | ESM-2 |
| Paste latency | <100 ms | <100 ms | <100 ms | 1-3 s GPU / 10-30 s CPU |
| Coloring | word category | target class, MW, QED, MoA | Nextstrain clade, date | kingdom/compartment, organism |
| Path tracing | yes | yes | yes | yes |
| Add-points | yes | yes | yes | yes |
| 3D structure viewer | — | — | — | Mol* + AlphaFold |
| Guest gallery | optional | optional | curated variants (Wuhan, Delta, Omicron, JN.1, KP.2) | curated proteins (e.g., known mito ortholog pairs) |

### PG1 — Word embeddings (warm-up, ~3 min)

Already built (`examples/playground_server.py`). Work needed: drop the floating panel and migrate to the shared drawer UI.

**Scripted wow moments:**

- Type `jazz` → semantic neighbors light up.
- Trace `jazz -> engineering` → see the conceptual bridge along the path.
- Paste `serendipity` → cluster of "lucky-discovery" words.

### PG2 — ChEMBL molecules (chemistry hero, ~7 min)

Already built (`scripts/playground_chembl.py`). Work needed: drop the floating panel, migrate to the shared drawer UI, expand the detail panel with an RDKit 2D structure rendering.

**Scripted wow moments:**

- Paste imatinib → kinase-inhibitor cluster.
- Switch coloring to target class / MoA → narrative.
- Trace `imatinib -> vemurafenib` → scaffold-hop along the path.
- Add-points: paste a Roche-relevant compound and watch it land in its scaffold neighborhood.

### PG3 — SARS-CoV-2 spike evolution (new, ~5 min)

**Data already on disk** (`examples/data/sars_cov2_spike/`): aligned spike FASTA, Nextstrain metadata, global ML tree (for offline validation).

**Build:**

- Reuse `examples/sars_cov2_spike_tmap.py` for the base map (AA k-mer Jaccard + MinHash + OGDF tree). Save the model and a metadata parquet to `examples/data/sars_cov2_spike/playground/`.
- Build script: `scripts/build_spike_tmap.py`.

**Coloring layers:**

- Nextstrain clade (categorical, primary)
- Collection date (continuous, secondary — drives the "watch evolution unfold" narrative)
- Country (categorical, optional)

**Paste & path:**

- Paste a new spike protein sequence → shingle → MinHash → kneighbors. Sub-100 ms.
- Path mode: `Wuhan-Hu-1 -> JN.1` (or any two strain names / pasted sequences). Trace the MST.

**Scripted wow moments:**

- Color by clade → narrate the clade tree.
- Switch to date → watch the lineage emergence visually.
- Paste a 2024-2025 spike sequence (JN.1, KP.2) → lands among Omicrons.
- Trace `Wuhan -> JN.1` → see stepping-stone clades.

**No 3D viewer** (per Albert's call: every node is the same fold, so it adds little).

### PG4 — Cross-kingdom protein function (new, ~5-7 min)

**Data already on disk** (`examples/data/endosymbiosis/` and `examples/data/embeddings.npz`):

- `dataset.fasta` — 7,258 sequences.
- `dataset_metadata.tsv` — accession, organism, source, domain, compartment.
- `embeddings.npz` — ESM-2 (650M) embeddings, mean-pooled, accession-aligned.

**Build:**

- Reuse `examples/mixed_proteome_tmap.py` logic to fit TMAP (cosine, USearch HNSW backend). Save model + metadata to `examples/data/endosymbiosis/playground/`.
- Build script: `scripts/build_endosymbiosis_tmap.py`.

**Coloring layers:**

- Domain (categorical, primary): Bacteria-alpha, Bacteria-cyano, Eukaryota-mito, Eukaryota-cyto.
- Organism (categorical, secondary).
- GO term or functional class (categorical, optional — needs lookup).
- Sequence length (continuous, optional).

**Paste & path:**

- Paste a single protein sequence → run ESM-2 mean-pooled embedding on the fly → kneighbors on the saved index. FastAPI lazy-loads ESM-2 once at startup.
- Path mode: pick two proteins (by accession, sequence, or click). Trace MST.

**Guest gallery:**

A pre-computed set of "famous" or Roche-relevant proteins (mitochondrial enzymes with known bacterial orthologs, antibiotics targets, druggable kinases). User can pick from a dropdown if they don't want to paste. Eliminates ESM-2 latency for scripted demo moments.

**3D structure viewer (Mol*):**

- On hover/click of a node, fetch the AlphaFold structure: `https://alphafold.ebi.ac.uk/files/AF-{accession}-F1-model_v4.pdb`.
- Display in an expandable drawer pane.
- For path mode: show structures of path endpoints side by side, optionally aligned with TM-align (deferred to follow-up).

**Scripted wow moments:**

- Paste a bacterial enzyme (acetyltransferase) → top NN is a eukaryotic mitochondrial enzyme. Click both → see AlphaFold structures side by side, visually similar folds.
- Trace path α-proteobacterial enzyme → mitochondrial human enzyme → "TMAP recovers the endosymbiosis story."

## Architecture

Single Next.js site (`tmap-website`) provides the chrome, navigation tabs, and drawer UI. A single FastAPI backend (`tmap-playgrounds-api`) serves all four playgrounds. The TmapViz JS is embedded via iframe in each playground page so we don't reimplement scatter + lasso + tooltip in React.

```
+------------------------------------------------------------------+
| Browser: localhost:3000/playgrounds/{slug}                       |
| +--------------------------------------------------------------+ |
| | Next.js page shell                                           | |
| |   nav tabs (Words | ChEMBL | Spike | Proteins)               | |
| |   +--------------------------------------------+             | |
| |   | iframe -> /api/static/{slug}/index.html    |             | |
| |   |   (the TmapViz HTML)                       |             | |
| |   +--------------------------------------------+             | |
| |   bottom drawer (React)                                      | |
| |     search input | results list | detail card | (Mol* for #4)| |
| +--------------------------------------------------------------+ |
+--------------------------------|---------------------------------+
                                 | fetch /api/playgrounds/{slug}/...
                                 v
+------------------------------------------------------------------+
| FastAPI (localhost:8000)                                         |
|   /playgrounds/{slug}/query?q=...                                |
|   /playgrounds/{slug}/path?a=...&b=...                           |
|   /playgrounds/{slug}/add?seq=...                                |
|   /playgrounds/{slug}/gallery   (optional pre-computed list)     |
|   /static/{slug}/index.html  (serves the TmapViz files)          |
+------------------------------------------------------------------+
```

### Components & boundaries

**FastAPI backend** (lives in this repo at `src/tmap/playgrounds/`):

- One module per playground (`word_pg.py`, `chembl_pg.py`, `spike_pg.py`, `protein_pg.py`).
- Each exposes a class with `query(q, k) -> Results`, `path(a, b) -> Path`, `add(item) -> NewPoint`, plus playground-specific helpers (encode, gallery).
- Each playground lazy-loads its model on first hit; the ESM-2 model loads at startup if PG4 is enabled (slow init worth doing once).
- A FastAPI app wires those modules into HTTP routes.
- Cross-origin: configure CORS to allow `localhost:3000`.
- Serves static TmapViz HTML under `/static/{slug}/`.

**Next.js shell** (`tmap-website`):

- New route group: `app/playgrounds/page.tsx` (index) and `app/playgrounds/[slug]/page.tsx` (per-playground).
- Per-playground page = a server component that defines metadata + a client component (`PlaygroundShell`) that renders nav, iframe, and drawer.
- `PlaygroundShell` accepts a `slug` and a slug-specific drawer component (`WordDrawer`, `ChemblDrawer`, `SpikeDrawer`, `ProteinDrawer`).
- Drawer talks to FastAPI via `fetch('/api/...')`. In dev, Next.js rewrites `/api/playgrounds/*` to `localhost:8000/playgrounds/*`.

**TmapViz HTML** (per playground):

- Generated once by a build script in the `tmap2` repo (`scripts/build_*_tmap.py`).
- Files copied to `src/tmap/playgrounds/static/{slug}/` (served by FastAPI).
- Modified to expose `window._tmap_scatterplot` so the drawer (cross-frame via `postMessage`) can call `.select(indices)` and read viewport for path overlay.
- Path overlay: a transparent canvas inside the iframe receives draw commands from the drawer via `postMessage`.

### Drawer interactions

The drawer is the only React UI component beyond the page shell. It owns:

- `<SearchInput>` — query box, hint text.
- `<ResultsList>` — top-k neighbors with click-to-select.
- `<DetailCard>` — slug-specific (molecule structure / spike clade summary / protein metadata + Mol*).
- `<GalleryPicker>` — optional, for PG4.

State machine: idle → querying → showing-results → showing-path. Esc clears.

### Cross-frame coordination

Iframe → parent: forward viewport changes for path-overlay scaling.
Parent → iframe: `{type: 'select', indices}`, `{type: 'draw-path', nodes}`, `{type: 'clear'}`.

A small JS shim (`scripts/playground-bridge.js`) is injected into each TmapViz HTML to handle `postMessage`. This replaces the current direct-injection panel approach in `playground_chembl.py` and `playground_server.py`.

## Data flow (paste a SMILES in PG2)

1. User types SMILES, presses Enter in drawer input.
2. Drawer `fetch('/api/playgrounds/chembl/query?q=...')`.
3. Next.js rewrites to FastAPI `/playgrounds/chembl/query`.
4. ChemblPlayground.query encodes SMILES (RDKit Morgan) → `model.kneighbors(fps)`.
5. Returns `{results: [{idx, chembl_id, smiles, distance, scaffold, mw, qed, ...}, ...]}`.
6. Drawer renders results list, sends `{type: 'select', indices}` to iframe.
7. TmapViz highlights those points.

## Error handling

- Invalid input (unparseable SMILES, invalid sequence) → 400 with a human message; drawer shows the message inline.
- Missing model file → 503 with "playground not built yet" message.
- ESM-2 not loaded yet (PG4) → 503 with retry; drawer shows "loading model".
- During the live demo: errors are fine to surface; just keep messages short and not scary.

## Testing

- Unit tests per playground module: `query()`, `path()`, `add()` on a tiny synthetic model.
- Integration test: spin up FastAPI in-process, exercise each playground's full path-and-query loop.
- Manual: each "scripted wow moment" listed above gets a smoke test before the demo.

## Build & run

```bash
# in tmap2/
python scripts/build_word_tmap.py        # existing
python scripts/build_chembl_tmap.py      # existing
python scripts/build_spike_tmap.py       # new
python scripts/build_endosymbiosis_tmap.py  # new (uses cached ESM-2 embeddings)

# in tmap2/ — start backend
python -m tmap.playgrounds.serve --port 8000

# in tmap-website/
npm run dev   # localhost:3000

# Open localhost:3000/playgrounds
```

## Out of scope for this spec

- Public deployment (Vercel + remote FastAPI host) — separate spec when needed.
- TM-align of two AlphaFold structures in PG4 — nice-to-have, follow-up.
- A SwissProt enzyme-class build (dataset β) — follow-up if PG4 lands and we want a broader story.
- Hover-card AlphaFold structures (only show on click for now).
- Animation of clade emergence by date in PG3 — nice-to-have.

## Open questions

None blocking. The two intentionally deferred:

- Where the FastAPI process runs in production. For the demo it's local; for the post-demo leave-behind, options are fly.io / Railway / a Roche-internal box.
- Whether the AlphaFold structure URLs work reliably from the demo network (Roche-internal). If not, pre-download a small set of "guest gallery" structures and serve them from FastAPI.
