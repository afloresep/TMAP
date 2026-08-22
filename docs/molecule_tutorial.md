# Molecule Tutorial

This is the clearest end-to-end TMAP workflow for chemistry:

1. Load SMILES from `examples/cluster_65053.csv`
2. Compute Morgan fingerprints
3. Compute molecular properties and scaffolds
4. Fit a Jaccard TMAP
5. Explore it in Jupyter, save HTML, or serve it locally

## Requirements

```bash
python -m pip install .
python -m pip install rdkit jupyter-scatter
```

`rdkit` is needed for fingerprints and molecular properties. `jupyter-scatter` is only needed for notebook widgets.

## 1. Load A Molecule Table

```python
import pandas as pd

df = pd.read_csv("../examples/cluster_65053.csv", nrows=3000)
smiles = df["smiles"].tolist()

print(f"{len(smiles):,} molecules")
```

Use a small subset first. Once the workflow is working, remove `nrows=3000`.

## 2. Compute Fingerprints, Properties, And Scaffolds

```python
from tmap.utils import fingerprints_from_smiles, molecular_properties, murcko_scaffolds

fps = fingerprints_from_smiles(smiles, fp_type="morgan", radius=2, n_bits=2048)
props = molecular_properties(smiles, properties=["mw", "logp", "n_rings", "qed"])
scaffolds = murcko_scaffolds(smiles)
```

`fingerprints_from_smiles` returns a binary matrix for `metric="jaccard"`.

## 3. Fit TMAP

```python
from tmap import TMAP

model = TMAP(
    metric="jaccard",
    n_neighbors=20,
    n_permutations=512,
    kc=50,
    seed=42,
).fit(fps)

print(model.embedding_.shape)
print(model.tree_.edges.shape)
```

## 4. Build A Visualization

```python
viz = model.to_tmapviz()
viz.title = "Cluster 65053"
viz.add_smiles(smiles)
viz.add_color_layout("MW", props["mw"].tolist(), color="viridis")
viz.add_color_layout("LogP", props["logp"].tolist(), color="plasma")
viz.add_color_layout("Ring Count", props["n_rings"].tolist(), categorical=True, color="tab10")
viz.add_color_layout("QED", props["qed"].tolist(), color="magma")
viz.add_label("Murcko Scaffold", scaffolds.tolist())
```

This gives you:

- molecule structures in tooltips
- multiple property views
- scaffold labels in the tooltip

## 5. Explore In A Notebook

```python
widget = viz.to_widget(width=1000, height=650, controls=True)
widget.show()
```

Use this when you want to switch color layers, filter categories, and inspect points interactively in Jupyter.

Note: notebook widgets do not draw tree edges yet. HTML export does.

## 6. Save A Shareable HTML File

```python
path = viz.write_html("cluster_65053.html")
print(path)
```

This writes one self-contained HTML file that you can open in a browser.

## 7. Serve A Larger Map Locally

```python
viz.serve(port=8050)
```

Use `serve()` when the map is large enough that one HTML file becomes inconvenient. `serve()` writes static assets and starts a small local HTTP server.

## 8. Save The Model Too

```python
model.save("cluster_65053_model.pkl")
loaded = TMAP.load("cluster_65053_model.pkl")
```

This is useful when you want to keep the fitted tree and coordinates for later analysis.

## 9. Script Version

If you want the same workflow as a script, run:

```bash
python examples/chemistry/molecules_tmap.py --nrows 3000 --output examples/cluster_65053.html
```

Add `--serve` if you want it to start a local viewer.
