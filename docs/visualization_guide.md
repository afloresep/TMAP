# Visualization Guide

`TmapViz` is the public visualization object in TMAP.

The most useful mental model is simple:

- `to_widget(...)` for notebooks
- `write_html(...)` for one shareable file
- `serve(...)` for larger local maps

This guide uses the molecule example from `examples/cluster_65053.csv`.

## Build A Base Visualization

```python
import pandas as pd
from tmap import TMAP
from tmap.utils import fingerprints_from_smiles, molecular_properties, murcko_scaffolds

df = pd.read_csv("../examples/cluster_65053.csv", nrows=3000)
smiles = df["smiles"].tolist()

fps = fingerprints_from_smiles(smiles, fp_type="morgan", radius=2, n_bits=2048)
props = molecular_properties(smiles, properties=["mw", "logp", "n_rings", "qed"])
scaffolds = murcko_scaffolds(smiles)

model = TMAP(metric="jaccard", n_neighbors=20, seed=42).fit(fps)

viz = model.to_tmapviz()
viz.title = "Cluster 65053"
viz.add_smiles(smiles)
viz.add_color_layout("MW", props["mw"].tolist(), color="viridis")
viz.add_color_layout("LogP", props["logp"].tolist(), color="plasma")
viz.add_color_layout("Ring Count", props["n_rings"].tolist(), categorical=True, color="tab10")
viz.add_color_layout("QED", props["qed"].tolist(), color="magma")
viz.add_label("Murcko Scaffold", scaffolds.tolist())
```

## Notebook Workflow

Use notebook widgets when you want fast iteration in Jupyter.

```python
widget = viz.to_widget(width=1000, height=650, controls=True)
widget.show()
```

Good for:

- switching color layouts
- filtering categories
- inspecting tooltips
- staying inside a notebook

Current limitation:

- notebook widgets do not draw tree edges yet

## HTML Workflow

Use HTML when you want one file you can send to someone else.

```python
path = viz.write_html("cluster_65053.html")
print(path)
```

Good for:

- sharing a result
- opening the map in any browser
- keeping edge rendering

## Local Server Workflow

Use `serve()` when the map gets large and one HTML file becomes awkward.

```python
viz.serve(port=8050)
```

This writes static assets and starts a small local HTTP server at `http://127.0.0.1:8050`.

Good for:

- larger maps
- faster reloads during local work
- lazy loading of data files

## Adding Data Columns

### Numeric Properties

```python
viz.add_color_layout("MW", props["mw"].tolist(), color="viridis")
viz.add_color_layout("LogP", props["logp"].tolist(), color="plasma")
```

Use this for values you want to color continuously.

### Categorical Properties

```python
viz.add_color_layout(
    "Ring Count",
    props["n_rings"].tolist(),
    categorical=True,
    color="tab10",
)
```

Use this for discrete labels or small integer sets.

### Filter-Only Columns

```python
viz.add_filter("MW", props["mw"].tolist())
viz.add_filter("Ring Count", props["n_rings"].tolist(), categorical=True)
```

Use this when people should be able to filter on a column, but it does not need
to be one of the colors they can switch between. Numbers get a slider, groups
get one clickable bar each. Choosing colors is the slow part of
`add_color_layout`, and this skips it, so extra filter columns cost very little.

Color layouts can already be filtered on, so `add_filter` columns go into the
panel on top of them, not instead of them. To say exactly what the panel shows,
set `filterable`:

```python
viz.filterable = ["MW", "LogP"]
```

Pass `add_as_label=False` to keep a filter column out of the tooltip.

### Tooltip Labels

```python
viz.add_label("Murcko Scaffold", scaffolds.tolist())
```

Use labels for text that should appear in the tooltip but should not control colors.

### Molecular Structures

```python
viz.add_smiles(smiles)
```

This enables SMILES structure rendering in tooltips.

## Common Patterns

### Quick Notebook First, HTML Second

```python
widget = viz.to_widget(width=1000, height=650, controls=True)
widget.show()

viz.write_html("cluster_65053.html")
```

### Save Model And Visualization

```python
model.save("cluster_65053.pkl")
viz.write_html("cluster_65053.html")
```

### Start Small

For tutorials, start with `nrows=1000` or `nrows=3000`. Once the workflow is working, use more rows.
