# Examples

Runnable scripts grouped by data type. Each one is self-contained: run it from
the repository root and it downloads or caches whatever it needs under
`examples/data/`, then writes its output to `examples/`.

```bash
python examples/chemistry/molecules_tmap.py --nrows 3000
```

## Chemistry

| Example | Description |
|---------|-------------|
| [`chemistry/molecules_tmap.py`](chemistry/molecules_tmap.py) | The simplest chemistry example: SMILES → fingerprints → interactive map, with molecular properties and Murcko scaffolds |
| [`chemistry/smiles_tmap.py`](chemistry/smiles_tmap.py) | The same map built step by step: `MinHash` → `LSHForest` → OGDF layout → `TmapViz` |
| [`chemistry/molecules_tmap_legacy.py`](chemistry/molecules_tmap_legacy.py) | `molecules_tmap.py` with the adaptive layout and untangle post-pass switched off, for comparison |

## Images

| Example | Description |
|---------|-------------|
| [`images/pet_breed_audit.py`](images/pet_breed_audit.py) | Audit an image classifier: ResNet-50 embeddings of Oxford-IIIT Pets, a linear probe, and tree analysis of where it fails |
| [`images/mnist_cosine_tmap.py`](images/mnist_cosine_tmap.py) | MNIST digits with the cosine metric, including paths between similar digits |
| [`images/emnist_characters_tmap.py`](images/emnist_characters_tmap.py) | Handwritten digits and letters together, showing where OCR confuses the two |
| [`images/flowers_tmap.py`](images/flowers_tmap.py) | Oxford Flowers 102: morphological gradients across species |
| [`images/cub200_birds_tmap.py`](images/cub200_birds_tmap.py) | CUB-200 birds: morphological paths across 200 species |
| [`images/wikiart_tmap.py`](images/wikiart_tmap.py) | WikiArt paintings coloured by artistic style |

## Proteins

| Example | Description |
|---------|-------------|
| [`proteins/esm_atlas_tmap.py`](proteins/esm_atlas_tmap.py) | Two views of ESMC's protein space: raw embeddings and SAE features, with predicted structures in the pinned cards |

## Text

| Example | Description |
|---------|-------------|
| [`text/word_embeddings_tmap.py`](text/word_embeddings_tmap.py) | ~800 common English words embedded with a sentence-transformer |
| [`text/word_embeddings_50k.py`](text/word_embeddings_50k.py) | The same idea at 50,000 WordNet nouns |

## Layout internals

| Example | Description |
|---------|-------------|
| [`layout/untangle_demo.py`](layout/untangle_demo.py) | Before-and-after figure showing what the crossing-reduction (untangle) post-pass does to a layout |

## Data

- `cluster_65053.csv` — ~6k SMILES from an Enamine chemical cluster. Used by the
  chemistry examples, the layout demo, and several docs and notebooks.
- `data/` — datasets and cached embeddings downloaded by the examples on first
  run. Not tracked in git, and can be deleted to reclaim the space.

Outputs (`.html`, `.png`) are written to `examples/` and are not tracked either.

## Notebooks

Step-by-step walkthroughs live in [`notebooks/`](../notebooks) rather than here.
