"""Parallel chemistry utilities for fingerprint and property computation.
Requires ``rdkit`` (install via ``pip install rdkit``).
"""

from __future__ import annotations

import importlib.util
import logging
import multiprocessing
import os
import sys
from typing import Any

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# default params (set via worker initializers for multiprocessing)
_FP_RADIUS: int = 2
_FP_NBITS: int = 1024
_DRFP_PARAMS: dict[str, Any] = {}
_MHFP_PARAMS: dict[str, Any] = {}
_PROP_NAMES: list[str] = []

AVAILABLE_PROPERTIES: list[str] = [
    "mw",
    "logp",
    "n_rings",
    "hba",
    "hbd",
    "tpsa",
    "n_rotatable_bonds",
    "n_heavy_atoms",
    "fraction_csp3",
    "n_aromatic_rings",
    "n_heteroatoms",
    "formal_charge",
    "qed",
]

AVAILABLE_REACTION_PROPERTIES: list[str] = [
    "delta_mw",
    "delta_heavy_atoms",
    "delta_rings",
    "delta_aromatic_rings",
    "atom_economy",
    "delta_logp",
]


def _init_fp_worker(radius: int, n_bits: int) -> None:
    global _FP_RADIUS, _FP_NBITS
    _FP_RADIUS = radius
    _FP_NBITS = n_bits


def _init_drfp_worker(n_folded_length: int, radius: int, min_radius: int, rings: bool) -> None:
    global _DRFP_PARAMS
    _DRFP_PARAMS = {
        "n_folded_length": n_folded_length,
        "radius": radius,
        "min_radius": min_radius,
        "rings": rings,
    }


def _init_mhfp_worker(
    length: int,
    radius: int,
    rings: bool,
    kekulize: bool,
    sanitize: bool,
) -> None:
    global _MHFP_PARAMS
    _MHFP_PARAMS = {
        "length": length,
        "radius": radius,
        "rings": rings,
        "kekulize": kekulize,
        "sanitize": sanitize,
    }


def _init_props_worker(prop_names: list[str]) -> None:
    global _PROP_NAMES
    _PROP_NAMES = prop_names


# Fingerprint batch functions (called inside Pool workers)


def _morgan_fp_batch(smiles_batch: list[str]) -> tuple[NDArray[np.uint8], NDArray[np.bool_]]:
    """Process a batch of SMILES, return the fingerprints plus one True/False flag per input."""
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=_FP_RADIUS, fpSize=_FP_NBITS)
    fps = []
    valid = np.zeros(len(smiles_batch), dtype=bool)
    for i, smi in enumerate(smiles_batch):
        if not smi:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fps.append(gen.GetFingerprintAsNumPy(mol).astype(np.uint8))
            valid[i] = True
    if fps:
        return np.stack(fps), valid
    return np.empty((0, _FP_NBITS), dtype=np.uint8), valid


def _mqn_fp_batch(smiles_batch: list[str]) -> tuple[NDArray[np.int16], NDArray[np.bool_]]:
    """Process a batch of SMILES, return MQN fingerprints (42 counts) plus one flag per input."""
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors

    fps = []
    valid = np.zeros(len(smiles_batch), dtype=bool)
    for i, smi in enumerate(smiles_batch):
        if not smi:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fps.append(np.array(rdMolDescriptors.MQNs_(mol), dtype=np.int16))
            valid[i] = True
    if fps:
        return np.stack(fps), valid
    return np.empty((0, 42), dtype=np.int16), valid


def _mxfp_fp_batch(smiles_batch: list[str]) -> tuple[NDArray[np.int64], NDArray[np.bool_]]:
    """Process a batch of SMILES, return MXFP fingerprints (217-dim) plus one flag per input."""
    from mxfp.mxfp import MXFPCalculator

    calc = MXFPCalculator()
    fps = []
    valid = np.zeros(len(smiles_batch), dtype=bool)
    for i, smi in enumerate(smiles_batch):
        if not smi:
            continue
        try:
            fp = calc.mxfp_from_smiles(smi)
            if fp is not None:
                fps.append(fp)
                valid[i] = True
        except Exception:
            pass
    if fps:
        return np.stack(fps), valid
    return np.empty((0, 217), dtype=np.int64), valid


def _drfp_fp_batch(smiles_batch: list[str]) -> tuple[NDArray[np.uint8], NDArray[np.bool_]]:
    """Process a batch of reaction SMILES, return DRFP fingerprints plus one flag per input."""
    from drfp import DrfpEncoder

    valid = np.array([bool(smi) for smi in smiles_batch], dtype=bool)
    clean = [smi for smi in smiles_batch if smi]
    if not clean:
        n_bits = _DRFP_PARAMS.get("n_folded_length", 2048)
        return np.empty((0, n_bits), dtype=np.uint8), valid
    fps_list = DrfpEncoder.encode(clean, **_DRFP_PARAMS)
    return np.array(fps_list, dtype=np.uint8), valid


def _mhfp_fp_batch(smiles_batch: list[str]) -> tuple[NDArray[np.uint8], NDArray[np.bool_]]:
    """Process a batch of SMILES, return SECFP fingerprints plus one flag per input.

    SECFP is MHFP's folded binary variant: a uint8 {0,1} vector suitable
    for the Jaccard-on-binary path used by the other fingerprint types.
    """
    from mhfp.encoder import MHFPEncoder
    from rdkit import Chem

    encoder = MHFPEncoder()
    length = _MHFP_PARAMS.get("length", 2048)
    fps: list[NDArray[np.uint8]] = []
    valid = np.zeros(len(smiles_batch), dtype=bool)
    for i, smi in enumerate(smiles_batch):
        if not smi:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = encoder.secfp_from_smiles(
            smi,
            length=length,
            radius=_MHFP_PARAMS.get("radius", 3),
            rings=_MHFP_PARAMS.get("rings", True),
            kekulize=_MHFP_PARAMS.get("kekulize", True),
            sanitize=_MHFP_PARAMS.get("sanitize", False),
        )
        if fp is not None:
            fps.append(np.asarray(fp, dtype=np.uint8))
            valid[i] = True
    if fps:
        return np.stack(fps), valid
    return np.empty((0, length), dtype=np.uint8), valid


# Property batch functions (called inside Pool workers)


def _mol_props_batch(smiles_batch: list[str]) -> NDArray[np.float64]:
    """Process a batch, return (len(batch), n_props) array. Invalid rows are NaN."""
    from rdkit import Chem
    from rdkit.Chem import QED, Descriptors, rdMolDescriptors

    computers = {
        "mw": lambda mol: Descriptors.ExactMolWt(mol),
        "logp": lambda mol: Descriptors.MolLogP(mol),
        "n_rings": lambda mol: mol.GetNumBonds() - mol.GetNumAtoms() + len(Chem.GetMolFrags(mol)),
        "hba": lambda mol: rdMolDescriptors.CalcNumHBA(mol),
        "hbd": lambda mol: rdMolDescriptors.CalcNumHBD(mol),
        "tpsa": lambda mol: rdMolDescriptors.CalcTPSA(mol),
        "n_rotatable_bonds": lambda mol: rdMolDescriptors.CalcNumRotatableBonds(mol),
        "n_heavy_atoms": lambda mol: float(mol.GetNumHeavyAtoms()),
        "fraction_csp3": lambda mol: rdMolDescriptors.CalcFractionCSP3(mol),
        "n_aromatic_rings": lambda mol: rdMolDescriptors.CalcNumAromaticRings(mol),
        "n_heteroatoms": lambda mol: float(rdMolDescriptors.CalcNumHeteroatoms(mol)),
        "formal_charge": lambda mol: float(Chem.GetFormalCharge(mol)),
        "qed": lambda mol: QED.qed(mol),
    }

    props = _PROP_NAMES
    out = np.full((len(smiles_batch), len(props)), np.nan, dtype=np.float64)
    for i, smi in enumerate(smiles_batch):
        if not smi:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            for j, name in enumerate(props):
                out[i, j] = computers[name](mol)
    return out


def _rxn_props_batch(rxn_smiles_batch: list[str]) -> NDArray[np.float64]:
    """Process a batch of reaction SMILES, return (len(batch), n_props) array."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors

    props = _PROP_NAMES
    out = np.full((len(rxn_smiles_batch), len(props)), np.nan, dtype=np.float64)

    for i, rxn in enumerate(rxn_smiles_batch):
        if not rxn:
            continue
        # Split reaction SMILES: reactants>reagents>products or reactants>>products
        parts = rxn.split(">")
        if len(parts) < 2:
            continue
        reactant_str, product_str = parts[0].strip(), parts[-1].strip()
        if not reactant_str or not product_str:
            continue

        r_mols = [Chem.MolFromSmiles(s) for s in reactant_str.split(".") if s]
        p_mols = [Chem.MolFromSmiles(s) for s in product_str.split(".") if s]
        r_mols = [m for m in r_mols if m is not None]
        p_mols = [m for m in p_mols if m is not None]
        if not r_mols or not p_mols:
            continue

        def _sum(mols: list, fn) -> float:  # type: ignore[type-arg]
            return sum(fn(m) for m in mols)

        def _nrings(m):
            return m.GetNumBonds() - m.GetNumAtoms() + len(Chem.GetMolFrags(m))

        r_mw = _sum(r_mols, Descriptors.ExactMolWt)
        p_mw = _sum(p_mols, Descriptors.ExactMolWt)

        computers = {
            "delta_mw": p_mw - r_mw,
            "delta_heavy_atoms": (
                _sum(p_mols, lambda m: m.GetNumHeavyAtoms())
                - _sum(r_mols, lambda m: m.GetNumHeavyAtoms())
            ),
            "delta_rings": (_sum(p_mols, _nrings) - _sum(r_mols, _nrings)),
            "delta_aromatic_rings": (
                _sum(p_mols, rdMolDescriptors.CalcNumAromaticRings)
                - _sum(r_mols, rdMolDescriptors.CalcNumAromaticRings)
            ),
            "atom_economy": (p_mw / r_mw * 100.0) if r_mw > 0 else np.nan,
            "delta_logp": (_sum(p_mols, Descriptors.MolLogP) - _sum(r_mols, Descriptors.MolLogP)),
        }

        for j, name in enumerate(props):
            out[i, j] = computers[name]

    return out


# Utilities


def _get_mp_context() -> multiprocessing.context.BaseContext:  # type:ignore
    if sys.platform == "darwin":
        return multiprocessing.get_context("spawn")
    return multiprocessing.get_context("forkserver")


def _default_n_workers() -> int:
    return min(os.cpu_count() or 1, 12)


def _warn_dropped(valid: NDArray[np.bool_]) -> None:
    """Log a warning when some SMILES could not be read."""
    n_dropped = int((~valid).sum())
    if n_dropped:
        logger.warning(
            "%d/%d SMILES could not be parsed and were skipped. "
            "Pass return_valid=True to find out which ones.",
            n_dropped,
            len(valid),
        )


def _split_into_chunks(lst: list[Any], n_chunks: int) -> list[list[Any]]:
    """Split list into roughly equal chunks."""
    k, m = divmod(len(lst), n_chunks)
    return [lst[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n_chunks)]


# Fingerprint helpers with built-in parallelization (no Pool wrapper)


def _map4_fingerprints(
    smiles: list[str], n_workers: int, **kwargs: Any
) -> tuple[NDArray[np.uint8], NDArray[np.bool_]]:
    """Compute MAP4 fingerprints using the map4 package.

    MAP4 (MinHashed Atom-Pair fingerprint of radius 2) encodes molecules
    by extracting all atom-pair shingles at multiple radii, MinHashing
    them, and folding the signature into a fixed-length binary vector.
    That folded output feeds TMAP's binary-Jaccard path directly.
    """
    try:
        from map4 import MAP4
    except ImportError:
        raise ImportError(
            "map4 is required for fp_type='map4'. Install with: pip install map4"
        ) from None
    from rdkit import Chem

    dimensions = kwargs.get("dimensions", 1024)
    radius = kwargs.get("radius", 2)
    include_duplicated_shingles = kwargs.get("include_duplicated_shingles", False)
    seed = kwargs.get("seed", 75434278)

    calc = MAP4(
        dimensions=dimensions,
        radius=radius,
        include_duplicated_shingles=include_duplicated_shingles,
        seed=seed,
    )

    mols = []
    valid = np.zeros(len(smiles), dtype=bool)
    for i, smi in enumerate(smiles):
        if not smi:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            mols.append(mol)
            valid[i] = True

    if not mols:
        return np.empty((0, dimensions), dtype=np.uint8), valid

    fps = calc.calculate_many(mols, number_of_threads=n_workers)
    return np.asarray(fps, dtype=np.uint8), valid


# fingerprints_from_smiles


def fingerprints_from_smiles(
    smiles: list[str],
    fp_type: str = "morgan",
    n_workers: int | None = None,
    return_valid: bool = False,
    **kwargs: Any,
) -> NDArray | tuple[NDArray, NDArray[np.bool_]]:
    """Compute molecular fingerprints in parallel.

    Parameters
    ----------
    smiles : list[str]
        Input SMILES strings. For ``fp_type='drfp'``, these should be
        reaction SMILES (``reactants>>products``).
    fp_type : str, default ``'morgan'``
        Fingerprint type:

        - ``'morgan'`` -- Morgan circular fingerprints (``uint8``).
          kwargs: ``radius=2``, ``n_bits=2048``.
        - ``'mqn'`` -- Molecular Quantum Numbers, 42 integer counts
          (``int16``). No additional kwargs.
        - ``'mxfp'`` -- Macromolecule Extended Fingerprint, 217-dim
          pharmacophoric distance counts (``int64``). Requires the
          ``mxfp`` package. No additional kwargs.
        - ``'map4'`` -- MinHashed Atom-Pair fingerprint, folded to a
          binary vector (``uint8``, values 0/1). Requires the ``map4``
          package. kwargs: ``dimensions=1024``, ``radius=2``,
          ``include_duplicated_shingles=False``, ``seed=75434278``.
        - ``'mhfp'`` -- SECFP, the folded binary variant of MHFP
          (``uint8``, values 0/1). Requires the ``mhfp`` package.
          kwargs: ``length=2048``, ``radius=3``, ``rings=True``,
          ``kekulize=True``, ``sanitize=False``.
        - ``'drfp'`` -- Differential Reaction Fingerprint for chemical
          reactions (``uint8``). Requires the ``drfp`` package.
          kwargs: ``n_folded_length=2048``, ``radius=3``,
          ``min_radius=0``, ``rings=True``.
    n_workers : int or None
        Number of parallel workers. Defaults to ``min(cpu_count, 12)``.
    return_valid : bool, default False
        If True, also return one True/False flag per input SMILES saying
        whether it produced a fingerprint. Use it to line the fingerprints
        up with :func:`molecular_properties` and :func:`murcko_scaffolds`,
        which always return one entry per input.
    **kwargs
        Passed to the fingerprint generator.

    Returns
    -------
    fps : ndarray of shape ``(n_valid, n_bits)``
        Fingerprint matrix. Invalid SMILES are skipped (a warning is
        logged with the count). Dtype depends on ``fp_type``: ``uint8``
        for Morgan/DRFP/MAP4/MHFP, ``int16`` for MQN, ``int64`` for
        MXFP.
    valid : ndarray of shape ``(len(smiles),)``, dtype ``bool``
        Returned only when ``return_valid=True``. ``valid[i]`` is True
        when ``smiles[i]`` could be read. Indexing any per-input array
        with it drops the same entries that ``fps`` is missing, so both
        end up the same length.

    Examples
    --------
    >>> from tmap.utils import fingerprints_from_smiles
    >>> fps = fingerprints_from_smiles(["CCO", "c1ccccc1"], fp_type="morgan")
    >>> fps.shape
    (2, 2048)

    >>> fps = fingerprints_from_smiles(["CCO"], fp_type="mqn")
    >>> fps.shape
    (1, 42)
    >>> fps.dtype
    dtype('int16')

    Unreadable SMILES get no row, so ask for the flags when other
    per-molecule arrays have to match up:

    >>> smiles = ["CCO", "not_a_molecule", "c1ccccc1"]
    >>> fps, valid = fingerprints_from_smiles(smiles, return_valid=True)
    >>> valid
    array([ True, False,  True])
    >>> fps.shape
    (2, 2048)
    """
    n = len(smiles)
    if n == 0:
        # Nothing to do, but still hand back an array of the width and type
        # this fp_type would normally produce.
        empty_shapes = {
            "morgan": (kwargs.get("n_bits", 2048), np.uint8),
            "mqn": (42, np.int16),
            "mxfp": (217, np.int64),
            "drfp": (kwargs.get("n_folded_length", 2048), np.uint8),
            "map4": (kwargs.get("dimensions", 1024), np.uint8),
            "mhfp": (kwargs.get("length", 2048), np.uint8),
        }
        ncols, dt = empty_shapes.get(fp_type, (0, np.uint8))
        empty = np.empty((0, ncols), dtype=dt)
        return (empty, np.zeros(0, dtype=bool)) if return_valid else empty

    if n_workers is None:
        n_workers = _default_n_workers()
    n_workers = min(n_workers, n)

    # MAP4 has built-in parallelization via calculate_many()
    if fp_type == "map4":
        fps, valid = _map4_fingerprints(smiles, n_workers=n_workers, **kwargs)
        _warn_dropped(valid)
        return (fps, valid) if return_valid else fps

    # All other fingerprints use our Pool wrapper
    ctx = _get_mp_context()
    chunks = _split_into_chunks(smiles, n_workers)

    if fp_type == "morgan":
        radius = kwargs.get("radius", 2)
        n_bits = kwargs.get("n_bits", 2048)
        with ctx.Pool(n_workers, initializer=_init_fp_worker, initargs=(radius, n_bits)) as pool:
            batch_results = pool.map(_morgan_fp_batch, chunks)
    elif fp_type == "mqn":
        with ctx.Pool(n_workers) as pool:
            batch_results = pool.map(_mqn_fp_batch, chunks)
    elif fp_type == "mxfp":
        if importlib.util.find_spec("mxfp") is None:
            raise ImportError("mxfp is required for fp_type='mxfp'. Install with: pip install mxfp")
        with ctx.Pool(n_workers) as pool:
            batch_results = pool.map(_mxfp_fp_batch, chunks)
    elif fp_type == "drfp":
        if importlib.util.find_spec("drfp") is None:
            raise ImportError("drfp is required for fp_type='drfp'. Install with: pip install drfp")
        n_folded_length = kwargs.get("n_folded_length", 2048)
        drfp_radius = kwargs.get("radius", 3)
        min_radius = kwargs.get("min_radius", 0)
        rings = kwargs.get("rings", True)
        with ctx.Pool(
            n_workers,
            initializer=_init_drfp_worker,
            initargs=(n_folded_length, drfp_radius, min_radius, rings),
        ) as pool:
            batch_results = pool.map(_drfp_fp_batch, chunks)
    elif fp_type == "mhfp":
        if importlib.util.find_spec("mhfp") is None:
            raise ImportError("mhfp is required for fp_type='mhfp'. Install with: pip install mhfp")
        mhfp_args = (
            kwargs.get("length", 2048),
            kwargs.get("radius", 3),
            kwargs.get("rings", True),
            kwargs.get("kekulize", True),
            kwargs.get("sanitize", False),
        )
        with ctx.Pool(
            n_workers,
            initializer=_init_mhfp_worker,
            initargs=mhfp_args,
        ) as pool:
            batch_results = pool.map(_mhfp_fp_batch, chunks)
    else:
        raise ValueError(
            f"Unsupported fp_type={fp_type!r}. "
            "Use 'morgan', 'mqn', 'mxfp', 'map4', 'mhfp', or 'drfp'."
        )

    # Each worker got one slice of the input list, and the slices are in order,
    # so joining the results back together restores the original order. Workers
    # that parsed nothing return an empty block of the right width, which joins
    # in without changing anything.
    fps = np.concatenate([arr for arr, _ in batch_results])
    valid = np.concatenate([mask for _, mask in batch_results])

    _warn_dropped(valid)
    return (fps, valid) if return_valid else fps


# Scaffolds


def _scaffolds_batch(smiles_batch: list[str]) -> list[str]:
    """Return Murcko scaffold SMILES for a batch. Invalid SMILES -> empty string."""
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    out: list[str] = []
    for smi in smiles_batch:
        if not smi:
            out.append("")
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            try:
                out.append(MurckoScaffold.MurckoScaffoldSmiles(mol=mol))
            except Exception:
                out.append("")
        else:
            out.append("")
    return out


def murcko_scaffolds(
    smiles: list[str],
    n_workers: int | None = None,
) -> NDArray[np.object_]:
    """Compute Murcko scaffolds in parallel.

    The Murcko scaffold is the core ring system plus linkers of a
    molecule -- useful for grouping/coloring by chemical series.

    Parameters
    ----------
    smiles : list[str]
        Input SMILES strings.
    n_workers : int or None
        Number of parallel workers. Defaults to ``min(cpu_count, 12)``.

    Returns
    -------
    ndarray of shape ``(len(smiles),)``, dtype ``object``
        Scaffold SMILES for each input. Invalid SMILES produce ``""``.
    """
    n = len(smiles)
    if n == 0:
        return np.empty(0, dtype=object)

    if n_workers is None:
        n_workers = _default_n_workers()
    n_workers = min(n_workers, n)

    ctx = _get_mp_context()
    chunks = _split_into_chunks(smiles, n_workers)

    with ctx.Pool(n_workers) as pool:
        batch_results = pool.map(_scaffolds_batch, chunks)

    flat: list[str] = []
    for batch in batch_results:
        flat.extend(batch)

    result = np.empty(n, dtype=object)
    result[:] = flat
    n_valid = sum(1 for s in flat if s)
    print(f"  [Scaffolds] {n_valid:,}/{n:,} valid")
    return result


# Molecular properties


def molecular_properties(
    smiles: list[str],
    properties: list[str] | None = None,
    n_workers: int | None = None,
) -> dict[str, NDArray[np.float64]]:
    """Compute molecular properties in parallel.

    Parameters
    ----------
    smiles : list[str]
        Input SMILES strings.
    properties : list[str] or None
        Which properties to compute. Defaults to all in
        ``AVAILABLE_PROPERTIES``. Options: ``'mw'``, ``'logp'``,
        ``'n_rings'``, ``'hba'``, ``'hbd'``, ``'tpsa'``,
        ``'n_rotatable_bonds'``, ``'n_heavy_atoms'``,
        ``'fraction_csp3'``, ``'n_aromatic_rings'``,
        ``'n_heteroatoms'``, ``'formal_charge'``.
    n_workers : int or None
        Number of parallel workers. Defaults to ``min(cpu_count, 12)``.

    Returns
    -------
    dict[str, ndarray]
        Each key is a property name, each value is an ndarray of
        length ``len(smiles)``. Invalid SMILES produce ``NaN``.
    """
    if properties is None:
        properties = list(AVAILABLE_PROPERTIES)
    else:
        bad = [p for p in properties if p not in AVAILABLE_PROPERTIES]
        if bad:
            raise ValueError(f"Unknown properties: {bad}. Available: {AVAILABLE_PROPERTIES}")

    n = len(smiles)
    if n == 0:
        return {p: np.empty(0, dtype=np.float64) for p in properties}

    if n_workers is None:
        n_workers = _default_n_workers()
    n_workers = min(n_workers, n)

    ctx = _get_mp_context()
    chunks = _split_into_chunks(smiles, n_workers)

    with ctx.Pool(n_workers, initializer=_init_props_worker, initargs=(properties,)) as pool:
        batch_results = pool.map(_mol_props_batch, chunks)

    props = np.concatenate(batch_results)  # (n, len(properties))
    print(f"  [Props] {n:,} done, {np.isnan(props[:, 0]).sum():,} invalid")

    return {name: props[:, j] for j, name in enumerate(properties)}


# Reaction properties


def reaction_properties(
    rxn_smiles: list[str],
    properties: list[str] | None = None,
    n_workers: int | None = None,
) -> dict[str, NDArray[np.float64]]:
    """Compute reaction-level descriptors in parallel.

    Each reaction SMILES is split on ``>`` into reactant and product
    sides. Properties are computed as deltas (product - reactant) or
    ratios across all molecules on each side.

    Parameters
    ----------
    rxn_smiles : list[str]
        Reaction SMILES in ``reactants>>products`` or
        ``reactants>reagents>products`` format.
    properties : list[str] or None
        Which properties to compute. Defaults to all in
        ``AVAILABLE_REACTION_PROPERTIES``:

        - ``'delta_mw'`` -- change in total molecular weight.
        - ``'delta_heavy_atoms'`` -- change in heavy atom count.
        - ``'delta_rings'`` -- ring-forming (+) or ring-breaking (-).
        - ``'delta_aromatic_rings'`` -- aromatization change.
        - ``'atom_economy'`` -- product MW / reactant MW * 100.
        - ``'delta_logp'`` -- change in lipophilicity.
    n_workers : int or None
        Number of parallel workers. Defaults to ``min(cpu_count, 12)``.

    Returns
    -------
    dict[str, ndarray]
        Each key is a property name, each value is an ndarray of
        length ``len(rxn_smiles)``. Unparseable reactions produce
        ``NaN``.

    Examples
    --------
    >>> from tmap.utils import reaction_properties
    >>> rxn = ["CCO.CC(=O)O>>CC(=O)OCC.O"]
    >>> props = reaction_properties(rxn)
    >>> props["delta_rings"]
    array([0.])
    """
    if properties is None:
        properties = list(AVAILABLE_REACTION_PROPERTIES)
    else:
        bad = [p for p in properties if p not in AVAILABLE_REACTION_PROPERTIES]
        if bad:
            raise ValueError(
                f"Unknown reaction properties: {bad}. Available: {AVAILABLE_REACTION_PROPERTIES}"
            )

    n = len(rxn_smiles)
    if n == 0:
        return {p: np.empty(0, dtype=np.float64) for p in properties}

    if n_workers is None:
        n_workers = _default_n_workers()
    n_workers = min(n_workers, n)

    ctx = _get_mp_context()
    chunks = _split_into_chunks(rxn_smiles, n_workers)

    with ctx.Pool(n_workers, initializer=_init_props_worker, initargs=(properties,)) as pool:
        batch_results = pool.map(_rxn_props_batch, chunks)

    props = np.concatenate(batch_results)  # (n, len(properties))
    n_invalid = np.isnan(props[:, 0]).sum()
    if n_invalid > 0:
        logger.warning("%d/%d reaction SMILES could not be parsed.", int(n_invalid), n)

    return {name: props[:, j] for j, name in enumerate(properties)}
