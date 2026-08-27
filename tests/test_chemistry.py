"""Tests for tmap.utils.chemistry.

These check that the three per-molecule helpers can be kept in step.
``fingerprints_from_smiles`` skips any SMILES it cannot read, while
``molecular_properties`` and ``murcko_scaffolds`` keep one entry for
every input, so callers need the flags to line the results back up.
"""

import numpy as np
import pytest

rdkit = pytest.importorskip("rdkit")

from tmap.utils.chemistry import (  # noqa: E402
    fingerprints_from_smiles,
    molecular_properties,
    murcko_scaffolds,
)

VALID = ["CCO", "c1ccccc1", "CC(=O)O"]
MIXED = ["CCO", "not_a_molecule", "c1ccccc1"]


class TestReturnValidMask:
    def test_default_returns_bare_array(self):
        fps = fingerprints_from_smiles(VALID, n_workers=1)
        assert isinstance(fps, np.ndarray)
        assert fps.shape == (3, 2048)

    def test_mask_has_one_entry_per_input(self):
        fps, valid = fingerprints_from_smiles(MIXED, n_workers=1, return_valid=True)
        assert valid.dtype == np.bool_
        assert valid.tolist() == [True, False, True]
        assert len(valid) == len(MIXED)
        assert int(valid.sum()) == len(fps)

    def test_all_valid_mask_is_all_true(self):
        fps, valid = fingerprints_from_smiles(VALID, n_workers=1, return_valid=True)
        assert valid.all()
        assert len(fps) == len(VALID)

    def test_empty_input(self):
        fps, valid = fingerprints_from_smiles([], return_valid=True)
        assert fps.shape == (0, 2048)
        assert valid.shape == (0,)

    def test_all_invalid(self):
        fps, valid = fingerprints_from_smiles(["nope", ""], n_workers=1, return_valid=True)
        assert fps.shape == (0, 2048)
        assert not valid.any()
        assert len(valid) == 2

    def test_mask_spans_multiple_worker_chunks(self):
        # More SMILES than workers, with bad entries in different slices, so
        # the flags from each worker have to be joined back in input order.
        smiles = ["CCO", "bad_1", "c1ccccc1", "bad_2", "CC(=O)O", "CCN"]
        fps, valid = fingerprints_from_smiles(smiles, n_workers=3, return_valid=True)
        assert valid.tolist() == [True, False, True, False, True, True]
        assert len(fps) == 4

    def test_mqn_mask(self):
        fps, valid = fingerprints_from_smiles(MIXED, fp_type="mqn", n_workers=1, return_valid=True)
        assert fps.shape == (2, 42)
        assert valid.tolist() == [True, False, True]

    def test_warns_about_dropped_rows(self, caplog):
        with caplog.at_level("WARNING", logger="tmap.utils.chemistry"):
            fingerprints_from_smiles(MIXED, n_workers=1)
        assert "1/3" in caplog.text
        assert "return_valid" in caplog.text


class TestAlignmentAcrossHelpers:
    def test_mask_realigns_properties_and_scaffolds(self):
        fps, valid = fingerprints_from_smiles(MIXED, n_workers=1, return_valid=True)
        props = molecular_properties(MIXED, ["mw"], n_workers=1)
        scaffolds = murcko_scaffolds(MIXED, n_workers=1)

        assert len(props["mw"]) == len(MIXED)
        assert len(scaffolds) == len(MIXED)

        mw = props["mw"][valid]
        kept_scaffolds = scaffolds[valid]
        assert len(mw) == len(fps) == len(kept_scaffolds)
        assert not np.isnan(mw).any()

        # Row 1 of fps is benzene, so the property left at index 1 after
        # dropping the bad entry has to be benzene's too.
        benzene_mw = molecular_properties(["c1ccccc1"], ["mw"], n_workers=1)["mw"][0]
        assert mw[1] == pytest.approx(benzene_mw)
        assert kept_scaffolds[1] == "c1ccccc1"
