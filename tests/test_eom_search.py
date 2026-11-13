"""A converged EOM eigenpair can still omit lower invariant symmetry blocks."""
from types import SimpleNamespace

import numpy as np
from scipy.linalg import eigvals

from quantum_lab.engines.correlated import _eom_block_roots, _packed_eom_irreps


def response_settings():
    return SimpleNamespace(conv_tol=1e-10, max_cycle=100, max_space=40,
                           max_memory=256, verbose=0)


def solve_blocks(matrix, blocks, requested):
    def matvec(vectors):
        return [matrix @ vector for vector in vectors]
    candidates = []
    for indices in blocks:
        flags, energies, vectors = _eom_block_roots(
            matvec, matrix.diagonal(), np.asarray(indices), requested, response_settings(),
        )
        assert len(flags) == len(energies) == len(vectors) == min(requested, len(indices))
        assert all(flags)
        for energy, vector in zip(energies, vectors, strict=True):
            vector = np.asarray(vector)
            assert vector.shape == (matrix.shape[0],)
            vector = vector / np.linalg.norm(vector)
            # Full-space residual detects incorrect lifting or projection, not
            # just whether a smaller projected block happened to converge.
            assert np.linalg.norm(matrix @ vector - energy * vector) < 1e-8
            candidates.append(float(energy))
    return np.sort(candidates)[:requested]


def test_search_finds_lower_root_in_sector_missing_from_lowest_diagonal_guess():
    matrix = np.array([[1.0, 0.0, 0.0, 0.0],
                       [0.0, 8.0, 0.0, 0.0],
                       [0.0, 0.0, 4.0, 6.0],
                       [0.0, 0.0, 2.0, 4.0]])
    initial_index = int(np.argmin(matrix.diagonal()))
    initial = np.eye(4)[initial_index]
    # This exact eigenvector has zero residual, but is not the lowest root.
    assert np.linalg.norm(matrix @ initial - initial) == 0
    expected = np.sort(eigvals(matrix).real)[:1]
    assert expected[0] < 1
    actual = solve_blocks(matrix, [[0, 1], [2, 3]], requested=1)
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=0)


def test_packed_singlet_labels_include_a_symmetry_represented_only_by_doubles():
    # One occupied A1 orbital, B1/B2 virtuals: singles are B1/B2, whereas the
    # packed (B2,B1) double excitation belongs to A2 (numeric XOR code 3).
    labels = _packed_eom_irreps(np.array([0, 1, 2]), nocc=1)
    np.testing.assert_array_equal(labels, [1, 2, 0, 3, 0])
    matrix = np.diag([2.0, 3.0, 4.0, 0.25, 6.0])
    blocks = [np.flatnonzero(labels == irrep) for irrep in np.unique(labels)]
    actual = solve_blocks(matrix, blocks, requested=1)
    np.testing.assert_allclose(actual, [0.25], atol=1e-12, rtol=0)


def test_equal_energy_roots_in_distinct_symmetry_blocks_keep_multiplicity():
    matrix = np.diag([1.0, 2.0, 1.0, 3.0])
    actual = solve_blocks(matrix, [[0, 1], [2, 3]], requested=2)
    np.testing.assert_allclose(actual, [1.0, 1.0], atol=1e-12, rtol=0)
