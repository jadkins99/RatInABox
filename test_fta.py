"""
Tests for FTA PyTorch implementations against the paper:
  "Fuzzy Tiling Activations: A Simple Approach to Learning Sparse Representations Online"
  Pan, Banman, White (ICLR 2021)

Tests both:
  - fta.py (FTA class): direct port of the TF reference code
  - fta_pytorch.py (FuzzyTilingActivation class): alternative PyTorch implementation

Reference TF code: reproduceRL/agents/network/ftann.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import unittest


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def tf_fta_numpy(z, c, delta, eta):
    """
    Pure-NumPy reference matching the TF/PyTorch Iplus_eta in float32.

    The TF code uses:  cast(x <= eta) * x + cast(x > eta)
    All arithmetic is done in float32 to match PyTorch tensor behaviour,
    avoiding float32/float64 boundary mismatches at s == eta.

    Args:
        z: scalar input
        c: 1-D numpy array of tile centers  (float32)
        delta: scalar tile width
        eta: scalar fuzziness
    Returns:
        1-D numpy array, same length as c  (float32)
    """
    z = np.float32(z)
    delta = np.float32(delta)
    eta = np.float32(eta)

    term1 = np.maximum(c - z, np.float32(0.0))
    term2 = np.maximum(z - delta - c, np.float32(0.0))
    s = term1 + term2  # float32

    I_eta_plus = (s <= eta).astype(np.float32) * s + (s > eta).astype(np.float32)

    phi = np.float32(1.0) - I_eta_plus
    return phi


# ---------------------------------------------------------------------------
# Tests for fta.py  (FTA class — direct port of TF code)
# ---------------------------------------------------------------------------

class TestFTA_CoreMath(unittest.TestCase):
    """Verify fta.py against paper equations and TF reference."""

    # ---- Paper worked example (page 4) ------------------------------------
    # [l, u] = [0, 1],  delta = 0.25,  k = 4
    # c = (0, 0.25, 0.5, 0.75)
    # TA(0.3) = (0, 1, 0, 0)           (hard tiling, eta=0)
    # FTA(0.3) with eta=0.25 => (0.95, 1, 0.8, 0)

    def _make_fta(self, n_tiles, input_min, input_max, eta):
        from fta import FTA
        params = {
            'n_tiles': n_tiles,
            'n_tilings': 1,
            'fta_input_min': input_min,
            'fta_input_max': input_max,
            'fta_eta': eta,
        }
        return FTA(params, input_dim=1)

    def test_paper_example_fta_output(self):
        """FTA(0.3) with k=4, [0,1], eta=0.25 should be [0.95, 1, 0.8, 0]."""
        fta = self._make_fta(4, 0.0, 1.0, 0.25)
        z = torch.tensor([[0.3]])
        out = fta(z).detach().numpy().flatten()
        expected = np.array([0.95, 1.0, 0.8, 0.0])
        np.testing.assert_allclose(out, expected, atol=1e-5,
            err_msg="FTA(0.3) does not match the paper's worked example")

    def test_paper_example_hard_tiling(self):
        """With eta→0 the TA should produce a one-hot: TA(0.3) = (0,1,0,0)."""
        fta = self._make_fta(4, 0.0, 1.0, 1e-7)
        z = torch.tensor([[0.3]])
        out = fta(z).detach().numpy().flatten()
        # bin 1 should be ~1, others ~0
        self.assertAlmostEqual(out[1], 1.0, places=4)
        for i in [0, 2, 3]:
            self.assertAlmostEqual(out[i], 0.0, places=4,
                msg=f"Bin {i} should be 0 for hard tiling, got {out[i]}")

    def test_tile_centers_match_paper_eq1(self):
        """Eq (1): c = (l, l+delta, l+2*delta, ..., u-delta)."""
        fta = self._make_fta(4, 0.0, 1.0, 0.25)
        expected = np.array([0.0, 0.25, 0.5, 0.75], dtype=np.float32)
        np.testing.assert_allclose(fta.c_vec.numpy(), expected, atol=1e-6)

    def test_tile_delta(self):
        """tile_delta should equal (u - l) / k."""
        fta = self._make_fta(10, -1.0, 1.0, 0.2)
        expected_delta = 2.0 / 10.0
        self.assertAlmostEqual(fta.tile_delta.item(), expected_delta, places=5)

    def test_matches_numpy_reference(self):
        """FTA output should match our pure-numpy paper reference for many z."""
        fta = self._make_fta(10, 0.0, 1.0, 0.1)
        c = np.linspace(0.0, 1.0, 10, endpoint=False).astype(np.float32)
        delta = 0.1
        eta = 0.1
        for z_val in np.linspace(-0.5, 1.5, 41):
            z_t = torch.tensor([[z_val]], dtype=torch.float32)
            torch_out = fta(z_t).detach().numpy().flatten()
            np_out = tf_fta_numpy(z_val, c, delta, eta)
            np.testing.assert_allclose(torch_out, np_out, atol=1e-5,
                err_msg=f"Mismatch at z={z_val}")


class TestFTA_Sparsity(unittest.TestCase):
    """Theorem 1: ||phi_eta(z)||_0 <= 2*floor(eta/delta) + 3."""

    def _make_fta(self, n_tiles, input_min, input_max, eta):
        from fta import FTA
        params = {
            'n_tiles': n_tiles, 'n_tilings': 1,
            'fta_input_min': input_min, 'fta_input_max': input_max,
            'fta_eta': eta,
        }
        return FTA(params, input_dim=1)

    def test_sparsity_bound_eta_eq_delta(self):
        """eta = delta => at most 2*1+3 = 5 nonzero entries per input dim."""
        k = 20
        fta = self._make_fta(k, 0.0, 1.0, 1.0 / k)
        delta = 1.0 / k
        eta = delta
        bound = 2 * int(eta / delta) + 3  # = 5

        for z_val in np.linspace(0.0, 1.0, 100):
            z_t = torch.tensor([[z_val]])
            out = fta(z_t).detach().numpy().flatten()
            nnz = int(np.sum(out > 1e-7))
            self.assertLessEqual(nnz, bound,
                msg=f"z={z_val}: {nnz} nonzero > bound {bound}")

    def test_sparsity_bound_eta_lt_delta(self):
        """eta < delta => floor(eta/delta)=0, bound = 3."""
        k = 20
        delta = 1.0 / k
        eta = delta / 2.0
        fta = self._make_fta(k, 0.0, 1.0, eta)
        bound = 2 * int(eta / delta) + 3  # = 3

        for z_val in np.linspace(0.0, 1.0, 100):
            z_t = torch.tensor([[z_val]])
            out = fta(z_t).detach().numpy().flatten()
            nnz = int(np.sum(out > 1e-7))
            self.assertLessEqual(nnz, bound,
                msg=f"z={z_val}: {nnz} nonzero > bound {bound}")


class TestFTA_Gradients(unittest.TestCase):
    """Gradient flow through fta.py's FTA layer."""

    def _make_fta(self, n_tiles, input_min, input_max, eta):
        from fta import FTA
        params = {
            'n_tiles': n_tiles, 'n_tilings': 1,
            'fta_input_min': input_min, 'fta_input_max': input_max,
            'fta_eta': eta,
        }
        return FTA(params, input_dim=1)

    def test_gradient_nonzero_at_boundary(self):
        """At a tile boundary the gradient of individual bins should be nonzero."""
        fta = self._make_fta(4, 0.0, 1.0, 0.25)
        z = torch.tensor([[0.3]], requires_grad=True)
        out = fta(z)
        # bin 0 has value 0.95 — its gradient w.r.t. z should be nonzero
        out[0, 0].backward()
        self.assertNotEqual(z.grad.item(), 0.0,
            msg="Gradient should be nonzero at bin boundary region")

    def test_gradient_zero_inside_bin(self):
        """Inside a bin (val=0 => FTA=1), gradient of that bin should be 0."""
        fta = self._make_fta(4, 0.0, 1.0, 0.25)
        # z = 0.375 is exactly in the middle of bin 1 [0.25, 0.5]
        z = torch.tensor([[0.375]], requires_grad=True)
        out = fta(z)
        out[0, 1].backward()
        self.assertAlmostEqual(z.grad.item(), 0.0, places=5,
            msg="Gradient should be zero in the flat interior of a bin")

    def test_gradient_zero_far_outside(self):
        """For z far outside the range, all bin gradients should be zero."""
        fta = self._make_fta(10, 0.0, 1.0, 0.1)
        z = torch.tensor([[5.0]], requires_grad=True)
        out = fta(z)
        out.sum().backward()
        self.assertAlmostEqual(z.grad.item(), 0.0, places=5,
            msg="Gradient should be zero for out-of-range input")

    def test_gradient_flows_through_network(self):
        """Pre-FTA linear layer should receive nonzero gradients."""
        from fta import FTA
        params = {
            'n_tiles': 10, 'n_tilings': 1,
            'fta_input_min': -2.0, 'fta_input_max': 2.0, 'fta_eta': 0.4,
        }
        model = nn.Sequential(
            nn.Linear(4, 8),
            FTA(params, input_dim=8),
            nn.Linear(80, 1),
        )
        x = torch.randn(16, 4)
        loss = model(x).sum()
        loss.backward()
        grad_norm = model[0].weight.grad.norm().item()
        self.assertGreater(grad_norm, 0.0,
            msg="Pre-FTA layer should receive nonzero gradients")


class TestFTA_Learning(unittest.TestCase):
    """Bin activation histogram should change during learning."""

    def test_histogram_changes_during_training(self):
        from fta import FTA
        params = {
            'n_tiles': 10, 'n_tilings': 1,
            'fta_input_min': -2.0, 'fta_input_max': 2.0, 'fta_eta': 0.4,
        }
        torch.manual_seed(42)
        model = nn.Sequential(
            nn.Linear(4, 8),
            FTA(params, input_dim=8),
            nn.Linear(80, 1),
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        x = torch.randn(32, 4)
        target = torch.randn(32, 1)

        with torch.no_grad():
            hist_before = model[1](model[0](x)).mean(dim=0).clone()

        for _ in range(200):
            optimizer.zero_grad()
            F.mse_loss(model(x), target).backward()
            optimizer.step()

        with torch.no_grad():
            hist_after = model[1](model[0](x)).mean(dim=0).clone()

        diff = (hist_before - hist_after).abs().sum().item()
        self.assertGreater(diff, 1.0,
            msg="Bin activation histogram should change substantially during learning")


class TestFTA_OutputShape(unittest.TestCase):
    """Output dimension should be input_dim * n_tiles * n_tilings."""

    def test_single_tiling(self):
        from fta import FTA
        params = {'n_tiles': 10, 'n_tilings': 1,
                  'fta_input_min': -1.0, 'fta_input_max': 1.0, 'fta_eta': 0.2}
        fta = FTA(params, input_dim=5)
        x = torch.randn(8, 5)
        out = fta(x)
        self.assertEqual(out.shape, (8, 5 * 10))

    def test_multi_tiling(self):
        from fta import FTA
        params = {'n_tiles': 10, 'n_tilings': 3,
                  'fta_input_min': -1.0, 'fta_input_max': 1.0, 'fta_eta': 0.2}
        fta = FTA(params, input_dim=5)
        x = torch.randn(8, 5)
        out = fta(x)
        self.assertEqual(out.shape, (8, 5 * 10 * 3))


class TestFTA_QNN(unittest.TestCase):
    """Test the FTA_QNN network (mirrors TF create_fta_qnn)."""

    def test_architecture_no_relu_before_fta(self):
        """Paper Figure 2: h2 = FTA(h1 W2) — no ReLU between fc2 and FTA."""
        from fta import FTA_QNN
        params = {'n_tiles': 10, 'n_tilings': 1,
                  'fta_input_min': -1.0, 'fta_input_max': 1.0, 'fta_eta': 0.2}
        model = FTA_QNN(4, 2, 64, 32, params)
        # Trace the forward: fc1 -> relu -> fc2 -> fta (no relu before fta)
        x = torch.randn(1, 4)
        h1 = F.relu(model.fc1(x))
        phi_pre = model.fc2(h1)
        # phi_pre should contain negative values (no ReLU applied)
        self.assertTrue((phi_pre < 0).any().item(),
            msg="fc2 output should NOT have ReLU; it feeds directly into FTA")

    def test_forward_shapes(self):
        from fta import FTA_QNN
        params = {'n_tiles': 10, 'n_tilings': 1,
                  'fta_input_min': -1.0, 'fta_input_max': 1.0, 'fta_eta': 0.2}
        model = FTA_QNN(4, 2, 64, 32, params)
        x = torch.randn(8, 4)
        q_vals, max_q, max_idx, sparse_phi = model(x)
        self.assertEqual(q_vals.shape, (8, 2))
        self.assertEqual(sparse_phi.shape, (8, 32 * 10))


# ---------------------------------------------------------------------------
# Tests for fta_pytorch.py  (FuzzyTilingActivation — the buggy file)
# ---------------------------------------------------------------------------

class TestFuzzyTilingActivation_Bugs(unittest.TestCase):
    """
    Verify that fta_pytorch.py has the known bugs.
    These tests document the problems so they can be fixed.
    """

    def test_cannot_instantiate_due_to_import(self):
        """fta_pytorch.py imports delaunay_triangles from shapely — nonsensical."""
        with self.assertRaises(Exception):
            from fta_pytorch import FuzzyTilingActivation
            # Even if import succeeds, instantiation should fail
            FuzzyTilingActivation(input_dim=2, delta=10)

    def test_sigmoid_formula_lacks_sparsity(self):
        """
        The sigmoid formula  sigmoid((x-c)/eta) - sigmoid((x-(c+w))/eta)
        never produces exact zeros, violating the sparsity guarantee
        (Theorem 1) from the paper.
        """
        # Manually replicate fta_pytorch.py's forward formula
        centers = torch.tensor([0.0, 0.25, 0.5, 0.75])
        tile_width = 0.25
        eta = 0.25

        z = torch.tensor([[0.3]])  # scalar input
        lower = torch.sigmoid((z - centers) / eta)
        upper = torch.sigmoid((z - (centers + tile_width)) / eta)
        sigmoid_out = (lower - upper).detach().numpy().flatten()

        # Paper FTA should have exactly 1 zero bin for z=0.3
        # (bin 3 is 0.0).  Sigmoid version has NO exact zeros.
        exact_zeros = np.sum(np.abs(sigmoid_out) < 1e-7)
        self.assertEqual(exact_zeros, 0,
            msg="Sigmoid formula should have NO exact zeros (this IS the bug)")

    def test_sigmoid_formula_wrong_magnitude(self):
        """
        The paper's FTA produces values near 0 and 1.
        The sigmoid formula produces values ≪ 1 (around 0.03–0.24).
        """
        centers = torch.tensor([0.0, 0.25, 0.5, 0.75])
        tile_width = 0.25
        eta = 0.25

        z = torch.tensor([[0.3]])
        lower = torch.sigmoid((z - centers) / eta)
        upper = torch.sigmoid((z - (centers + tile_width)) / eta)
        sigmoid_out = (lower - upper).detach().numpy().flatten()

        # Max activation should be near 1.0 for correct FTA
        self.assertLess(sigmoid_out.max(), 0.3,
            msg="Sigmoid formula peak is far below 1.0 — wrong magnitude")

    def test_sigmoid_formula_does_not_match_paper(self):
        """
        Direct comparison: sigmoid FTA vs paper's FTA for z=0.3.
        Paper expects [0.95, 1.0, 0.8, 0.0].
        """
        centers = torch.tensor([0.0, 0.25, 0.5, 0.75])
        tile_width = 0.25
        eta = 0.25

        z = torch.tensor([[0.3]])
        lower = torch.sigmoid((z - centers) / eta)
        upper = torch.sigmoid((z - (centers + tile_width)) / eta)
        sigmoid_out = (lower - upper).detach().numpy().flatten()

        expected = np.array([0.95, 1.0, 0.8, 0.0])
        max_err = np.abs(sigmoid_out - expected).max()
        self.assertGreater(max_err, 0.5,
            msg="Sigmoid formula should NOT match the paper's expected output")


# ---------------------------------------------------------------------------
# Cross-check: fta.py vs TF reference code logic
# ---------------------------------------------------------------------------

class TestFTA_vs_TFReference(unittest.TestCase):
    """
    Verify fta.py matches the TF reference (reproduceRL/agents/network/ftann.py)
    by checking key intermediate computations.
    """

    def test_sum_relu_matches_tf(self):
        """
        TF _sum_relu: relu(c - x) + relu(x - delta - c)
        PyTorch should be identical.
        """
        from fta import FTA
        params = {'n_tiles': 4, 'n_tilings': 1,
                  'fta_input_min': 0.0, 'fta_input_max': 1.0, 'fta_eta': 0.25}
        fta = FTA(params, input_dim=1)

        z = 0.3
        c = np.array([0.0, 0.25, 0.5, 0.75], dtype=np.float32)
        delta = 0.25

        # Expected from paper page 4
        expected_term1 = np.array([0.0, 0.0, 0.2, 0.45])
        expected_term2 = np.array([0.05, 0.0, 0.0, 0.0])
        expected_sum = expected_term1 + expected_term2

        # PyTorch computation
        x_t = torch.tensor([[[z]]])  # (1, 1, 1)
        c_t = fta.c_vec
        d_t = fta.tile_delta
        result = fta._sum_relu(c_t, x_t, d_t).detach().numpy().flatten()

        np.testing.assert_allclose(result, expected_sum, atol=1e-5)

    def test_Iplus_eta_matches_tf(self):
        """
        TF: cast(x<=eta)*x + cast(x>eta)
        PyTorch Iplus_eta should give the same result.
        """
        from fta import FTA
        params = {'n_tiles': 4, 'n_tilings': 1,
                  'fta_input_min': 0.0, 'fta_input_max': 1.0, 'fta_eta': 0.25}
        fta = FTA(params, input_dim=1)

        test_vals = torch.tensor([0.0, 0.05, 0.1, 0.25, 0.3, 0.5, 1.0, 2.0])
        eta = 0.25

        for v in test_vals:
            x = v.unsqueeze(0)
            result = fta.Iplus_eta(x, eta).item()
            # Expected: x if x <= eta, else 1.0
            expected = v.item() if v.item() <= eta else 1.0
            self.assertAlmostEqual(result, expected, places=5,
                msg=f"Iplus_eta({v.item()}, {eta}) = {result}, expected {expected}")

    def test_paper_settings_match_tf_defaults(self):
        """
        Paper Section 5.1: [l,u]=[-20,20], delta=eta=2.0, k=20.
        TF defaults: fta_input_max=20, fta_input_min=-20, n_tiles=20, fta_eta=2.0.
        """
        from fta import FTA
        params = {'n_tiles': 20, 'n_tilings': 1,
                  'fta_input_max': 20.0, 'fta_input_min': -20.0, 'fta_eta': 2.0}
        fta = FTA(params, input_dim=1)

        self.assertEqual(fta.n_tiles, 20)
        self.assertAlmostEqual(fta.fta_eta, 2.0, places=5)
        self.assertAlmostEqual(fta.tile_delta.item(), 2.0, places=5)
        # c should go from -20 to 18 in steps of 2
        expected_c = np.linspace(-20.0, 20.0, 20, endpoint=False).astype(np.float32)
        np.testing.assert_allclose(fta.c_vec.numpy(), expected_c, atol=1e-5)


if __name__ == '__main__':
    unittest.main(verbosity=2)
