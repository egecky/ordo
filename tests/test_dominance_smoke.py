import unittest

import torch

from sdmorl.hk import L_hat


class DominanceSmokeTest(unittest.TestCase):
    def test_matching_returns_have_zero_gap_and_finite_gradient(self):
        returns = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float32)
        threshold = torch.tensor([0.5, 0.5], dtype=torch.float32, requires_grad=True)

        gap = L_hat(
            returns,
            returns,
            threshold,
            k=2,
            smooth="softplus",
            smooth_param=0.5,
        )
        gap.backward()

        self.assertAlmostEqual(gap.item(), 0.0, places=7)
        self.assertTrue(torch.isfinite(threshold.grad).all().item())


if __name__ == "__main__":
    unittest.main()
