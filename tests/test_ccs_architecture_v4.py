import unittest

import torch

from ccs_model.architecture_v4 import CcsMultiTaskV4, remap_source_grade
from ccs_model.metrics_v4 import area_metrics, binary_metrics, severity_metrics


class CcsArchitectureV4Test(unittest.TestCase):
    def test_grade_mapping(self) -> None:
        source = torch.tensor([0, 1, 2, 3])
        self.assertEqual(remap_source_grade(source).tolist(), [0, 0, 1, 2])

    def test_forward_shapes_and_collapsed_probability(self) -> None:
        model = CcsMultiTaskV4(pretrained=False, decoder_width=32).eval()
        image = torch.rand(2, 3, 128, 128)
        with torch.no_grad():
            output = model(
                image,
                wtw_mm=torch.tensor([11.3, 11.8]),
                wtw_diameter_px=torch.tensor([100.0, 104.0]),
            )
        self.assertEqual(tuple(output.severity_logits.shape), (2, 3))
        self.assertEqual(tuple(output.binary_logit.shape), (2,))
        self.assertEqual(tuple(output.opening_mask_probability.shape), (2, 1, 128, 128))
        self.assertEqual(tuple(output.wtw_endpoints_xy.shape), (2, 2, 2))
        self.assertEqual(tuple(output.opening_area_mm2.shape), (2,))
        self.assertTrue(torch.allclose(output.opening_area_mm2, output.opening_area_regression_mm2))
        self.assertTrue(
            torch.allclose(output.area_scale_diameter_px, torch.tensor([100.0, 104.0]))
        )
        self.assertTrue(torch.all(output.opening_area_mm2 > 0))
        self.assertTrue(
            torch.allclose(
                output.collapsed_probability,
                output.severity_probabilities[:, 1:].sum(dim=1),
            )
        )

    def test_metrics_include_three_class_binary_and_area(self) -> None:
        severity = severity_metrics(
            [0, 1, 2, 2], [0, 1, 1, 2], [0.1, 0.7, 0.6, 0.9]
        )
        self.assertIn("moderate_or_severe", severity)
        self.assertIn("auroc", severity["moderate_or_severe"])
        self.assertIn("auprc", severity["moderate_or_severe"])
        self.assertEqual(severity["labels"], ["light", "moderate", "severe"])
        area = area_metrics([10.0, 12.0], [9.0, 14.0])
        self.assertAlmostEqual(area["mae_mm2"], 1.5)
        self.assertAlmostEqual(area["signed_bias_mm2"], 0.5)
        binary = binary_metrics([0, 0, 1, 1], [0, 1, 1, 1], [0.1, 0.6, 0.7, 0.9])
        self.assertIn("auroc", binary)
        self.assertIn("auprc", binary)


if __name__ == "__main__":
    unittest.main()
