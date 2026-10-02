import unittest

import torch

from ccs_model.architecture_document_v1 import CcsDocumentMultiTaskV1
from ccs_model.train_document_v1 import contraction_class


class CcsDocumentV1Test(unittest.TestCase):
    def test_thresholds(self):
        self.assertEqual(contraction_class(20.1), 0)
        self.assertEqual(contraction_class(20.0), 1)
        self.assertEqual(contraction_class(16.1), 1)
        self.assertEqual(contraction_class(16.0), 2)

    def test_forward(self):
        model = CcsDocumentMultiTaskV1(pretrained=False).eval()
        with torch.no_grad():
            output = model(torch.rand(2, 3, 128, 128), torch.tensor([11.3, 12.0]))
        self.assertEqual(tuple(output.opacity_logits.shape), (2, 4))
        self.assertEqual(tuple(output.contraction_logits.shape), (2, 3))
        self.assertEqual(tuple(output.opening_area_mm2.shape), (2,))
        self.assertTrue(torch.all(output.opening_area_mm2 > 0))


if __name__ == "__main__":
    unittest.main()
