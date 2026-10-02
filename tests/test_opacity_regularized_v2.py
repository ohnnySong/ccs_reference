import unittest
import torch
from ccs_model.opacity_regularized_v2 import OpacityRegularizedV2


class OpacityRegularizedV2Test(unittest.TestCase):
    def test_output_shape(self):
        model=OpacityRegularizedV2().eval()
        with torch.no_grad(): output=model(torch.rand(2,3,128,128))
        self.assertEqual(tuple(output.shape),(2,4))


if __name__=="__main__": unittest.main()
