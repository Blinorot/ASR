import unittest

import numpy as np
from hw_asr.metric.utils import calc_cer, calc_wer

# based on seminar materials

class TestMetrics(unittest.TestCase):
    def test_cer_and_wer(self):
        for target, pred, expected_wer, expected_cer in [
            ("if you can not measure it you can not improve it", 
            "if you can nt measure t yo can not i", 
            0.454, 0.25),
            ("if you cant describe what you are doing as a process you dont know what youre doing", 
            "if you cant describe what you are doing as a process you dont know what youre doing", 
            0.0, 0.0),
            ("one measurement is worth a thousand expert opinions", 
            "one  is worth thousand opinions", 
            0.375, 0.392)
        ]:
            wer = calc_wer(target, pred)
            cer = calc_cer(target, pred)
            wer_result = np.isclose(wer, expected_wer, atol=1e-3), f"true: {target}, pred: {pred}, expected wer {expected_wer} != your wer {wer}"
            cer_result = np.isclose(cer, expected_cer, atol=1e-3), f"true: {target}, pred: {pred}, expected cer {expected_cer} != your cer {cer}"
            self.assertTrue(wer_result)
            self.assertTrue(cer_result)
