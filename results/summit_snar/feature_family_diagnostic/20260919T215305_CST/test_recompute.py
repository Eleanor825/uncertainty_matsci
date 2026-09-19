import unittest
import recompute as r


class MetricsContracts(unittest.TestCase):
    def test_tied_scores_AP_and_undefined_AUC(self):
        m=r.metrics([0,1],[.5,.5]);self.assertEqual(m['auroc'],.5);self.assertEqual(m['error_auprc'],.5)
        self.assertIsNone(r.metrics([1,1],[.2,.8])['auroc'])

    def test_known_rank_AP_Brier(self):
        m=r.metrics([0,0,1,1],[.1,.4,.35,.8]);self.assertEqual(m['auroc'],.75)
        self.assertAlmostEqual(m['error_auprc'],5/6);self.assertAlmostEqual(m['brier'],.158125)

    def test_bin_boundaries_and_empty_unknown(self):
        b=r.calibration_bins([0,1,1],[0.,.5,1.]);self.assertEqual([x['count'] for x in b],[1,0,0,0,0,1,0,0,0,1])
        self.assertIsNone(b[1]['mean_predicted_no_HVI']);self.assertEqual(b[9]['mean_predicted_no_HVI'],1.)


if __name__=='__main__':unittest.main()
