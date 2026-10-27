import sys, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent)); sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from src.segmentation import segment, lesion_features, iou
import make_synthetic_leaves as syn


class T(unittest.TestCase):
    def setUp(self): self.rng = np.random.default_rng(0)

    def test_healthy_leaf_has_negligible_lesion(self):
        img, _, _ = syn.make_leaf(self.rng, "healthy"); self.assertLess(lesion_features(img)[0], 0.02)

    def test_blight_has_more_lesion_area_than_spot(self):
        sp = np.mean([lesion_features(syn.make_leaf(self.rng, "leaf_spot")[0])[0] for _ in range(20)])
        bl = np.mean([lesion_features(syn.make_leaf(self.rng, "blight")[0])[0] for _ in range(20)])
        self.assertGreater(bl, sp)

    def test_spot_has_more_components_than_blight(self):
        sp = np.mean([lesion_features(syn.make_leaf(self.rng, "leaf_spot")[0])[1] for _ in range(20)])
        bl = np.mean([lesion_features(syn.make_leaf(self.rng, "blight")[0])[1] for _ in range(20)])
        self.assertGreater(sp, bl)

    def test_iou_good_on_diseased(self):
        v = []
        for c in ("leaf_spot", "blight"):
            for _ in range(15):
                img, gt, _ = syn.make_leaf(self.rng, c); v.append(iou(segment(img)[1], gt))
        self.assertGreater(np.mean(v), 0.6)

    def test_feature_dim(self):
        self.assertEqual(lesion_features(syn.make_leaf(self.rng, "blight")[0]).shape, (16,))

    def test_iou_edge_cases(self):
        a = np.zeros((4, 4), bool); self.assertEqual(iou(a, a), 1.0); a[0, 0] = True; self.assertEqual(iou(a, a), 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
