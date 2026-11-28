from copy import deepcopy
from hashlib import sha256
import json
import unittest

import numpy as np

from research.flexible_scheduling.range_channel import packet, release_for_primary


class ScopedRange(unittest.TestCase):
    def setUp(self):
        self.source = {"width": 100, "height": 100, "intrinsics": {"fx": 50, "fy": 50, "cx": 50, "cy": 50},
                       "world_T_camera": np.eye(4).tolist()}
        self.box = [.25, .25, .75, .75]
        self.args = {"reference_digest": "a" * 64, "capture_digest": "b" * 64, "source_kind": "onboard-projection"}
        self.permission = {"explicit_range_grant": True, "actual_request": True, "branch": "unresolved",
                           "source_digest": "a" * 64, "allowed_region": self.box}

    def test_two_camera_visibility_and_original_depth(self):
        capture = deepcopy(self.source)
        capture["world_T_camera"][0][3] = 1.
        capture["world_T_camera"][2][3] = 1.
        # The first two occupy one reference bin. The other points are outside the scope,
        # behind the lens, or in the source ROI but outside the capture FOV.
        positions = [[0., 0., 4.], [0., 0., 6.], [3., 0., 4.], [0., 0., -2.], [0., 0., 1.2]]
        m, selected, captured_pixels = packet(self.source, capture, self.box, positions, **self.args)
        self.assertEqual(selected.tolist(), [0, 1])
        self.assertAlmostEqual(captured_pixels[0][0], 50 - 50/3)
        self.assertEqual(m["sample_count"], 2)
        self.assertEqual(m["bins"], [{"col": 6, "row": 4, "n": 2, "min_m": 4., "median_m": 5., "max_m": 6.}])
        self.assertEqual(release_for_primary(measurement=m, **self.permission), m)

    def test_permission_source_and_missing_returns(self):
        m, _, _ = packet(self.source, self.source, self.box, [[0., 0., 3.]], **self.args)
        for change in ({"explicit_range_grant": False}, {"actual_request": False}, {"branch": "negative"},
                       {"branch": "positive"}, {"source_digest": "b"*64}, {"allowed_region": [.2,.2,.8,.8]}):
            self.assertIsNone(release_for_primary(measurement=m, **{**self.permission, **change}))
        empty, _, _ = packet(self.source, self.source, self.box, [], **self.args)
        self.assertEqual(empty["bins"], [])
        self.assertIsNone(release_for_primary(measurement=empty, **self.permission))

    def test_even_a_rehashed_free_text_or_count_mismatch_is_withheld(self):
        m, _, _ = packet(self.source, self.source, self.box, [[0.,0.,3.]], **self.args)
        for field, value in (("identity","EXAMPLE"), ("median_m","3 metres"), ("n",2), ("min_m",4)):
            altered = deepcopy(m)
            altered["bins"][0][field] = value
            del altered["packet_sha256"]
            altered["packet_sha256"] = sha256(json.dumps(altered, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            self.assertIsNone(release_for_primary(measurement=altered, **self.permission))
        tampered = {**m, "capture_sha256": "c"*64}
        self.assertIsNone(release_for_primary(measurement=tampered, **self.permission))
