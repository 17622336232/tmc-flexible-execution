import unittest

from research.flexible_scheduling.material_binding import bind_identity


class ImageSourceAssociation(unittest.TestCase):
    def setUp(self):
        self.camera=dict(width=100,height=100,intrinsics=dict(fx=50,fy=50,cx=50,cy=50),
                         world_T_camera=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
        self.surface=dict(reference_quad=[[30,35],[65,35],[65,52],[30,52]],depth_m=3.)
        self.row=dict(value="EQ-07",confidence=95.,center=[.45,.46])

    def test_literal_must_belong_to_the_requested_region(self):
        self.assertEqual(bind_identity(self.camera,self.camera,self.surface,(.25,.3,.7,.75),[self.row]),self.row)
        self.assertIsNone(bind_identity(self.camera,self.camera,self.surface,(.72,.3,.95,.75),[self.row]))

    def test_conflicts_and_degenerate_sources_abstain(self):
        box=(.1,.1,.9,.9)
        self.assertIsNone(bind_identity(self.camera,self.camera,self.surface,box,[self.row,{**self.row,"value":"EQ-19"}]))
        self.assertIsNone(bind_identity(self.camera,self.camera,None,box,[self.row]))
        self.assertIsNone(bind_identity(self.camera,self.camera,{**self.surface,"reference_quad":[[4,4]]*4},box,[self.row]))
        self.assertIsNone(bind_identity(self.camera,self.camera,self.surface,box,[{**self.row,"confidence":60}]))

    def test_measured_camera_motion_changes_correspondence(self):
        moved={**self.camera,"world_T_camera":[[1,0,0,1.5],[0,1,0,0],[0,0,1,0],[0,0,0,1]]}
        self.assertIsNone(bind_identity(self.camera,moved,self.surface,(.4,.4,.5,.5),[self.row]))
