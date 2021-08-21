# Enter your Python code here
import pya
from pya import Point, Vector, DPoint, DVector
from pya import DSimplePolygon, SimplePolygon, DPolygon, Polygon, Region
from pya import Trans, DTrans, CplxTrans, DCplxTrans, ICplxTrans
from pya import Path

from importlib import reload
import classLib

reload(classLib)

from classLib.baseClasses import ElementBase, ComplexBase
from classLib.coplanars import CPWParameters, CPWRLPath, CPW2CPW, Bridge1
from classLib.shapes import XmonCross, Circle, Rectangle
from classLib.resonators import EMResonatorTL3QbitWormRLTailXmonFork
from classLib.josJ import AsymSquidParams, AsymSquid
from classLib.chipTemplates import CHIP_10x10_12pads, FABRICATION
from classLib.chipDesign import ChipDesign
from classLib.marks import MarkBolgar

# imports for docstrings generation
from classLib.coplanars import CPW, CPWArc

from typing import List, Dict, Union, Optional
from classLib.contactPads import ContactPad

from classLib.helpers import fill_holes, split_polygons

from math import cos, sin, tan, atan2, pi, degrees
import itertools

FABRICATION.OVERETCHING = 0.0e3  # nm


class FluxLineEnd(ElementBase):
    def __init__(self, origin, fc_cpw_params, width, trans_in=None):  # width = 5e3

        self._fc_cpw_params: Union[CPW, CPWParameters, CPWArc] = fc_cpw_params
        self._width = width

        super().__init__(origin, trans_in)

        self.start = self.connections[0]
        self.end = self.connections[1]

    def init_regions(self):
        # flux CPW width and gap
        f_cpw = self._fc_cpw_params
        p1 = DPoint(-f_cpw.b / 2, -f_cpw.width)
        p2 = p1 + DPoint(0, f_cpw.width)
        p3 = p2 + DPoint(f_cpw.gap, 0)
        p4 = p3 + DPoint(0, -3 / 2 * f_cpw.width)
        metal_points = [p1, p2, p3, p4]
        self.metal_region.insert(Region(DSimplePolygon(metal_points)))

        empty_points = [DPoint(f_cpw.width / 2, 0),
                        DPoint(f_cpw.width / 2, f_cpw.width),
                        DPoint(-self._width / 2, f_cpw.width),
                        DPoint(-self._width / 2, f_cpw.width + f_cpw.gap),
                        DPoint(self._width / 2, f_cpw.width + f_cpw.gap),
                        DPoint(self._width / 2, f_cpw.width),
                        DPoint(f_cpw.width / 2 + f_cpw.gap, f_cpw.width),
                        DPoint(f_cpw.width / 2 + f_cpw.gap, 0)]

        empty_region = Region(DSimplePolygon(empty_points))
        self.empty_region.insert(empty_region)

        self.connections = [DPoint(0, 0), DPoint(0, f_cpw.width + f_cpw.gap)]


class MDriveLineEnd(ComplexBase):

    def __init__(self, z0_cpw, z1_cpw_params, transition_length, z1_length, trans_in=None):
        """
        Makes transition to thin Z at the end of the md coplanar.

        Parameters
        ----------
        z0_cpw : Union[CPW, CPWArc]
            last part of the md coplanar
        z1_cpw_params : Union[CPWParameters, CPW]
            parameters of the thin coplanar
        transition_length : float
            length of cpw2cpw transition
        z1_length : float
            length of the z1 thin coplanar
        trans_in : DCplxTrans
            initial transformation
        """
        self.z0_cpw: Union[CPW, CPWArc] = z0_cpw
        self.z1_params: Union[CPWParameters, CPW] = z1_cpw_params
        self.transition_length: float = transition_length
        self.z1_length: float = z1_length

        # attributes to be constructed later
        self.transition_cpw2cpw: CPW2CPW = None
        self.z1_cpw: CPW = None
        self.z1_cpw_open_end: CPW = None

        super().__init__(self.z0_cpw.end, trans_in)

        self.start = self.connections[0]
        self.end = self.connections[1]

    def init_primitives(self):
        origin = DPoint(0, 0)
        transition_end = origin + DPoint(self.transition_length, 0)
        alpha = self.z0_cpw.angle_connections[1]

        self.transition_cpw2cpw = CPW2CPW(
            self.z0_cpw, self.z1_params, origin, transition_end,
            trans_in=DCplxTrans(1, degrees(alpha), False, 0, 0)
        )
        self.primitives["transition_cpw2cpw"] = self.transition_cpw2cpw

        z1_cpw_end = self.transition_cpw2cpw.end + DPoint(self.z1_length, 0)
        self.z1_cpw = CPW(cpw_params=self.z1_params,
                          start=self.transition_cpw2cpw.end, end=z1_cpw_end,
                          trans_in=DCplxTrans(1, degrees(alpha), False, 0, 0))
        self.primitives["z1_cpw"] = self.z1_cpw

        # open-ended
        self.z1_cpw_open_end = CPW(
            0, gap=self.z1_params.b / 2,
            start=self.z1_cpw.end, end=self.z1_cpw.end + DPoint(self.z1_params.b / 2, 0),
            trans_in=DCplxTrans(1, degrees(alpha), False, 0, 0)
        )
        self.primitives["z1_cpw_open_end"] = self.z1_cpw_open_end

        self.connections = [self.transition_cpw2cpw.start, self.z1_cpw.end]


class TestStructurePads(ComplexBase):
    def __init__(self, center, trans_in=None):
        self.center = center
        self.rectangle_a = 200e3 + 2 * FABRICATION.OVERETCHING
        self.gnd_gap = 20e3 - 2 * FABRICATION.OVERETCHING
        self.rectangles_gap = 20e3 - 2 * FABRICATION.OVERETCHING
        super().__init__(center, trans_in)

    def init_primitives(self):
        center = DPoint(0, 0)

        ## empty rectangle ##
        empty_width = self.rectangle_a + 2 * self.gnd_gap
        empty_height = 2 * self.rectangle_a + 2 * self.gnd_gap + self.rectangles_gap
        # bottom-left point of rectangle
        bl_point = center - DPoint(empty_width / 2, empty_height / 2)
        self.empty_rectangle = Rectangle(
            bl_point,
            empty_width, empty_height, inverse=True
        )
        self.primitives["empty_rectangle"] = self.empty_rectangle

        ## top rectangle ##
        # bottom-left point of rectangle
        bl_point = center + DPoint(-self.rectangle_a / 2, self.rectangles_gap / 2)
        self.top_rec = Rectangle(bl_point, self.rectangle_a, self.rectangle_a)
        self.primitives["top_rec"] = self.top_rec

        ## bottom rectangle ##
        # bottom-left point of rectangle
        bl_point = center + DPoint(
            -self.rectangle_a / 2,
            - self.rectangles_gap / 2 - self.rectangle_a
        )
        self.bot_rec = Rectangle(bl_point, self.rectangle_a, self.rectangle_a)
        self.primitives["bot_rec"] = self.bot_rec

        self.connections = [center]

    def _refresh_named_connections(self):
        self.center = self.connections[0]


class Design4QSquare(ChipDesign):
    def __init__(self, cell_name):
        super().__init__(cell_name)
        info_el2 = pya.LayerInfo(3, 0)  # for DC contact deposition
        self.region_el2 = Region()
        self.layer_el2 = self.layout.layer(info_el2)

        info_bridges1 = pya.LayerInfo(4, 0)  # bridge photo layer 1
        self.region_bridges1 = Region()
        self.layer_bridges1 = self.layout.layer(info_bridges1)

        info_bridges2 = pya.LayerInfo(5, 0)  # bridge photo layer 2
        self.region_bridges2 = Region()
        self.layer_bridges2 = self.layout.layer(info_bridges2)

        # layer with polygons that will protect structures located
        # on the `self.region_el` - e-beam litography layer
        info_el_protection = pya.LayerInfo(6, 0)
        self.region_el_protection = Region()
        self.layer_el_protection = self.layout.layer(info_el_protection)

        self.lv.add_missing_layers()  # has to call it once more to add new layers

        ### ADDITIONAL VARIABLES SECTION START ###
        # chip rectangle and contact pads
        self.chip = CHIP_10x10_12pads
        self.chip.pcb_gap -= 2 * FABRICATION.OVERETCHING
        self.chip.pcb_width += 2 * FABRICATION.OVERETCHING
        self.chip.pcb_Z = CPWParameters(self.chip.pcb_width, self.chip.pcb_gap)

        self.chip_box: pya.DBox = self.chip.box
        self.z_md_fl: CPWParameters = CPWParameters(11e3, 5.7e3)  # Z = 50.09 E_eff = 6.235 (E = 11.45)
        self.ro_Z: CPWParameters = self.chip.chip_Z
        self.contact_pads: list[ContactPad] = self.chip.get_contact_pads(
            [self.z_md_fl] * 10 + [self.ro_Z] * 2, FABRICATION.OVERETCHING
        )

        # readout line parameters
        self.ro_line_turn_radius: float = 200e3
        self.ro_line_dy: float = 1600e3
        self.cpwrl_ro_line: CPWRLPath = None
        self.Z0: CPWParameters = CHIP_10x10_12pads.chip_Z

        ## resonator parameters ##
        # resonators objects list, 4 resonators in total
        self.resonators: List[EMResonatorTL3QbitWormRLTailXmonFork] = [None] * 4
        # distance between nearest resonators central conductors centers
        # constant step between resonators origin points along x-axis.
        self.resonators_dx: float = 900e3
        self.L_coupling_list: list[float] = [1e3 * x for x in [310, 320, 320, 310]]
        # corresponding to resonanse freq is linspaced in interval [6,9) GHz
        self.L0 = 1150e3
        self.L1_list = [1e3 * x for x in [91.3339, 133.001, 137.77, 79.9156]]
        self.r = 60e3
        self.N_coils = [3] * len(self.resonators)
        self.L2_list = [self.r] * len(self.resonators)
        self.L3_list = [0e3] * len(self.resonators)  # to be constructed
        self.L4_list = [self.r] * len(self.resonators)
        self.width_res = 20e3
        self.gap_res = 10e3
        self.Z_res = CPWParameters(self.width_res, self.gap_res)
        self.to_line_list = [56e3] * len(self.resonators)
        self.fork_metal_width = 10e3
        self.fork_gnd_gap = 15e3
        self.xmon_fork_gnd_gap = 14e3
        # resonator-fork parameters
        # for coarse C_qr evaluation
        self.fork_y_spans = [x * 1e3 for x in [78.3046, 26.2982, 84.8277, 35.3751]]

        # xmon parameters
        self.xmon_x_distance: float = 545e3  # from simulation of g_12
        # for fine C_qr evaluation
        self.xmon_dys_Cg_coupling = [14e3] * len(self.resonators)
        self.xmons: list[XmonCross] = [None] * len(self.resonators)

        self.cross_len_x = 180e3
        self.cross_width_x = 60e3
        self.cross_gnd_gap_x = 20e3
        self.cross_len_y = 155e3
        self.cross_width_y = 60e3
        self.cross_gnd_gap_y = 20e3

        # squids
        self.squids: List[AsymSquid] = [None] * len(self.resonators)
        self.test_squids: List[AsymSquid] = []
        self.el_dc_contacts: List[List[ElementBase, ElementBase]] = []

        # md and flux lines attributes
        self.shift_fl_y = self.cross_len_y + 60e3
        self.shift_md_x = 60e3
        self.shift_md_y = 510e3

        self.cpwrl_md1: List[Union[CPWRLPath, CPW]] = []
        self.cpwrl_md1_end: MDriveLineEnd = None
        self.cpwrl_fl1: List[Union[CPWRLPath, CPW]] = []
        self.cpwrl_fl1_end: FluxLineEnd = None

        self.cpwrl_md2: List[Union[CPWRLPath, CPW]] = []
        self.cpwrl_md2_end: MDriveLineEnd = None
        self.cpwrl_fl2: List[Union[CPWRLPath, CPW]] = []
        self.cpwrl_fl2_end: FluxLineEnd = None

        self.cpwrl_md3: List[Union[CPWRLPath, CPW]] = []
        self.cpwrl_md3_end: MDriveLineEnd = None
        self.cpwrl_fl3: List[Union[CPWRLPath, CPW]] = []
        self.cpwrl_fl3_end: FluxLineEnd = None

        self.cpwrl_md4: List[Union[CPWRLPath, CPW]] = []
        self.cpwrl_md4_end: MDriveLineEnd = None
        self.cpwrl_fl4: List[Union[CPWRLPath, CPW]] = []
        self.cpwrl_fl4_end: FluxLineEnd = None

        # marks
        self.marks: List[MarkBolgar] = []
        ### ADDITIONAL VARIABLES SECTION END ###

    def draw(self, design_params=None):
        self.draw_chip()

        '''
            Only creating object. This is due to the drawing of xmons and resonators require
        draw xmons, then draw resonators and then draw additional xmons. This is
        ugly and that how this was before migrating to `ChipDesign` based code structure
            This is also the reason why `self.__init__` is flooded with design parameters that
        are used across multiple drawing functions.

        TODO: This drawings sequence can be decoupled in the future.
        '''
        self.draw_xmons()
        self.draw_josephson_loops()
        self.draw_el_protection()

        self.draw_resonators()
        self.draw_md_and_flux_lines()
        self.draw_readout_waveguide()

        self.draw_test_structures()

        self.draw_el_dc_contacts()

        self.draw_photo_el_marks()
        self.draw_bridges()
        self.draw_pinning_holes()
        self.inverse_destination(self.region_ph)
        # self.split_polygons_in_layers(max_pts=180)

    def _transfer_regs2cell(self):
        # this too methods assumes that all previous drawing
        # functions are placing their object on regions
        # in order to avoid extensive copying of the polygons
        # to/from cell.shapes during the logic operations on
        # polygons
        self.cell.shapes(self.layer_ph).insert(self.region_ph)
        self.cell.shapes(self.layer_el).insert(self.region_el)
        self.cell.shapes(self.layer_el2).insert(self.region_el2)
        self.cell.shapes(self.layer_bridges1).insert(self.region_bridges1)
        self.cell.shapes(self.layer_bridges2).insert(self.region_bridges2)
        self.cell.shapes(self.layer_el_protection).insert(self.region_el_protection)
        self.lv.zoom_fit()

    def draw_chip(self):
        self.region_bridges2.insert(self.chip_box)

        self.region_ph.insert(self.chip_box)
        for contact_pad in self.contact_pads:
            contact_pad.place(self.region_ph)

    def draw_xmons(self):
        chip_center = DPoint(self.chip.dx / 2, self.chip.dy / 2)

        xmon_centers = [None] * 4
        grid_basis_v1 = DVector(self.xmon_x_distance, 0)
        grid_basis_v2 = DVector(0, self.xmon_x_distance)
        grid_origin = chip_center - (grid_basis_v1 + grid_basis_v2) / 2
        for i, (v2_idx, v1_idx) in enumerate(itertools.product(range(2), range(2))):
            xmon_centers[i] = grid_origin + v1_idx * grid_basis_v1 + v2_idx * grid_basis_v2
            self.xmons[i] = XmonCross(
                xmon_centers[i], self.cross_len_x,
                self.cross_width_x + 2 * FABRICATION.OVERETCHING,
                self.cross_gnd_gap_x - 2 * FABRICATION.OVERETCHING,
                sideY_length=self.cross_len_y,
                sideY_width=self.cross_width_y + 2 * FABRICATION.OVERETCHING,
                sideY_gnd_gap=self.cross_gnd_gap_y - 2 * FABRICATION.OVERETCHING
            )
            self.xmons[i].place(self.region_ph)

    def draw_josephson_loops(self):
        new_pars_squid = AsymSquidParams(
            pad_r=5e3, pads_distance=30e3,
            p_ext_width=10e3, p_ext_r=200,
            sq_len=15e3, sq_area=200e6,
            j_width_1=95, j_width_2=348,
            intermediate_width=500, b_ext=1e3, j_length=94, n=20,
            bridge=180, j_length_2=250
        )
        # place left squids
        for i, xmon_cross in enumerate(self.xmons):
            # for left xmon crosses
            if i == 0 or i == 2:
                squid_center = (xmon_cross.cpw_lempt.end + xmon_cross.cpw_lempt.start) / 2
                squid = AsymSquid(squid_center, new_pars_squid, 0, trans_in=DTrans.R270)
            # for right xmon crosses
            elif i == 1 or i == 3:
                squid_center = (xmon_cross.cpw_rempt.end + xmon_cross.cpw_rempt.start) / 2
                squid = AsymSquid(squid_center, new_pars_squid, 0, trans_in=DTrans.R90)
            self.squids[i] = squid
            squid.place(self.region_el)

    def draw_resonators(self):
        fork_x_span = self.cross_width_y + 2 * (self.xmon_fork_gnd_gap + self.fork_metal_width)

        ### RESONATORS TAILS CALCULATIONS SECTION START ###
        # key to the calculations can be found in hand-written format here:
        # https://drive.google.com/file/d/1wFmv5YmHAMTqYyeGfiqz79a9kL1MtZHu/view?usp=sharing

        # x span between left long vertical line and
        # right-most center of central conductors
        resonators_widths = [2 * self.r + L_coupling for L_coupling in self.L_coupling_list]
        x1 = 2 * self.resonators_dx + resonators_widths[2] / 2 - 2 * self.xmon_x_distance
        x2 = x1 + self.xmon_x_distance - self.resonators_dx
        x3 = resonators_widths[2] / 2
        x4 = 3 * self.resonators_dx - (x1 + 3 * self.xmon_x_distance)
        x5 = 4 * self.resonators_dx - (x1 + 4 * self.xmon_x_distance)

        res_tail_shape = "LRLRL"
        tail_turn_radiuses = self.r
        # list corrected for resonator-qubit coupling geomtry, so all transmons centers are placed
        # along single horizontal line
        self.L0_list = [self.L0 - xmon_dy_Cg_coupling for xmon_dy_Cg_coupling in self.xmon_dys_Cg_coupling]
        self.L2_list[1] += 0
        self.L2_list[2] += 3 * self.Z_res.b
        self.L2_list[3] += 6 * self.Z_res.b

        self.L3_list[1] = x2
        self.L3_list[0] = x3
        self.L3_list[2] = x4
        self.L3_list[3] = x5

        self.L4_list[1] += 6 * self.Z_res.b
        self.L4_list[0] += 6 * self.Z_res.b
        self.L4_list[2] += 3 * self.Z_res.b
        tail_segment_lengths_list = [[L2, L3, L4 + FABRICATION.OVERETCHING] for L2, L3, L4 in
                                     zip(self.L2_list, self.L3_list, self.L4_list)]
        tail_turn_angles_list = [
            [pi / 2, -pi / 2],
            [pi / 2, -pi / 2],
            [pi / 2, -pi / 2],
            [-pi / 2, pi / 2],
            [-pi / 2, pi / 2],
        ]
        tail_trans_in_list = [
            Trans.R270,
            Trans.R270,
            Trans.R270,
            Trans.R270,
            Trans.R270
        ]
        ### RESONATORS TAILS CALCULATIONS SECTION END ###

        res_pars_list = list(
            zip(
                self.L1_list, self.to_line_list, self.L_coupling_list,
                self.fork_y_spans,
                tail_segment_lengths_list, tail_turn_angles_list, tail_trans_in_list,
                self.L0_list, self.N_coils,
                self.xmon_dys_Cg_coupling
            )
        )
        resonator_cpw = CPWParameters(self.Z_res.width + 2 * FABRICATION.OVERETCHING,
                                      self.Z_res.gap - 2 * FABRICATION.OVERETCHING)
        for i, (xmon_cross, res_param) in enumerate(zip(self.xmons, res_pars_list)):
            L1 = res_param[0]
            L_coupling = res_param[2]
            fork_y_span = res_param[3]
            tail_segment_lengths = res_param[4]
            tail_turn_angles = res_param[5]
            tail_trans_in = res_param[6]
            L0 = res_param[7]
            n_coils = res_param[8]
            xmon_dy_Cg_coupling = res_param[9]

            origin = DPoint(0, 0)
            res = EMResonatorTL3QbitWormRLTailXmonFork(
                resonator_cpw, origin, L_coupling, L0, L1, self.r, n_coils,
                tail_shape=res_tail_shape, tail_turn_radiuses=tail_turn_radiuses,
                tail_segment_lengths=tail_segment_lengths,
                tail_turn_angles=tail_turn_angles, tail_trans_in=tail_trans_in,
                fork_x_span=fork_x_span + 2 * FABRICATION.OVERETCHING, fork_y_span=fork_y_span,
                fork_metal_width=self.fork_metal_width + 2 * FABRICATION.OVERETCHING,
                fork_gnd_gap=self.fork_gnd_gap - 2 * FABRICATION.OVERETCHING
            )

            # for bottom crosses
            if i == 0 or i == 1:
                res_dx = xmon_cross.center.x + (
                        (res.fork_x_cpw.start.x + res.fork_x_cpw.end.x) / 2 - origin.x
                )
                res_dy = xmon_cross.center.y + (
                        (res.fork_x_cpw.start.y + res.fork_x_cpw.end.y) / 2 - res.fork_metal_width / 2
                        - origin.y - xmon_dy_Cg_coupling -
                        (self.cross_len_y + self.cross_width_x / 2 +
                         min(self.cross_gnd_gap_y, self.xmon_fork_gnd_gap)) - FABRICATION.OVERETCHING
                )
                res.make_trans(DCplxTrans(1, 180, False, res_dx, res_dy))
                res.place(self.region_ph)
            # for top crosses
            elif i == 2 or i == 3:
                res_dx = xmon_cross.center.x - (
                        (res.fork_x_cpw.start.x + res.fork_x_cpw.end.x) / 2 - origin.x
                )
                res_dy = xmon_cross.center.y - (
                        (res.fork_x_cpw.start.y + res.fork_x_cpw.end.y) / 2 - res.fork_metal_width / 2
                        - origin.y - xmon_dy_Cg_coupling -
                        (self.cross_len_y + self.cross_width_x / 2 +
                         min(self.cross_gnd_gap_y, self.xmon_fork_gnd_gap)) - FABRICATION.OVERETCHING
                )
                res.make_trans(DCplxTrans(1, 0, False, res_dx, res_dy))
                res.place(self.region_ph)
            self.resonators[i] = res

            # correct xmon-cross metal erased by resonator placing
            xmon_cross_metal = XmonCross(
                xmon_cross.center,
                xmon_cross.sideX_length, xmon_cross.sideX_width, 0, 0,
                xmon_cross.sideY_length, xmon_cross.sideY_width, 0, 0
            )
            xmon_cross_metal.place(self.region_ph)
            self.region_ph.merge()

    def draw_md_and_flux_lines(self):
        """
        Drawing of md (microwave drive) and flux tuning lines for 5 qubits
        Returns
        -------

        """
        # TODO: MDriveLineEnd - working bad with overetching
        md_fl_together_l = 200e3
        md_end_l = md_fl_together_l / 2
        md_cross_shift_transversal = 60e3
        md_cross_shift_directed = 38e3
        md_fl_turn_radius = 100e3

        z_md_fl_corrected = CPWParameters(
            self.z_md_fl.width + 2 * FABRICATION.OVERETCHING,
            self.z_md_fl.gap - 2 * FABRICATION.OVERETCHING
        )

        # flux line 1
        fl1_diag = (
                (self.xmons[0].cpw_lempt.end - DPoint(md_fl_together_l, 0)) -
                (self.contact_pads[2].end + DPoint(md_fl_together_l, 0))
        )
        fl1_diag_len = fl1_diag.abs()
        fl1_diag_angle = atan2(fl1_diag.y, fl1_diag.x)
        self.cpwrl_fl1.append(
            CPWRLPath(
                self.contact_pads[2].end, "LRLRL", z_md_fl_corrected,
                [md_fl_turn_radius] * 2,
                [
                    md_fl_together_l,
                    fl1_diag_len,
                    md_fl_together_l
                ],
                [fl1_diag_angle, -fl1_diag_angle],
            )
        )
        self.cpwrl_fl1[-1].place(self.region_ph)

        self.cpwrl_fl1_end = FluxLineEnd(self.cpwrl_fl1[-1].end, z_md_fl_corrected, z_md_fl_corrected.width, Trans.R270)
        self.cpwrl_fl1_end.place(self.region_ph)

        # microwave drive line 1
        md1_diag = (
                (self.xmons[0].cpw_lempt.end - DPoint(md_fl_together_l, md_cross_shift_transversal)) -
                (self.contact_pads[3].end + DPoint(0, md_fl_together_l))
        )
        md1_diag_len = md1_diag.abs()
        md1_diag_angle = atan2(md1_diag.y, md1_diag.x)
        self.cpwrl_md1.append(
            CPWRLPath(
                self.contact_pads[3].end, "LRLRL", z_md_fl_corrected,
                [md_fl_turn_radius] * 2,
                [
                    md_fl_together_l,
                    md1_diag_len,
                    md_fl_together_l - md_end_l
                ],
                [-(pi / 2 - md1_diag_angle), -md1_diag_angle],
                Trans.R90
            )
        )
        self.cpwrl_md1[-1].place(self.region_ph)

        self.cpwrl_md1_end = MDriveLineEnd(
            list(self.cpwrl_md1[-1].primitives.values())[-1], z_md_fl_corrected, md_end_l / 4,
                                                                                 md_end_l - md_end_l / 4 - md_cross_shift_directed
        )
        self.cpwrl_md1_end.place(self.region_ph)

        """
        Control line pairs 2 and 4 are intersecting with the readout waveguide.
        Intersections are close enough and equally spaced along vertical axis.
        """
        intersections_dy = 300e3
        # terminals along vertical line. To the left from readout line
        # Indexing starts from design's top.
        t2_left = self.xmons[3].cpw_rempt.end + DPoint(self.resonators_dx, 0)
        t1_left = t2_left + DPoint(0, intersections_dy)
        t3_left = self.xmons[1].cpw_rempt.end + DPoint(self.resonators_dx, 0)
        t4_left = t3_left + DPoint(0, -intersections_dy)

        # terminals along vertical line. To the right from readout line
        # Indexing starts from design's top.
        t1_right = t1_left + DPoint(2 / 3 * self.resonators_dx, 0)
        t2_right = t2_left + DPoint(2 / 3 * self.resonators_dx, 0)
        t3_right = t3_left + DPoint(2 / 3 * self.resonators_dx, 0)
        t4_right = t4_left + DPoint(2 / 3 * self.resonators_dx, 0)

        # flux line 2
        self.cpwrl_fl2.append(
            CPW(start=t3_left, end=self.xmons[1].cpw_rempt.end, cpw_params=z_md_fl_corrected)
        )
        self.cpwrl_fl2[-1].place(self.region_ph)

        self.cpwrl_fl2_end = FluxLineEnd(self.cpwrl_fl2[-1].end, z_md_fl_corrected, z_md_fl_corrected.width, Trans.R90)
        self.cpwrl_fl2_end.place(self.region_ph)

        self.cpwrl_fl2.append(
            CPW(start=t3_left, end=t3_right, cpw_params=z_md_fl_corrected)
        )
        self.cpwrl_fl2[-1].place(self.region_ph)

        fl2_diag = (
                (t3_right + DPoint(md_fl_together_l, 0)) -
                (self.contact_pads[6].end - DPoint(md_fl_together_l, 0))
        )
        fl2_diag_len = fl2_diag.abs()
        fl2_diag_angle = atan2(fl2_diag.y, fl2_diag.x)
        self.cpwrl_fl2.append(
            CPWRLPath(
                self.contact_pads[6].end, "LRLRL", z_md_fl_corrected,
                [md_fl_turn_radius] * 2,
                [
                    md_fl_together_l,
                    fl2_diag_len,
                    md_fl_together_l
                ],
                [-(pi - fl2_diag_angle), (pi - fl2_diag_angle)],
                Trans.R180
            )
        )
        self.cpwrl_fl2[-1].place(self.region_ph)

        # microwave drive line 2
        md2_diag = (
                (self.xmons[1].cpw_rempt.end + DPoint(md_fl_together_l, -md_cross_shift_transversal)) -
                (t4_left - DPoint(md_fl_together_l, 0))
        )
        md2_diag_len = md2_diag.abs()
        md2_diag_angle = atan2(md2_diag.y, md2_diag.x)
        self.cpwrl_md2.append(
            CPWRLPath(
                t4_left, "LRLRL", z_md_fl_corrected,
                [md_fl_turn_radius] * 2,
                [
                    md_fl_together_l,
                    md2_diag_len,
                    md_fl_together_l - md_end_l
                ],
                [-(pi - md2_diag_angle), (pi - md2_diag_angle)],
                Trans.R180
            )
        )
        self.cpwrl_md2[-1].place(self.region_ph)

        self.cpwrl_md2_end = MDriveLineEnd(
            list(self.cpwrl_md2[-1].primitives.values())[-1], z_md_fl_corrected, md_end_l / 4,
                                                                                 md_end_l - md_end_l / 4 - md_cross_shift_directed
        )
        self.cpwrl_md2_end.place(self.region_ph)

        self.cpwrl_md2.append(
            CPW(start=t4_left, end=t4_right, cpw_params=z_md_fl_corrected)
        )
        self.cpwrl_md2[-1].place(self.region_ph)

        md2_diag = (
                (t4_right + DPoint(md_fl_together_l, 0)) -
                (self.contact_pads[5].end + DPoint(0, md_fl_together_l))
        )
        md2_diag_len = md2_diag.abs()
        md2_diag_angle = atan2(md2_diag.y, md2_diag.x)
        self.cpwrl_md2.append(
            CPWRLPath(
                self.contact_pads[5].end, "LRLRL", z_md_fl_corrected,
                [md_fl_turn_radius] * 2,
                [
                    md_fl_together_l,
                    md2_diag_len,
                    md_fl_together_l
                ],
                [md2_diag_angle - pi / 2, pi - md2_diag_angle],
                Trans.R90
            )
        )
        self.cpwrl_md2[-1].place(self.region_ph)

        # flux line 3
        fl3_diag = (
                (self.xmons[2].cpw_lempt.end - DPoint(md_fl_together_l, 0)) -
                (self.contact_pads[1].end + DPoint(md_fl_together_l, 0))
        )
        fl3_diag_len = fl3_diag.abs()
        fl3_diag_angle = atan2(fl3_diag.y, fl3_diag.x)
        self.cpwrl_fl3.append(
            CPWRLPath(
                self.contact_pads[1].end, "LRLRL", z_md_fl_corrected,
                [md_fl_turn_radius] * 2,
                [
                    md_fl_together_l,
                    fl3_diag_len,
                    md_fl_together_l
                ],
                [fl3_diag_angle, -fl3_diag_angle],
            )
        )
        self.cpwrl_fl3[-1].place(self.region_ph)

        self.cpwrl_fl3_end = FluxLineEnd(self.cpwrl_fl3[-1].end, z_md_fl_corrected, z_md_fl_corrected.width, Trans.R270)
        self.cpwrl_fl3_end.place(self.region_ph)

        # place microwave drive line 3
        md3_diag = (
                (self.xmons[2].cpw_lempt.end + DPoint(-md_fl_together_l, md_cross_shift_transversal)) -
                (self.contact_pads[0].end + DPoint(md_fl_together_l, 0))
        )
        md3_diag_len = md3_diag.abs()
        md3_diag_angle = atan2(md3_diag.y, md3_diag.x)
        self.cpwrl_md3.append(
            CPWRLPath(
                self.contact_pads[0].end, "LRLRL", z_md_fl_corrected,
                [md_fl_turn_radius] * 2,
                [
                    md_fl_together_l,
                    md3_diag_len,
                    md_fl_together_l - md_end_l
                ],
                [md3_diag_angle, -md3_diag_angle]
            )
        )
        self.cpwrl_md3[-1].place(self.region_ph)

        self.cpwrl_md3_end = MDriveLineEnd(
            list(self.cpwrl_md3[-1].primitives.values())[-1], z_md_fl_corrected, md_end_l / 4,
                                                                                 md_end_l - md_end_l / 4 - md_cross_shift_directed
        )
        self.cpwrl_md3_end.place(self.region_ph)

        # flux line 4
        self.cpwrl_fl4.append(
            CPW(start=t2_left, end=self.xmons[3].cpw_rempt.end, cpw_params=z_md_fl_corrected)
        )
        self.cpwrl_fl4[-1].place(self.region_ph)
        self.cpwrl_fl4_end = FluxLineEnd(self.cpwrl_fl4[-1].end, z_md_fl_corrected, z_md_fl_corrected.width, Trans.R90)
        self.cpwrl_fl4_end.place(self.region_ph)

        self.cpwrl_fl4.append(
            CPW(start=t2_left, end=t2_right, cpw_params=z_md_fl_corrected)
        )
        self.cpwrl_fl4[-1].place(self.region_ph)

        fl4_diag = (
                (t2_right + DPoint(md_fl_together_l, 0)) -
                (self.contact_pads[7].end - DPoint(md_fl_together_l, 0))
        )
        fl4_diag_len = fl4_diag.abs()
        fl4_diag_angle = atan2(fl4_diag.y, fl4_diag.x)
        self.cpwrl_fl4.append(
            CPWRLPath(
                self.contact_pads[7].end, "LRLRL", z_md_fl_corrected,
                [md_fl_turn_radius] * 2,
                [
                    md_fl_together_l,
                    fl4_diag_len,
                    md_fl_together_l
                ],
                [fl4_diag_angle - pi, -(fl4_diag_angle - pi)],
                Trans.R180
            )
        )
        self.cpwrl_fl4[-1].place(self.region_ph)

        # microwave drive line 4
        md4_diag = (
                (self.xmons[3].cpw_rempt.end + DPoint(md_fl_together_l, md_cross_shift_transversal)) -
                (t1_left - DPoint(md_fl_together_l, 0))
        )
        md4_diag_len = md4_diag.abs()
        md4_diag_angle = atan2(md4_diag.y, md4_diag.x)
        self.cpwrl_md4.append(
            CPWRLPath(
                t1_left, "LRLRL", z_md_fl_corrected,
                [md_fl_turn_radius] * 2,
                [
                    md_fl_together_l,
                    md4_diag_len,
                    md_fl_together_l - md_end_l
                ],
                [(md4_diag_angle + pi), -(md4_diag_angle + pi)],
                Trans.R180
            )
        )
        self.cpwrl_md4[-1].place(self.region_ph)

        self.cpwrl_md4_end = MDriveLineEnd(
            list(self.cpwrl_md4[-1].primitives.values())[-1], z_md_fl_corrected, md_end_l / 4,
                                                                                 md_end_l - md_end_l / 4 - md_cross_shift_directed
        )
        self.cpwrl_md4_end.place(self.region_ph)

        self.cpwrl_md4.append(
            CPW(start=t1_left, end=t1_right, cpw_params=z_md_fl_corrected)
        )
        self.cpwrl_md4[-1].place(self.region_ph)

        md4_diag = (
                (t1_right + DPoint(md_fl_together_l, 0)) -
                (self.contact_pads[8].end - DPoint(md_fl_together_l, 0))
        )
        md4_diag_len = md4_diag.abs()
        md4_diag_angle = atan2(md4_diag.y, md4_diag.x)
        self.cpwrl_md4.append(
            CPWRLPath(
                self.contact_pads[8].end, "LRLRL", z_md_fl_corrected,
                [md_fl_turn_radius] * 2,
                [
                    md_fl_together_l,
                    md4_diag_len,
                    md_fl_together_l
                ],
                [(md4_diag_angle + pi), -(md4_diag_angle + pi)],
                Trans.R180
            )
        )
        self.cpwrl_md4[-1].place(self.region_ph)

    def draw_readout_waveguide(self):
        '''
        Subdividing horizontal waveguide adjacent to resonators into several waveguides.
        Even segments of this adjacent waveguide are adjacent to resonators.
        Bridges will be placed on odd segments later.

        Returns
        -------
        None
        '''
        # place readout waveguide
        ro_line_turn_radius = self.ro_line_turn_radius
        ro_line_dy = self.ro_line_dy

        ## calculating segment lengths of subdivided coupling part of ro coplanar ##

        # value that need to be added to `L_coupling` to get width of resonators bbox.
        def get_res_extension(resonator: EMResonatorTL3QbitWormRLTailXmonFork):
            return resonator.Z0.b + 2 * resonator.r

        def get_res_width(resonator: EMResonatorTL3QbitWormRLTailXmonFork):
            return (resonator.L_coupling + get_res_extension(resonator))

        self.ro_line_segs_list = []
        dy1 = (self.resonators[0].connections[0] - self.contact_pads[4].end).y - self.to_line_list[0]
        dx1 = self.contact_pads[4].connections[0].x - \
              (self.resonators[0].connections[0].x - self.resonators[0].L_coupling -
               get_res_extension(self.resonators[0]) / 2)
        seg1 = CPWRLPath(
            self.contact_pads[4].end, "LRLRLRL", self.ro_Z,
            [self.ro_line_turn_radius] * 3,
            [dy1 / 2, 2 * dx1, dy1 / 2, dx1],
            [pi / 2, -pi / 2, -pi / 2],
            trans_in=Trans.R90
        )
        seg1.place(self.region_ph)
        self.ro_line_segs_list.append(seg1)

        seg2 = CPW(
            start=seg1.end, end=seg1.end + DPoint(get_res_width(self.resonators[0]), 0),
            cpw_params=self.ro_Z
        )
        seg2.place(self.region_ph)
        self.ro_line_segs_list.append(seg2)

        seg3 = CPW(
            start=seg2.end,
            end=seg2.end + DPoint(
                self.resonators_dx -
                (get_res_extension(self.resonators[0]) + get_res_extension(self.resonators[1])) / 2 -
                self.resonators[1].L_coupling,
                0
            ),
            cpw_params=self.ro_Z
        )
        seg3.place(self.region_ph)
        self.ro_line_segs_list.append(seg3)

        seg4 = CPW(
            start=seg3.end,
            end=seg3.end + DPoint(get_res_width(self.resonators[1]), 0),
            cpw_params=self.ro_Z
        )
        seg4.place(self.region_ph)
        self.ro_line_segs_list.append(seg4)

        seg5_dx1 = self.resonators_dx
        seg5_dy = self.resonators[3].connections[0].y - seg4.end.y + self.to_line_list[3]
        seg5_dx2 = seg5_dx1 - \
                   (
                           (self.resonators[3].connections[0].x + self.resonators[3].L_coupling +
                            get_res_extension(self.resonators[3]) / 2) -
                           (self.resonators[1].connections[0].x + get_res_extension(self.resonators[1]) / 2)
                   )

        ro_bridges_l = 50e3
        seg5 = CPWRLPath(
            seg4.end, "LRL", self.ro_Z,
            [self.ro_line_turn_radius],
            [seg5_dx1, self.cpwrl_md2[1].end.y - seg4.end.y - ro_bridges_l / 2],
            [pi / 2]
        )
        seg5.place(self.region_ph)
        self.ro_line_segs_list.append(seg5)

        def open_cpw_ends(cpw, sides="both"):
            if sides == "both":
                CPW(width=0, gap=cpw.b / 2, start=cpw.start, end=cpw.start + DPoint(0, cpw.gap)).place(self.region_ph)
                CPW(width=0, gap=cpw.b / 2, start=cpw.end, end=cpw.end - DPoint(0, cpw.gap)).place(self.region_ph)
            elif sides == "start":
                CPW(width=0, gap=cpw.b / 2, start=cpw.start, end=cpw.start + DPoint(0, cpw.gap)).place(self.region_ph)
            elif sides == "end":
                CPW(width=0, gap=cpw.b / 2, start=cpw.end, end=cpw.end - DPoint(0, cpw.gap)).place(self.region_ph)

        open_cpw_ends(list(seg5.primitives.values())[-1], sides="end")

        seg6 = CPW(
            start=seg5.end + DPoint(0, ro_bridges_l),
            end=seg5.end + DPoint(0, ro_bridges_l) +
                DPoint(0, self.cpwrl_fl2[1].end.y - self.cpwrl_md2[1].end.y - ro_bridges_l),
            cpw_params=self.ro_Z
        )
        seg6.place(self.region_ph)
        self.ro_line_segs_list.append(seg6)

        open_cpw_ends(seg6)

        bridge56 = Bridge1(
            (seg5.end + seg6.start)/2, gnd2gnd_dy=100e3
        )
        bridge56.place(self.region_bridges1, region_name="bridges_1")
        bridge56.place(self.region_bridges2, region_name="bridges_2")

        seg7 = CPW(
            start=seg6.end + DPoint(0, ro_bridges_l),
            end=seg6.end + DPoint(0, ro_bridges_l) +
                DPoint(0, self.cpwrl_fl4[1].end.y - self.cpwrl_fl2[1].end.y - ro_bridges_l),
            cpw_params=self.ro_Z
        )
        seg7.place(self.region_ph)
        self.ro_line_segs_list.append(seg7)

        open_cpw_ends(seg7)

        bridge67 = Bridge1(
            (seg6.end + seg7.start) / 2, gnd2gnd_dy=100e3
        )
        bridge67.place(self.region_bridges1, region_name="bridges_1")
        bridge67.place(self.region_bridges2, region_name="bridges_2")

        seg8 = CPW(
            start=seg7.end + DPoint(0, ro_bridges_l),
            end=seg7.end + DPoint(0, ro_bridges_l) +
                DPoint(0, self.cpwrl_md4[1].end.y - self.cpwrl_fl4[1].end.y - ro_bridges_l),
            cpw_params=self.ro_Z
        )
        seg8.place(self.region_ph)
        self.ro_line_segs_list.append(seg8)

        open_cpw_ends(seg8)

        bridge78 = Bridge1(
            (seg7.end + seg8.start) / 2, gnd2gnd_dy=100e3
        )
        bridge78.place(self.region_bridges1, region_name="bridges_1")
        bridge78.place(self.region_bridges2, region_name="bridges_2")

        seg9 = CPWRLPath(
            seg8.end + DPoint(0, ro_bridges_l), "LRL", self.ro_Z,
            [self.ro_line_turn_radius],
            [
                self.resonators[3].connections[0].y + self.to_line_list[3] - seg8.end.y - ro_bridges_l,
                seg5_dx2
            ],
            [pi / 2],
            Trans.R90
        )
        seg9.place(self.region_ph)
        self.ro_line_segs_list.append(seg9)

        open_cpw_ends(list(seg9.primitives.values())[-1], sides="start")

        bridge89 = Bridge1(
            (seg8.end + seg9.start) / 2, gnd2gnd_dy=100e3
        )
        bridge89.place(self.region_bridges1, region_name="bridges_1")
        bridge89.place(self.region_bridges2, region_name="bridges_2")

        seg10 = CPW(
            start=seg9.end,
            end=seg9.end + DPoint(-get_res_width(self.resonators[3]), 0),
            cpw_params=self.ro_Z
        )
        seg10.place(self.region_ph)
        self.ro_line_segs_list.append(seg10)

        seg11 = CPW(
            start=seg10.end,
            end=seg10.end + DPoint(
                get_res_extension(self.resonators[3]) / 2 -
                abs(self.resonators[2].connections[0].x - self.resonators[3].connections[0].x) +
                self.resonators[2].L_coupling + get_res_extension(self.resonators[2]) / 2,
                0
            ),
            cpw_params=self.ro_Z
        )
        seg11.place(self.region_ph)
        self.ro_line_segs_list.append(seg11)

        seg12 = CPW(
            start=seg11.end,
            end=seg11.end + DPoint(-get_res_width(self.resonators[2]), 0),
            cpw_params=self.ro_Z
        )
        seg12.place(self.region_ph)
        self.ro_line_segs_list.append(seg12)

        dy9 = (self.contact_pads[10].end - self.resonators[2].connections[0]).y - self.to_line_list[2]
        dx9 = self.contact_pads[10].connections[0].x - \
              (self.resonators[2].connections[0].x -
               get_res_extension(self.resonators[2]) / 2)
        seg13 = CPWRLPath(
            seg12.end, "LRLRLRL", self.ro_Z,
            [self.ro_line_turn_radius] * 3, [dx9, dy9 / 2, 2 * dx9, dy9 / 2],
            [-pi / 2, -pi / 2, pi / 2],
            Trans.R180
        )
        seg13.place(self.region_ph)
        self.ro_line_segs_list.append(seg13)

    def draw_test_structures(self):
        new_pars_squid = AsymSquidParams(
            pad_r=5e3, pads_distance=30e3,
            p_ext_width=10e3, p_ext_r=200,
            sq_len=15e3, sq_area=200e6,
            j_width_1=94, j_width_2=347,
            intermediate_width=500, b_ext=1e3, j_length=94, n=20,
            bridge=180, j_length_2=250
        )

        struct_centers = [DPoint(1e6, 4e6), DPoint(8.7e6, 6.2e6), DPoint(6.2e6, 1.5e6)]
        for struct_center in struct_centers:
            ## JJ test structures ##
            # test structure with big critical current
            test_struct1 = TestStructurePads(struct_center)
            test_struct1.place(self.region_ph)
            text_reg = pya.TextGenerator.default_generator().text("48.32 nA", 0.001, 50, False, 0, 0)
            text_bl = test_struct1.empty_rectangle.origin + DPoint(
                test_struct1.gnd_gap, -4 * test_struct1.gnd_gap
            )
            text_reg.transform(ICplxTrans(1.0, 0, False, text_bl.x, text_bl.y))
            self.region_ph -= text_reg
            test_jj = AsymSquid(test_struct1.center, new_pars_squid, side=1)
            self.test_squids.append(test_jj)
            test_jj.place(self.region_el)

            # test structure with low critical current
            test_struct2 = TestStructurePads(struct_center + DPoint(0.3e6, 0))
            test_struct2.place(self.region_ph)
            text_reg = pya.TextGenerator.default_generator().text("9.66 nA", 0.001, 50, False, 0, 0)
            text_bl = test_struct2.empty_rectangle.origin + DPoint(
                test_struct2.gnd_gap, -4 * test_struct2.gnd_gap
            )
            text_reg.transform(ICplxTrans(1.0, 0, False, text_bl.x, text_bl.y))
            self.region_ph -= text_reg
            test_jj = AsymSquid(test_struct2.center, new_pars_squid, side=-1)
            self.test_squids.append(test_jj)
            test_jj.place(self.region_el)

            # test structure for bridge DC contact
            test_struct3 = TestStructurePads(struct_center + DPoint(0.6e6, 0))
            test_struct3.place(self.region_ph)
            text_reg = pya.TextGenerator.default_generator().text("DC", 0.001, 50, False, 0, 0)
            text_bl = test_struct3.empty_rectangle.origin + DPoint(
                test_struct3.gnd_gap, -4 * test_struct3.gnd_gap
            )
            text_reg.transform(ICplxTrans(1.0, 0, False, test_struct3.center.x, text_bl.y))
            self.region_ph -= text_reg

            test_bridges = []
            for i in range(3):
                bridge = Bridge1(test_struct3.center + DPoint(50e3 * (i - 1), 0),
                                 gnd_touch_dx=20e3)
                test_bridges.append(bridge)
                bridge.place(self.region_bridges1, region_name="bridges_1")
                bridge.place(self.region_bridges2, region_name="bridges_2")

        # bandages test structures
        test_dc_el2_centers = [
            DPoint(2.5e6, 2.4e6),
            DPoint(3.2e6, 7.3e6),
            DPoint(9.0e6, 3.8e6)
        ]
        for struct_center in test_dc_el2_centers:
            test_struct1 = TestStructurePads(struct_center)
            test_struct1.place(self.region_ph)
            text_reg = pya.TextGenerator.default_generator().text("Bandage", 0.001, 40, False, 0, 0)
            text_bl = test_struct1.empty_rectangle.origin + DPoint(
                test_struct1.gnd_gap, -4 * test_struct1.gnd_gap
            )
            text_reg.transform(ICplxTrans(1.0, 0, False, text_bl.x, text_bl.y))
            self.region_ph -= text_reg

            rec_width = 10e3
            rec_height = test_struct1.rectangles_gap + 2 * FABRICATION.OVERETCHING + 2 * rec_width
            p1 = struct_center - DVector(rec_width / 2, rec_height / 2)
            dc_rec = Rectangle(p1, rec_width, rec_height)
            dc_rec.place(self.region_el2)

    def draw_el_dc_contacts(self):
        for squid in (self.squids + self.test_squids):
            squid_dir = squid.p_ext_up.start - squid.p_ext_down.start
            squid_dir /= squid_dir.abs()
            squid_dir_angle_degree = 180 * atan2(squid_dir.y, squid_dir.x) / pi - 90
            # turn squid dir to 90 degrees
            left_orth_squid_dir = DCplxTrans(1, 90, False, 0, 0) * squid_dir

            # place DC contact pads
            dc_rec_delta = 1.2e3
            dWidth = -0.4e3
            rec_width = squid.params.p_ext_width + dWidth
            dHeight = 1.2 * (squid.p_ext_up.start - squid.p_ext_up.end).abs()
            rec_height = (squid.p_ext_up.start - squid.p_ext_up.end).abs() + dHeight
            rec_top_origin = squid.p_ext_up.end + dc_rec_delta * squid_dir + (
                        squid.params.p_ext_width / 2 + dWidth / 2) * left_orth_squid_dir
            rec_bot_origin = squid.p_ext_down.start - (dc_rec_delta + dHeight) * squid_dir + (
                        squid.params.p_ext_width / 2 + dWidth / 2) * left_orth_squid_dir
            self.el_dc_contacts = [
                # top DC
                Rectangle(
                    rec_top_origin, rec_width, rec_height,
                    trans_in=DCplxTrans(1, squid_dir_angle_degree, False, 0, 0)
                ),
                Rectangle(
                    rec_bot_origin, rec_width, rec_height,
                    trans_in=DCplxTrans(1, squid_dir_angle_degree, False, 0, 0)
                )
            ]
            for rec in self.el_dc_contacts:
                rec.place(self.region_el2)

            # cutoff rectangles to enlarge perimeter of transition:
            # photolygraphy metal - squid substrate metal
            cut_rec_width = 2 / 3 * squid.params.p_ext_width
            dWidth = cut_rec_width - squid.params.p_ext_width
            cut_rec_height = (squid.p_ext_up.start - squid.p_ext_up.end).abs()
            cut_rec_top_origin = squid.p_ext_up.start + \
                                 (squid.params.p_ext_width / 2 + dWidth / 2) * left_orth_squid_dir
            cut_rec_bot_origin = squid.p_ext_down.start + \
                                 (squid.params.p_ext_width / 2 + dWidth / 2) * left_orth_squid_dir
            self.cutoff_rectangles = [
                Rectangle(
                    cut_rec_top_origin, cut_rec_width, cut_rec_height,
                    trans_in=DCplxTrans(1, squid_dir_angle_degree, True, 0, 0),
                    inverse=True
                ),
                Rectangle(
                    cut_rec_bot_origin, cut_rec_width, cut_rec_height,
                    trans_in=DCplxTrans(1, squid_dir_angle_degree, False, 0, 0),
                    inverse=True
                )
            ]
            for cut_rec in self.cutoff_rectangles:
                cut_rec.place(self.region_ph)

    def draw_el_protection(self):
        protection_a = 300e3
        for squid in (self.squids + self.test_squids):
            self.region_el_protection.insert(
                pya.Box().from_dbox(
                    pya.DBox(
                        squid.origin - 0.5 * DVector(protection_a, protection_a),
                        squid.origin + 0.5 * DVector(protection_a, protection_a)
                    )
                )
            )

    def draw_photo_el_marks(self):
        marks_centers = [
            DPoint(1e6, 9e6), DPoint(1e6, 1e6),
            DPoint(9e6, 1e6), DPoint(9e6, 9e6),
            DPoint(6e6, 8e6), DPoint(1e6, 6e6)
        ]
        for mark_center in marks_centers:
            self.marks.append(
                MarkBolgar(mark_center, overetching=FABRICATION.OVERETCHING)
            )
            self.marks[-1].place(self.region_ph)

    def draw_bridges(self):
        bridges_step = 150e3

        # for resonators
        for resonator in self.resonators:
            for name, res_primitive in resonator.primitives.items():
                if "coil0" in name:
                    # skip L_coupling coplanar.
                    # bridgyfy all in "coil0" except for the first cpw that
                    # is adjacent to readout line and has length equal to `L_coupling`
                    for primitive in list(res_primitive.primitives.values())[1:]:
                        Bridge1.bridgify_CPW(
                            primitive, bridges_step,
                            dest=self.region_bridges1, dest2=self.region_bridges2
                        )

                    continue
                elif "fork" in name:  # skip fork primitives
                    continue
                else:  # bridgify everything else
                    Bridge1.bridgify_CPW(
                        res_primitive, bridges_step,
                        dest=self.region_bridges1, dest2=self.region_bridges2
                    )

        # for contact wires
        # avoid readout bridges
        ro_bridges_pts = self.ro_line_segs_list
        for key, val in self.__dict__.items():
            if "cpwrl_md" in key:
                if isinstance(val, list):
                    for cpw in val:
                        Bridge1.bridgify_CPW(
                            cpw, bridges_step,
                            dest=self.region_bridges1, dest2=self.region_bridges2,
                            avoid_points=[squid.origin for squid in self.squids],
                            avoid_distance=200e3
                        )
                elif isinstance(val, (CPW, CPWArc, CPWRLPath)):
                    cpw = val
                    Bridge1.bridgify_CPW(
                        cpw, bridges_step,
                        dest=self.region_bridges1, dest2=self.region_bridges2,
                        avoid_points=[squid.origin for squid in self.squids],
                        avoid_distance=200e3
                    )
            elif "cpwrl_fl" in key:
                if isinstance(val, list):
                    for cpw in val:
                        Bridge1.bridgify_CPW(
                            cpw, bridges_step,
                            dest=self.region_bridges1, dest2=self.region_bridges2,
                            avoid_points=[squid.origin for squid in self.squids],
                            avoid_distance=500e3
                        )
                elif isinstance(val, (CPW, CPWArc, CPWRLPath)):
                    cpw = val
                    Bridge1.bridgify_CPW(
                        cpw, bridges_step,
                        dest=self.region_bridges1, dest2=self.region_bridges2,
                        avoid_points=[squid.origin for squid in self.squids],
                        avoid_distance=500e3
                    )
        # for readout waveguide
        bridgified_primitives_idxs = [0, 2, 4, 5, 6, 7, 8, 10, 12]
        for idx, cpw in enumerate(self.ro_line_segs_list):
            if idx in bridgified_primitives_idxs:
                Bridge1.bridgify_CPW(
                    cpw, bridges_step,
                    dest=self.region_bridges1, dest2=self.region_bridges2
                )

    def draw_pinning_holes(self):
        selection_region = Region(
            pya.Box(Point(100e3, 100e3), Point(101e3, 101e3))
        )
        tmp_ph = self.region_ph.dup()
        other_regs = tmp_ph.select_not_interacting(selection_region)
        reg_to_fill = self.region_ph.select_interacting(selection_region)
        filled_reg = fill_holes(reg_to_fill)

        self.region_ph = filled_reg + other_regs

    def split_polygons_in_layers(self, max_pts=200):
        self.region_ph = split_polygons(self.region_ph, max_pts)
        self.region_bridges2 = split_polygons(self.region_bridges2, max_pts)
        for poly in self.region_ph:
            if poly.num_points() > max_pts:
                print("exists photo")
        for poly in self.region_ph:
            if poly.num_points() > max_pts:
                print("exists bridge2")


if __name__ == "__main__":
    design = Design4QSquare("testScript")
    design.draw()
    design.show()
