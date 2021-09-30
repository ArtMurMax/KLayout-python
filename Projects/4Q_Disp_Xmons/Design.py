__version__ = "0.3.0.1"

# Enter your Python code here
from math import cos, sin, tan, atan2, pi, degrees
import itertools
from typing import List, Dict, Union, Optional

import pya
from pya import Cell
from pya import Point, Vector, DPoint, DVector, DSimplePolygon, \
    SimplePolygon, DPolygon, Polygon, Region
from pya import Trans, DTrans, CplxTrans, DCplxTrans, ICplxTrans, DPath

from importlib import reload
import classLib

reload(classLib)

from classLib.baseClasses import ElementBase, ComplexBase
from classLib.coplanars import CPWParameters, CPW, DPathCPW, \
    CPWRLPath, Bridge1
from classLib.shapes import XmonCross, Rectangle
from classLib.resonators import EMResonatorTL3QbitWormRLTailXmonFork
from classLib.josJ import AsymSquidDCFluxParams, AsymSquidDCFlux, AsymSquid
from classLib.chipTemplates import CHIP_10x10_12pads, FABRICATION
from classLib.chipDesign import ChipDesign
from classLib.marks import MarkBolgar
from classLib.contactPads import ContactPad
from classLib.helpers import fill_holes, split_polygons

import copy

# 0.0 - for development
# 0.8e3 - estimation for fabrication by Bolgar photolytography etching
# recipe
FABRICATION.OVERETCHING = 0.8e3
SQUID_PARAMETERS = AsymSquidDCFluxParams(
    pad_r=5e3, pads_distance=30e3,
    contact_pad_width=10e3, contact_pad_ext_r=200,
    sq_len=15e3, sq_area=200e6,
    j1_dx=95, j2_dx=348,
    inter_leads_width=500, b_ext=2e3, j1_dy=94, n=20,
    bridge=180, j2_dy=250,
    flux_line_dx=30e3, flux_line_dy=10e3, flux_line_outer_width=0.6e3,
    # 4e3
    flux_line_contact_width=5e3,
    flux_line_inner_width=370, flux_line_IO_transition_L=100
)


class TestStructurePads(ComplexBase):
    def __init__(self, center, trans_in=None):
        self.center = center
        self.rectangle_a = 200e3
        self.gnd_gap = 20e3
        self.rectangles_gap = 20e3

        self.empty_rectangle: Rectangle = None
        self.top_rec: Rectangle = None
        self.bot_rec: Rectangle = None
        super().__init__(center, trans_in)

    def init_primitives(self):
        center = DPoint(0, 0)

        ## empty rectangle ##
        empty_width = self.rectangle_a + 2 * self.gnd_gap
        empty_height = 2 * self.rectangle_a + 2 * self.gnd_gap + \
                       self.rectangles_gap
        # bottom-left point of rectangle
        bl_point = center - DPoint(empty_width / 2, empty_height / 2)
        self.empty_rectangle = Rectangle(
            bl_point,
            empty_width, empty_height, inverse=True
        )
        self.primitives["empty_rectangle"] = self.empty_rectangle

        ## top rectangle ##
        # bottom-left point of rectangle
        bl_point = center + DPoint(-self.rectangle_a / 2,
                                   self.rectangles_gap / 2)
        self.top_rec = Rectangle(
            bl_point, self.rectangle_a, self.rectangle_a
        )
        self.primitives["top_rec"] = self.top_rec

        ## bottom rectangle ##
        # bottom-left point of rectangle
        bl_point = center + DPoint(
            -self.rectangle_a / 2,
            - self.rectangles_gap / 2 - self.rectangle_a
        )
        self.bot_rec = Rectangle(
            bl_point, self.rectangle_a, self.rectangle_a
        )
        self.primitives["bot_rec"] = self.bot_rec

        self.connections = [center]

    def _refresh_named_connections(self):
        self.center = self.connections[0]


class Design5Q(ChipDesign):
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

        # has to call it once more to add new layers
        self.lv.add_missing_layers()

        ### ADDITIONAL VARIABLES SECTION START ###
        # chip rectangle and contact pads
        self.chip = CHIP_10x10_12pads
        self.chip_box: pya.DBox = self.chip.box
        # Z = 50.09 E_eff = 6.235 (E = 11.45)
        self.z_md_fl: CPWParameters = CPWParameters(11e3, 5.7e3)
        self.ro_Z: CPWParameters = self.chip.chip_Z
        self.contact_pads: list[ContactPad] = self.chip.get_contact_pads(
            [self.z_md_fl] * 10 + [self.ro_Z] * 2
        )

        # readout line parameters
        self.ro_line_turn_radius: float = 200e3
        self.ro_line_dy: float = 1600e3
        self.cpwrl_ro_line: CPWRLPath = None
        self.Z0: CPWParameters = CHIP_10x10_12pads.chip_Z

        # resonators objects list
        self.resonators: List[EMResonatorTL3QbitWormRLTailXmonFork] = []
        # distance between nearest resonators central conductors centers
        # constant step between resonators origin points along x-axis.
        self.resonators_dx: float = 900e3
        # resonator parameters
        self.L_coupling_list: list[float] = [
            1e3 * x for x in [310, 320, 320, 310, 300]
        ]
        # corresponding to resonanse freq is linspaced in interval [6,9) GHz
        self.L0 = 1150e3
        self.L1_list = [
            1e3 * x for x in [50.7218, 96.3339, 138.001, 142.77, 84.9156]
        ]
        self.r = 60e3
        self.N_coils = [3] * len(self.L1_list)
        self.L2_list = [self.r] * len(self.L1_list)
        self.L3_list = [0e3] * len(self.L1_list)  # to be constructed
        self.L4_list = [self.r] * len(self.L1_list)
        self.width_res = 20e3
        self.gap_res = 10e3
        self.Z_res = CPWParameters(self.width_res, self.gap_res)
        self.to_line_list = [56e3] * len(self.L1_list)
        self.fork_metal_width = 10e3
        self.fork_gnd_gap = 15e3
        self.xmon_fork_gnd_gap = 14e3
        # resonator-fork parameters
        # for coarse C_qr evaluation
        self.fork_y_spans = [
            x * 1e3 for x in [8.73781, 78.3046, 26.2982, 84.8277, 35.3751]
        ]

        # xmon parameters
        self.xmon_x_distance: float = 545e3  # from simulation of g_12
        # for fine C_qr evaluation
        self.xmon_dys_Cg_coupling = [14e3] * 5
        self.xmons: list[XmonCross] = []

        self.cross_len_x = 180e3
        self.cross_width_x = 60e3
        self.cross_gnd_gap_x = 20e3
        self.cross_len_y = 155e3
        self.cross_width_y = 60e3
        self.cross_gnd_gap_y = 20e3

        # fork at the end of resonator parameters
        self.fork_x_span = self.cross_width_y + 2 * (
                self.xmon_fork_gnd_gap + self.fork_metal_width)

        # squids
        self.squids: List[AsymSquidDCFlux] = []
        self.test_squids: List[AsymSquidDCFlux] = []

        # el-dc concacts attributes
        # required extension of el_dc contacts into the el polygon
        self.dc_cont_el_ext = 0.0e3  # 1.8e3
        # required extension into the photo polygon
        self.dc_cont_ph_ext = 1.4e3
        # minimum distance from el_dc contact polygon perimeter points to
        # dc + el polygons perimeter. (el-dc contact polygon lies within
        # el + dc polygons)
        self.dc_cont_clearance = 1e3  # [nm] = [m*1e-9]
        # el-photo layer cut box width
        self.dc_cont_cut_width = \
            SQUID_PARAMETERS.flux_line_contact_width - 2e3
        # cut box clearance from initial el ^ ph regions
        self.dc_cont_cut_clearance = 1e3
        self.el_dc_contacts: List[List[ElementBase, ...]] = []

        # microwave and flux drive lines parameters
        self.ctr_lines_turn_radius = 100e3
        self.cont_lines_y_ref: float = None  # nm

        self.md234_cross_bottom_dy = 55e3
        self.md234_cross_bottom_dx = 60e3

        self.cpwrl_md1: DPathCPW = None
        self.cpwrl_fl1: DPathCPW = None

        self.cpwrl_md2: DPathCPW = None
        self.cpwrl_fl2: DPathCPW = None

        self.cpwrl_md3: DPathCPW = None
        self.cpwrl_fl3: DPathCPW = None

        self.cpwrl_md4: DPathCPW = None
        self.cpwrl_fl4: DPathCPW = None

        self.cpwrl_md5: DPathCPW = None
        self.cpwrl_fl5: DPathCPW = None

        self.cpw_fl_lines: List[DPathCPW] = []
        self.cpw_md_lines: List[DPathCPW] = []

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
        self.create_resonator_objects()
        self.draw_readout_waveguide()
        self.draw_xmons_and_resonators()

        self.draw_josephson_loops()

        self.draw_microwave_drvie_lines()
        self.draw_flux_control_lines()

        self.draw_test_structures()
        self.draw_el_dc_contacts()
        self.draw_el_protection()

        self.draw_photo_el_marks()
        self.draw_bridges()
        self.draw_pinning_holes()
        self.extend_photo_overetching()
        self.inverse_destination(self.region_ph)
        self.split_polygons_in_layers(max_pts=180)

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
        self.cell.shapes(self.layer_el_protection).insert(
            self.region_el_protection)
        self.lv.zoom_fit()

    def draw_chip(self):
        self.region_bridges2.insert(self.chip_box)

        self.region_ph.insert(self.chip_box)
        for contact_pad in self.contact_pads:
            contact_pad.place(self.region_ph)

    def create_resonator_objects(self):
        ### RESONATORS TAILS CALCULATIONS SECTION START ###
        # key to the calculations can be found in hand-written format here:
        # https://drive.google.com/file/d/1wFmv5YmHAMTqYyeGfiqz79a9kL1MtZHu/view?usp=sharing

        # x span between left long vertical line and
        # right-most center of central conductors
        resonators_widths = [2 * self.r + L_coupling for L_coupling in
                             self.L_coupling_list]
        x1 = 2 * self.resonators_dx + resonators_widths[
            2] / 2 - 2 * self.xmon_x_distance
        x2 = x1 + self.xmon_x_distance - self.resonators_dx
        x3 = resonators_widths[2] / 2
        x4 = 3 * self.resonators_dx - (x1 + 3 * self.xmon_x_distance)
        x5 = 4 * self.resonators_dx - (x1 + 4 * self.xmon_x_distance)

        res_tail_shape = "LRLRL"
        tail_turn_radiuses = self.r
        # list corrected for resonator-qubit coupling geomtry, so all transmons centers are placed
        # along single horizontal line
        self.L0_list = [self.L0 - xmon_dy_Cg_coupling for
                        xmon_dy_Cg_coupling in self.xmon_dys_Cg_coupling]
        self.L2_list[0] += 6 * self.Z_res.b
        self.L2_list[1] += 0
        self.L2_list[3] += 3 * self.Z_res.b
        self.L2_list[4] += 6 * self.Z_res.b

        self.L3_list[0] = x1
        self.L3_list[1] = x2
        self.L3_list[2] = x3
        self.L3_list[3] = x4
        self.L3_list[4] = x5

        self.L4_list[1] += 6 * self.Z_res.b
        self.L4_list[2] += 6 * self.Z_res.b
        self.L4_list[3] += 3 * self.Z_res.b
        tail_segment_lengths_list = [[L2, L3, L4]
                                     for L2, L3, L4 in
                                     zip(self.L2_list, self.L3_list,
                                         self.L4_list)]
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

        pars = list(
            zip(
                self.L1_list, self.to_line_list, self.L_coupling_list,
                self.fork_y_spans,
                tail_segment_lengths_list, tail_turn_angles_list,
                tail_trans_in_list,
                self.L0_list, self.N_coils
            )
        )
        for res_idx, params in enumerate(pars):
            # parameters exctraction
            L1 = params[0]
            to_line = params[1]
            L_coupling = params[2]
            fork_y_span = params[3]
            tail_segment_lengths = params[4]
            tail_turn_angles = params[5]
            tail_trans_in = params[6]
            L0 = params[7]
            n_coils = params[8]

            # deduction for resonator placements
            # under condition that Xmon-Xmon distance equals
            # `xmon_x_distance`
            worm_x = self.contact_pads[-1].end.x + (
                    res_idx + 1 / 2) * self.resonators_dx
            worm_y = self.contact_pads[
                         -1].end.y - self.ro_line_dy - to_line

            self.resonators.append(
                EMResonatorTL3QbitWormRLTailXmonFork(
                    self.Z_res, DPoint(worm_x, worm_y), L_coupling,
                    L0,
                    L1, self.r, n_coils,
                    tail_shape=res_tail_shape,
                    tail_turn_radiuses=tail_turn_radiuses,
                    tail_segment_lengths=tail_segment_lengths,
                    tail_turn_angles=tail_turn_angles,
                    tail_trans_in=tail_trans_in,
                    fork_x_span=self.fork_x_span,
                    fork_y_span=fork_y_span,
                    fork_metal_width=self.fork_metal_width,
                    fork_gnd_gap=self.fork_gnd_gap
                )
            )
            self.resonators[-1].place(self.region_ph)
        # print([self.L0 - xmon_dy_Cg_coupling for xmon_dy_Cg_coupling in  self.xmon_dys_Cg_coupling])
        # print(self.L1_list)
        # print(self.L2_list)
        # print(self.L3_list)
        # print(self.L4_list)

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
        def get_res_extension(
                resonator: EMResonatorTL3QbitWormRLTailXmonFork):
            return resonator.Z0.b + 2 * resonator.r

        def get_res_width(resonator: EMResonatorTL3QbitWormRLTailXmonFork):
            return (resonator.L_coupling + get_res_extension(resonator))

        res_line_segments_lengths = [
            self.resonators[0].origin.x - self.contact_pads[-1].end.x
            - get_res_extension(self.resonators[0]) / 2
        ]  # length from bend to first bbox of first resonator
        for i, resonator in enumerate(self.resonators[:-1]):
            resonator_extension = get_res_extension(resonator)
            resonator_width = get_res_width(resonator)
            next_resonator_extension = get_res_extension(
                self.resonators[i + 1])
            # order of adding is from left to right (imagine chip geometry in your head to follow)
            res_line_segments_lengths.extend(
                [
                    resonator_width,
                    # `resonator_extension` accounts for the next resonator extension
                    # in this case all resonator's extensions are equal
                    self.resonators_dx - (
                            resonator_width - resonator_extension / 2) - next_resonator_extension / 2
                ]
            )
        res_line_segments_lengths.extend(
            [
                get_res_width(self.resonators[-1]),
                self.resonators_dx / 2
            ]
        )
        # first and last segment will have length `self.resonator_dx/2`
        res_line_total_length = sum(res_line_segments_lengths)
        segment_lengths = [ro_line_dy] + res_line_segments_lengths + \
                          [ro_line_dy / 2,
                           res_line_total_length - self.chip.pcb_feedline_d,
                           ro_line_dy / 2]

        self.cpwrl_ro_line = CPWRLPath(
            self.contact_pads[-1].end, shape="LR" + ''.join(
                ['L'] * len(res_line_segments_lengths)) + "RLRLRL",
            cpw_parameters=self.Z0,
            turn_radiuses=[ro_line_turn_radius] * 4,
            segment_lengths=segment_lengths,
            turn_angles=[pi / 2, pi / 2, pi / 2, -pi / 2],
            trans_in=Trans.R270
        )
        self.cpwrl_ro_line.place(self.region_ph)

    def draw_xmons_and_resonators(self):
        for resonator, fork_y_span, xmon_dy_Cg_coupling in \
                list(zip(
                    self.resonators,
                    self.fork_y_spans,
                    self.xmon_dys_Cg_coupling
                )):
            xmon_center = \
                (
                    resonator.fork_x_cpw.start + resonator.fork_x_cpw.end
                ) / 2 + \
                DVector(
                    0,
                    -xmon_dy_Cg_coupling - resonator.fork_metal_width / 2
                )
            # changes start #
            xmon_center += DPoint(
                0,
                -(
                        self.cross_len_y + self.cross_width_x / 2 +
                        self.cross_gnd_gap_y
                )
            )
            self.xmons.append(
                XmonCross(
                    xmon_center,
                    sideX_length=self.cross_len_x,
                    sideX_width=self.cross_width_x,
                    sideX_gnd_gap=self.cross_gnd_gap_x,
                    sideY_length=self.cross_len_y,
                    sideY_width=self.cross_width_y,
                    sideY_gnd_gap=self.cross_gnd_gap_y,
                    sideX_face_gnd_gap=self.cross_gnd_gap_x,
                    sideY_face_gnd_gap=self.cross_gnd_gap_y
                )
            )
            self.xmons[-1].place(self.region_ph)
            resonator.place(self.region_ph)
            xmonCross_corrected = XmonCross(
                xmon_center,
                sideX_length=self.cross_len_x,
                sideX_width=self.cross_width_x,
                sideX_gnd_gap=self.cross_gnd_gap_x,
                sideY_length=self.cross_len_y,
                sideY_width=self.cross_width_y,
                sideY_gnd_gap=max(
                    0,
                    self.fork_x_span - 2 * self.fork_metal_width -
                    self.cross_width_y -
                    max(self.cross_gnd_gap_y, self.fork_gnd_gap)
                ) / 2
            )
            xmonCross_corrected.place(self.region_ph)

    def draw_josephson_loops(self):
        # place left squid
        xmon0 = self.xmons[0]
        xmon0_xmon5_loop_shift = self.cross_len_x / 3
        center1 = DPoint(
            xmon0.cpw_l.end.x + xmon0_xmon5_loop_shift,
            xmon0.center.y - (xmon0.sideX_width + xmon0.sideX_gnd_gap) / 2
        )
        squid = AsymSquidDCFlux(center1, SQUID_PARAMETERS, 0)
        self.squids.append(squid)
        squid.place(self.region_el)

        # place intermediate squids
        for xmon_cross in self.xmons[1:-1]:
            squid_center = (
                xmon_cross.cpw_bempt.start + xmon_cross.cpw_bempt.end
            ) / 2
            squid = AsymSquidDCFlux(squid_center, SQUID_PARAMETERS, 0)
            self.squids.append(squid)
            squid.place(self.region_el)

        # place right squid
        xmon5 = self.xmons[4]
        center5 = DPoint(
            xmon5.cpw_r.end.x - xmon0_xmon5_loop_shift,
            xmon5.center.y - (xmon5.sideX_width + xmon5.sideX_gnd_gap) / 2
        )
        squid = AsymSquidDCFlux(center5, SQUID_PARAMETERS, 0)
        self.squids.append(squid)
        squid.place(self.region_el)

    def draw_microwave_drvie_lines(self):
        self.cont_lines_y_ref = self.xmons[0].cpw_bempt.end.y - 200e3

        tmp_reg = self.region_ph

        # place caplanar line 1md
        _p1 = self.contact_pads[0].end
        _p2 = _p1 + DPoint(1e6, 0)
        _p3 = self.xmons[0].cpw_l.end + DVector(-1e6, 0)
        _p4 = self.xmons[0].cpw_l.end + DVector(-66.2e3, 0)
        _p5 = _p4 + DVector(11.2e3, 0)
        self.cpwrl_md1 = DPathCPW(
            points=[_p1, _p2, _p3, _p4, _p5],
            cpw_parameters=[self.z_md_fl] * 5 + [
                CPWParameters(width=0, gap=self.z_md_fl.b / 2)
            ],
            turn_radiuses=self.ctr_lines_turn_radius
        )
        self.cpwrl_md1.place(tmp_reg)
        self.cpw_md_lines.append(self.cpwrl_md1)

        # place caplanar line 2md
        _p1 = self.contact_pads[3].end
        _p2 = _p1 + DPoint(0, 500e3)
        _p3 = DPoint(
            self.xmons[1].cpw_b.end.x + self.md234_cross_bottom_dx,
            self.cont_lines_y_ref
        )
        _p4 = DPoint(
            _p3.x,
            self.xmons[1].cpw_b.end.y - self.md234_cross_bottom_dy
        )
        _p5 = _p4 + DPoint(0, 10e3)
        self.cpwrl_md2 = DPathCPW(
            points=[_p1, _p2, _p3, _p4, _p5],
            cpw_parameters=[self.z_md_fl] * 5 + [
                CPWParameters(width=0, gap=self.z_md_fl.b / 2)
            ],
            turn_radiuses=self.ctr_lines_turn_radius
        )
        self.cpwrl_md2.place(tmp_reg)
        self.cpw_md_lines.append(self.cpwrl_md2)

        # place caplanar line 3md
        _p1 = self.contact_pads[5].end
        _p2 = _p1 + DPoint(0, 500e3)
        _p3 = _p2 + DPoint(-2e6, 2e6)
        _p5 = DPoint(
            self.xmons[2].cpw_b.end.x + self.md234_cross_bottom_dx,
            self.cont_lines_y_ref
        )
        _p4 = DPoint(_p5.x, _p5.y - 1e6)
        _p6 = DPoint(
            _p5.x,
            self.xmons[2].cpw_b.end.y - self.md234_cross_bottom_dy
        )
        _p7 = _p6 + DPoint(0, 10e3)
        self.cpwrl_md3 = DPathCPW(
            points=[_p1, _p2, _p3, _p4, _p5, _p6, _p7],
            cpw_parameters=[self.z_md_fl] * 8 + [
                CPWParameters(width=0, gap=self.z_md_fl.b / 2)
            ],
            turn_radiuses=self.ctr_lines_turn_radius
        )
        self.cpwrl_md3.place(tmp_reg)
        self.cpw_md_lines.append(self.cpwrl_md3)

        # place caplanar line 4md
        _p1 = self.contact_pads[7].end
        _p2 = _p1 + DPoint(-3e6, 0)
        _p3 = DPoint(
            self.xmons[3].cpw_b.end.x + self.md234_cross_bottom_dx,
            self.cont_lines_y_ref
        )
        _p4 = DPoint(
            _p3.x,
            self.xmons[3].cpw_b.end.y - self.md234_cross_bottom_dy
        )
        _p5 = _p4 + DPoint(0, 10e3)
        self.cpwrl_md4 = DPathCPW(
            points=[_p1, _p2, _p3, _p4, _p5],
            cpw_parameters=[self.z_md_fl] * 5 + [
                CPWParameters(width=0, gap=self.z_md_fl.b / 2)
            ],
            turn_radiuses=self.ctr_lines_turn_radius
        )
        self.cpwrl_md4.place(tmp_reg)
        self.cpw_md_lines.append(self.cpwrl_md4)

        # place caplanar line 5md
        _p1 = self.contact_pads[9].end
        _p2 = _p1 + DPoint(0, -0.5e6)
        _p3 = _p2 + DPoint(1e6, -1e6)
        _p4 = self.xmons[4].cpw_r.end + DVector(1e6, 0)
        _p5 = self.xmons[4].cpw_r.end + DVector(66.2e3, 0)
        _p6 = _p5 + DVector(11.2e3, 0)
        self.cpwrl_md5 = DPathCPW(
            points=[_p1, _p2, _p3, _p4, _p5, _p6],
            cpw_parameters=[self.z_md_fl] * 8 + [
                CPWParameters(width=0, gap=self.z_md_fl.b / 2)
            ],
            turn_radiuses=self.ctr_lines_turn_radius,
        )
        self.cpwrl_md5.place(tmp_reg)
        self.cpw_md_lines.append(self.cpwrl_md5)

    def draw_flux_control_lines(self):
        tmp_reg = self.region_ph

        # place caplanar line 1 fl
        _p1 = self.contact_pads[1].end
        _p2 = self.contact_pads[1].end + DPoint(1e6, 0)
        _p3 = DPoint(
            self.squids[0].bot_dc_flux_line_left.start.x,
            self.cont_lines_y_ref
        )
        _p4 = DPoint(
            _p3.x,
            self.xmons[0].center.y - self.xmons[0].cpw_l.b / 2
        )
        self.cpwrl_fl1 = DPathCPW(
            points=[_p1, _p2, _p3, _p4],
            cpw_parameters=self.z_md_fl,
            turn_radiuses=self.ctr_lines_turn_radius,
        )
        self.cpwrl_fl1.place(tmp_reg)
        self.cpw_fl_lines.append(self.cpwrl_fl1)

        # place caplanar line 2 fl
        _p1 = self.contact_pads[2].end
        _p2 = self.contact_pads[2].end + DPoint(1e6, 0)
        _p3 = DPoint(
            self.squids[1].bot_dc_flux_line_left.start.x,
            self.cont_lines_y_ref
        )
        _p4 = DPoint(
            _p3.x,
            self.xmons[1].cpw_bempt.end.y
        )
        self.cpwrl_fl2 = DPathCPW(
            points=[_p1, _p2, _p3, _p4],
            cpw_parameters=self.z_md_fl,
            turn_radiuses=self.ctr_lines_turn_radius,
        )
        self.cpwrl_fl2.place(tmp_reg)
        self.cpw_fl_lines.append(self.cpwrl_fl2)

        # place caplanar line 3 fl
        _p1 = self.contact_pads[4].end
        _p2 = self.contact_pads[4].end + DPoint(0, 1e6)
        _p3 = _p2 + DPoint(-1e6, 1e6)
        _p4 = DPoint(
            self.squids[2].bot_dc_flux_line_left.start.x,
            self.cont_lines_y_ref
        )
        _p5 = DPoint(
            _p4.x,
            self.xmons[2].cpw_bempt.end.y
        )
        self.cpwrl_fl3 = DPathCPW(
            points=[_p1, _p2, _p3, _p4, _p5],
            cpw_parameters=self.z_md_fl,
            turn_radiuses=self.ctr_lines_turn_radius,
        )
        self.cpwrl_fl3.place(tmp_reg)
        self.cpw_fl_lines.append(self.cpwrl_fl3)

        # place caplanar line 4 fl
        _p1 = self.contact_pads[6].end
        _p2 = self.contact_pads[6].end + DPoint(-1.5e6, 0)
        _p3 = DPoint(
            self.squids[3].bot_dc_flux_line_left.start.x,
            self.cont_lines_y_ref
        )
        _p4 = DPoint(
            _p3.x,
            self.xmons[3].cpw_bempt.end.y
        )
        self.cpwrl_fl4 = DPathCPW(
            points=[_p1, _p2, _p3, _p4],
            cpw_parameters=self.z_md_fl,
            turn_radiuses=self.ctr_lines_turn_radius,
        )
        self.cpwrl_fl4.place(tmp_reg)
        self.cpw_fl_lines.append(self.cpwrl_fl4)

        # place caplanar line 5 fl
        _p1 = self.contact_pads[8].end
        _p2 = self.contact_pads[8].end + DPoint(-0.3e6, 0)
        _p4 = DPoint(
            self.squids[4].bot_dc_flux_line_left.start.x,
            self.cont_lines_y_ref
        )
        _p3 = _p4 + DVector(2e6, 0)
        _p5 = DPoint(
            _p4.x,
            self.xmons[4].center.y - self.xmons[4].cpw_l.b / 2
        )
        self.cpwrl_fl5 = DPathCPW(
            points=[_p1, _p2, _p3, _p4, _p5],
            cpw_parameters=self.z_md_fl,
            turn_radiuses=self.ctr_lines_turn_radius,
        )
        self.cpwrl_fl5.place(tmp_reg)
        self.cpw_fl_lines.append(self.cpwrl_fl5)

    def draw_el_dc_contacts(self):
        test_samples_el2: List[Region] = []
        test_samples_ph_cut: List[Region] = []
        for squid, xmon, fl_line in zip(self.squids, self.xmons,
                                        self.cpw_fl_lines):
            # dc contact pad has to be completely
            # inside union of both  e-beam and photo deposed
            # metal regions.
            # `self.dc_cont_clearance` represents minimum distance
            # from dc contact pad`s perimeter to the perimeter of the
            # e-beam and photo-deposed metal perimeter.
            top_rect1 = CPW(
                start=squid.pad_top.center,
                end=squid.ph_el_conn_pad.end + DPoint(
                    0, self.dc_cont_clearance
                ),
                width=squid.ph_el_conn_pad.width -
                      2 * self.dc_cont_clearance,
                gap=0
            )

            if xmon == self.xmons[0]:
                end = DPoint(
                    squid.origin.x, xmon.center.y - xmon.cpw_l.width / 2
                )
            elif xmon == self.xmons[-1]:
                end = DPoint(
                    squid.origin.x, xmon.center.y - xmon.cpw_r.width / 2
                )
            else:
                end = xmon.cpw_b.end
            end += DPoint(0, self.dc_cont_clearance)

            top_rect2 = CPW(
                start=squid.ph_el_conn_pad.start +
                      DPoint(0, squid.pad_top.r + self.dc_cont_ph_ext),
                end=end,
                width=2 * (squid.pad_top.r + self.dc_cont_ph_ext),
                gap=0
            )

            left_bottom = list(
                squid.bot_dc_flux_line_left.primitives.values()
            )[0]
            bot_left_shape1 = CPW(
                start=fl_line.end + DPoint(0, -self.dc_cont_clearance),
                end=left_bottom.start,
                width=left_bottom.width - 2 * self.dc_cont_clearance,
                gap=0
            )

            bot_left_shape2 = CPW(
                start=fl_line.end + DPoint(0, -self.dc_cont_clearance),
                end=left_bottom.start + DPoint(0, -self.dc_cont_ph_ext),
                width=left_bottom.width + 2 * self.dc_cont_ph_ext,
                gap=0
            )

            right_bottom = list(
                squid.bot_dc_flux_line_right.primitives.values()
            )[0]
            bot_right_shape1 = CPW(
                start=DPoint(right_bottom.start.x, fl_line.end.y) +
                      DPoint(
                          0,
                          self.dc_cont_clearance
                      ),
                end=right_bottom.start + DPoint(0, -self.dc_cont_ph_ext),
                width=right_bottom.width - 2 * self.dc_cont_clearance,
                gap=0
            )

            bot_right_shape2 = CPW(
                start=DPoint(right_bottom.start.x, fl_line.end.y) +
                DPoint(
                    0,
                    -self.dc_cont_clearance
                ),
                end=right_bottom.start + DPoint(0, -self.dc_cont_ph_ext),
                width=right_bottom.width + 2 * self.dc_cont_ph_ext,
                gap=0
            )

            self.el_dc_contacts.append(
                [
                    top_rect1, top_rect2,
                    bot_left_shape1, bot_left_shape2,
                    bot_right_shape1, bot_right_shape2
                ]
            )

            test_sample_reg_el2 = Region()
            for contact in self.el_dc_contacts[-1]:
                contact.place(self.region_el2)
                contact.place(test_sample_reg_el2)
            test_samples_el2.append(test_sample_reg_el2)

            # DC contacts has to have intersection with empty
            # layer in photo litography. This is needed in order
            # to ensure that first e-beam layer does not
            # broke at the step between substrate and
            # photolytography polygons.
            # Following rectangle pads are cutted from photo region
            # to ensure DC contacts are covering aforementioned level step.

            test_ph_cut = Region()
            # Rectangle to cut for top DC contact pad
            rec_top = CPW(
                start=squid.ph_el_conn_pad.start + DPoint(
                    0, -self.dc_cont_cut_clearance
                ),
                end=squid.ph_el_conn_pad.end,
                width=0,
                gap=self.dc_cont_cut_width/2
            )
            rec_top.place(self.region_ph)
            test_ph_cut |= rec_top.empty_region

            # Rectangle for bottom left DC contact pad
            left_bot = CPW(
                start=fl_line.end + DPoint(
                    0, self.cross_gnd_gap_x/6
                ),
                end=left_bottom.start + DPoint(
                    0, self.dc_cont_cut_clearance
                ),
                width=0,
                gap=self.dc_cont_cut_width/2
            )
            left_bot.place(self.region_ph)
            test_ph_cut |= left_bot.empty_region

            # Rectangle for bottom right DC contact pad
            right_bot = CPW(
                start=DPoint(right_bottom.start.x, fl_line.end.y) +
                DPoint(
                    0, self.cross_gnd_gap_x
                ),
                end=right_bottom.start + DPoint(
                    0, self.dc_cont_cut_clearance
                ),
                width=0,
                gap=self.dc_cont_cut_width/2
            )
            right_bot.place(self.region_ph)
            test_ph_cut |= right_bot.empty_region

            test_samples_ph_cut.append(test_ph_cut)

            self.region_el2.merge()

        for squid in self.test_squids:
            trans = DCplxTrans(
                1, 0, False,
                -self.squids[0].origin + squid.origin
            )
            test_reg_el2 = test_samples_el2[0].dup().transformed(trans)
            self.region_el2 |= test_reg_el2

            cut_reg_ph = test_samples_ph_cut[0].dup().transformed(trans)
            self.region_ph -= cut_reg_ph

    def draw_test_structures(self):
        struct_centers = [DPoint(1e6, 4e6), DPoint(8.7e6, 5.7e6),
                          DPoint(6.5e6, 2.7e6)]
        for struct_center in struct_centers:
            ## JJ test structures ##
            # test structure with big critical current
            test_struct1 = TestStructurePads(struct_center)
            test_struct1.place(self.region_ph)
            text_reg = pya.TextGenerator.default_generator().text(
                "48.32 nA", 0.001, 50, False, 0, 0)
            text_bl = test_struct1.empty_rectangle.origin + DPoint(
                test_struct1.gnd_gap, -4 * test_struct1.gnd_gap
            )
            text_reg.transform(
                ICplxTrans(1.0, 0, False, text_bl.x, text_bl.y))
            self.region_ph -= text_reg
            test_jj = AsymSquidDCFlux(
                test_struct1.center, SQUID_PARAMETERS, side=1
            )
            self.test_squids.append(test_jj)
            test_jj.place(self.region_el)

            # test structure with low critical current
            test_struct2 = TestStructurePads(
                struct_center + DPoint(0.3e6, 0))
            test_struct2.place(self.region_ph)
            text_reg = pya.TextGenerator.default_generator().text(
                "9.66 nA", 0.001, 50, False, 0, 0)
            text_bl = test_struct2.empty_rectangle.origin + DPoint(
                test_struct2.gnd_gap, -4 * test_struct2.gnd_gap
            )
            text_reg.transform(
                ICplxTrans(1.0, 0, False, text_bl.x, text_bl.y))
            self.region_ph -= text_reg
            test_jj = AsymSquidDCFlux(
                test_struct2.center, SQUID_PARAMETERS, side=-1
            )
            self.test_squids.append(test_jj)
            test_jj.place(self.region_el)

            # test structure for bridge DC contact
            test_struct3 = TestStructurePads(
                struct_center + DPoint(0.6e6, 0))
            test_struct3.place(self.region_ph)
            text_reg = pya.TextGenerator.default_generator().text(
                "DC", 0.001, 50, False, 0, 0
            )
            text_bl = test_struct3.empty_rectangle.origin + DPoint(
                test_struct3.gnd_gap, -4 * test_struct3.gnd_gap
            )
            text_reg.transform(
                ICplxTrans(1.0, 0, False, test_struct3.center.x, text_bl.y)
            )
            self.region_ph -= text_reg

            test_bridges = []
            for i in range(3):
                bridge = Bridge1(
                    test_struct3.center + DPoint(50e3 * (i - 1), 0),
                    gnd_touch_dx=20e3)
                test_bridges.append(bridge)
                bridge.place(self.region_bridges1, region_name="bridges_1")
                bridge.place(self.region_bridges2, region_name="bridges_2")

        # bandages test structures
        test_dc_el2_centers = [
            DPoint(2.5e6, 2.4e6),
            DPoint(4.2e6, 1.6e6),
            DPoint(9.0e6, 3.8e6)
        ]
        for struct_center in test_dc_el2_centers:
            test_struct1 = TestStructurePads(struct_center)
            test_struct1.place(self.region_ph)
            text_reg = pya.TextGenerator.default_generator().text(
                "Bandage", 0.001, 40, False, 0, 0)
            text_bl = test_struct1.empty_rectangle.origin + DPoint(
                test_struct1.gnd_gap, -4 * test_struct1.gnd_gap
            )
            text_reg.transform(
                ICplxTrans(1.0, 0, False, text_bl.x, text_bl.y))
            self.region_ph -= text_reg

            rec_width = 10e3
            rec_height = test_struct1.rectangles_gap + 2 * rec_width
            p1 = struct_center - DVector(rec_width / 2, rec_height / 2)
            dc_rec = Rectangle(p1, rec_width, rec_height)
            dc_rec.place(self.region_el2)

    def draw_el_protection(self):
        protection_a = 300e3
        for squid in (self.squids + self.test_squids):
            self.region_el_protection.insert(
                pya.Box().from_dbox(
                    pya.DBox(
                        squid.origin - 0.5 * DVector(protection_a,
                                                     protection_a),
                        squid.origin + 0.5 * DVector(protection_a,
                                                     protection_a)
                    )
                )
            )

    def draw_photo_el_marks(self):
        marks_centers = [
            DPoint(1e6, 9e6), DPoint(1e6, 1e6),
            DPoint(9e6, 1e6), DPoint(9e6, 9e6),
            DPoint(8e6, 4e6), DPoint(1e6, 6e6)
        ]
        for mark_center in marks_centers:
            self.marks.append(
                MarkBolgar(mark_center)
            )
            self.marks[-1].place(self.region_ph)

    def draw_bridges(self):
        bridges_step = 150e3
        fl_bridges_step = 150e3

        # for resonators
        for resonator in self.resonators:
            for name, res_primitive in resonator.primitives.items():
                if "coil0" in name:
                    # skip L_coupling coplanar.
                    # bridgyfy all in "coil0" except for the first cpw that
                    # is adjacent to readout line and has length equal to `L_coupling`
                    for primitive in list(
                            res_primitive.primitives.values())[1:]:
                        Bridge1.bridgify_CPW(
                            primitive, bridges_step,
                            dest=self.region_bridges1,
                            dest2=self.region_bridges2
                        )

                    continue
                elif "fork" in name:  # skip fork primitives
                    continue
                else:  # bridgify everything else
                    Bridge1.bridgify_CPW(
                        res_primitive, bridges_step,
                        dest=self.region_bridges1,
                        dest2=self.region_bridges2
                    )

        # for contact wires
        for key, val in self.__dict__.items():
            if "cpwrl_md" in key:
                Bridge1.bridgify_CPW(
                    val, bridges_step,
                    dest=self.region_bridges1, dest2=self.region_bridges2
                )
            elif "cpwrl_fl" in key:
                Bridge1.bridgify_CPW(
                    val, fl_bridges_step,
                    dest=self.region_bridges1, dest2=self.region_bridges2,
                    avoid_points=[squid.origin for squid in self.squids],
                    avoid_distance=400e3
                )

        for cpw_fl in self.cpw_fl_lines:
            bridge_center1 = cpw_fl.end + DVector(0, -40e3)
            br = Bridge1(center=bridge_center1, trans_in=Trans.R90)
            br.place(dest=self.region_bridges1, region_name="bridges_1")
            br.place(dest=self.region_bridges2, region_name="bridges_2")

        # for readout waveguide
        bridgified_primitives_idxs = list(range(2))
        bridgified_primitives_idxs += list(
            range(2, 2 * (len(self.resonators) + 1) + 1, 2))
        bridgified_primitives_idxs += list(range(
            2 * (len(self.resonators) + 1) + 1,
            len(self.cpwrl_ro_line.primitives.values()))
        )
        for idx, primitive in enumerate(
                self.cpwrl_ro_line.primitives.values()):
            if idx in bridgified_primitives_idxs:
                Bridge1.bridgify_CPW(
                    primitive, bridges_step,
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

    def extend_photo_overetching(self):
        tmp_reg = Region()
        ep = pya.EdgeProcessor()
        for poly in self.region_ph.each():
            tmp_reg.insert(
                ep.simple_merge_p2p(
                    [
                        poly.sized(
                            FABRICATION.OVERETCHING,
                            FABRICATION.OVERETCHING,
                            2
                        )
                    ],
                    False,
                    False,
                    1
                )
            )
        self.region_ph = tmp_reg

    def split_polygons_in_layers(self, max_pts=200):
        self.region_ph = split_polygons(self.region_ph, max_pts)
        self.region_bridges2 = split_polygons(self.region_bridges2,
                                              max_pts)
        for poly in self.region_ph:
            if poly.num_points() > max_pts:
                print("exists photo")
        for poly in self.region_ph:
            if poly.num_points() > max_pts:
                print("exists bridge2")


if __name__ == "__main__":
    design = Design5Q("testScript")
    design.draw()
    design.show()
