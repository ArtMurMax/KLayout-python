__version__ = "v.0.3.0.8.T1"

'''
Description:
This program is made to generate litography blueprints for testing of the 
main series chips. E.g. this one is based on v.0.3.0.8 Design.py

e.g. version v.0.3.0.8. Test layout is marked as T1, where `T` stands for `test`.
T1 geometry parameters mainly follows those
from v.0.3.0.8 design if not mensioned otherwise in `changes log`. 
Resonators frequencies, shapes, Q-factors,
readout and resonator coplanar geometries,
Xmon crosses shape, qubit-resonator coupling capacitances,
SQUID loops and test structure geomtries.


Changes log 
v.0.3.0.8.T1
1. Added 1 aditional resonator with identical SQUID, resonator frequency is 7.52 + 0.08 = 7.60 GHz
2. Added 2 resonators with Xmon Crosses but without SQUID's. Frequencies 7.68 and 7.76 GHz correspondingly.
3. Coplanar bridges are NOT used for this test version.
'''

# Enter your Python code here
from math import cos, sin, tan, atan2, pi, degrees
import itertools
from typing import List, Dict, Union, Optional

import numpy as np

import pya
from pya import Cell
from pya import Point, Vector, DPoint, DVector, DEdge, \
    DSimplePolygon, \
    SimplePolygon, DPolygon, DBox, Polygon, Region
from pya import Trans, DTrans, CplxTrans, DCplxTrans, ICplxTrans, DPath

from importlib import reload
import classLib

reload(classLib)

from classLib.baseClasses import ElementBase, ComplexBase
from classLib.coplanars import CPWParameters, CPW, DPathCPW, \
    CPWRLPath, Bridge1
from classLib.shapes import XmonCross, Rectangle, CutMark
from classLib.resonators import EMResonatorTL3QbitWormRLTailXmonFork
from classLib.josJ import AsymSquidOneLegParams, AsymSquidOneLegRigid
from classLib.chipTemplates import CHIP_10x5_8pads, FABRICATION
from classLib.chipDesign import ChipDesign
from classLib.marks import MarkBolgar
from classLib.contactPads import ContactPad
from classLib.helpers import fill_holes, split_polygons, extended_region

import sonnetSim

reload(sonnetSim)
from sonnetSim import SonnetLab, SonnetPort, SimulationBox

import copy

# 0.0 - for development
# 0.8e3 - estimation for fabrication by Bolgar photolytography etching
# recipe
FABRICATION.OVERETCHING = 0.6e3
SQUID_PARAMETERS = AsymSquidOneLegParams(
    pad_r=5e3, pads_distance=60e3,
    contact_pad_width=10e3, contact_pad_ext_r=200,
    sq_len=5e3, sq_area=200e6,
    j1_dx=114.551, j2_dx=398.086,
    j1_dy=114.551, j2_dy=250,
    bridge=180, b_ext=2e3,
    inter_leads_width=500,
    n=20,
    flux_line_dx=100e3, flux_line_dy=25e3, flux_line_outer_width=2e3,
    flux_line_inner_width=370,
    flux_line_contact_width=20e3
)


class TestStructurePadsSquare(ComplexBase):
    def __init__(self, center, trans_in=None, square_a=200e3,
                 gnd_gap=20e3, squares_gap=20e3):
        self.center = center
        self.rectangle_a = square_a
        self.gnd_gap = gnd_gap
        self.rectangles_gap = squares_gap

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


class Design5QTest(ChipDesign):
    def __init__(self, cell_name):
        super().__init__(cell_name)
        dc_bandage_layer_i = pya.LayerInfo(3, 0)  # for DC contact deposition
        self.dc_bandage_reg = Region()
        self.dc_bandage_layer = self.layout.layer(dc_bandage_layer_i)

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
        self.chip = CHIP_10x5_8pads
        self.chip_box: pya.DBox = self.chip.box
        # Z = 50.09 E_eff = 6.235 (E = 11.45)
        self.z_md_fl: CPWParameters = CPWParameters(11e3, 5.7e3)
        self.ro_Z: CPWParameters = self.chip.chip_Z
        self.contact_pads: list[ContactPad] = self.chip.get_contact_pads(
            [self.ro_Z] + [self.z_md_fl] * 3 + [self.ro_Z] + [self.z_md_fl] * 3
        )

        # readout line parameters
        self.ro_line_dy: float = 1600e3
        self.cpwrl_ro_line: CPWRLPath = None
        self.Z0: CPWParameters = CHIP_10x5_8pads.chip_Z

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
        self.L0 = 1000e3
        self.L1_list = [
            1e3 * x for x in [58.4246, 20.374, 76.41, 74.3824, 26.0267]
        ]
        self.r = 60e3
        self.N_coils = [2, 3, 3, 3, 3]
        self.L2_list = [self.r] * len(self.L1_list)
        self.L3_list = [0e3] * len(self.L1_list)  # to be constructed
        self.L4_list = [self.r] * len(self.L1_list)
        self.width_res = 20e3
        self.gap_res = 10e3
        self.Z_res = CPWParameters(self.width_res, self.gap_res)
        self.to_line_list = [58e3] * len(self.L1_list)
        self.fork_metal_width = 10e3
        self.fork_gnd_gap = 15e3
        self.xmon_fork_gnd_gap = 14e3
        # resonator-fork parameters
        # for coarse C_qr evaluation
        self.fork_y_spans = [
            x * 1e3 for x in [35.044, 87.202, 42.834, 90.72, 46.767]
        ]

        # 4 additional resonators based on resonator with idx 2, but
        # only frequency is changed (7.58, 7.66, 7.84) GHz correspondingly
        self.add_res_based_idx = 2
        self.L1_list += [x*1e3 for x in [22.7979, 51.3575, 46.1034]]
        self.L_coupling_list += [self.L_coupling_list[
                                     self.add_res_based_idx]] * 3
        self.N_coils += [self.N_coils[self.add_res_based_idx]] * 3
        self.L2_list += [self.L2_list[self.add_res_based_idx]] * 3
        self.L3_list += [self.L3_list[self.add_res_based_idx]] * 3
        self.L4_list += [self.L4_list[self.add_res_based_idx]] * 3
        self.to_line_list += [self.to_line_list[self.add_res_based_idx]] * 3
        self.fork_y_spans += [self.fork_y_spans[self.add_res_based_idx]] * 3

        # xmon parameters
        self.xmon_x_distance: float = 545e3  # from simulation of g_12
        # for fine C_qr evaluation
        self.xmon_dys_Cg_coupling = [14e3] * len(self.L1_list)
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
        self.squids: List[AsymSquidOneLegRigid] = []
        self.test_squids: List[AsymSquidOneLegRigid] = []
        # minimal distance between squid loop and photo layer
        self.squid_ph_clearance = 1.5e3

        # el-dc concacts attributes
        # e-beam polygon has to cover hole in photoregion and also
        # overlap photo region by the following amount
        self.el_overlaps_ph_by = 2e3
        # required clearance of dc contacts from squid perimeter
        self.dc_cont_el_clearance = 2e3  # 1.8e3
        # required clearance of dc contacts from photo layer polygon
        # perimeter
        self.dc_cont_ph_clearance = 2e3
        # required extension into photo region over the hole cutted
        self.dc_cont_ph_ext = 10e3

        # microwave and flux drive lines parameters
        self.ctr_lines_turn_radius = 40e3
        self.cont_lines_y_ref: float = 300e3  # nm

        # distance between microwave-drive central coplanar line
        # to the face of Xmon's cross metal. Assuming that microwave
        # drive CPW's end comes to cross simmetrically
        self.md_line_to_cross_metal = 80e3

        self.flLine_squidLeg_gap = 5e3
        self.flux_lines_x_shifts: List[float] = [None] * len(self.L1_list)
        self.current_line_width = 3.5e3 - 2 * FABRICATION.OVERETCHING

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

    def draw(self):
        """

        Parameters
        ----------
        res_f_Q_sim_idx : int
            resonator index to draw. If not None, design will contain only
            readout waveguide and resonator with corresponding index (from 0 to 4),
            as well as corresponding Xmon Cross.
        design_params : object
            design parameters to customize

        Returns
        -------
        None
        """
        self.draw_chip()
        '''
            Only creating object. This is due to the drawing of xmons and resonators require
        draw xmons, then draw resonators and then draw additional xmons. This is
        ugly and that how this was before migrating to `ChipDesign` based code structure
            This is also the reason why `self.__init__` is flooded with design parameters that
        are used across multiple drawing functions.

        TODO: This drawings sequence can be decoupled in the future.
        '''
        self.draw_readout_waveguide()

        self.create_resonator_objects()
        self.draw_xmons_and_resonators()

        self.draw_josephson_loops()

        # self.draw_microwave_drvie_lines()
        self.draw_flux_control_lines()

        self.draw_test_structures()
        self.draw_el_dc_contacts()
        self.draw_el_protection()

        self.draw_photo_el_marks()
        # self.draw_bridges()
        self.draw_pinning_holes()
        # v.0.3.0.8 p.12 - ensure that contact pads consist no wholes
        for contact_pad in self.contact_pads:
            contact_pad.place(self.region_ph)
            self.region_ph.merge()
        self.extend_photo_overetching()
        self.inverse_destination(self.region_ph)
        self.draw_cut_marks()
        self.resolve_holes()  # convert to gds acceptable polygons (without inner holes)
        self.split_polygons_in_layers(max_pts=180)

    def draw_for_res_f_and_Q_sim(self, res_idx):
        """
        Function draw part of design that will be cropped and simulateed to obtain resonator`s frequency and Q-factor.
        Resonators are enumerated starting from 0.
        Parameters
        ----------
        res_f_Q_sim_idx : int
            resonator index to draw. If not None, design will contain only
            readout waveguide and resonator with corresponding index (from 0 to 4),
            as well as corresponding Xmon Cross.
        design_params : object
            design parameters to customize

        Returns
        -------
        None
        """
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
        self.draw_xmons_and_resonators(res_idx=res_idx)

    def draw_for_Cqr_simulation(self, res_idx):
        """
        Function draw part of design that will be cropped and simulateed to obtain capacity value of capacitive
        coupling between qubit and resonator.
        Resonators are enumerated starting from 0.
        Parameters
        ----------
        res_f_Q_sim_idx : int
            resonator index to draw. If not None, design will contain only
            readout waveguide and resonator with corresponding index (from 0 to 4),
            as well as corresponding Xmon Cross.
        design_params : object
            design parameters to customize

        Returns
        -------
        None
        """
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
        self.draw_xmons_and_resonators(res_idx=res_idx)

    def _transfer_regs2cell(self):
        # this too methods assumes that all previous drawing
        # functions are placing their object on regions
        # in order to avoid extensive copying of the polygons
        # to/from cell.shapes during the logic operations on
        # polygons
        self.cell.shapes(self.layer_ph).insert(self.region_ph)
        self.cell.shapes(self.layer_el).insert(self.region_el)
        self.cell.shapes(self.dc_bandage_layer).insert(self.dc_bandage_reg)
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

    def draw_cut_marks(self):
        chip_box_poly = DPolygon(self.chip_box)
        for point in chip_box_poly.each_point_hull():
            CutMark(origin=point).place(self.region_ph)

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
        for i in range(5, 8):
            self.L0_list[i] = self.L0_list[self.add_res_based_idx]

        self.L2_list[0] += 6 * self.Z_res.b
        self.L2_list[1] += 0
        self.L2_list[3] += 3 * self.Z_res.b
        self.L2_list[4] += 6 * self.Z_res.b
        self.L2_list[5] += 6 * self.Z_res.b

        self.L3_list[0] = x1
        self.L3_list[1] = x2
        self.L3_list[2] = x3
        self.L3_list[3] = x4
        self.L3_list[4] = x5
        for i in range(5, 8):
            self.L3_list[i] = self.L3_list[self.add_res_based_idx]

        self.L4_list[1] += 6 * self.Z_res.b
        self.L4_list[2] += 6 * self.Z_res.b
        self.L4_list[3] += 3 * self.Z_res.b
        for i in range(5, 8):
            self.L4_list[i] = self.L4_list[self.add_res_based_idx]

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
            [pi / 2, -pi / 2]
        ]
        tail_turn_angles_list += [tail_turn_angles_list[
                                      self.add_res_based_idx]] * 2

        tail_trans_in_list = [
            Trans.R270,
            Trans.R270,
            Trans.R270,
            Trans.R270,
            Trans.R270,
            Trans.R270
        ]
        tail_trans_in_list += [tail_trans_in_list[
                                   self.add_res_based_idx]] * 2
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

        worm_x_list = [x * 1e6 for x in [1, 2.7, 3.5, 4.35, 7.6, 6.5, 5.5, 8.5]]
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
            worm_x = worm_x_list[res_idx]
            worm_y = self.chip.dy / 2 + to_line * (-1) ** (res_idx)

            if res_idx % 2 == 0:
                trans = DTrans.M0
            else:
                trans = DTrans.R0
            self.resonators.append(
                EMResonatorTL3QbitWormRLTailXmonFork(
                    self.Z_res, DPoint(worm_x, worm_y), L_coupling,
                    L0=L0,
                    L1=L1, r=self.r, N=n_coils,
                    tail_shape=res_tail_shape,
                    tail_turn_radiuses=tail_turn_radiuses,
                    tail_segment_lengths=tail_segment_lengths,
                    tail_turn_angles=tail_turn_angles,
                    tail_trans_in=tail_trans_in,
                    fork_x_span=self.fork_x_span,
                    fork_y_span=fork_y_span,
                    fork_metal_width=self.fork_metal_width,
                    fork_gnd_gap=self.fork_gnd_gap,
                    trans_in=trans
                )
            )
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
        ro_line_dy = self.ro_line_dy

        self.cpwrl_ro_line = CPW(
            start=self.contact_pads[0].end,
            end=self.contact_pads[4].end,
            cpw_params=self.ro_Z
        )
        self.cpwrl_ro_line.place(self.region_ph)

    def draw_xmons_and_resonators(self, res_idx=None):
        """
        Fills photolitography Region() instance with resonators
        and Xmons crosses structures.

        Parameters
        ----------
        res_idx : int
            draw only particular resonator (if passed)
            used in resonator simulations.


        Returns
        -------
        None
        """
        for current_res_idx, (
                resonator, fork_y_span, xmon_dy_Cg_coupling) in \
                enumerate(zip(self.resonators, self.fork_y_spans,
                              self.xmon_dys_Cg_coupling)):
            if current_res_idx % 2 == 0:
                m = -1
            else:
                m = 1
            xmon_center = \
                (
                        resonator.fork_x_cpw.start + resonator.fork_x_cpw.end
                ) / 2 + \
                m * DVector(
                    0,
                    -xmon_dy_Cg_coupling - resonator.fork_metal_width / 2
                )
            # changes start #
            xmon_center += DPoint(
                0,
                -m * (
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
            if (res_idx is None) or (res_idx == current_res_idx):
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
        for res_idx, xmon_cross in enumerate(self.xmons[:-2]):
            if res_idx % 2 == 0:
                squid_center = xmon_cross.cpw_tempt.end
                squid_center += DPoint(
                    0,
                    -(SQUID_PARAMETERS.sq_len / 2 +
                      SQUID_PARAMETERS.flux_line_inner_width / 2 +
                      self.squid_ph_clearance)
                )
                trans = DTrans.M0
            else:
                squid_center = xmon_cross.cpw_bempt.end
                squid_center += DPoint(
                    0,
                    SQUID_PARAMETERS.sq_len / 2 +
                    SQUID_PARAMETERS.flux_line_inner_width / 2 +
                    self.squid_ph_clearance
                )
                trans = DTrans.R0
            squid = AsymSquidOneLegRigid(squid_center, SQUID_PARAMETERS, 0,
                                         leg_side=-1, trans_in=trans)
            self.squids.append(squid)
            squid.place(self.region_el)

    def draw_microwave_drvie_lines(self):

        tmp_reg = self.region_ph

        # place caplanar line 1md
        _p1 = self.contact_pads[7].end
        _p2 = DPoint(_p1.x, self.xmons[0].origin.y)
        _p3 = DPoint(self.xmons[0].cpw_r.end.x + self.md_line_to_cross_metal, _p2.y)
        self.cpwrl_md1 = DPathCPW(
            points=[_p1, _p2, _p3],
            cpw_parameters=self.z_md_fl,
            turn_radiuses=self.ctr_lines_turn_radius
        )
        self.cpwrl_md1.place(tmp_reg)
        self.cpw_md_lines.append(self.cpwrl_md1)

        # place caplanar line 2md
        _p1 = self.contact_pads[1].end
        _p2 = DPoint(_p1.x, self.xmons[1].origin.y)
        _p3 = DPoint(self.xmons[1].cpw_l.end.x - self.md_line_to_cross_metal, _p2.y)
        self.cpwrl_md1 = DPathCPW(
            points=[_p1, _p2, _p3],
            cpw_parameters=self.z_md_fl,
            turn_radiuses=self.ctr_lines_turn_radius
        )
        self.cpwrl_md1.place(tmp_reg)
        self.cpw_md_lines.append(self.cpwrl_md1)

        # place caplanar line 3md
        _p1 = self.contact_pads[6].end
        _p2 = DPoint(_p1.x, self.xmons[2].origin.y)
        _p3 = DPoint(self.xmons[2].cpw_r.end.x + self.md_line_to_cross_metal, _p2.y)
        self.cpwrl_md1 = DPathCPW(
            points=[_p1, _p2, _p3],
            cpw_parameters=self.z_md_fl,
            turn_radiuses=self.ctr_lines_turn_radius
        )
        self.cpwrl_md1.place(tmp_reg)
        self.cpw_md_lines.append(self.cpwrl_md1)

        # place caplanar line 4md
        _p1 = self.contact_pads[2].end
        _p2 = DPoint(_p1.x, self.xmons[3].origin.y)
        _p3 = DPoint(self.xmons[3].cpw_l.end.x - self.md_line_to_cross_metal, _p2.y)
        self.cpwrl_md1 = DPathCPW(
            points=[_p1, _p2, _p3],
            cpw_parameters=self.z_md_fl,
            turn_radiuses=self.ctr_lines_turn_radius
        )
        self.cpwrl_md1.place(tmp_reg)
        self.cpw_md_lines.append(self.cpwrl_md1)

        # place caplanar line 5md
        _p1 = self.contact_pads[5].end
        _p2 = DPoint(_p1.x, self.xmons[4].origin.y)
        _p3 = DPoint(self.xmons[4].cpw_r.end.x + self.md_line_to_cross_metal, _p2.y)
        self.cpwrl_md1 = DPathCPW(
            points=[_p1, _p2, _p3],
            cpw_parameters=self.z_md_fl,
            turn_radiuses=self.ctr_lines_turn_radius
        )
        self.cpwrl_md1.place(tmp_reg)
        self.cpw_md_lines.append(self.cpwrl_md1)

        # place caplanar line 6md
        _p1 = self.contact_pads[3].end
        _p2 = DPoint(_p1.x, self.xmons[5].origin.y)
        _p3 = DPoint(self.xmons[5].cpw_l.end.x - self.md_line_to_cross_metal, _p2.y)
        self.cpwrl_md1 = DPathCPW(
            points=[_p1, _p2, _p3],
            cpw_parameters=self.z_md_fl,
            turn_radiuses=self.ctr_lines_turn_radius
        )
        self.cpwrl_md1.place(tmp_reg)
        self.cpw_md_lines.append(self.cpwrl_md1)

    def draw_flux_control_lines(self):
        tmp_reg = self.region_ph

        # calculate flux line end horizontal shift from center of the
        # squid loop
        self.flux_lines_x_shifts = [
            -(squid.bot_inter_lead_dx / 2 + self.z_md_fl.b / 2) for squid
            in
            self.squids]

        # place caplanar line 1 fl
        _p1 = self.contact_pads[1].end
        _p2 = self.contact_pads[1].end + DPoint(0, 0.5e6)
        _p3 = DPoint(
            self.squids[1].origin.x + self.flux_lines_x_shifts[1],
            self.squids[1].origin.y - self.cont_lines_y_ref
        )

        _p4 = DPoint(_p3.x, self.xmons[1].cpw_bempt.end.y)
        self.cpwrl_fl1 = DPathCPW(
            points=[_p1, _p2, _p3, _p4],
            cpw_parameters=self.z_md_fl,
            turn_radiuses=self.ctr_lines_turn_radius,
        )
        self.cpwrl_fl1.place(tmp_reg)
        self.cpw_fl_lines.append(self.cpwrl_fl1)

        def draw_inductor_for_fl_line(design, fl_line_idx):
            cpwrl_fl = self.cpw_fl_lines[fl_line_idx]
            cpwrl_fl_inductor_start = cpwrl_fl.end + \
                                      DVector(0,
                                              -design.current_line_width / 2)
            cpwrl_fl_inductor = CPW(
                cpw_params=CPWParameters(
                    width=design.current_line_width, gap=0
                ),
                start=cpwrl_fl_inductor_start,
                end=
                cpwrl_fl_inductor_start + DVector(
                    2 * abs(design.flux_lines_x_shifts[fl_line_idx]), 0
                )
            )
            cpwrl_fl_inductor.place(design.region_ph)
            cpwrl_fl_inductor_empty_box = Rectangle(
                origin=cpwrl_fl.end +
                       DVector(
                           0,
                           -design.current_line_width - 2 * design.z_md_fl.gap
                       ),
                width=cpwrl_fl_inductor.dr.abs(),
                height=2 * design.z_md_fl.gap,
                inverse=True
            )
            cpwrl_fl_inductor_empty_box.place(design.region_ph)

        draw_inductor_for_fl_line(self, 0)

        # place caplanar line 1 fl
        _p1 = self.contact_pads[2].end
        _p2 = self.contact_pads[2].end + DPoint(0, 0.5e6)
        _p3 = DPoint(
            self.squids[3].origin.x + self.flux_lines_x_shifts[3],
            self.squids[3].origin.y - self.cont_lines_y_ref
        )

        _p4 = DPoint(_p3.x, self.xmons[3].cpw_bempt.end.y)
        self.cpwrl_fl1 = DPathCPW(
            points=[_p1, _p2, _p3, _p4],
            cpw_parameters=self.z_md_fl,
            turn_radiuses=self.ctr_lines_turn_radius,
        )
        self.cpwrl_fl1.place(tmp_reg)
        self.cpw_fl_lines.append(self.cpwrl_fl1)

        draw_inductor_for_fl_line(self, 1)

        # place caplanar line 1 fl
        _p1 = self.contact_pads[3].end
        _p2 = self.contact_pads[3].end + DPoint(0, 0.5e6)
        _p3 = DPoint(
            self.squids[5].origin.x + self.flux_lines_x_shifts[1],
            self.squids[5].origin.y - self.cont_lines_y_ref
        )

        _p4 = DPoint(_p3.x, self.xmons[5].cpw_bempt.end.y)
        self.cpwrl_fl1 = DPathCPW(
            points=[_p1, _p2, _p3, _p4],
            cpw_parameters=self.z_md_fl,
            turn_radiuses=self.ctr_lines_turn_radius,
        )
        self.cpwrl_fl1.place(tmp_reg)
        self.cpw_fl_lines.append(self.cpwrl_fl1)

        draw_inductor_for_fl_line(self, 2)

    def draw_test_structures(self):
        # DRAW CONCTACT FOR BANDAGES WITH 5um CLEARANCE
        def augment_with_bandage_test_contacts(test_struct: TestStructurePadsSquare,
                                               test_jj: AsymSquidOneLegRigid = None):
            if test_jj.leg_side == 1:
                # DRAW FOR RIGHT LEG
                clearance = 5e3
                cb_left = CPW(
                    width=test_struct.rectangles_gap - clearance,
                    gap=0,
                    start=DPoint(
                        test_struct.top_rec.p1.x,
                        test_struct.bot_rec.p2.y
                        + test_struct.rectangles_gap / 2
                    ),
                    end=DPoint(
                        test_jj.origin.x - test_jj.top_ph_el_conn_pad.width / 2,
                        test_struct.bot_rec.p2.y +
                        test_struct.rectangles_gap / 2
                    )
                )
                cb_left.place(self.region_el)

                # DRAW RIGHT ONE
                clearance = 5e3
                cb_right = CPW(
                    width=test_struct1.rectangles_gap - clearance,
                    gap=0,
                    start=DPoint(
                        test_struct.top_rec.p2.x,
                        test_struct.bot_rec.p2.y
                        + test_struct.rectangles_gap / 2
                    ),
                    end=DPoint(
                        list(test_jj.bot_dc_flux_line_right.primitives.values())[1].end.x,
                        test_struct.bot_rec.p2.y +
                        test_struct.rectangles_gap / 2
                    )
                )
                cb_right.place(self.region_el)
            if test_jj.leg_side == -1:
                # DRAW FOR RIGHT LEG
                clearance = 5e3
                cb_left = CPW(
                    width=test_struct.rectangles_gap - clearance,
                    gap=0,
                    start=DPoint(
                        test_struct.top_rec.p1.x,
                        test_struct.bot_rec.p2.y
                        + test_struct.rectangles_gap / 2
                    ),
                    end=DPoint(
                        list(test_jj.bot_dc_flux_line_left.primitives.values())[1].end.x,
                        test_struct.bot_rec.p2.y +
                        test_struct.rectangles_gap / 2
                    )
                )
                cb_left.place(self.region_el)

                # DRAW RIGHT ONE
                clearance = 5e3
                cb_right = CPW(
                    width=test_struct.rectangles_gap - clearance,
                    gap=0,
                    start=DPoint(
                        test_struct.top_rec.p2.x,
                        test_struct.bot_rec.p2.y
                        + test_struct.rectangles_gap / 2
                    ),
                    end=DPoint(
                        test_jj.origin.x + test_jj.top_ph_el_conn_pad.width / 2,
                        test_struct.bot_rec.p2.y +
                        test_struct.rectangles_gap / 2
                    )
                )
                cb_right.place(self.region_el)

        struct_centers = [DPoint(1.5e6, 1.5e6), DPoint(5.2e6, 1.5e6),
                          DPoint(2e6, 3e6)]
        self.test_squids_pads = []
        for struct_center in struct_centers:
            ## JJ test structures ##

            # test structure with big critical current (#1)
            test_struct1 = TestStructurePadsSquare(
                struct_center,
                # gnd gap in test structure is now equal to
                # the same of first xmon cross, where polygon is placed
                squares_gap=self.xmons[0].sideY_face_gnd_gap
            )
            self.test_squids_pads.append(test_struct1)
            test_struct1.place(self.region_ph)
            text_reg = pya.TextGenerator.default_generator().text(
                "48.32 nA", 0.001, 50, False, 0, 0)
            text_bl = test_struct1.empty_rectangle.origin + DPoint(
                test_struct1.gnd_gap, -4 * test_struct1.gnd_gap
            )
            text_reg.transform(
                ICplxTrans(1.0, 0, False, text_bl.x, text_bl.y))
            self.region_ph -= text_reg

            # DRAW TEST SQUID
            squid_center = DPoint(test_struct1.center.x,
                                  test_struct1.bot_rec.p2.y
                                  )
            squid_center += DPoint(
                0,
                SQUID_PARAMETERS.sq_len / 2 +
                SQUID_PARAMETERS.flux_line_inner_width / 2 +
                self.squid_ph_clearance
            )
            test_jj = AsymSquidOneLegRigid(
                squid_center, SQUID_PARAMETERS, side=1,
                leg_side=1
            )
            self.test_squids.append(test_jj)
            test_jj.place(self.region_el)
            augment_with_bandage_test_contacts(test_struct1, test_jj)

            # test structure with low critical current
            test_struct2 = TestStructurePadsSquare(
                struct_center + DPoint(0.3e6, 0))
            self.test_squids_pads.append(test_struct2)
            test_struct2.place(self.region_ph)
            text_reg = pya.TextGenerator.default_generator().text(
                "9.66 nA", 0.001, 50, False, 0, 0)
            text_bl = test_struct2.empty_rectangle.origin + DPoint(
                test_struct2.gnd_gap, -4 * test_struct2.gnd_gap
            )
            text_reg.transform(
                ICplxTrans(1.0, 0, False, text_bl.x, text_bl.y))
            self.region_ph -= text_reg

            squid_center = DPoint(test_struct2.center.x,
                                  test_struct2.bot_rec.p2.y
                                  )
            squid_center += DPoint(
                0,
                SQUID_PARAMETERS.sq_len / 2 +
                SQUID_PARAMETERS.flux_line_inner_width / 2 +
                self.squid_ph_clearance
            )
            test_jj = AsymSquidOneLegRigid(
                squid_center, SQUID_PARAMETERS, side=-1,
                leg_side=-1
            )
            self.test_squids.append(test_jj)
            test_jj.place(self.region_el)
            augment_with_bandage_test_contacts(test_struct2, test_jj)

            # test structure for bridge DC contact
            test_struct3 = TestStructurePadsSquare(
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
            DPoint(1.1e6, 1.8e6),
            DPoint(5.5e6, 2.0e6),
            DPoint(2.6e6, 3.5e6)
        ]
        for struct_center in test_dc_el2_centers:
            test_struct1 = TestStructurePadsSquare(struct_center)
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
            dc_rec.place(self.dc_bandage_reg)

    def draw_el_dc_contacts(self):
        """
        TODO: add documentation and rework this function. It has to
            operate only with upper and lower polygons that are
             attached to the Tmon loop and has nothing to do with whether
             it is xmon and flux line or test pads squares.
        Returns
        -------

        """
        from itertools import chain
        for squid, contact in chain(
                zip(self.squids, self.xmons),
                zip(self.test_squids, self.test_squids_pads)
        ):

            # for optimization of boolean operations in the vicinitiy of
            # a squid
            photo_vicinity = self.region_ph & Region(
                DBox(
                    squid.origin + DPoint(100e3, 100e3),
                    squid.origin - DPoint(100e3, 100e3)
                )
            )
            # dc contact pad has to be completely
            # inside union of both  e-beam and photo deposed
            # metal regions.
            # `self.dc_cont_clearance` represents minimum distance
            # from dc contact pad`s perimeter to the perimeter of the
            # e-beam and photo-deposed metal perimeter.

            # collect all bottom contacts

            # philling `cut_regs` array that consists of metal regions
            # to be cutted from photo region
            cut_regs = [squid.top_ph_el_conn_pad.metal_region,
                        squid.pad_top.metal_region]
            for bot_contact in [squid.bot_dc_flux_line_right,
                                squid.bot_dc_flux_line_left]:
                # some bottom parts of squid can be missing (for test pads)
                if bot_contact is None:
                    continue
                bot_cut_regs = [
                    primitive.metal_region for primitive in
                    list(bot_contact.primitives.values())[:2]
                ]
                bot_cut_regs = bot_cut_regs[0] + bot_cut_regs[1]
                cut_regs.append(bot_cut_regs)

            # creating bandages
            for cut_reg in cut_regs:
                # find only those polygons in photo-layer that overlap with
                # bandage an return as a `Region()`
                contact_reg = cut_reg.pull_overlapping(photo_vicinity)
                # cut from photo region
                self.region_ph -= cut_reg

                # draw shrinked bandage on top of el region
                el_extension = extended_region(cut_reg, self.el_overlaps_ph_by) & contact_reg
                self.region_el += el_extension
                # self.region_el.merge()

                # correction of extension into the SQUID loop
                hwidth = squid.top_ph_el_conn_pad.width / 2 + self.dc_cont_el_clearance
                fix_box = DBox(
                    squid.origin + DPoint(-hwidth, 0),
                    squid.origin + DPoint(hwidth, squid.params.sq_len / 2 - squid.params.inter_leads_width / 2)
                )
                self.region_el -= Region(fix_box)

                el_bandage = extended_region(
                    cut_reg,
                    -self.dc_cont_el_clearance
                )
                self.dc_bandage_reg += el_bandage
                self.dc_bandage_reg.merge()

                # draw extended bandage in photo region
                ph_bandage = extended_region(
                    cut_reg, self.dc_cont_ph_ext)
                ph_bandage = ph_bandage & contact_reg
                ph_bandage = extended_region(
                    ph_bandage,
                    -self.dc_cont_ph_clearance
                )
                self.dc_bandage_reg += ph_bandage
                self.dc_bandage_reg.merge()

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
            DPoint(0.5e6, 0.5e6), DPoint(0.5e6, 4.5e6),
            DPoint(9.5e6, 0.5e6), DPoint(9.5e6, 4.5e6),
            DPoint(7.7e6, 1.7e6), DPoint(4.6e6, 3.2e6)
        ]
        for mark_center in marks_centers:
            self.marks.append(
                MarkBolgar(mark_center)
            )
            self.marks[-1].place(self.region_ph)

    def draw_bridges(self):
        bridges_step = 130e3
        fl_bridges_step = 130e3

        # for resonators
        for resonator in self.resonators:
            for name, res_primitive in resonator.primitives.items():
                if "coil" in name:
                    subprimitives = res_primitive.primitives
                    for primitive_name, primitive in subprimitives.items():
                        # place bridges only at arcs of coils
                        # but not on linear segments
                        if "arc" in primitive_name:
                            Bridge1.bridgify_CPW(
                                primitive, bridges_step,
                                dest=self.region_bridges1,
                                dest2=self.region_bridges2
                            )
                    continue
                elif "fork" in name:  # skip fork primitives
                    continue
                else:
                    # bridgify everything else except "arc1"
                    # resonator.primitives["arc1"] is arc that connects
                    # L_coupling with long vertical line for
                    # `EMResonatorTL3QbitWormRLTailXmonFork`
                    if name == "arc1":
                        continue
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
                    dest=self.region_bridges1, dest2=self.region_bridges2,
                    avoid_points=[squid.origin for squid in self.squids],
                    avoid_distance=200e3
                )
            elif "cpwrl_fl" in key:
                Bridge1.bridgify_CPW(
                    val, fl_bridges_step,
                    dest=self.region_bridges1, dest2=self.region_bridges2,
                    avoid_points=[squid.origin for squid in self.squids],
                    avoid_distance=400e3
                )

        for i, cpw_fl in enumerate(self.cpw_fl_lines):
            dy = 220e3
            bridge_center1 = cpw_fl.end + DVector(0, -dy)
            br = Bridge1(center=bridge_center1, trans_in=Trans.R90)
            br.place(dest=self.region_bridges1, region_name="bridges_1")
            br.place(dest=self.region_bridges2, region_name="bridges_2")

        # for readout waveguide
        bridgified_primitives_idxs = list(range(2))
        bridgified_primitives_idxs += list(
            range(2, 2 * (len(self.resonators) + 1) + 1, 2))
        bridgified_primitives_idxs += list(range(
            2 * (len(self.resonators) + 1) + 1,
            len(self.cpwrl_ro_line.primitives.values())
            )
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
        filled_reg = fill_holes(reg_to_fill, d=40e3, width=15e3,
                                height=15e3)

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

    # TODO: add layer or region arguments to the functions wich end with "..._in_layers()"
    def resolve_holes(self):
        for reg in (
                self.region_ph, self.region_bridges1, self.region_bridges2,
                self.region_el, self.dc_bandage_reg,
                self.region_el_protection):
            tmp_reg = Region()
            for poly in reg:
                tmp_reg.insert(poly.resolved_holes())
            reg.clear()
            reg |= tmp_reg

        # TODO: the following code is not working (region_bridges's polygons remain the same)
        # for poly in chain(self.region_bridges2):
        #     poly.resolve_holes()

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

    def get_resonator_length(self, res_idx):
        resonator = self.resonators[res_idx]
        res_length = resonator.L_coupling


def simulate_resonators_f_and_Q():
    freqs_span_corase = 1.0  # GHz
    corase_only = False
    freqs_span_fine = 0.050
    dl_list = [0e3, 15e3, -15e3]
    # dl_list = [0e3]
    from itertools import product

    for dl, resonator_idx in list(product(
            dl_list,
            range(8)
    )):
        fine_resonance_success = False
        freqs_span = freqs_span_corase

        design = Design5QTest("testScript")
        design.L1_list = [L1 + dl for L1 in design.L1_list]
        design.draw_for_res_f_and_Q_sim(resonator_idx)
        estimated_freq = \
            design.resonators[resonator_idx].get_approx_frequency(
                refractive_index=np.sqrt(6.26423)
            )
        # print("start drawing")
        print(estimated_freq)

        crop_box = (
                design.resonators[resonator_idx].metal_region +
                design.resonators[resonator_idx].empty_region +
                design.xmons[resonator_idx].metal_region +
                design.xmons[resonator_idx].empty_region
        ).bbox()

        # further from resonator edge of the readout CPW
        if resonator_idx%2==0:
            crop_box.bottom += design.Z_res.b / 2 - design.to_line_list[
                resonator_idx] - design.Z0.b / 2
        else:
            crop_box.top += -design.Z_res.b / 2 + design.to_line_list[
                resonator_idx] + design.Z0.b / 2

        box_extension = 100e3
        crop_box.bottom -= box_extension
        crop_box.top += box_extension
        crop_box.left -= box_extension
        crop_box.right += box_extension

        ### MESH CALCULATION SECTION START ###
        arr1 = np.round(np.array(
            design.to_line_list) - design.Z0.b / 2 - design.Z_res.b / 2)
        arr2 = np.array([box_extension, design.Z0.gap, design.Z0.width,
                         design.Z_res.width, design.Z_res.gap])
        arr = np.hstack((arr1, arr2))
        resolution_dy = np.gcd.reduce(arr.astype(int))
        # print(arr)
        # print(resolution_dy)
        # resolution_dy = 8e3
        resolution_dx = 2e3

        # cut part of the ground plane due to rectangular mesh in Sonnet
        if resonator_idx%2==0:
            crop_box.top = crop_box.top + int(crop_box.height() % resolution_dy)
        else:
            crop_box.bottom = crop_box.bottom - int(crop_box.height() % resolution_dy)
        print(crop_box.top, " ", crop_box.bottom)
        print(crop_box.height() / resolution_dy)
        ### MESH CALCULATION SECTION END ###

        design.crop(crop_box, region=design.region_ph)

        if resonator_idx%2==0:
            design.sonnet_ports = [
                DPoint(crop_box.left,
                       crop_box.bottom + box_extension + design.Z0.b / 2),
                DPoint(crop_box.right,
                       crop_box.bottom + box_extension + design.Z0.b / 2)
            ]
        else:
            design.sonnet_ports = [
                DPoint(crop_box.left,
                       crop_box.top - box_extension - design.Z0.b / 2),
                DPoint(crop_box.right,
                       crop_box.top - box_extension - design.Z0.b / 2)
            ]
        # transforming cropped box to the origin
        dr = DPoint(0, 0) - crop_box.p1
        design.transform_region(
            design.region_ph,
            DTrans(dr.x, dr.y),
            trans_ports=True
        )
        # [print((port.x, port.y)) for port in design.sonnet_ports]
        # transfer design`s regions shapes to the corresponding layers in layout
        design.show()
        # show layout in UI window
        design.lv.zoom_fit()

        import os
        project_dir = os.path.dirname(__file__)
        design.layout.write(
            os.path.join(project_dir, f"res_f_Q_{resonator_idx}_{dl}_um.gds")
        )

        ### RESONANCE FINDING SECTION START ###
        while not fine_resonance_success:
            # fine_resonance_success = True  # NOTE: FOR DEBUG
            ### SIMULATION SECTION START ###
            ml_terminal = SonnetLab()
            # print("starting connection...")
            from sonnetSim.cMD import CMD

            ml_terminal._send(CMD.SAY_HELLO)
            ml_terminal.clear()
            simBox = SimulationBox(
                crop_box.width(), crop_box.height(),
                crop_box.width() / resolution_dx,
                crop_box.height() / resolution_dy
            )

            # if freqs_span == freqs_span_corase:
            ml_terminal.set_boxProps(simBox)
            # print("sending cell and layer")
            from sonnetSim.pORT_TYPES import PORT_TYPES

            ports = [
                SonnetPort(design.sonnet_ports[0], PORT_TYPES.BOX_WALL),
                SonnetPort(design.sonnet_ports[1], PORT_TYPES.BOX_WALL)
            ]
            ml_terminal.set_ports(ports)
            ml_terminal.send_polygons(design.cell, design.layer_ph)
            ml_terminal.set_ABS_sweep(estimated_freq - freqs_span / 2,
                                      estimated_freq + freqs_span / 2)
            print(f"simulating...{resonator_idx}")
            result_path = ml_terminal.start_simulation(wait=True)
            ml_terminal.release()

            """
            intended to be working ONLY IF:
            s12 is monotonically increasing or decreasing over the chosen frequency band.
            That generally holds true for circuits with single resonator.
            """
            with open(result_path.decode('ascii'), "r",
                      newline='') as file:
                # exctracting s-parameters in csv format
                # though we do not have csv module
                rows = [row.split(',') for row in
                        list(file.readlines())[8:]]
                freqs = [float(row[0]) for row in rows]  # rows in GHz
                df = freqs[1] - freqs[0]  # frequency error
                s12_list = [float(row[3]) + 1j * float(row[4]) for row in
                            rows]
                s12_abs_list = [abs(s12) for s12 in s12_list]
                min_freq_idx, min_s21_abs = min(enumerate(s12_abs_list),
                                                key=lambda x: x[1])
                min_freq = freqs[min_freq_idx]
                min_freq_idx = len(s12_abs_list) / 2  # Note: FOR DEBUG

            # processing the results
            if min_freq_idx == 0:
                # local minimum is located to the left of current interval
                # => shift interval to the left and try again
                derivative = (s12_list[1] - s12_list[0]) / df
                second_derivative = (s12_list[2] - 2 * s12_list[1] +
                                     s12_list[0]) / df ** 2
                print('resonance located the left of the current interval')
                # try adjacent interval to the left
                estimated_freq -= freqs_span
                continue
            elif min_freq_idx == (len(freqs) - 1):
                # local minimum is located to the right of current interval
                # => shift interval to the right and try again
                derivative = (s12_list[-1] - s12_list[-2]) / df
                second_derivative = (s12_list[-1] - 2 * s12_list[-2] +
                                     s12_list[-3]) / df ** 2
                print(
                    'resonance located the right of the current interval')
                # try adjacent interval to the right
                estimated_freq += freqs_span
                continue
            else:
                # local minimum is within current interval
                print(f"fr = {min_freq:3.5} GHz,  fr_err = {df:.5}")
                estimated_freq = min_freq
                if freqs_span == freqs_span_corase:
                    if corase_only:
                        # terminate simulation after corase simulation
                        fine_resonance_success = True
                    else:
                        # go to fine approximation step
                        freqs_span = freqs_span_fine
                        continue
                elif freqs_span == freqs_span_fine:
                    # fine approximation ended, go to saving the result
                    fine_resonance_success = True  # breaking frequency locating cycle condition is True

            # unreachable code:
            # TODO: add approximation of the resonance if minimum is nonlocal during corase approximation
            # fr_approx = (2*derivative/second_derivative) + min_freq
            # B = -4*derivative**3/second_derivative**2
            # A = min_freq - 2*derivative**2/second_derivative
            # print(f"fr = {min_freq:3.3} GHz,  fr_err = not implemented(")
            ### RESONANCE FINDING SECTION END  ###

            ### RESULT SAVING SECTION START ###
            import shutil
            import os
            import csv

            # geometry parameters gathering
            res_params = design.resonators[
                resonator_idx].get_geometry_params_dict(prefix="worm_")
            Z0_params = design.Z0.get_geometry_params_dict(
                prefix="S21Line_")

            from collections import OrderedDict

            all_params = OrderedDict(
                itertools.chain(
                    res_params.items(),
                    Z0_params.items(),
                    {
                        "to_line, um": design.to_line_list[
                                           resonator_idx] / 1e3,
                        "filename": None,
                        "resonator_idx": resonator_idx
                    }.items()
                )
            )

            # creating directory with simulation results
            results_dirname = "resonators_S21"
            results_dirpath = os.path.join(project_dir, results_dirname)

            output_metaFile_path = os.path.join(
                results_dirpath,
                "resonator_waveguide_Q_freq_meta.csv"
            )
            try:
                # creating directory
                os.mkdir(results_dirpath)
            except FileExistsError:
                # directory already exists
                with open(output_metaFile_path, "r+",
                          newline='') as csv_file:
                    reader = csv.reader(csv_file)
                    existing_entries_n = len(list(reader))
                    all_params["filename"] = "result_" + str(
                        existing_entries_n) + ".csv"

                    writer = csv.writer(csv_file)
                    # append new values row to file
                    writer.writerow(list(all_params.values()))
            else:
                '''
                    Directory did not exist and has been created sucessfully.
                    So we create fresh meta-file.
                    Meta-file contain simulation parameters and corresponding
                    S-params filename that is located in this directory
                '''
                with open(output_metaFile_path, "w+", newline='') as csv_file:
                    writer = csv.writer(csv_file)
                    # create header of the file
                    writer.writerow(list(all_params.keys()))
                    # add first parameters row
                    reader = csv.reader(csv_file)
                    existing_entries_n = len(list(reader))
                    all_params["filename"] = "result_1.csv"
                    writer.writerow(list(all_params.values()))
            finally:
                # copy result from sonnet folder and rename it accordingly
                shutil.copy(
                    result_path.decode("ascii"),
                    os.path.join(results_dirpath, all_params["filename"])
                )
            ### RESULT SAVING SECTION END ###


def simulate_Cqr():
    resolution_dx = 1e3
    resolution_dy = 1e3
    dl_list = [10e3, 0, -10e3]
    # dl_list = [0e3]
    from itertools import product

    for dl, res_idx in list(
            product(
                dl_list, range(5)
            )
    ):
        ### DRAWING SECTION START ###
        design = Design5QTest("testScript")
        design.fork_y_spans = [fork_y_span + dl for fork_y_span in design.fork_y_spans]
        design.draw_for_Cqr_simulation(res_idx=res_idx)

        worm = design.resonators[res_idx]
        xmonCross = design.xmons[res_idx]
        worm_start = list(worm.primitives.values())[0].start

        # draw open end at the resonators start
        p1 = worm_start - DVector(design.Z_res.b / 2, 0)
        rec = Rectangle(p1, design.Z_res.b, design.Z_res.b / 2, inverse=True)
        rec.place(design.region_ph)

        if worm_start.x < xmonCross.center.x:
            dr = (worm_start - xmonCross.cpw_r.end)
        else:
            dr = (worm_start - xmonCross.cpw_l.end)
        dr.x = abs(dr.x)
        dr.y = abs(dr.y)

        box_side_x = 8 * xmonCross.sideX_length
        box_side_y = 8 * xmonCross.sideY_length
        dv = DVector(box_side_x / 2, box_side_y / 2)

        crop_box = pya.Box().from_dbox(pya.Box(
            xmonCross.center + dv,
            xmonCross.center + (-1) * dv
        ))
        design.crop(crop_box)
        dr = DPoint(0, 0) - crop_box.p1

        # finding the furthest edge of cropped resonator`s central line polygon
        # sonnet port will be attached to this edge
        reg1 = worm.metal_region & Region(crop_box)
        reg1.merge()
        max_distance = 0
        port_pt = None
        for poly in reg1.each():
            for edge in poly.each_edge():
                edge_center = (edge.p1 + edge.p2) / 2
                dp = edge_center - xmonCross.cpw_b.end
                d = max(abs(dp.x), abs(dp.y))
                if d > max_distance:
                    max_distance = d
                    port_pt = edge_center
        design.sonnet_ports.append(port_pt)
        design.sonnet_ports.append(xmonCross.cpw_b.end)

        design.transform_region(design.region_ph, DTrans(dr.x, dr.y), trans_ports=True)

        design.show()
        design.lv.zoom_fit()
        ### DRAWING SECTION END ###

        ### SIMULATION SECTION START ###
        ml_terminal = SonnetLab()
        # print("starting connection...")
        from sonnetSim.cMD import CMD

        ml_terminal._send(CMD.SAY_HELLO)
        ml_terminal.clear()
        simBox = SimulationBox(
            crop_box.width(),
            crop_box.height(),
            crop_box.width() / resolution_dx,
            crop_box.height() / resolution_dy
        )

        ml_terminal.set_boxProps(simBox)
        # print("sending cell and layer")
        from sonnetSim.pORT_TYPES import PORT_TYPES

        ports = [
            SonnetPort(design.sonnet_ports[0], PORT_TYPES.AUTOGROUNDED),
            SonnetPort(design.sonnet_ports[1], PORT_TYPES.AUTOGROUNDED)
        ]
        # for sp in ports:
        #     print(sp.point)
        ml_terminal.set_ports(ports)

        ml_terminal.send_polygons(design.cell, design.layer_ph)
        ml_terminal.set_linspace_sweep(0.01, 0.01, 1)
        # print("simulating...")
        result_path = ml_terminal.start_simulation(wait=True)
        ml_terminal.release()

        ### SIMULATION SECTION END ###

        import shutil
        import os
        import csv

        project_dir = os.path.dirname(__file__)

        ### CALCULATE C_QR CAPACITANCE SECTION START ###
        C12 = None
        with open(result_path.decode("ascii"), "r") as csv_file:
            data_rows = list(csv.reader(csv_file))
            ports_imps_row = data_rows[6]
            R = float(ports_imps_row[0].split(' ')[1])
            data_row = data_rows[8]
            freq0 = float(data_row[0])

            s = [[0, 0], [0, 0]]  # s-matrix
            # print(data_row)
            for i in range(0, 2):
                for j in range(0, 2):
                    s[i][j] = complex(float(data_row[1 + 2 * (i * 2 + j)]), float(data_row[1 + 2 * (i * 2 + j) + 1]))
            import math

            y11 = 1 / R * (1 - s[0][0]) / (1 + s[0][0])
            C1 = -1e15 / (2 * math.pi * freq0 * 1e9 * (1 / y11).imag)
            # formula taken from https://en.wikipedia.org/wiki/Admittance_parameters#Two_port
            delta = (1 + s[0][0]) * (1 + s[1][1]) - s[0][1] * s[1][0]
            y21 = -2 * s[1][0] / delta * 1 / R
            C12 = 1e15 / (2 * math.pi * freq0 * 1e9 * (1 / y21).imag)

        print(design.fork_y_spans[res_idx] / 1e3, design.xmon_dys_Cg_coupling[res_idx] / 1e3, C12, C1)
        ### CALCULATE C_QR CAPACITANCE SECTION START ###

        ### SAVING REUSLTS SECTION START ###
        design.layout.write(
            os.path.join(project_dir, f"Cqr_{res_idx}_{dl}_um.gds")
        )
        output_filepath = os.path.join(project_dir, "Xmon_resonator_Cqr_results.csv")
        if os.path.exists(output_filepath):
            # append data to file
            with open(output_filepath, "a", newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(
                    [res_idx, design.fork_y_spans[res_idx] / 1e3, C12, C1]
                )
        else:
            # create file, add header, append data
            with open(output_filepath, "w", newline='') as csv_file:
                writer = csv.writer(csv_file)
                # create header of the file
                writer.writerow(
                    ["res_idx", "fork_y_span, um", "C12, fF", "C1, fF"])
                writer.writerow(
                    [res_idx, design.fork_y_spans[res_idx] / 1e3, C12, C1]
                )

        ### SAVING REUSLTS SECTION END ###


if __name__ == "__main__":
    design = Design5QTest("testScript")
    design.draw()
    design.show()
    # simulate_resonators_f_and_Q()
    # simulate_Cqr()
