'''
Changes log

v.0.0.0.1
Based on 5Q_3.1.1.2
'''
# Enter your Python code here
from math import cos, sin, tan, atan2, pi, degrees
import itertools
from typing import List, Dict, Union, Optional
from numbers import Number
from copy import deepcopy
import os
import shutil
import csv

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
    CPWRLPath, Bridge1, CPW2CPW
from classLib.shapes import TmonT, Rectangle, CutMark
from classLib.resonators import EMResonatorTL3QbitWormRLTailXmonFork
from classLib.josJ import AsymSquid, AsymSquidParams
from classLib.chipTemplates import CHIP_16p5x16p5_20pads, FABRICATION
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
FABRICATION.OVERETCHING = 0.0e3
PROJECT_DIR = os.path.dirname(__file__)


class TestStructurePadsSquare(ComplexBase):
    def __init__(self, center, trans_in=None, square_a=200e3,
                 gnd_gap=20e3, pads_gap=20e3):
        self.center = center
        self.rectangle_a = square_a
        self.gnd_gap = gnd_gap
        self.pads_gap = pads_gap

        self.empty_rectangle: Rectangle = None
        self.top_rec: Rectangle = None
        self.bot_rec: Rectangle = None
        super().__init__(center, trans_in)

    def init_primitives(self):
        center = DPoint(0, 0)

        ## empty rectangle ##
        empty_width = self.rectangle_a + 2 * self.gnd_gap
        empty_height = 2 * self.rectangle_a + 2 * self.gnd_gap + \
                       self.pads_gap
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
                                   self.pads_gap / 2)
        self.top_rec = Rectangle(
            bl_point, self.rectangle_a, self.rectangle_a
        )
        self.primitives["top_rec"] = self.top_rec

        ## bottom rectangle ##
        # bottom-left point of rectangle
        bl_point = center + DPoint(
            -self.rectangle_a / 2,
            - self.pads_gap / 2 - self.rectangle_a
        )
        self.bot_rec = Rectangle(
            bl_point, self.rectangle_a, self.rectangle_a
        )
        self.primitives["bot_rec"] = self.bot_rec

        self.connections = [center]

    def _refresh_named_connections(self):
        self.center = self.connections[0]


SQUID_PARS = AsymSquidParams(
    band_ph_tol=1e3,
    squid_dx=14.2e3,
    squid_dy=10e3,
    TC_dx=2.5e3 * np.sqrt(2) + 1e3,
    TC_dy=5e3*np.sqrt(2)/2 + 1e3,
    TCW_dy=6e3,
    TCW_dx=0.5e3,
    BCW_dy=0e3,
    BC_dy=5e3*np.sqrt(2)/2 + 1e3,
    BC_dx=2.5e3 * np.sqrt(2) + 1e3
)

VERT_ARR_SHIFT = DVector(-50e3, -150e3)


class Design8Q(ChipDesign):
    def __init__(self, cell_name):
        super().__init__(cell_name)
        self.__version = "8Q_0.0.0.1"
        # print(self.__version)
        dc_bandage_layer_i = pya.LayerInfo(3,
                                           0)  # for DC contact deposition
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
        self.chip = CHIP_16p5x16p5_20pads
        self.chip_box: pya.DBox = self.chip.box
        # Z = 49.5656 E_eff = 6.30782 (E = 11.45)
        self.z_fl1: CPWParameters = CPWParameters(10e3, 5.7e3)
        self.z_md1: CPWParameters = CPWParameters(10e3, 5.7e3)
        # Z = 50.136  E_eff = 6.28826 (E = 11.45)
        self.z_fl2: CPWParameters = CPWParameters(10e3, 5.7e3)
        self.z_md2: CPWParameters = CPWParameters(4e3, 4e3)
        # flux line widths at the end of flux line
        self.flux2ground_left_width = 2e3
        self.flux2ground_right_width = 4e3
        # squid parameters
        self.squid_pars = AsymSquidParams()
        # readout coplanar transverse geometry parameters
        self.ro_Z: CPWParameters = self.chip.chip_Z
        # contact pads obejct array
        self.contact_pads: list[ContactPad] = self.chip.get_contact_pads(
            [self.ro_Z] * 2 + [self.z_md1, self.z_fl1] * 4 + [self.ro_Z] * 2 + [self.z_md1, self.z_fl1] * 4,
            back_metal_gap=200e3,
            back_metal_width=0e3,
            pad_length=700e3,
            transition_len=250e3
        )
        # xmon parameters
        self.NQUBITS = 8
        self.xmon_x_distance: float = 722e3  # from simulation of g_12
        # distance between open end (excluding fork) of resonator
        # and cross polygons
        self.xmon_res_d = 254e3
        self.xmon_dys_Cg_coupling = [14e3] * self.NQUBITS
        self.xmons: list[TmonT] = []
        self.xmons_corrected: list[TmonT] = []

        # C11 = 107
        self.cross_len_x = 270e3
        self.cross_width_x = 45e3  # from C11 sim
        self.cross_gnd_gap_x_list = np.array([60e3] * self.NQUBITS)
        self.cross_gnd_gap_face_x = 35e3
        self.cross_len_y = 180e3
        self.cross_width_y = 45e3  # from C11 sim
        self.cross_gnd_gap_y_list = [
            1e3 * x for x in [26.955, 48.462, 27.669, 49.494, 28.311, 53.563, 29.764, 55.887]
        ]
        self.cross_gnd_gap_face_y = 30e3

        # readout line parameters
        self.ro_line_turn_radius: float = 100e3
        self.ro_line_dy: float = 1600e3
        self.cpwrl_ro_line1: DPathCPW = None
        # primitive indexes of `self.cpwrl_ro_line1` that will be covered
        # with airbridges in `draw_bridges`
        self.cpwrl_ro_line1_idxs2bridgify: list[int] = []
        self.cpwrl_ro_line2: DPathCPW = None
        # primitive indexes of `self.cpwrl_ro_line2` that will be covered
        # with airbridges in `draw_bridges`
        self.cpwrl_ro_line2_indxs2bridgify: list[int] = []
        self.Z0: CPWParameters = self.chip.chip_Z

        # resonators objects and parameters
        self.resonators: List[EMResonatorTL3QbitWormRLTailXmonFork] = []
        # distance between nearest resonators central conductors centers
        # constant step between resonators origin points along x-axis.
        self.resonators_dx: float = self.xmon_x_distance
        # resonator parameters
        self.L_coupling_list: list[float] = [
            1e3 * x for x in [310, 320, 320, 310] * 2
        ]
        # corresponding to resonanse freq is linspaced in
        # interval [7.2,7.44] GHz
        self.L0 = 986e3
        # long vertical line length
        self.L0_list = [self.L0] * self.NQUBITS
        # from f_res, Q_res simulations
        # horizontal coil line length
        self.L1_list = [
            1e3 * x for x in
            [114.5219, 95.1897, 99.0318, 83.7159, 88.8686, 70.3649,74.0874, 59.6982]
        ]
        # curvature radius of resonators CPW turns
        self.res_r = 60e3
        # coil consist of L1, 2r, L1, 2r segment
        self.N_coils = [3, 3, 3, 3] * 2
        # another vertical line connected to L0
        self.L2_list = [self.res_r] * len(self.L1_list)
        # horizontal line connected to L2
        self.L3_list = [0e3] * len(self.L1_list)  # to be constructed
        # vertical line connected to L3
        self.L4_list = [self.res_r] * len(self.L1_list)
        # Z = 51.0, E_eff = 6.29
        self.Z_res = CPWParameters(10e3, 6e3)
        self.to_line_list = [45e3] * len(self.L1_list)
        # fork at the end of resonator parameters
        self.fork_metal_width = 15e3
        self.fork_gnd_gap = 10e3
        self.xmon_fork_gnd_gap = 14e3
        self.fork_x_span = self.cross_width_y + + 2 * \
                           (self.xmon_fork_gnd_gap + self.fork_metal_width)
        # resonator-fork parameters
        # from simulation of g_qr
        self.fork_y_span_list = [
            x * 1e3 for x in [33.18, 91.43, 39.36, 95.31, 44.34, 96.58, 49.92, 99.59]
        ]

        # bandages
        self.bandage_width = 2.5e3 * np.sqrt(2)
        self.bandage_height = 5e3 * np.sqrt(2)
        self.bandage_r_outer = 2e3
        self.bandage_r_inner = 2e3
        self.bandage_curve_pts_n = 40
        self.bandages_regs_list = []

        # squids
        self.squids: List[AsymSquid] = []
        self.test_squids: List[AsymSquid] = []

        ''' SQUID POSITIONING AND PARAMETERS SECTION STARAT '''
        # vertical shift of every squid local origin coordinates
        # this line puts squid center on the
        # "outer capacitor plate's edge" of the Tmon shunting capacitance.
        self.squid_vertical_shifts_list = \
            -np.array(self.cross_gnd_gap_x_list/2)
        _shift_into_substrate = 1.5e3
        # next step is to make additional shift such that top of the
        # BC polygon is located at the former squid center position with
        # additional 1.5 um shift in direction of substrate
        _shift_to_BC_center = SQUID_PARS.shadow_gap/2 + \
                             SQUID_PARS.SQLBT_dy + SQUID_PARS.SQB_dy + \
                             SQUID_PARS.BCW_dy + _shift_into_substrate
        self.squid_vertical_shifts_list += _shift_to_BC_center
        # self.squid_vertical_shifts_list[4:] -= _shift_to_BC_center

        # now make `SQUID_PARS.TCW.dy` such that bottom of TC
        # sticks out into subtrate by 1.5 um
        SQUID_PARS.TCW_dy = self.cross_gnd_gap_x_list[0] - \
                            (_shift_into_substrate +
                             SQUID_PARS.BCW_dy + SQUID_PARS.SQLBT_dy +
                             SQUID_PARS.shadow_gap +
                             SQUID_PARS.SQLTJJ_dy +
                             SQUID_PARS.SQLTT_dy + SQUID_PARS.SQT_dy
                             + _shift_into_substrate)
        ''' SQUID POSITIONING AND PARAMETERS SECTION END '''

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
        self.ctr_lines_turn_radius = 100e3
        self.ctrLine_2_qLine_d = 800e3
        self.ctr_lines_y_ref: float = None  # nm

        self.flLine_squidLeg_gap = 5e3
        self.flux_lines_x_shifts: List[float] = \
            [
                -SQUID_PARS.squid_dx / 2 - SQUID_PARS.SQLBT_dx / 2 -
                self.z_fl2.width / 2 +
                SQUID_PARS.BC_dx / 2 + SQUID_PARS.band_ph_tol
            ] * len(self.L1_list)

        # curve that is used for intermediate parking points for
        # control line of qubits group №1
        self._ic_alpha_start1 = (- 1 + 1 / 6) * np.pi  # ic === intermediate curve
        self._ic_alpha_end1 = -1 / 2 * np.pi
        self._ic_alpha_list1 = np.linspace(self._ic_alpha_start1,
                                           self._ic_alpha_end1, 7)
        self._ic_r1 = 3 / 2 * 3 * self.xmon_x_distance
        self.ic_pts1 = []

        # curve that is used for intermediate parking points for
        # control line of qubits group №2
        self._ic_alpha_start2 = 1 / 2 * np.pi
        self._ic_alpha_end2 = 1 / 6 * np.pi
        self._ic_alpha_list2 = np.linspace(self._ic_alpha_start2,
                                           self._ic_alpha_end2, 7)
        self._ic_r2 = self._ic_r1
        self.ic_pts2 = []

        # shift from middle of cross bottom finder
        # where md line should end
        # for qubits 0-3
        self.md_line_end_shift_x = -80e3
        self.md_line_end_shift_y = -129559
        self.md_line_end_shift = DVector(self.md_line_end_shift_x, self.md_line_end_shift_y)
        # distance from end of md control line (metal)
        # to cross (metal) for qubits 0 and 7
        self.md07_x_dist = 28686

        # length of the smoothing part between normal thick and end-thin cpw for md line
        self.md_line_cpw12_smoothhing = 10e3

        # length ofa thin part of the microwave drive line end
        self.md_line_cpw2_len = 550e3

        self.cpwrl_md0: DPathCPW = None
        self.cpwrl_fl0: DPathCPW = None

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

        self.cpwrl_md6: DPathCPW = None
        self.cpwrl_fl6: DPathCPW = None

        self.cpwrl_md7: DPathCPW = None
        self.cpwrl_fl7: DPathCPW = None

        self.cpw_md_lines: List[DPathCPW] = []
        self.cpw_fl_lines: List[DPathCPW] = []

        # marks
        self.marks: List[MarkBolgar] = []

        # length scale for most indents
        self.l_scale = max(3 * self.ro_Z.b, self.ro_line_turn_radius)

        # test structures
        self.test_squids_pads: List[TestStructurePadsSquare] = []
        ### ADDITIONAL VARIABLES SECTION END ###

    def draw(self, design_parameters=None):
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
        self.create_resonator_objects()
        self.draw_xmons_and_resonators()
        self.draw_readout_waveguide()

        self.draw_josephson_loops()

        self.draw_microwave_drvie_lines()
        self.draw_flux_control_lines()
        # self.draw_coupling_res()

        self.draw_test_structures()
        self.draw_express_test_structures_pads()
        self.draw_bandages()
        self.draw_recess()
        self.region_el.merge()
        self.draw_el_protection()
        # #
        self.draw_photo_el_marks()
        self.draw_bridges()
        self.draw_pinning_holes()
        # # v.0.3.0.8 p.12 - ensure that contact pads has no holes
        for contact_pad in self.contact_pads:
            contact_pad.place(self.region_ph)
        self.draw_cut_marks()
        self.extend_photo_overetching()
        self.inverse_destination(self.region_ph)
        # convert to gds acceptable polygons (without inner holes)
        self.resolve_holes()
        # convert to litograph readable format. Litograph can't handle
        # polygons with more than 200 vertices.
        self.split_polygons_in_layers(max_pts=180)

    def draw_for_res_f_and_Q_sim(self, res_idxs2Draw):
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
        self.draw_xmons_and_resonators(res_idxs2Draw=res_idxs2Draw)
        self.draw_readout_waveguide()

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
        self.draw_xmons_and_resonators(res_idxs2Draw=[res_idx])

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
        resonators_widths = [2 * self.res_r + L_coupling for L_coupling in
                             self.L_coupling_list]
        L3_arr = [0.0] * 8
        L3_arr[0] = (
                resonators_widths[0] / 2
        )
        for i in range(1, self.NQUBITS):
            L3_arr[i] = L3_arr[i - 1] + self.xmon_x_distance - \
                        self.resonators_dx

        res_tail_shape = "LRLRL"
        tail_turn_radiuses = self.res_r

        # along single horizontal line
        # self.L2_list[0] += 6 * self.Z_res.b
        # self.L2_list[1] += 0
        # self.L2_list[3] += 3 * self.Z_res.b

        # self.L3_list[0] = x1
        # self.L3_list[1] = x2
        # self.L3_list[2] = x3
        # self.L3_list[3] = x4

        # self.L4_list[1] += 6 * self.Z_res.b
        # self.L4_list[2] += 6 * self.Z_res.b
        # self.L4_list[3] += 3 * self.Z_res.b

        # for vertical xmon line
        # self.L2_list[4] += 6 * self.Z_res.b
        # self.L2_list[5] += 0
        # self.L2_list[7] += 3 * self.Z_res.b

        # self.L3_list[4] = x5
        # self.L3_list[5] = x2
        # self.L3_list[6] = x3
        # self.L3_list[7] = x4
        self.L3_list = np.array(self.L3_list) + np.array(L3_arr)

        # self.L4_list[5] += 6 * self.Z_res.b

        tail_segment_lengths_list = [[L2, L3, L4]
                                     for L2, L3, L4 in
                                     zip(self.L2_list, self.L3_list,
                                         self.L4_list)]
        tail_turn_angles_list = [
            [np.pi / 2, -np.pi / 2],
            [np.pi / 2, -np.pi / 2],
            [np.pi / 2, -np.pi / 2],
            [np.pi / 2, -np.pi / 2]
        ]
        tail_turn_angles_list = tail_turn_angles_list + list(reversed(
            tail_turn_angles_list))
        tail_trans_in_list = [
                                 Trans.R270,
                                 Trans.R270,
                                 Trans.R270,
                                 Trans.R270
                             ] * 2
        ''' RESONATORS TAILS CALCULATIONS SECTION END '''

        pars = list(
            zip(
                self.L1_list, self.to_line_list, self.L_coupling_list,
                self.fork_y_span_list,
                tail_segment_lengths_list, tail_turn_angles_list,
                tail_trans_in_list,
                self.L0_list, self.N_coils
            )
        )

        # deduction for resonator placements
        # under condition that Xmon-Xmon distance equals
        # `xmon_x_distance`
        worm_x = []
        worm_y = []  # calculated based on resonator dimensions
        for res_idx in range(int(self.NQUBITS)):
            worm_x.append(
                self.chip_box.center().x + (
                        res_idx - 3.5) * self.resonators_dx
            )

        # create horizontal qubit line resonators twice
        # copy with idx=[4,5,6,7] will be transformed versions
        # of [0,1,2,3] resonators
        for res_idx, params in enumerate(pars):
            # print(res_idx)
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

            res = EMResonatorTL3QbitWormRLTailXmonFork(
                self.Z_res, DPoint(worm_x[res_idx], 0),
                L_coupling,
                L0=L0,
                L1=L1, r=self.res_r, N=n_coils,
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
            self.resonators.append(res)
            worm_y.append(
                self.chip_box.center().y + self.xmon_res_d +
                abs((res.start - res.end).y)
            )
            res.make_trans(DCplxTrans(1, 0, False, 0, worm_y[-1]))

        # print([self.L0 - xmon_dy_Cg_coupling for xmon_dy_Cg_coupling in  self.xmon_dys_Cg_coupling])
        # print(self.L1_list)
        # print(self.L2_list)
        # print(self.L3_list)
        # print(self.L4_list)

    def draw_xmons_and_resonators(self, res_idxs2Draw: List[int] = None):
        """
        Fills photolitography Region() instance with resonators
        and crosses.

        Parameters
        ----------
        res_idx2Draw : int
            draw only particular resonator (if passed)
            used in resonator simulations.


        Returns
        -------
        None
        """
        # draw first 4 Xmons
        it_list = list(
            enumerate(
                zip(self.resonators, self.fork_y_span_list)
            )
        )
        for res_idx, (res, fork_y_span) in it_list:
            xmon_center = res.end + DVector(0, -self.xmon_res_d)
            xmonCross = TmonT(
                xmon_center,
                sideX_length=self.cross_len_x,
                sideX_width=self.cross_width_x,
                sideX_gnd_gap=self.cross_gnd_gap_x_list[res_idx],
                sideY_length=self.cross_len_y,
                sideY_width=self.cross_width_y,
                sideY_gnd_gap=self.cross_gnd_gap_y_list[res_idx],
                sideX_face_gnd_gap=self.cross_gnd_gap_face_x,
                sideY_face_gnd_gap=self.cross_gnd_gap_face_y
            )

            self.xmons.append(xmonCross)
            xmonCross_corrected = TmonT(
                xmon_center,
                sideX_length=self.cross_len_x,
                sideX_width=self.cross_width_x,
                sideX_gnd_gap=self.cross_gnd_gap_x_list[res_idx],
                sideX_face_gnd_gap=self.cross_gnd_gap_face_x,
                sideY_length=self.cross_len_y,
                sideY_width=self.cross_width_y,
                sideY_gnd_gap=max(
                    0,
                    self.fork_x_span - 2 * self.fork_metal_width -
                    self.cross_width_y -
                    max(self.cross_gnd_gap_y_list[res_idx], self.fork_gnd_gap)
                ) / 2
            )
            self.xmons_corrected.append(xmonCross_corrected)

            # resonators transforms before `place()` logic
            if res_idx >= 4:  # transform resonator
                trans1 = DCplxTrans(1, 0, False, -DVector(xmon_center))
                trans2 = DCplxTrans(1, 0, True, 0, 0)
                trans3 = DCplxTrans(1, 0, False, DVector(xmon_center))
                res_trans = trans3 * trans2 * trans1
                res.make_trans(res_trans)
                self.xmons[-1].make_trans(res_trans)
                self.xmons_corrected[-1].make_trans(res_trans)

            # used in simulation regime
            # Draw only custom resonators with indexes from `res_idxs2Draw`
            if res_idxs2Draw is None:
                pass
            else:
                if res_idx not in res_idxs2Draw:
                    continue

            # print(res_idx)
            # print(self.cross_len_x)
            # print(self.cross_width_x)
            # print(self.cross_len_y)
            # print()
            # place xmon
            self.xmons[-1].place(self.region_ph)
            # place resonator with fork. May corrupt xmon cross due to
            # ground space around fork teeth.
            res.place(self.region_ph)
            # repair xmon cross
            xmonCross_corrected.place(self.region_ph)

    def draw_readout_waveguide(self):
        """
            Subdividing horizontal waveguide adjacent to resonators into
            several waveguides.
            Even segments of this adjacent waveguide are adjacent to
            resonators.
            Bridges will be placed on odd segments later.

            Returns
            -------
            None
            """

        ## calculating segment lengths of subdivided coupling  part of
        # ro coplanar ##

        # value that need to be added to `L_coupling`  to get width of
        # resonators bbox.
        def get_res_extension(
                resonator: EMResonatorTL3QbitWormRLTailXmonFork):
            return resonator.Z0.b + 2 * resonator.r

        def get_res_width(
                resonator: EMResonatorTL3QbitWormRLTailXmonFork):
            return (resonator.L_coupling + get_res_extension(
                resonator))

        # 1st readout line
        p1 = self.contact_pads[1].end
        p_last = self.contact_pads[0].end
        # start of readout
        p2 = p1 + DVector(2 * self.l_scale, 0)
        p3 = self.resonators[0].start + \
             DVector(
                 -get_res_extension(self.resonators[0]) / 2 -
                 2 * self.ro_line_turn_radius,
                 self.to_line_list[0]
             )
        p4 = self.resonators[3].start + \
             DVector(
                 -get_res_extension(self.resonators[3]) / 2 +
                 get_res_width(self.resonators[3]) +
                 2 * self.ro_line_turn_radius,
                 self.to_line_list[3]
             )
        p5 = p4 + DVector(0, 2 * self.l_scale)
        p6 = p_last + DVector(2 * self.l_scale, 0)
        self.cpwrl_ro_line1 = DPathCPW(
            points=[p1, p2, p3, p4, p5, p6, p_last],
            cpw_parameters=self.ro_Z,
            turn_radiuses=self.ro_line_turn_radius
        )
        self.cpwrl_ro_line1.place(self.region_ph)

        # 2nd readout line
        p1 = self.contact_pads[10].end
        p_last = self.contact_pads[11].end
        # start of readout
        p2 = p1 + DVector(-2 * self.l_scale, 0)
        p4 = self.resonators[4].start + \
             DVector(
                 -get_res_extension(self.resonators[4]) / 2 -
                 2 * self.ro_line_turn_radius,
                 -self.to_line_list[4]
             )
        p3 = p4 + DVector(0, -2 * self.l_scale)
        p5 = self.resonators[7].start + \
             DVector(
                 -get_res_extension(self.resonators[7]) / 2 +
                 get_res_width(self.resonators[7]) +
                 2 * self.ro_line_turn_radius,
                 -self.to_line_list[7]
             )
        p6 = p_last + DVector(-2 * self.l_scale, 0)
        self.cpwrl_ro_line2 = DPathCPW(
            points=[p1, p2, p3, p4, p5, p6, p_last],
            cpw_parameters=self.ro_Z,
            turn_radiuses=self.ro_line_turn_radius
        )
        self.cpwrl_ro_line2.place(self.region_ph)

    def draw_josephson_loops(self):
        # place left squid
        dx = SQUID_PARS.SQB_dx / 2 - SQUID_PARS.SQLBT_dx / 2
        pars_local = deepcopy(SQUID_PARS)
        pars_local.bot_wire_x = [-dx, dx]
        pars_local.SQB_dy = 0

        for xmon_idx, xmon in enumerate(self.xmons):
            if xmon_idx < 4:  # 1st group
                squid_center = xmon.cpw_bempt.center() + \
                               DVector(0, self.squid_vertical_shifts_list[xmon_idx])
                squid = AsymSquid(
                    squid_center, pars_local
                )
                self.squids.append(squid)
                squid.place(self.region_el)
            elif xmon_idx >= 4:  # 2nd group
                squid_center = xmon.cpw_bempt.center() + \
                               DVector(0,
                                       -self.squid_vertical_shifts_list[
                                           xmon_idx])
                squid = AsymSquid(
                    squid_center, pars_local,
                    trans_in=Trans.R180
                )
                self.squids.append(squid)
                squid.place(self.region_el)

        self.region_el.merge()
        # refuse rounding for new technology process "Daria K. 22.03.2022"
        # self.region_el.round_corners(0.5e3, 0.5e3, 40)

    def _help_routing_ctr_lines(self):
        if len(self.xmons) < 8:
            raise RuntimeError("xmons to draw control lines to are not "
                               "drawn themselves yet. Fill `self.xmons` "
                               "list with appropriate data structures")
        else:
            self.ctr_lines_y_ref1 = self.xmons[0].center.y - \
                                    self.ctrLine_2_qLine_d
            self.ctr_lines_y_ref2 = self.xmons[-1].center.y + \
                                    self.ctrLine_2_qLine_d
            if len(self.ic_pts1) == 0:
                ic_center1 = self.xmons[3].center
                for alpha in self._ic_alpha_list1:
                    self.ic_pts1.append(
                        ic_center1 +
                        self._ic_r1 * DVector(np.cos(alpha), np.sin(alpha))
                    )
            if len(self.ic_pts2) == 0:
                ic_center2 = self.xmons[4].center
                for alpha in self._ic_alpha_list2:
                    self.ic_pts2.append(
                        ic_center2 +
                        self._ic_r2 * DVector(np.cos(alpha), np.sin(alpha))
                    )

    def draw_microwave_drvie_lines(self):
        self._help_routing_ctr_lines()
        r_turn = self.ctr_lines_turn_radius
        ''' for qubit group №1 '''
        # place caplanar line 0md
        p1 = self.contact_pads[2].end
        p2 = self.xmons[0].cpw_lempt.end + DVector(-self.md07_x_dist, 0)
        self.cpwrl_md0 = DPathCPW(
            points=[p1, p2],
            cpw_parameters=[self.z_md1],
            turn_radiuses=r_turn
        )
        self.cpw_md_lines.append(self.cpwrl_md0)

        # place md1-md3
        for q_idx, (cp_idx, ccurve_pt) in enumerate(
                zip([4, 6, 8], self.ic_pts1[1::2]),
                start=1
        ):
            cp = self.contact_pads[cp_idx]
            cross = self.xmons[q_idx]
            p1 = cp.end

            cp_dv = cp.end - cp.start
            cp_dv /= cp_dv.abs()

            p2 = p1 + 2 * r_turn * cp_dv
            cross_dv = cross.cpw_b.dr.dup()
            cross_dv /= cross_dv.abs()
            p3 = ccurve_pt
            p4 = DPoint(
                self.xmons[q_idx].cpw_b.end.x + self.md_line_end_shift.x,
                self.ctr_lines_y_ref1
            )
            p5 = DPoint(
                p4.x,
                self.xmons[q_idx].cpw_b.end.y + self.md_line_end_shift.y
            )
            md_dpath = DPathCPW(
                points=[p1, p2, p3, p4, p5],
                cpw_parameters=[self.z_md1] * 7,
                turn_radiuses=r_turn
            )
            self.__setattr__("cpwrl_md_" + str(q_idx), md_dpath)
            self.cpw_md_lines.append(md_dpath)

        ''' for qubits group №2 '''
        # place md4-md7
        for q_idx, (cp_idx, ccurve_pt) in enumerate(
                zip([18, 16, 14], self.ic_pts2[1::2]),
                start=4
        ):
            cp = self.contact_pads[cp_idx]
            cross = self.xmons[q_idx]
            p1 = cp.end

            cp_dv = cp.end - cp.start
            cp_dv /= cp_dv.abs()

            p2 = p1 + 2 * r_turn * cp_dv
            cross_dv = cross.cpw_b.dr.dup()
            cross_dv /= cross_dv.abs()
            p3 = ccurve_pt
            p4 = DPoint(
                self.xmons[q_idx].cpw_b.end.x - self.md_line_end_shift.x,
                self.ctr_lines_y_ref2
            )
            p5 = DPoint(
                p4.x,
                self.xmons[q_idx].cpw_b.end.y - self.md_line_end_shift.y
            )
            md_dpath = DPathCPW(
                points=[p1, p2, p3, p4, p5],
                cpw_parameters=[self.z_md1] * 7,
                turn_radiuses=r_turn
            )
            self.__setattr__("cpwrl_md_" + str(q_idx), md_dpath)
            self.cpw_md_lines.append(md_dpath)

        # place caplanar line 7md
        p1 = self.contact_pads[12].end
        p2 = self.xmons[-1].cpw_rempt.end + DVector(self.md07_x_dist, 0)
        self.cpwrl_md7 = DPathCPW(
            points=[p1, p2],
            cpw_parameters=[self.z_md1],
            turn_radiuses=r_turn
        )
        self.cpw_md_lines.append(self.cpwrl_md7)
        for cpw_md_line in self.cpw_md_lines:
            self.modify_md_line_end_and_place(
                cpw_md_line, mod_length=self.md_line_cpw2_len, smoothing=self.md_line_cpw12_smoothhing
            )

    def modify_md_line_end_and_place(self, md_line: DPathCPW, mod_length=100e3, smoothing=20e3):
        """
        Changes coplanar for `mod_length` length from the end of `md_line`.
        Transition region length along the `md_line` is controlled by passing `smoothing` value.

        Notes
        -------
        Unhandled behaviour if transition (smoothing) region overlaps with any number of `CPWArc`s.

        Parameters
        ----------
        md_line : DPathCPW
            line object to modify
        mod_length : float
            length counting from the end of line to be modified
        smoothing : float
            length of smoothing for CPW2CPW transition between normal-thick and end-thin parts

        Returns
        -------
        None
        """
        # make flux line wider to have a less chance to misalign
        # bandage-eBeam-photo layers at the qubit bandage region.
        last_lines = {}
        total_length = 0
        for key, primitive in reversed(list(md_line.primitives.items())):
            total_length += primitive.length()
            last_lines[key] = (primitive)
            if total_length > mod_length:
                break

        # iterate from the end of the line
        # TODO: this is not working, objects already drawn in their corresponding coordinate system
        for key, primitive in list(last_lines.items())[:-1]:
            primitive.width = self.z_md2.width
            primitive.gap = self.z_md2.gap

        transition_line = list(last_lines.values())[-1]
        length_to_mod_left = mod_length - sum([primitive.length() for primitive in list(last_lines.values())[:-1]])
        # divide line into 3 sections with proportions `alpha_i`
        beta_2 = smoothing
        beta_3 = length_to_mod_left
        beta_1 = transition_line.length() - (beta_2 + beta_3)
        transition_line_dv = transition_line.end - transition_line.start
        # tangent vector
        transition_line_dv_s = transition_line_dv / transition_line_dv.abs()
        p1 = transition_line.start
        p2 = p1 + beta_1 * transition_line_dv_s
        p3 = p2 + beta_2 * transition_line_dv_s
        p4 = p3 + beta_3 * transition_line_dv_s
        cpw_transition_line1 = CPW(
            start=p1,
            end=p2,
            cpw_params=self.z_md1
        )
        cpw_transition = CPW2CPW(
            Z0=self.z_md1,
            Z1=self.z_md2,
            start=p2, end=p3
        )
        cpw_transition_line2 = CPW(
            start=p3 - 2 * transition_line_dv_s,  # rounding error correction
            end=p4,
            cpw_params=self.z_md2
        )

        transition_line_name = list(last_lines.keys())[-1]
        new_primitives = {}
        for key, primitive in md_line.primitives.items():
            if key != transition_line_name:
                new_primitives[key] = primitive
            elif key == transition_line_name:
                new_primitives[transition_line_name + "_1"] = cpw_transition_line1
                new_primitives[transition_line_name + "_2"] = cpw_transition
                new_primitives[transition_line_name + "_3"] = cpw_transition_line2

        # make open-circuit end for md line end
        cpw_end = list(last_lines.values())[0]
        end_dv = cpw_end.end - cpw_end.start
        end_dv_s = end_dv / end_dv.abs()
        p1 = cpw_end.end
        p2 = p1 + cpw_end.b / 2 * end_dv_s
        md_open_end = CPW(
            start=p1, end=p2,
            width=0, gap=cpw_transition_line2.b / 2
        )
        new_primitives["md_open_end"] = md_open_end
        md_line.primitives = new_primitives
        md_line.connections[1] = list(md_line.primitives.values())[-1].end
        md_line._refresh_named_connections()
        md_line.place(self.region_ph)

    def draw_flux_control_lines(self):
        self._help_routing_ctr_lines()
        r_turn = self.ctr_lines_turn_radius

        # place fl0,fl1,fl2,fl3
        for q_idx, (cp_idx, ccurve_pt) in enumerate(
                zip([3, 5, 7, 9], self.ic_pts1[::2]),
                start=0
        ):
            cp = self.contact_pads[cp_idx]
            cross = self.xmons[q_idx]
            squid = self.squids[q_idx]
            p1 = cp.end

            # direction from contact towards chip center
            cp_dv = cp.end - cp.start
            cp_dv /= cp_dv.abs()

            p2 = p1 + 2 * r_turn * cp_dv
            # direction from xmon center towards squid loop
            cross_dv = cross.cpw_t.dr.dup()
            cross_dv /= cross_dv.abs()

            p3 = ccurve_pt
            p4 = DPoint(
                squid.center.x + self.flux_lines_x_shifts[q_idx],
                self.ctr_lines_y_ref1
            )
            p5 = DPoint(
                p4.x,
                cross.cpw_bempt.end.y
            )

            fl_dpath = DPathCPW(
                points=[p1, p2, p3, p4, p5],
                cpw_parameters=self.z_fl1,
                turn_radiuses=r_turn
            )
            self.__setattr__("cpwrl_fl" + str(q_idx), fl_dpath)
            self.cpw_fl_lines.append(fl_dpath)

        # place fl4,fl5,fl6, fl7
        for q_idx, (cp_idx, ccurve_pt) in enumerate(
                zip([-1, -3, -5, -7], self.ic_pts2[::2]),
                start=4
        ):
            cp = self.contact_pads[cp_idx]
            cross = self.xmons[q_idx]
            squid = self.squids[q_idx]
            p1 = cp.end

            # direction from contact towards chip center
            cp_dv = cp.end - cp.start
            cp_dv /= cp_dv.abs()

            p2 = p1 + 2 * r_turn * cp_dv
            # direction from xmon center towards squid loop
            cross_dv = cross.cpw_t.dr.dup()
            cross_dv /= cross_dv.abs()

            p3 = ccurve_pt
            p4 = DPoint(
                squid.center.x - self.flux_lines_x_shifts[q_idx],
                self.ctr_lines_y_ref2
            )
            p5 = DPoint(
                p4.x,
                cross.cpw_bempt.end.y
            )

            fl_dpath = DPathCPW(
                points=[p1, p2, p3, p4, p5],
                cpw_parameters=self.z_fl1,
                turn_radiuses=r_turn
            )
            self.__setattr__("cpwrl_fl" + str(q_idx), fl_dpath)
            self.cpw_fl_lines.append(fl_dpath)
        #
        # # place caplanar line 7 fl
        # p1 = self.contact_pads[-6].end
        # p2 = p1 + DVector(-2 * r_turn, 0)
        # # TODO: hardcoded value 100e3
        # p3 = DPoint(p2.x, self.ctr_lines_y_ref + 100e3)
        # p4 = DPoint(
        #     self.squids[7].center.x + self.flux_lines_x_shifts[7],
        #     p3.y
        # )
        # p5 = DPoint(
        #     p4.x,
        #     self.xmons[7].center.y - self.xmons[7].cpw_r.b / 2
        # )
        # self.cpwrl_fl7 = DPathCPW(
        #     points=[p1, p2, p3, p4, p5],
        #     cpw_parameters=self.z_fl1,
        #     turn_radiuses=r_turn
        # )
        # self.cpw_fl_lines.append(self.cpwrl_fl7)

        for flux_line in self.cpw_fl_lines:
            self.modify_flux_line_end_and_place(flux_line)
        self.region_ph.merge()

    def modify_flux_line_end_and_place(self, flux_line: DPathCPW):
        # make flux line wider to have a less chance to misalign
        # bandage-eBeam-photo layers at the qubit bandage region.
        last_line = list(flux_line.primitives.values())[-1]
        last_line_name = list(flux_line.primitives.keys())[-1]

        # divide line into 3 sections with proportions `alpha_i`
        alpha_1 = 0.6
        alpha_2 = 0.3
        alpha_3 = 1 - (alpha_1 + alpha_2)
        last_line_dv = last_line.end - last_line.start
        last_line_dv_s = last_line_dv / last_line_dv.abs()
        p1 = last_line.start
        p2 = p1 + alpha_1 * last_line_dv
        p3 = p2 + alpha_2 * last_line_dv
        p4 = p3 + alpha_3 * last_line_dv
        cpw_normal = CPW(
            start=p1,
            end=p2,
            width=self.z_fl1.width,
            gap=self.z_fl1.gap
        )
        cpw_transition = CPW2CPW(
            Z0=self.z_fl1,
            Z1=self.z_fl2,
            start=p2, end=p3
        )
        cpw_thick = CPW(
            start=p3 - 2 * last_line_dv_s,  # rounding error correction
            end=p4,
            width=self.z_fl2.width,
            gap=self.z_fl2.gap
        )

        del flux_line.primitives[last_line_name]
        flux_line.primitives[last_line_name + "_1"] = cpw_normal
        flux_line.primitives[last_line_name + "_2"] = cpw_transition
        flux_line.primitives[last_line_name + "_3"] = cpw_thick

        # tangent vector
        last_line_dv_s = last_line_dv / last_line_dv.abs()
        # normal vector (90deg counterclockwise rotated tangent vector)
        last_line_dv_n = DVector(-last_line_dv_s.y, last_line_dv_s.x)

        # draw mutual inductance for squid, shunted to ground to the right
        # (view from flux line start towards squid)
        p1 = cpw_thick.end - cpw_thick.width / 2 * last_line_dv_n - \
             self.flux2ground_right_width / 2 * last_line_dv_s
        p2 = p1 - cpw_thick.gap * last_line_dv_n
        inductance_cpw = CPW(
            start=p1, end=p2,
            width=self.flux2ground_right_width, gap=0
        )
        flux_line.primitives["fl_end_ind_right"] = inductance_cpw

        # connect central conductor to ground to the left (view from
        # flux line start towards squid)
        p1 = cpw_thick.end + cpw_thick.width / 2 * last_line_dv_n - \
             self.flux2ground_left_width / 2 * last_line_dv_s
        p2 = p1 + cpw_thick.gap * last_line_dv_n
        flux_gnd_cpw = CPW(
            start=p1, end=p2,
            width=self.flux2ground_left_width, gap=0
        )
        flux_line.primitives["fl_end_ind_left"] = flux_gnd_cpw
        flux_line.connections[1] = list(flux_line.primitives.values())[-3].end
        flux_line._refresh_named_connections()
        flux_line.place(self.region_ph)

    def draw_coupling_res(self):
        # TODO: draw appropriate coupling
        self.coupling_cpw = CPW(
            start=self.xmons[3].cpw_rempt.end +
                  DPoint(self.coupling_resonator_dx, 0),
            end=self.xmons[4].cpw_lempt.end +
                DPoint(-self.coupling_resonator_dx, 0),
            cpw_params=self.Z_res
        )
        self.coupling_cpw.place(self.region_ph)

        self.cpw_empty1 = CPW(
            width=0, gap=self.Z_res.b / 2,
            start=self.coupling_cpw.start,
            end=self.coupling_cpw.start + DVector(-self.Z_res.b / 2, 0)
        )
        self.cpw_empty1.place(self.region_ph)

        self.cpw_empty2 = CPW(
            width=0, gap=self.Z_res.b / 2,
            start=self.coupling_cpw.end,
            end=self.coupling_cpw.end + DVector(self.Z_res.b / 2, 0)
        )
        self.cpw_empty2.place(self.region_ph)

    def draw_test_structures(self):
        struct_centers = [DPoint(4e6, 9.0e6), DPoint(12.0e6, 9.0e6),
                          DPoint(12.0e6, 7.1e6)]
        for struct_center in struct_centers:
            ## JJ test structures ##
            dx = SQUID_PARS.SQB_dx / 2 - SQUID_PARS.SQLBT_dx / 2

            # test structure with big critical current (#1)
            test_struct1 = TestStructurePadsSquare(
                struct_center,
                # gnd gap in test structure is now equal to
                # the same of first xmon cross, where polygon is placed
                gnd_gap=20e3,
                pads_gap=self.xmons[0].sideX_gnd_gap
            )
            self.test_squids_pads.append(test_struct1)
            test_struct1.place(self.region_ph)

            text_reg = pya.TextGenerator.default_generator().text(
                "56 nA", 0.001, 25, False, 0, 0)
            text_bl = test_struct1.empty_rectangle.p1 - DVector(0, 20e3)
            text_reg.transform(
                ICplxTrans(1.0, 0, False, text_bl.x, text_bl.y))
            self.region_ph -= text_reg

            pars_local = deepcopy(SQUID_PARS)
            pars_local.SQRBT_dx = 0
            pars_local.SQRBJJ_dy = 0
            pars_local.bot_wire_x = [-dx]

            squid_center = test_struct1.center
            test_jj = AsymSquid(
                squid_center + DVector(0,
                                       self.squid_vertical_shifts_list[0]),
                pars_local
            )
            self.test_squids.append(test_jj)
            test_jj.place(self.region_el)

            # test structure with low critical current (#2)
            test_struct2 = TestStructurePadsSquare(
                struct_center + DPoint(0.3e6, 0),
                gnd_gap=20e3,
                pads_gap=self.xmons[0].sideX_gnd_gap
            )
            self.test_squids_pads.append(test_struct2)
            test_struct2.place(self.region_ph)

            text_reg = pya.TextGenerator.default_generator().text(
                "11 nA", 0.001, 25, False, 0, 0)
            text_bl = test_struct2.empty_rectangle.p1 - DVector(0, 20e3)
            text_reg.transform(
                ICplxTrans(1.0, 0, False, text_bl.x, text_bl.y))
            self.region_ph -= text_reg

            pars_local = deepcopy(SQUID_PARS)
            pars_local.SQLBT_dx = 0
            pars_local.SQLBJJ_dy = 0
            pars_local.bot_wire_x = [dx]

            squid_center = test_struct2.center
            test_jj = AsymSquid(
                squid_center + DVector(0,
                                       self.squid_vertical_shifts_list[0]),
                pars_local
            )
            self.test_squids.append(test_jj)
            test_jj.place(self.region_el)

            # test structure for bridge DC contact (#3)
            test_struct3 = TestStructurePadsSquare(
                struct_center + DPoint(0.6e6, 0))
            test_struct3.place(self.region_ph)
            text_reg = pya.TextGenerator.default_generator().text(
                "DC", 0.001, 25, False, 0, 0
            )
            text_bl = test_struct3.empty_rectangle.p1 - DVector(0, 20e3)
            text_reg.transform(
                ICplxTrans(1.0, 0, False, text_bl.x, text_bl.y)
            )
            self.region_ph -= text_reg

            test_bridges = []
            for i in range(3):
                bridge = Bridge1(
                    test_struct3.center + DPoint(50e3 * (i - 1), 0),
                    gnd_touch_dx=20e3
                )
                test_bridges.append(bridge)
                bridge.place(self.region_bridges1, region_name="bridges_1")
                bridge.place(self.region_bridges2, region_name="bridges_2")

        # bandages test structures
        test_dc_el2_centers = [
            DPoint(3.3e6, 9.0e6),
            DPoint(13.3e6, 9.0e6),
            DPoint(13.3e6, 7.1e6)
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
            rec_height = test_struct1.pads_gap + 2 * rec_width
            p1 = struct_center - DVector(rec_width / 2, rec_height / 2)
            dc_rec = Rectangle(p1, rec_width, rec_height)
            dc_rec.place(self.dc_bandage_reg)

    def draw_express_test_structures_pads(self):
        for squid, test_pad in zip(
                self.test_squids,
                self.test_squids_pads
        ):
            thick_pad_dx = 2 * test_pad.gnd_gap
            if squid.squid_params.SQRBJJ_dy == 0:
                ## only left JJ is present ##
                # test pad expanded to the right
                p1 = squid.TCW.center()
                thin_pad_dx = test_pad.top_rec.p2.x - p1.x - thick_pad_dx
                p2 = p1 + DVector(thin_pad_dx, 0)
                etc1 = CPW(
                    start=p1, end=p2,
                    width=1e3,
                    gap=0
                )
                etc1.place(self.region_el)

                p3 = DPoint(p2.x, test_pad.center.y)
                p4 = DPoint(test_pad.top_rec.p2.x, p3.y)
                etc2 = CPW(
                    start=p3, end=p4,
                    width=test_pad.pads_gap - 4e3,
                    gap=0
                )
                etc2.place(self.region_el)

                # test pad expanded to the left
                p1 = squid.SQLBT.center()
                thin_pad_dx = p1.x - test_pad.top_rec.p1.x - thick_pad_dx
                p2 = p1 - DVector(thin_pad_dx, 0)
                etc3 = CPW(
                    start=p1, end=p2,
                    width=1e3,  # TODO: hardcoded value
                    gap=0
                )
                etc3.place(self.region_el)

                p3 = DPoint(p2.x, test_pad.center.y)
                p4 = DPoint(test_pad.top_rec.p1.x, test_pad.center.y)
                etc4 = CPW(
                    start=p3, end=p4,
                    width=test_pad.pads_gap - 4e3,
                    gap=0
                )
                etc4.place(self.region_el)

            elif squid.squid_params.SQLBJJ_dy == 0:
                ## only right leg is present ##
                # test pad expanded to the left
                p1 = squid.TCW.center()
                thin_pad_dx = p1.x - test_pad.top_rec.p1.x - thick_pad_dx
                p2 = p1 + DVector(-thin_pad_dx, 0)
                etc1 = CPW(
                    start=p1, end=p2,
                    width=1e3,
                    gap=0
                )
                etc1.place(self.region_el)

                p3 = DPoint(p2.x, test_pad.center.y)
                p4 = DPoint(test_pad.top_rec.p1.x, p3.y)
                etc2 = CPW(
                    start=p3, end=p4,
                    width=test_pad.pads_gap - 4e3,
                    gap=0
                )
                etc2.place(self.region_el)

                # test pad expanded to the right
                p1 = squid.SQRBT.center()
                thin_pad_dx = test_pad.top_rec.p2.x - p1.x - thick_pad_dx
                p2 = p1 + DVector(thin_pad_dx, 0)
                etc3 = CPW(
                    start=p1, end=p2,
                    width=1e3,  # TODO: hardcoded value
                    gap=0
                )
                etc3.place(self.region_el)

                p3 = DPoint(p2.x, test_pad.center.y)
                p4 = DPoint(test_pad.top_rec.p2.x, test_pad.center.y)
                etc4 = CPW(
                    start=p3, end=p4,
                    width=test_pad.pads_gap - 4e3,
                    gap=0
                )
                etc4.place(self.region_el)

    def draw_bandages(self):
        """
        Returns
        -------

        """
        from itertools import chain
        for squid, contact in chain(
                zip(self.squids, self.xmons),
                zip(self.test_squids, self.test_squids_pads)
        ):
            # dc contact pad has to be completely
            # inside union of both  e-beam and photo deposed
            # metal regions.
            # `self.dc_cont_clearance` represents minimum distance
            # from dc contact pad`s perimeter to the perimeter of the
            # e-beam and photo-deposed metal perimeter.
            self.bandages_regs_list += self.draw_squid_bandage(
                squid,
                shift2sq_center=0
            )
            # collect all bottom contacts

    def draw_squid_bandage(self, test_jj: AsymSquid = None,
                           shift2sq_center=0):
        # squid direction from bottom to top
        squid_BT_dv = test_jj.TC.start - test_jj.TC.end
        squid_BT_dv_s = squid_BT_dv/squid_BT_dv.abs()  # normalized

        bandages_regs_list: List[Region] = []

        # top bandage
        top_bandage_reg = self._get_bandage_reg(
            center=test_jj.TC.start,
            shift=-shift2sq_center * squid_BT_dv_s
        )
        bandages_regs_list.append(top_bandage_reg)
        self.dc_bandage_reg += top_bandage_reg

        # bottom contacts
        for i, _ in enumerate(test_jj.squid_params.bot_wire_x):
            BC = getattr(test_jj, "BC" + str(i))
            bot_bandage_reg = self._get_bandage_reg(
                center=BC.end,
                shift=shift2sq_center * squid_BT_dv_s
            )
            bandages_regs_list.append(bot_bandage_reg)
            self.dc_bandage_reg += bot_bandage_reg
        return bandages_regs_list

    def _get_bandage_reg(self, center, shift: DVector = DVector(0, 0)):
        center += shift
        rect_lb = center + \
                  DVector(
                      -self.bandage_width / 2,
                      -self.bandage_height / 2
                  )
        bandage_reg = Rectangle(
            origin=rect_lb,
            width=self.bandage_width,
            height=self.bandage_height
        ).metal_region
        bandage_reg.round_corners(
            self.bandage_r_inner,
            self.bandage_r_outer,
            self.bandage_curve_pts_n
        )

        return bandage_reg

    def draw_recess(self):
        for squid in itertools.chain(self.squids, self.test_squids):
            # top recess
            recess_reg = squid.TC.metal_region.dup().size(-1e3)
            self.region_ph -= recess_reg

            # bottom recess(es)
            for i, _ in enumerate(squid.squid_params.bot_wire_x):
                BC = getattr(squid, "BC" + str(i))
                recess_reg = BC.metal_region.dup().size(-1e3)
                self.region_ph -= recess_reg

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
            DPoint(2.5e6, 11.5e6), DPoint(7.5e6, 11.5e6),
            DPoint(14.5e6, 11.5e6),
            DPoint(2.5e6, 5.0e6), DPoint(12.5e6, 5.5e6),
            DPoint(9.5e6, 5.0e6)
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
                cpwrl_md = val
                Bridge1.bridgify_CPW(
                    cpwrl_md, bridges_step,
                    dest=self.region_bridges1, dest2=self.region_bridges2,
                    avoid_points=[squid.origin for squid in self.squids]
                )
            elif "cpwrl_fl" in key:
                cpwrl_fl = val
                Bridge1.bridgify_CPW(
                    cpwrl_fl, fl_bridges_step,
                    dest=self.region_bridges1, dest2=self.region_bridges2,
                    avoid_points=[squid.origin for squid in self.squids],
                    avoid_distances=200e3
                )

        # close bridges for cpw_fl line
        for i, cpw_fl in enumerate(self.cpw_fl_lines):
            dy_list = [30e3, 130e3]
            for dy in dy_list:
                if i < 4:
                    pass
                elif i >= 4:
                    dy = -dy
                bridge_center1 = cpw_fl.end + DVector(0, -dy)
                br = Bridge1(center=bridge_center1, trans_in=Trans.R90)
                br.place(dest=self.region_bridges1,
                         region_name="bridges_1")
                br.place(dest=self.region_bridges2,
                         region_name="bridges_2")

        # close bridges for cpw_md line
        # for i, cpw_md in enumerate(self.cpw_md_lines):
        #     dy_list = [55e3, 200e3]
        #     for dy in dy_list:
        #         if i == 0:
        #             bridge_center1 = cpw_md.end + DVector(-dy, 0)
        #         elif i == 4:
        #             bridge_center1 = cpw_md.end + DVector(dy, 0)
        #             br = Bridge1(center=bridge_center1,
        #                          trans_in=None)
        #         else:
        #             bridge_center1 = cpw_md.end + DVector(0, -dy)
        #             br = Bridge1(center=bridge_center1,
        #                          trans_in=Trans.R90)
        #         br.place(dest=self.region_bridges1,
        #                  region_name="bridges_1")
        #         br.place(dest=self.region_bridges2,
        #                  region_name="bridges_2")

        # for readout waveguides
        avoid_points = []
        avoid_distances = []
        for res in self.resonators:
            av_pt = res.origin + DPoint(res.L_coupling / 2, 0)
            avoid_points.append(av_pt)
            av_dist = res.L_coupling / 2 + res.r + res.Z0.b / 2
            avoid_distances.append(av_dist)

        Bridge1.bridgify_CPW(
            self.cpwrl_ro_line1, bridges_step=bridges_step,
            dest=self.region_bridges1, dest2=self.region_bridges2,
            avoid_points=avoid_points, avoid_distances=avoid_distances
        )
        Bridge1.bridgify_CPW(
            self.cpwrl_ro_line2, bridges_step=bridges_step,
            dest=self.region_bridges1, dest2=self.region_bridges2,
            avoid_points=avoid_points, avoid_distances=avoid_distances
        )

    def draw_pinning_holes(self):
        # points that select polygons of interest if they were clicked at
        selection_pts = [
            Point(0.1e6, 0.1e6),
            Point(3.5e6, 11.2e6),
            (self.cpwrl_ro_line2.start + self.cpwrl_ro_line2.end) / 2
        ]

        # creating region of small boxes (polygon selection requires
        # regions)
        dv = DVector(1e3, 1e3)
        selection_regions = [
            Region(pya.Box(pt, pt + dv)) for pt in selection_pts
        ]
        selection_region = Region()
        for reg in selection_regions:
            selection_region += reg

        other_polys_reg = self.region_ph.dup().select_not_interacting(
            selection_region
        )

        reg_to_fill = self.region_ph.dup().select_interacting(
            selection_region
        )
        filled_reg = fill_holes(reg_to_fill, d=40e3, width=15e3,
                                height=15e3)

        self.region_ph = filled_reg + other_polys_reg

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

    # TODO: add layer or region
    #  arguments to the functions wich names end with "..._in_layers()"
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
        return res_length


def simulate_resonators_f_and_Q(resolution=(4e3,4e3)):
    resolution_dx = resolution[0]
    resolution_dy = resolution[1]
    freqs_span_corase = 1  # GHz
    corase_only = False
    freqs_span_fine = 0.050
    dl_list = [15e3, 0, -15e3]
    estimated_freqs = np.linspace(7.2, 7.76, 8)
    # dl_list = [0e3]
    from itertools import product
    for dl, (resonator_idx, predef_freq) in list(product(
            dl_list,
            zip(range(8), estimated_freqs),
    )):
        print()
        print("res №", resonator_idx)
        fine_resonance_success = False
        freqs_span = freqs_span_corase

        design = Design8Q("testScript")
        design.L1_list[resonator_idx] += dl
        # print(f"res length: {design.L1_list[resonator_idx]:3.5} um")
        design.draw_for_res_f_and_Q_sim(res_idxs2Draw=[resonator_idx])

        an_estimated_freq = \
            design.resonators[resonator_idx].get_approx_frequency(
                refractive_index=np.sqrt(6.26423)
            )
        # print(f"formula estimated freq: {an_estimated_freq:3.5} GHz")
        estimated_freq = predef_freq
        # print("start drawing")
        # print(f"previous result estimated freq: {estimated_freq:3.5} GHz")
        # print(design.resonators[resonator_idx].length(exception="fork"))

        crop_box = (
                design.resonators[resonator_idx].metal_region +
                design.resonators[resonator_idx].empty_region +
                design.xmons[resonator_idx].metal_region +
                design.xmons[resonator_idx].empty_region
        ).bbox()

        # center of the readout CPW
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
        crop_box.bottom = crop_box.bottom - int(
            crop_box.height() % resolution_dy)
        # print(crop_box.top, " ", crop_box.bottom)
        # print(crop_box.height() / resolution_dy)
        ### MESH CALCULATION SECTION END ###

        design.crop(crop_box, region=design.region_ph)

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

        # transfer design`s regions shapes to the corresponding layers in layout
        design.show()
        # show layout in UI window
        design.lv.zoom_fit()

        design.layout.write(
            os.path.join(PROJECT_DIR,
                         f"res_f_Q_{resonator_idx}_{dl}_um.gds")
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
            # print(f"simulating...{resonator_idx}")
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
                # min_freq_idx = len(s12_abs_list) / 2  # Note: FOR DEBUG
            print("min freq idx: ", min_freq_idx, "/", len(freqs))
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

            all_params = design.get_geometry_parameters()

            # creating directory with simulation results
            results_dirname = "resonators_S21"
            results_dirpath = os.path.join(PROJECT_DIR, results_dirname)

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
                with open(output_metaFile_path, "w+",
                          newline='') as csv_file:
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


def simulate_resonators_f_and_Q_together():
    freqs_span_corase = 1  # GHz
    corase_only = False
    freqs_span_fine = 0.050
    # dl_list = [15e3, 0, -15e3]
    estimated_freq = 7.5
    dl_list = [0e3]  # shift for every resonator
    res_idxs = [0, 1, 2, 3]
    fine_resonance_success = False
    freqs_span = freqs_span_corase

    design = Design8Q("testScript")
    for res_idx, dl in zip(res_idxs, dl_list):
        design.L1_list[res_idx] += dl

    design.draw_for_res_f_and_Q_sim(res_idxs2Draw=res_idxs)

    total_reg = Region()
    for res_idx in res_idxs:
        total_reg += design.resonators[res_idx].metal_region
        total_reg += design.resonators[res_idx].empty_region
        total_reg += design.xmons[res_idx].metal_region
        total_reg += design.xmons[res_idx].empty_region
    crop_box = total_reg.bbox()

    # center of the readout CPW
    crop_box.top += -design.Z_res.b / 2 + \
                    max(map(abs, design.to_line_list)) + design.Z0.b / 2
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
    # resolution_dy = 2e3
    resolution_dx = 2e3
    # print("resolution: ", resolution_dx,"x",resolution_dy, " um")

    # cut part of the ground plane due to rectangular mesh in Sonnet
    crop_box.bottom = crop_box.bottom - int(
        crop_box.height() % resolution_dy)
    # print(crop_box.top, " ", crop_box.bottom)
    # print(crop_box.height() / resolution_dy)
    ''' MESH CALCULATION SECTION END '''

    design.crop(crop_box, region=design.region_ph)

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

    # transfer design`s regions shapes to the corresponding layers in layout
    design.show()
    # show layout in UI window
    design.lv.zoom_fit()

    design.layout.write(
        os.path.join(PROJECT_DIR,
                     f"res_f_Q_{res_idxs}_{dl}_um.gds")
    )

    ''' SIMULATION SECTION START '''
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
    # print(f"simulating...{resonator_idx}")
    result_path = ml_terminal.start_simulation(wait=True)
    ml_terminal.release()
    ''' SIMULATION SECTION START '''

    ''' RESONANCE FINDING SECTION START '''
    # """
    # intended to be working ONLY IF:
    # s12 is monotonically increasing or decreasing over the chosen frequency band.
    # That generally holds true for circuits with single resonator.
    # """
    # with open(result_path.decode('ascii'), "r",
    #           newline='') as file:
    #     # exctracting s-parameters in csv format
    #     # though we do not have csv module
    #     rows = [row.split(',') for row in
    #             list(file.readlines())[8:]]
    #     freqs = [float(row[0]) for row in rows]  # rows in GHz
    #     df = freqs[1] - freqs[0]  # frequency error
    #     s12_list = [float(row[3]) + 1j * float(row[4]) for row in
    #                 rows]
    #     s12_abs_list = [abs(s12) for s12 in s12_list]
    #     min_freq_idx, min_s21_abs = min(enumerate(s12_abs_list),
    #                                     key=lambda x: x[1])
    #     min_freq = freqs[min_freq_idx]
    #     # min_freq_idx = len(s12_abs_list) / 2  # Note: FOR DEBUG
    # print("min freq idx: ", min_freq_idx, "/", len(freqs))
    # # processing the results
    # if min_freq_idx == 0:
    #     # local minimum is located to the left of current interval
    #     # => shift interval to the left and try again
    #     derivative = (s12_list[1] - s12_list[0]) / df
    #     second_derivative = (s12_list[2] - 2 * s12_list[1] +
    #                          s12_list[0]) / df ** 2
    #     print('resonance located the left of the current interval')
    #     # try adjacent interval to the left
    #     estimated_freq -= freqs_span
    # elif min_freq_idx == (len(freqs) - 1):
    #     # local minimum is located to the right of current interval
    #     # => shift interval to the right and try again
    #     derivative = (s12_list[-1] - s12_list[-2]) / df
    #     second_derivative = (s12_list[-1] - 2 * s12_list[-2] +
    #                          s12_list[-3]) / df ** 2
    #     print(
    #         'resonance located the right of the current interval')
    #     # try adjacent interval to the right
    #     estimated_freq += freqs_span
    # else:
    #     # local minimum is within current interval
    #     print(f"fr = {min_freq:3.5} GHz,  fr_err = {df:.5}")
    #     estimated_freq = min_freq
    #
    # # unreachable code:
    # # TODO: add approximation of the resonance if minimum is nonlocal during corase approximation
    # # fr_approx = (2*derivative/second_derivative) + min_freq
    # # B = -4*derivative**3/second_derivative**2
    # # A = min_freq - 2*derivative**2/second_derivative
    # # print(f"fr = {min_freq:3.3} GHz,  fr_err = not implemented(")
    ''' RESONANCE FINDING SECTION END '''

    ''' RESULT SAVING SECTION START '''
    # # geometry parameters gathering
    # # res_params = design.resonators[
    # #     resonator_idx].get_geometry_params_dict(prefix="worm_")
    # # Z0_params = design.Z0.get_geometry_params_dict(
    # #     prefix="S21Line_")
    # #
    # # from collections import OrderedDict
    # #
    # # all_params = design.get_geometry_parameters()
    #
    # # creating directory with simulation results
    # results_dirname = "resonators_S21"
    # results_dirpath = os.path.join(PROJECT_DIR, results_dirname)
    # #
    # # output_metaFile_path = os.path.join(
    # #     results_dirpath,
    # #     "resonator_waveguide_Q_freq_meta.csv"
    # # )
    # try:
    #     # creating directory
    #     os.mkdir(results_dirpath)
    # except FileExistsError:
    #     pass
    # finally:
    #     # copy result from sonnet folder and rename it accordingly
    #     filename = "res_together" + "-".join(map(str,res_idxs)) + ".cvs"
    #     shutil.copy(
    #         result_path.decode("ascii"),
    #         os.path.join(results_dirpath, filename)
    #     )
    ''' RESULT SAVING SECTION END '''


def simulate_Cqr(resolution=(4e3, 4e3), mode="Cqr"):
    # TODO: 1. make 2d geometry parameters mesh, for simultaneous finding of C_qr and C_q
    #  2. make 3d geometry optimization inside kLayout for simultaneous finding of C_qr, C_q and C_qq
    resolution_dx = resolution[0]
    resolution_dy = resolution[1]
    dl_list = np.linspace(-10e3, 10e3, 3)
    # dl_list = [0e3]
    from itertools import product

    for dl, res_idx in list(
            product(
                dl_list, range(8)
            )
    ):
        ### DRAWING SECTION START ###
        design = Design8Q("testScript")

            # adjusting `self.fork_y_spans` for C_qr
        if mode == "Cqr":
            design.fork_y_span_list = [fork_y_span + dl for fork_y_span in design.fork_y_span_list]
            save_fname = "Cqr_Cqr_results.csv"
        elif mode == "Cq":
            # adjusting `cross_gnd_gap_x` and same for y, to gain proper E_C
            design.cross_gnd_gap_y_list = [gnd_gap_y + dl for gnd_gap_y in design.cross_gnd_gap_y_list]
            save_fname = "Cqr_Cq_results.csv"

        # exclude coils from simulation (sometimes port is placed onto coil (TODO: fix)
        design.N_coils = [1] * design.NQUBITS
        design.draw_for_Cqr_simulation(res_idx=res_idx)

        worm = design.resonators[res_idx]
        xmonCross = design.xmons[res_idx]
        worm_start = list(worm.primitives.values())[0].start

        # draw open end at the resonators start
        p1 = worm_start - DVector(design.Z_res.b / 2, 0)
        rec = Rectangle(p1, design.Z_res.b, design.Z_res.b / 2,
                        inverse=True)
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
        design.sonnet_ports.append(xmonCross.cpw_l.end)

        design.transform_region(design.region_ph, DTrans(dr.x, dr.y),
                                trans_ports=True)

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
        print("simulating...")
        result_path = ml_terminal.start_simulation(wait=True)
        ml_terminal.release()

        ### SIMULATION SECTION END ###

        ### CALCULATE C_QR CAPACITANCE SECTION START ###
        C12 = None
        with open(result_path.decode("ascii"), "r") as csv_file:
            data_rows = list(csv.reader(csv_file))
            ports_imps_row = data_rows[6]
            R = float(ports_imps_row[0].split(' ')[1])
            data_row = data_rows[8]
            freq0 = float(data_row[0])

            s = [[0, 0], [0, 0]]  # s-matrix
            # print(data_row)`
            for i in range(0, 2):
                for j in range(0, 2):
                    s[i][j] = complex(
                        float(data_row[1 + 2 * (i * 2 + j)]),
                        float(data_row[1 + 2 * (i * 2 + j) + 1]))
            import math
            delta = (1 + s[0][0]) * (1 + s[1][1]) - s[0][1] * s[1][0]
            y11 = 1 / R * ((1 - s[0][0]) * (1 + s[1][1]) + s[0][1] * s[1][0]) / delta
            y22 = 1 / R * ((1 - s[1][1]) * (1 + s[0][0]) + s[0][1] * s[1][0]) / delta
            C1 = -1e15 / (2 * math.pi * freq0 * 1e9 * (1 / y11).imag)
            C2 = -1e15 / (2 * math.pi * freq0 * 1e9 * (1 / y22).imag)
            # formula taken from https://en.wikipedia.org/wiki/Admittance_parameters#Two_port
            y21 = -2 * s[1][0] / delta * 1 / R
            C12 = 1e15 / (2 * math.pi * freq0 * 1e9 * (1 / y21).imag)

        print("fork_y_span = ", design.fork_y_span_list[res_idx] / 1e3)
        print("C1 = ", C1)
        print("C12 = ", C12)
        print("C2 = ", C2)
        ### CALCULATE C_QR CAPACITANCE SECTION START ###

        ### SAVING REUSLTS SECTION START ###
        design.layout.write(
            os.path.join(PROJECT_DIR, f"Cqr_{res_idx}_{dl}_um.gds")
        )
        output_filepath = os.path.join(PROJECT_DIR,
                                       save_fname)
        if os.path.exists(output_filepath):
            # append data to file
            with open(output_filepath, "a", newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(
                    [res_idx, *list(design.get_geometry_parameters().values()), C12, C1]
                )
        else:
            # create file, add header, append data
            with open(output_filepath, "w", newline='') as csv_file:
                writer = csv.writer(csv_file)
                # create header of the file
                writer.writerow(
                    ["res_idx", *list(design.get_geometry_parameters().keys()), "C12, fF", "C1, fF"])
                writer.writerow(
                    [res_idx, *list(design.get_geometry_parameters().values()), C12, C1]
                )

        ### SAVING REUSLTS SECTION END ###


def simulate_Cqq(q1_idx, q2_idx, resolution=(5e3, 5e3)):
    resolution_dx, resolution_dy = resolution
    x_distance_dx_list = [-5e3, 0, 5e3]
    # x_distance_dx_list = [0]
    for x_distance in x_distance_dx_list:
        ''' DRAWING SECTION START '''
        design = Design8Q("testScript")
        design.N_coils = [1] * design.NQUBITS
        design.xmon_x_distance += x_distance
        design.resonators_dx += x_distance
        print("xmon_x_distance = ", design.xmon_x_distance)
        design.draw_chip()
        design.create_resonator_objects()
        design.draw_xmons_and_resonators([q1_idx, q2_idx])
        design.show()
        design.layout.write(
            os.path.join(PROJECT_DIR, f"Cqq_{q1_idx}_{q2_idx}_"
                                      f"{x_distance:.3f}_.gds")
        )

        design.layout.clear_layer(design.layer_ph)

        res1, res2 = design.resonators[q1_idx], design.resonators[q2_idx]
        cross1, cross2 = design.xmons[q1_idx], design.xmons[q2_idx]
        cross_corrected1, cross_corrected2 = design.xmons_corrected[q1_idx], design.xmons_corrected[q2_idx]
        design.draw_chip()
        cross1.place(design.region_ph)
        cross2.place(design.region_ph)
        # resonator polygons will be adjacent to grounded simulation box
        # thus they will be assumed to be grounded properly (in order to include correction for C11 from C_qr)
        res1.place(design.region_ph)
        res2.place(design.region_ph)

        # print(tmont_metal_width)
        # print(x_side_length)
        # print(list(cross1.get_geometry_params_dict().keys()))
        # print(list(cross1.get_geometry_params_dict().values()))
        # for key, val in cross1.get_geometry_params_dict().items():
        #     print(key, " = ", val)
        # print()
        # process edges of both objects to obtain the most distant edge centers
        # most distant edge centers will be chosen as ports points.
        # Hence, ports will be attached to edge pair with maximum distance.
        from itertools import product
        edgeCenter_cr1_best, edgeCenter_cr2_best = None, None
        max_distance = 0
        edge_centers_it = product(
            cross1.metal_region.edges().centers(0, 0).each(),
            cross2.metal_region.edges().centers(0, 0).each()
        )
        edge_centers_it = map(
            lambda edge_tuple: (edge_tuple[0].p1, edge_tuple[1].p1),
            edge_centers_it
        )
        for edgeCenter_cr1, edgeCenter_cr2 in edge_centers_it:
            centers_d = edgeCenter_cr1.distance(edgeCenter_cr2)
            if centers_d > max_distance:
                edgeCenter_cr1_best, edgeCenter_cr2_best = \
                    edgeCenter_cr1, edgeCenter_cr2
                max_distance = centers_d
            else:
                continue

        design.sonnet_ports.append(edgeCenter_cr1_best)
        design.sonnet_ports.append(edgeCenter_cr2_best)

        crop_box = (cross1.metal_region + cross2.metal_region).bbox()
        crop_box.left -= 4 * (cross1.sideX_length + cross2.sideX_length) / 2
        crop_box.bottom -= 4 * (cross1.sideY_length + cross2.sideY_length) / 2
        crop_box.right += 4 * (cross1.sideX_length + cross2.sideX_length) / 2
        crop_box.top += 4 * (cross1.sideY_length + cross2.sideY_length) / 2
        design.crop(crop_box)
        dr = DPoint(0, 0) - crop_box.p1

        design.transform_region(design.region_ph, DTrans(dr.x, dr.y),
                                trans_ports=True)

        design.show()
        design.lv.zoom_fit()
        '''DRAWING SECTION END'''

        '''SIMULATION SECTION START'''
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
        print("simulating...")
        result_path = ml_terminal.start_simulation(wait=True)
        ml_terminal.release()

        ### SIMULATION SECTION END ###

        ### CALCULATE CAPACITANCE SECTION START ###
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
                    s[i][j] = complex(float(data_row[1 + 2 * (i * 2 + j)]),
                                      float(data_row[1 + 2 * (i * 2 + j) + 1]))
            import math

            delta = (1 + s[0][0]) * (1 + s[1][1]) - s[0][1] * s[1][0]
            y11 = 1 / R * ((1 - s[0][0]) * (1 + s[1][1]) + s[0][1] * s[1][0]) / delta
            y22 = 1 / R * ((1 - s[1][1]) * (1 + s[0][0]) + s[0][1] * s[1][0]) / delta
            C1 = -1e15 / (2 * math.pi * freq0 * 1e9 * (1 / y11).imag)
            C2 = -1e15 / (2 * math.pi * freq0 * 1e9 * (1 / y22).imag)
            # formula taken from https://en.wikipedia.org/wiki/Admittance_parameters#Two_port
            y21 = -2 * s[1][0] / delta * 1 / R
            C12 = 1e15 / (2 * math.pi * freq0 * 1e9 * (1 / y21).imag)

        print("C_12 = ", C12)
        print("C1 = ", C1)
        print("C2 = ", C2)
        print()
        '''CALCULATE CAPACITANCE SECTION END'''

        '''SAVING REUSLTS SECTION START'''
        geometry_params = design.get_geometry_parameters()
        output_filepath = os.path.join(PROJECT_DIR, "Xmon_Cqq_results.csv")
        if os.path.exists(output_filepath):
            # append data to file
            with open(output_filepath, "a", newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(
                    [q1_idx, q2_idx, *list(geometry_params.values()),
                     design.xmon_x_distance / 1e3,
                     C1, C12]
                )
        else:
            # create file, add header, append data
            with open(output_filepath, "w", newline='') as csv_file:
                writer = csv.writer(csv_file)
                # create header of the file
                writer.writerow(
                    ["q1_idx", "q2_idx", *list(geometry_params.keys()),
                     "xmon_x_distance, um",
                     "C1, fF", "C12, fF"])
                writer.writerow(
                    [q1_idx, q2_idx, *list(geometry_params.values()),
                     design.xmon_x_distance / 1e3,
                     C1, C12]
                )
        '''SAVING REUSLTS SECTION END'''


# TODO: pattern copied
def simulate_md_Cg(md_idx, q_idx, resolution=(5e3, 5e3)):
    resolution_dx, resolution_dy = resolution
    # dl_list = np.linspace(-20e3, 20e3, 3)
    dl_list = [0]
    for dl in dl_list:
        design = Design8Q("testScript")
        if md_idx == 0 or md_idx == 7:
            design.md07_x_dist += dl
        elif 0 < md_idx < 7:
            design.md_line_end_shift.y += dl
            design.md_line_end_shift_y += dl

        design.draw_chip()
        design.create_resonator_objects()
        design.draw_xmons_and_resonators()
        design.draw_readout_waveguide()
        design.draw_josephson_loops()
        design.draw_microwave_drvie_lines()
        design.draw_flux_control_lines()

        design.layout.clear_layer(design.layer_ph)
        design.region_ph.clear()
        design.draw_chip()
        # draw requested qubit cross and md line
        crosses = []
        md_lines = []
        fl_lines = []
        resonators = []
        for res_idx in range(min(q_idx, md_idx), max(q_idx, md_idx) + 1):
            crosses.append(design.xmons[res_idx])
            md_lines.append(design.cpw_md_lines[res_idx])
            # fl_lines.append(design.cpw_fl_lines[res_idx])
            # resonators.append(design.resonators[res_idx])
            crosses[-1].place(design.region_ph)
            md_lines[-1].place(design.region_ph)
            # fl_lines[-1].place(design.region_ph)
            # resonators[-1].place(design.region_ph)

        # target xmon and md line
        t_xmon = design.xmons[q_idx]
        t_md_line = design.cpw_md_lines[md_idx]

        ''' CROP BOX CALCULATION SECTION START '''
        # qubit center
        qc = t_xmon.center
        # direction to end of microwave drive
        md_end = t_md_line.end

        dv = qc - md_end
        mid = (md_end + qc) / 2

        t_xmon_diag = np.sqrt(
            (2 * t_xmon.sideX_length + t_xmon.sideY_width +
             2 * t_xmon.sideX_face_gnd_gap) ** 2 +
            (2 * t_xmon.sideY_length + t_xmon.sideX_width +
             2 * t_xmon.sideY_face_gnd_gap)
        )
        if dv.x > 0:
            dx = t_xmon_diag
        else:
            dx = -t_xmon_diag

        if dv.y > 0:
            dy = t_xmon_diag
        else:
            dy = -t_xmon_diag
        p1 = mid - dv / 2 + DVector(-dx, -dy)
        p2 = mid + dv / 2 + DVector(dx, dy)

        crop_box = pya.Box().from_dbox(
            pya.Box(
                min(p1.x, p2.x), min(p1.y, p2.y),
                max(p1.x, p2.x), max(p1.y, p2.y)
            )
        )
        crop_box_reg = Region(crop_box)
        ''' CROP BOX CALCULATION SECTION END '''

        ''' SONNET PORTS POSITIONS CALCULATION SECTION START '''
        xmon_edges = t_xmon.metal_region.edges().centers(0, 0)
        md_line_creg = t_md_line.metal_region & crop_box_reg
        # md line polygon edges central points
        md_line_edge_cpts = md_line_creg.edges().centers(0, 0)

        # find md line port point that is adjacent to crop box
        md_port_pt = None
        for md_edge_cpt in md_line_edge_cpts.each():
            md_edge_cpt = md_edge_cpt.p1
            x = md_edge_cpt.x
            y = md_edge_cpt.y
            if (abs(y - crop_box.bottom) < 1) or \
                    (abs(y - crop_box.top) < 1) or \
                    (abs(x - crop_box.left) < 1) or \
                    (abs(x - crop_box.right) < 1):
                md_port_pt = md_edge_cpt
                break

        # find point on cross edges centers that is the furthest from md
        # line port point
        cross_port_pt = t_xmon.cpw_r.end
        design.sonnet_ports = [cross_port_pt, md_port_pt]
        ''' SONNET PORTS POSITIONS CALCULATION SECTION END '''

        design.crop(crop_box)
        dr = DPoint(0, 0) - crop_box.p1
        design.transform_region(design.region_ph, DTrans(dr.x, dr.y),
                                trans_ports=True)

        # visulize port positions with left-bottom corner of 100x100 um^2 boxes
        # for pt in design.sonnet_ports:
        #     design.region_ph.insert(pya.Box(pt, pt + DVector(100e3, 100e3)))

        design.show()
        design.lv.zoom_fit()
        design.layout.write(
            os.path.join(
                PROJECT_DIR,
                f"C_md_{md_idx}_q_{q_idx}_{dl}.gds"
            )
        )
        '''DRAWING SECTION END'''

        '''SIMULATION SECTION START'''
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
        print("simulating...")
        result_path = ml_terminal.start_simulation(wait=True)
        ml_terminal.release()

        ### SIMULATION SECTION END ###

        ### CALCULATE CAPACITANCE SECTION START ###
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
                    s[i][j] = complex(float(data_row[1 + 2 * (i * 2 + j)]),
                                      float(data_row[
                                                1 + 2 * (i * 2 + j) + 1]))
            import math

            delta = (1 + s[0][0]) * (1 + s[1][1]) - s[0][1] * s[1][0]
            y11 = 1 / R * ((1 - s[0][0]) * (1 + s[1][1]) + s[0][1] * s[1][0]) / delta
            y22 = 1 / R * ((1 - s[1][1]) * (1 + s[0][0]) + s[0][1] * s[1][0]) / delta
            C1 = -1e15 / (2 * math.pi * freq0 * 1e9 * (1 / y11).imag)
            C2 = -1e15 / (2 * math.pi * freq0 * 1e9 * (1 / y22).imag)
            # formula taken from https://en.wikipedia.org/wiki/Admittance_parameters#Two_port
            y21 = -2 * s[1][0] / delta * 1 / R
            C12 = 1e15 / (2 * math.pi * freq0 * 1e9 * (1 / y21).imag)

        print("C_12 = ", C12)
        print("C1 = ", C1)
        print("C2 = ", C2)
        print()
        '''CALCULATE CAPACITANCE SECTION END'''

        '''SAVING REUSLTS SECTION START'''
        geometry_params = design.get_geometry_parameters()
        output_filepath = os.path.join(PROJECT_DIR, f"Xmon_md_{md_idx}_Cmd.csv")
        if os.path.exists(output_filepath):
            # append data to file
            with open(output_filepath, "a", newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(
                    [q_idx, md_idx, *list(geometry_params.values()),
                     C1, C12, C2]
                )
        else:
            # create file, add header, append data
            with open(output_filepath, "w", newline='') as csv_file:
                writer = csv.writer(csv_file)
                # create header of the file
                writer.writerow(
                    ["q_idx", "md_idx", *list(geometry_params.keys()),
                     "C1, fF", "C12, fF", "C2, fF"])
                writer.writerow(
                    [q_idx, md_idx, *list(geometry_params.values()),
                     C1, C12, C2]
                )
        '''SAVING REUSLTS SECTION END'''


if __name__ == "__main__":
    ''' draw and show design for manual design evaluation '''
    design = Design8Q("testScript")
    design.draw()
    design.show()

    ''' C_qr sim '''
    # simulate_Cqr(resolution=(1e3, 1e3), mode="Cq")
    # simulate_Cqr(resolution=(1e3, 1e3), mode="Cqr")

    ''' Simulation of C_{q1,q2} in fF '''
    # simulate_Cqq(2, 3, resolution=(1e3, 1e3))

    ''' MD line C_qd for md1,..., md6 '''
    # for md_idx in [0,1]:
    #     for q_idx in range(2):
    #         simulate_md_Cg(md_idx=md_idx, q_idx=q_idx, resolution=(1e3, 1e3))

    ''' Resonators Q and f sim'''
    # simulate_resonators_f_and_Q(resolution=(2e3,2e3))

    ''' Resonators Q and f when placed together'''
    # simulate_resonators_f_and_Q_together()
