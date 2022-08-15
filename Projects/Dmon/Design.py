__version__ = "v.0.0.2.0"

'''
Description:
This program is made to generate litography blueprints for testing of the 
main series chips. E.g. this one is based on v.0.3.0.8 Design.py


Changes log
v.0.0.2.0
1. Testing several Dmons with different E_C
and 2 transmons with corresponding spectra.
2. No flux lines, nor flux CPW pads.

v.0.0.1.3
1. Qubit beneath the RO line repeat corresponding from the 5Q design.
Namely qubits 1, 3, 2 (starting from 0) from 5Q design
are located below the RO line. 

v.0.0.1.1, v.0.0.1.2
1. Due to up to date calculations, `C_qr` and hence 
`design.fork_y_spans` values are resimulated.


v.0.0.1.0
1. Added fluix lines for qubits (taken from 5q v.0.3.1.2). Bridges taken accordingly.
2. Contact pads are transformed by 90deg rotation in order to smooth
 the flux line.
3. Readout line y-shift is eliminated. RO line is straight now.

v.0.0.0.0
Copied fromv.0.3.0.8.T3
'''

# Enter your Python code here
from math import cos, sin, tan, atan2, pi, degrees
import itertools
from typing import List, Dict, Union, Optional
from copy import deepcopy
import csv
import os
import shutil

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
from classLib.shapes import XmonCross, Rectangle, CutMark
from classLib.resonators import EMResonatorTL3QbitWormRLTailXmonFork
from classLib.josJ import AsymSquidParams, AsymSquid
from classLib.chipTemplates import CHIP_10x5_8pads, FABRICATION
from classLib.chipDesign import ChipDesign
from classLib.marks import MarkBolgar
from classLib.contactPads import ContactPad
from classLib.helpers import fill_holes, split_polygons, extended_region

import sonnetSim

reload(sonnetSim)
from sonnetSim import SonnetLab, SonnetPort, SimulationBox


FABRICATION.OVERETCHING = 0.5e3
PROJECT_DIR = os.path.dirname(__file__)

SQUID_PARAMETERS = AsymSquidParams(
    band_ph_tol=1e3,
    squid_dx=14.2e3,
    squid_dy=7e3,
    TC_dx=9e3,
    TC_dy=7e3,
    TCW_dy=6e3,
    BCW_dy=0e3,
    BC_dy=7e3,
    BC_dx=7e3
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


class DesignDmon(ChipDesign):
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
        self.NQUBITS = 8
        # chip rectangle and contact pads
        self.chip = CHIP_10x5_8pads
        self.chip_box: pya.DBox = self.chip.box
        # Z = 50.09 E_eff = 6.235 (E = 11.45)
        self.z_md_fl: CPWParameters = CPWParameters(11e3, 5.7e3)
        # Z = 50.136  E_eff = 6.28826 (E = 11.45)
        self.z_md_fl2: CPWParameters = CPWParameters(10e3, 5.7e3)
        # flux line widths at the end of flux line
        self.flux2ground_left_width = 2e3
        self.flux2ground_right_width = 4e3
        self.ro_Z: CPWParameters = self.chip.chip_Z
        contact_pads_trans_list = [Trans.R0] + [Trans.R270] + 2 * [
            Trans.R90] + [Trans.R0] + [Trans.R270]*3
        for i, trans in enumerate(contact_pads_trans_list):
            contact_pads_trans_list[i] = DCplxTrans(DTrans(contact_pads_trans_list[i]))
            if trans == DTrans.R270:
                contact_pads_trans_list[i] = DCplxTrans(DVector(-self.chip.pad_length, self.chip.pcb_Z.b/2 + self.chip.back_metal_width))*contact_pads_trans_list[i]
            elif trans == DTrans.R90:
                contact_pads_trans_list[i] = DCplxTrans(DVector(-self.chip.pad_length, -self.chip.pcb_Z.b/2 -self.chip.back_metal_width))*contact_pads_trans_list[i]
        self.contact_pads: list[ContactPad] = self.chip.get_contact_pads(
            [self.ro_Z] + [self.z_md_fl] * 3 + [self.ro_Z] + [self.z_md_fl] * 3,
            cpw_trans_list=contact_pads_trans_list
        )

        # readout line parameters
        self.ro_line_dy: float = 1600e3
        # shifting readout line to the top due to absence of top pads
        self.cpwrl_ro_line: CPWRLPath = None
        # base coplanar waveguide parameters that correspond
        # to chip-end contact pads.
        self.Z0: CPWParameters = CHIP_10x5_8pads.chip_Z

        # xmon parameters
        self.xmon_x_distance: float = 722e3

        self.xmon_res_d_list = [40e3]*self.NQUBITS
        self.xmons: list[XmonCross] = []
        self.xmons_corrected: list[XmonCross] = []

        self.cross_len_x_list = np.array(
            [1e3*x for x in [129.905, 65.602, 35.098, 0.001, 0.001, 196.603, 233.873, 233.415]]
        )
        self.cross_width_x_list = np.array(
            [1e3*x for x in [16, 16, 16, 16, 16, 32, 56, 56]]
        )
        self.cross_len_y_list = np.array(
            [1e3*x for x in [245.0, 225.0, 211.0, 154.0, 154.0, 258.0, 267.0, 267.0 ]]
        )
        self.cross_width_y_list = np.array(
            [1e3*x for x in [16, 16, 16, 16, 16, 32, 56, 56]]
        )
        self.cross_gnd_gap_y_list = np.array(
            [1e3*x for x in [60]*self.NQUBITS]
        )
        self.cross_gnd_gap_x = 60e3
        self.cross_gnd_gap_face_y = 20e3

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
            [114.5219, 95.1897, 99.0318, 83.7159, 88.8686, 70.3649,
             74.0874, 59.6982]
        ]
        # curvature radius of resonators CPW turns
        self.res_r = 60e3
        # coil consist of L1, 2r, L1, 2r segment
        self.N_coils = [3]*self.NQUBITS
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
        self.fork_metal_width_list = np.array([10e3]*self.NQUBITS)
        self.fork_gnd_gap = 10e3
        self.xmon_fork_gnd_gap = 10e3
        # fork at the end of resonator parameters
        self.fork_x_span_list = self.cross_width_y_list + + 2 * \
                                (self.xmon_fork_gnd_gap + self.fork_metal_width_list)
        # resonator-fork parameters
        # from simulation of g_qr
        self.fork_y_span_list = [
            x * 1e3 for x in
            [53.6, 31.5, 13.7, 7.7, 9.7, 71.2, 75.3, 76.2]
        ]
        self.worm_x_list = [x * 1e6 for x in
                       [1, 2.7, 3.5, 4.35, 5.5, 6.5, 7.6, 8.5]]

        # squids
        self.squids: List[AsymSquid] = []
        self.test_squids: List[AsymSquid] = []
        # vertical shift of every squid local origin  coordinates
        self.squid_vertical_shift = 3e3
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

        # bandages
        self.bandage_width = 5e3
        self.bandage_height = 10e3
        self.bandage_r_outer = 2e3
        self.bandage_r_inner = 2e3
        self.bandage_curve_pts_n = 40
        self.bandages_regs_list = []

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
        # self.draw_flux_control_lines()

        self.draw_test_structures()
        self.draw_bandages()
        self.draw_recess()
        self.draw_el_protection()

        self.draw_photo_el_marks()
        # self.draw_bridges()
        # self.draw_pinning_holes()
        # for contact_pad in self.contact_pads:  # delete holes from
        #     # contact pads
        #     contact_pad.place(self.region_ph)
        # self.region_ph.merge()
        # self.extend_photo_overetching()
        # self.inverse_destination(self.region_ph)
        # self.draw_cut_marks()
        # self.resolve_holes()  # convert to gds acceptable polygons (without inner holes)
        # self.split_polygons_in_layers(max_pts=180)

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
        for i, contact_pad in enumerate(self.contact_pads):
            # draw only ROWG pads
            if (i==0) or (i==4):
             contact_pad.place(self.region_ph)

    def draw_cut_marks(self):
        chip_box_poly = DPolygon(self.chip_box)
        for point in chip_box_poly.each_point_hull():
            CutMark(origin=point).place(self.region_ph)

    def create_resonator_objects(self):
        ### RESONATORS TAILS CALCULATIONS SECTION START ###
        # key to the calculations can be found in hand-written format here:
        # https://drive.google.com/file/d/1wFmv5YmHAMTqYyeGfiqz79a9kL1MtZHu/view?usp=sharing
        # though, this calculations were implemented poorly
        # instead of \Delta = Const, implemented \Delta + S_i = const
        # see sketch for details

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

        # list corrected for resonator-qubit coupling geomtry, so all transmons centers are placed
        # along single horizontal line

        self.L3_list = np.array(self.L3_list) + np.array(L3_arr)

        tail_segment_lengths_list = [[L2, L3, L4]
                                     for L2, L3, L4 in
                                     zip(self.L2_list, self.L3_list,
                                         self.L4_list)]
        tail_turn_angles_list = [
            [np.pi / 2, -np.pi / 2]
        ]*8
        tail_trans_in_list = [Trans.R270] * 8
        ### RESONATORS TAILS CALCULATIONS SECTION END ###

        pars = list(
            zip(
                self.L1_list, self.to_line_list, self.L_coupling_list,
                self.fork_y_span_list,
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
            worm_x = self.worm_x_list[res_idx]
            worm_y = self.chip.dy / 2 + to_line * (-1) ** (res_idx)

            fork_x_span = self.fork_x_span_list[res_idx]
            fork_metal_width = self.fork_metal_width_list[res_idx]
            fork_gnd_gap = self.fork_gnd_gap
            if res_idx % 2 == 0:  # above RO line
                trans = DTrans.M0
            else:  # below RO line
                trans = DTrans.R0
            self.resonators.append(
                EMResonatorTL3QbitWormRLTailXmonFork(
                    Z0=self.Z_res,
                    start=DPoint(worm_x, worm_y),
                    L_coupling=L_coupling,
                    L0=L0,
                    L1=L1, r=self.res_r, N=n_coils,
                    tail_shape=res_tail_shape,
                    tail_turn_radiuses=tail_turn_radiuses,
                    tail_segment_lengths=tail_segment_lengths,
                    tail_turn_angles=tail_turn_angles,
                    tail_trans_in=tail_trans_in,
                    fork_x_span=fork_x_span,
                    fork_y_span=fork_y_span,
                    fork_metal_width=fork_metal_width,
                    fork_gnd_gap=fork_gnd_gap,
                    trans_in=trans
                )
            )

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
        self.cpwrl_ro_line = CPW(
            start=self.contact_pads[0].end, end=self.contact_pads[4].end,
            cpw_params=self.Z0
        )
        self.cpwrl_ro_line.place(self.region_ph)

    def draw_xmons_and_resonators(self, res_idxs2Draw: List[int] = None):
        """
        Fills photolitography Region() instance with resonators
        and Xmons crosses structures.

        Parameters
        ----------
        res_idx2Draw : int
            draw only particular resonator (if passed)
            used in resonator simulations.


        Returns
        -------
        None
        """
        it_list = list(
            enumerate(
                self.resonators
            )
        )
        for res_idx, res in it_list:
            d = self.xmon_res_d_list[res_idx] + self.cross_len_y_list[res_idx] + self.cross_width_x_list[res_idx] / 2 + \
                self.fork_metal_width_list[res_idx] + self.fork_gnd_gap
            if res_idx % 2 == 0:
                # above RO line
                xmon_center = res.end + DVector(0, d)
            else:
                # below RO line
                xmon_center = res.end + DVector(0, -d)
            self.xmons.append(
                XmonCross(
                    xmon_center,
                    sideX_length=self.cross_len_x_list[res_idx],
                    sideX_width=self.cross_width_x_list[res_idx],
                    sideX_gnd_gap=self.cross_gnd_gap_x,
                    sideY_length=self.cross_len_y_list[res_idx],
                    sideY_width=self.cross_width_y_list[res_idx],
                    sideY_gnd_gap=self.cross_gnd_gap_y_list[res_idx],
                    sideX_face_gnd_gap=self.cross_gnd_gap_x,
                    sideY_face_gnd_gap=self.cross_gnd_gap_face_y
                )
            )

            if res_idxs2Draw is None:
                pass
            else:
                if res_idx not in res_idxs2Draw:
                    continue

            self.xmons[-1].place(self.region_ph)
            res.place(self.region_ph)
            xmonCross_corrected = XmonCross(
                xmon_center,
                sideX_length=self.cross_len_x_list[res_idx],
                sideX_width=self.cross_width_x_list[res_idx],
                sideX_gnd_gap=self.cross_gnd_gap_x,
                sideY_length=self.cross_len_y_list[res_idx],
                sideY_width=self.cross_width_y_list[res_idx],
                sideY_gnd_gap=0
            )
            self.xmons_corrected.append(xmonCross_corrected)
            xmonCross_corrected.place(self.region_ph)

    def draw_josephson_loops(self):
        # place left squid
        dx = SQUID_PARAMETERS.SQB_dx / 2 - SQUID_PARAMETERS.SQLBT_dx / 2
        pars_local = deepcopy(SQUID_PARAMETERS)
        pars_local.bot_wire_x = [-dx, dx]
        pars_local.SQB_dy = 0
        for res_idx, xmon_cross in enumerate(self.xmons):
            if res_idx % 2 == 0:  # above RO line
                m = -1
                squid_center = (xmon_cross.cpw_tempt.end +
                                xmon_cross.cpw_tempt.start) / 2
                trans = DTrans.M0
            else:  # below RO line
                m = 1
                squid_center = (xmon_cross.cpw_bempt.end +
                                xmon_cross.cpw_bempt.start) / 2
                trans = DTrans.R0
            squid = AsymSquid(
                squid_center + m * DVector(0, -self.squid_vertical_shift),
                pars_local,
                trans_in=trans
            )
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
        self.flux_lines_x_shifts: List[float] = \
            [
                -SQUID_PARAMETERS.squid_dx/ 2 - SQUID_PARAMETERS.SQLBT_dx/ 2 -
                self.z_md_fl2.width/ 2 + SQUID_PARAMETERS.BC_dx / 2 +
                SQUID_PARAMETERS.band_ph_tol
            ] * len(self.L1_list)

        # place caplanar line 1 fl
        _p1 = self.contact_pads[1].end
        _p2 = DPoint(self.xmons[1].center.x + self.flux_lines_x_shifts[1], _p1.y)
        _p3 = DPoint(_p2.x, self.xmons[1].cpw_bempt.end.y)
        self.cpwrl_fl1 = DPathCPW(
            points=[_p1, _p2, _p3],
            cpw_parameters=self.z_md_fl,
            turn_radiuses=self.ctr_lines_turn_radius,
        )
        self.cpw_fl_lines.append(self.cpwrl_fl1)

        # place caplanar line 3 fl
        _p1 = self.contact_pads[2].end
        _p2 = DPoint(self.xmons[3].center.x + self.flux_lines_x_shifts[3], _p1.y)
        _p3 = DPoint(_p2.x, self.xmons[3].cpw_bempt.end.y)
        self.cpwrl_fl3 = DPathCPW(
            points=[_p1, _p2, _p3],
            cpw_parameters=self.z_md_fl,
            turn_radiuses=self.ctr_lines_turn_radius,
        )
        self.cpw_fl_lines.append(self.cpwrl_fl3)

        # place caplanar line 5 fl
        _p1 = self.contact_pads[3].end
        _p2 = DPoint(self.xmons[5].center.x + self.flux_lines_x_shifts[5], _p1.y)
        _p3 = DPoint(_p2.x, self.xmons[5].cpw_bempt.end.y)
        self.cpwrl_fl5 = DPathCPW(
            points=[_p1, _p2, _p3],
            cpw_parameters=self.z_md_fl,
            turn_radiuses=self.ctr_lines_turn_radius,
        )
        self.cpw_fl_lines.append(self.cpwrl_fl5)

        # place coplanar line 0 fl
        _p1 = self.contact_pads[-1].end
        _p2 = DPoint(self.xmons[0].center.x + self.flux_lines_x_shifts[0],
                     _p1.y)
        _p3 = DPoint(_p2.x, self.xmons[0].cpw_tempt.end.y)
        self.cpwrl_fl0 = DPathCPW(
            points=[_p1, _p2, _p3],
            cpw_parameters=self.z_md_fl,
            turn_radiuses=self.ctr_lines_turn_radius,
        )
        self.cpw_fl_lines.append(self.cpwrl_fl0)

        # place coplanar line 2 fl
        _p1 = self.contact_pads[-2].end
        _p2 = DPoint(self.xmons[2].center.x + self.flux_lines_x_shifts[2],
                     _p1.y)
        _p3 = DPoint(_p2.x, self.xmons[2].cpw_tempt.end.y)
        self.cpwrl_fl4 = DPathCPW(
            points=[_p1, _p2, _p3],
            cpw_parameters=self.z_md_fl,
            turn_radiuses=self.ctr_lines_turn_radius,
        )
        self.cpw_fl_lines.append(self.cpwrl_fl4)

        # place coplanar line 4 fl
        _p1 = self.contact_pads[-3].end
        _p2 = DPoint(self.xmons[4].center.x + self.flux_lines_x_shifts[4],
                     _p1.y)
        _p3 = DPoint(_p2.x, self.xmons[4].cpw_tempt.end.y)
        self.cpwrl_fl2 = DPathCPW(
            points=[_p1, _p2, _p3],
            cpw_parameters=self.z_md_fl,
            turn_radiuses=self.ctr_lines_turn_radius,
        )
        self.cpw_fl_lines.append(self.cpwrl_fl2)

        for i, flux_line in enumerate(self.cpw_fl_lines):
            if i < 3:  # for lower squids
                dir_y = 1
            else:  # for upper squids
                dir_y = -1
            self.modify_flux_line_end(flux_line, direction_y=dir_y)

    def modify_flux_line_end(self, flux_line: DPathCPW, direction_y=1):
        # make flux line wider to have a less chance to misalign
        # bandage-eBeam-photo layers at the qubit bandage region.
        last_line = list(flux_line.primitives.values())[-1]
        last_line_name = list(flux_line.primitives.keys())[-1]

        # divide line into 3 sections
        alpha_1 = 0.6
        alpha_2 = 0.3
        alpha_3 = 1 - (alpha_1 + alpha_2)
        p1 = last_line.start
        p2 = p1 + alpha_1*(last_line.end - last_line.start)
        p3 = p2 + alpha_2*(last_line.end - last_line.start)
        p4 = p3 + alpha_3*(last_line.end - last_line.start)
        cpw_normal = CPW(
            start=p1,
            end=p2,
            width=self.z_md_fl.width,
            gap=self.z_md_fl.gap
        )
        cpw_transition = CPW2CPW(
            Z0=self.z_md_fl,
            Z1=self.z_md_fl2,
            start=p2, end=p3
        )
        cpw_thick = CPW(
            # rounding error correction
            start=p3 - 2*(p4-p3)/(p4-p3).abs(),
            end=p4,
            width=self.z_md_fl2.width,
            gap=self.z_md_fl2.gap
        )

        del flux_line.primitives[last_line_name]
        flux_line.primitives[last_line_name + "_1"] = cpw_normal
        flux_line.primitives[last_line_name + "_2"] = cpw_transition
        flux_line.primitives[last_line_name + "_3"] = cpw_thick
        flux_line._refresh_named_connections()
        flux_line.place(self.region_ph)

        # draw mutual inductance for squid
        # connect central conductor to ground at right
        p1 = cpw_thick.end + DVector(
            cpw_thick.width / 2,
            -direction_y*self.flux2ground_right_width / 2
        )

        p2 = p1 + DVector(cpw_thick.gap, 0)
        inductance_cpw = CPW(
            start=p1, end=p2,
            width=self.flux2ground_right_width, gap=0
        )
        inductance_cpw.place(self.region_ph)

        # connect central conductor to ground at left
        p1 = cpw_thick.end + \
             DVector(
                 -cpw_thick.width / 2,
                 -direction_y*self.flux2ground_left_width / 2
             )
        p2 = p1 - DVector(cpw_thick.gap, 0)
        flux_gnd_cpw = CPW(
            start=p1, end=p2,
            width=self.flux2ground_left_width, gap=0
        )
        flux_gnd_cpw.place(self.region_ph)

    def draw_test_structures(self):
        # DRAW CONCTACT FOR BANDAGES WITH 5um CLEARANCE

        struct_centers = [DPoint(1.5e6, 1.5e6), DPoint(5.2e6, 1.5e6),
                          DPoint(2.2e6, 3.2e6)]
        self.test_squids_pads = []
        for struct_center in struct_centers:
            ## JJ test structures ##
            dx = SQUID_PARAMETERS.SQB_dx / 2 - SQUID_PARAMETERS.SQLBT_dx / 2

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
                "56 nA", 0.001, 25, False, 0, 0)
            text_bl = test_struct1.empty_rectangle.p1 - DVector(0, 20e3)
            text_reg.transform(
                ICplxTrans(1.0, 0, False, text_bl.x, text_bl.y))
            self.region_ph -= text_reg

            pars_local = deepcopy(SQUID_PARAMETERS)
            pars_local.SQRBT_dx = 0
            pars_local.SQRBJJ_dy = 0
            pars_local.bot_wire_x = [-dx]

            squid_center = test_struct1.center
            test_jj = AsymSquid(
                squid_center + DVector(0, -self.squid_vertical_shift),
                pars_local
            )
            self.test_squids.append(test_jj)
            test_jj.place(self.region_el)

            # test structure with low critical current
            test_struct2 = TestStructurePadsSquare(
                struct_center + DPoint(0.3e6, 0))
            self.test_squids_pads.append(test_struct2)
            test_struct2.place(self.region_ph)

            text_reg = pya.TextGenerator.default_generator().text(
                "11 nA", 0.001, 25, False, 0, 0)
            text_bl = test_struct2.empty_rectangle.p1 - DVector(0, 20e3)
            text_reg.transform(
                ICplxTrans(1.0, 0, False, text_bl.x, text_bl.y))
            self.region_ph -= text_reg

            pars_local = deepcopy(SQUID_PARAMETERS)
            pars_local.SQLBT_dx = 0
            pars_local.SQLBJJ_dy = 0
            pars_local.bot_wire_x = [dx]

            squid_center = test_struct2.center
            test_jj = AsymSquid(
                squid_center + DVector(0, -self.squid_vertical_shift),
                pars_local
            )
            self.test_squids.append(test_jj)
            test_jj.place(self.region_el)

            # test structure for bridge DC contact
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
            DPoint(6.7e6, 3.2e6),
            DPoint(3.6e6, 1.6e6),
            DPoint(9.0e6, 3.8e6)
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
            self.bandages_regs_list += self.draw_squid_bandage(squid)
            # collect all bottom contacts

    def draw_squid_bandage(self, test_jj: AsymSquid = None):
        bandages_regs_list: List[Region] = []

        import re
        # top bandage
        top_bandage_reg = self._get_bandage_reg(test_jj.TC.start)
        bandages_regs_list.append(top_bandage_reg)
        self.dc_bandage_reg += top_bandage_reg

        # bottom contacts
        for i, _ in enumerate(test_jj.squid_params.bot_wire_x):
            BC = getattr(test_jj, "BC" + str(i))
            bot_bandage_reg = self._get_bandage_reg(BC.end)
            bandages_regs_list.append(bot_bandage_reg)
            self.dc_bandage_reg += bot_bandage_reg
        return bandages_regs_list

    def _get_bandage_reg(self, center):
        rect_lb = center +\
                  DVector(
                      -self.bandage_width/2,
                      -self.bandage_height/2
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
            recess_reg = squid.TC.metal_region.dup().size(-1.5e3)
            self.region_ph -= recess_reg

            for i, _ in enumerate(squid.squid_params.bot_wire_x):
                BC = getattr(squid, "BC"+str(i))
                recess_reg = BC.metal_region.dup().size(-1.5e3)
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

        for i, cpw_fl in enumerate(self.cpw_fl_lines):
            Bridge1.bridgify_CPW(
                cpw_fl, bridges_step=bridges_step,
                dest=self.region_bridges1,
                dest2=self.region_bridges2,
                avoid_points=[cpw_fl.end],
                avoid_distances=130e3
            )
            dy_list = [30e3, 130e3]
            for dy in dy_list:
                if i < 3:
                    dy = dy
                else:
                    dy = -dy
                bridge_center1 = cpw_fl.end + DVector(0, -dy)
                br = Bridge1(center=bridge_center1, trans_in=Trans.R90)
                br.place(dest=self.region_bridges1, region_name="bridges_1")
                br.place(dest=self.region_bridges2, region_name="bridges_2")

        # for readout waveguide
        avoid_resonator_points = []
        for res in self.resonators:
            avoid_resonator_points.append(
                res.origin + DPoint(res.L_coupling/2, 0)
            )

        Bridge1.bridgify_CPW(
            self.cpwrl_ro_line, bridges_step,
            dest=self.region_bridges1, dest2=self.region_bridges2,
            avoid_points=avoid_resonator_points,
            avoid_distances=3 / 4 * max(self.L_coupling_list) + self.r
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
        return res_length


def simulate_resonators_f_and_Q(resolution=(4e3, 4e3)):
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

        design = DesignDmon("testScript")
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

    design = DesignDmon("testScript")
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


def simulate_Cqr(resolution=(4e3, 4e3), mode="Cq", pts=3, par_d=10e3):
    # TODO: 1. make 2d geometry parameters mesh, for simultaneous finding of C_qr and C_q
    #  2. make 3d geometry optimization inside kLayout for simultaneous finding of C_qr, C_q and C_qq
    resolution_dx = resolution[0]
    resolution_dy = resolution[1]
    # if linspace is requested with single point it will return
    # lhs border of the interval
    dl_list = np.linspace(-par_d/2, par_d/2, pts)
    # dl_list = [0e3]
    from itertools import product

    for dl, res_idx in list(
            product(
                dl_list, range(8)
            )
    ):
        ### DRAWING SECTION START ###
        design = DesignDmon("testScript")
        # adjusting `self.fork_y_span_list` for C_qr
        if mode == "Cqr":
            design.fork_y_span_list += dl
            # design.fork_metal_width_list += dl
            # design.fork_x_span_list += 2*dl
            save_fname = "Cqr_Cqr_results.csv"
        elif mode == "Cq":
            # adjusting `cross_len_x` to gain proper E_C
            # design.cross_width_y_list += dl
            # design.cross_width_x_list += dl
            # design.cross_len_x_list += dl
            design.cross_len_y_list += dl
            save_fname = "Cqr_Cq_results.csv"

        # exclude coils from simulation (sometimes port is placed onto coil (TODO: fix)
        design.N_coils = [0] * design.NQUBITS
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

        xc_bbx = xmonCross.empty_region.bbox()
        box_side_l = max(xc_bbx.height(), xc_bbx.width())
        dv = DVector(box_side_l, box_side_l)

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

        port_pt = None
        min_dist = 10e6
        for center_edge in reg1.edges().centers(0,0).each():
            edge_center = center_edge.p1
            dist = np.min(
                np.abs(
                    [edge_center.x - crop_box.left, edge_center.x - crop_box.right,
                     edge_center.y - crop_box.bottom, edge_center.y - crop_box.top]
                )
            )
            if dist < min_dist:
                min_dist = dist
                port_pt = edge_center
        design.sonnet_ports.append(port_pt)
        # place xmon port
        if xmonCross.sideX_length > xmonCross.sideY_length:
            design.sonnet_ports.append(xmonCross.cpw_l.end)
        elif res_idx%2 == 0:
            design.sonnet_ports.append(xmonCross.cpw_t.end)
        else:
            design.sonnet_ports.append(xmonCross.cpw_b.end)
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
            y11 = 1 / R * ((1 - s[0][0]) * (1 + s[1][1]) + s[0][1] * s[1][
                0]) / delta
            y22 = 1 / R * ((1 - s[1][1]) * (1 + s[0][0]) + s[0][1] * s[1][
                0]) / delta
            C1 = -1e15 / (2 * math.pi * freq0 * 1e9 * (1 / y11).imag)
            C2 = -1e15 / (2 * math.pi * freq0 * 1e9 * (1 / y22).imag)
            # formula taken from https://en.wikipedia.org/wiki/Admittance_parameters#Two_port
            y21 = -2 * s[1][0] / delta * 1 / R
            C12 = 1e15 / (2 * math.pi * freq0 * 1e9 * (1 / y21).imag)

        print(f"cross_x_len[{res_idx}] = {design.cross_len_x_list[res_idx] / 1e3}")
        print("C1 = ", C1)
        print("C12 = ", C12)
        print("C2 = ", C2)
        print("------------")  # 12 `-` whitespace
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
                    [res_idx,
                     *list(design.get_geometry_parameters().values()), C12,
                     C1]
                )
        else:
            # create file, add header, append data
            with open(output_filepath, "w", newline='') as csv_file:
                writer = csv.writer(csv_file)
                # create header of the file
                writer.writerow(
                    ["res_idx",
                     *list(design.get_geometry_parameters().keys()),
                     "C12, fF", "C1, fF"])
                writer.writerow(
                    [res_idx,
                     *list(design.get_geometry_parameters().values()), C12,
                     C1]
                )

        ### SAVING REUSLTS SECTION END ###


def simulate_Cqq(q1_idx, q2_idx, resolution=(5e3, 5e3)):
    resolution_dx, resolution_dy = resolution
    x_distance_dx_list = [-5e3, 0, 5e3]
    # x_distance_dx_list = [0]
    for x_distance in x_distance_dx_list:
        ''' DRAWING SECTION START '''
        design = DesignDmon("testScript")
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
        cross_corrected1, cross_corrected2 = design.xmons_corrected[
                                                 q1_idx], \
                                             design.xmons_corrected[q2_idx]
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
        crop_box.left -= 4 * (
                    cross1.sideX_length + cross2.sideX_length) / 2
        crop_box.bottom -= 4 * (
                    cross1.sideY_length + cross2.sideY_length) / 2
        crop_box.right += 4 * (
                    cross1.sideX_length + cross2.sideX_length) / 2
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
                                      float(data_row[
                                                1 + 2 * (i * 2 + j) + 1]))
            import math

            delta = (1 + s[0][0]) * (1 + s[1][1]) - s[0][1] * s[1][0]
            y11 = 1 / R * ((1 - s[0][0]) * (1 + s[1][1]) + s[0][1] * s[1][
                0]) / delta
            y22 = 1 / R * ((1 - s[1][1]) * (1 + s[0][0]) + s[0][1] * s[1][
                0]) / delta
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


def simulate_md_Cg(md_idx, q_idx, resolution=(5e3, 5e3)):
    resolution_dx, resolution_dy = resolution
    # dl_list = np.linspace(-20e3, 20e3, 3)
    dl_list = [0]
    for dl in dl_list:
        design = DesignDmon("testScript")
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
            y11 = 1 / R * ((1 - s[0][0]) * (1 + s[1][1]) + s[0][1] * s[1][
                0]) / delta
            y22 = 1 / R * ((1 - s[1][1]) * (1 + s[0][0]) + s[0][1] * s[1][
                0]) / delta
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
        output_filepath = os.path.join(PROJECT_DIR,
                                       f"Xmon_md_{md_idx}_Cmd.csv")
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
    # FABRICATION.OVERETCHING = 0.0e3
    # design = DesignDmon("testScript")
    # design.draw()
    # design.show()

    # design.save_as_gds2(
    #     os.path.join(
    #         PROJECT_DIR,
    #         "8Q_0.0.0.1_A482_A483_overetching_0um.gds"
    #     )
    # )
    #
    # FABRICATION.OVERETCHING = 0.5e3
    # design = DesignDmon("testScript")
    # design.draw()
    # design.show()
    # design.save_as_gds2(
    #     os.path.join(
    #         PROJECT_DIR,
    #         "8Q_0.0.0.1_A482_A483_overetching_0um5.gds"
    #     )
    # )

    ''' C_qr sim '''
    simulate_Cqr(resolution=(1e3, 1e3), mode="Cqr", pts=3, par_d=5e3)
    # simulate_Cqr(resolution=(1e3, 1e3), mode="Cq", pts=3, par_d=20e3)
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

