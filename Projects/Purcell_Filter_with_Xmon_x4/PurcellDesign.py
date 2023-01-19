__version__ = "v.0.0.1.0.PURCELL"

'''
Description:
Design for testing qubit lifetimes dependence on readout with purcell filter.
Based on 
1) Drawing logic & Parameters: "v.0.3.1.2.BANT3"

Changes log

'''

import pya
from pya import Point, DPoint, Vector, DVector, DSimplePolygon, SimplePolygon, DPolygon, Polygon, Region, Box, DBox
from pya import Trans, DTrans, CplxTrans, DCplxTrans, ICplxTrans

from importlib import reload

import classLib
reload(classLib)
from classLib.baseClasses import ComplexBase
from classLib.chipDesign import ChipDesign
from classLib.shapes import TmonT, Rectangle, XmonCross, CutMark
from classLib.coplanars import CPWParameters, CPW, DPathCPW, CPWRLPath, Bridge1, CPW2CPW
from classLib.baseClasses import ElementBase
from classLib.josJ import AsymSquid, AsymSquidParams
from classLib.chipTemplates import CHIP_14x14_20pads, FABRICATION
from classLib.marks import MarkBolgar
from classLib.contactPads import ContactPad
from classLib.resonators import EMResonatorTL3QbitWormRLTailXmonFork, EMResonatorTL3QbitWormRLTail
from classLib.purcell import CapacitorParams, EMResonatorCapCoupLineXmonFork, LinearCapacitor
from classLib.helpers import fill_holes, split_polygons, extended_region
import numpy as np
from collections import OrderedDict
from time import ctime
from xml.etree import ElementTree
from xml.etree.ElementTree import Element, SubElement, Comment, tostring
from xml.dom import minidom
import os
import csv
from copy import deepcopy
import shutil
import itertools
from pathlib import Path
import sonnetSim

reload(sonnetSim)
from sonnetSim import SonnetLab, SonnetPort, SimulationBox


# 0.0 - for development
# 0.8e3 - estimation for fabrication by Bolgar photolytography etching
# recipe
FABRICATION.OVERETCHING = 0.e3

refractive_index = np.sqrt(6.26423)
PROJECT_DIR = Path(r'C:\Users\Artem\Desktop\books\Phystech\LAQS\tasks')


class TestStructurePadsSquare(ComplexBase):
    def __init__(self, center, trans_in=None, square_a=200e3,
                 gnd_gap=20e3, squares_gap=20e3):
        self.center = center
        self.rectangle_a =square_a
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


SQUID_PARS = AsymSquidParams(
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


class Design8QTest(ChipDesign):

    def __init__(self, cell_name):
        super().__init__(cell_name)

        # for DC contact deposition
        dc_bandage_layer_i = pya.LayerInfo(3, 0)
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
        self.layer_el_protection = self.layout.layer(
            info_el_protection)

        # has to call it once more to add new layers
        self.lv.add_missing_layers()

        self.center = DPoint(14000e3 / 2, 14000e3 / 2)

        self.chip = CHIP_14x14_20pads
        self.chip_box: pya.DBox = self.chip.box
        self.cop_pars = CPWParameters(20e3, 12e3)
        self.tiny_flux_cop_pars = CPWParameters(4e3, 4e3)
        self.tiny_flux_cpw2cpw_len = 50e3
        self.readline_pads_idxs = [2, 12]
        self.feedline_pads_idxs = [5, 6, 7, 8, 15, 16, 17, 18]

        self.res_x_step = 2500e3
        self.simp_res_shift = -200e3

        ### Contact pads ###
        self.contact_pads: list[
            ContactPad] = self.chip.get_contact_pads(
            [self.cop_pars] * 20
        )

        ### Readline ###
        self.readline = CPW(start=self.contact_pads[2].end,
                            end=self.contact_pads[12].end,
                            cpw_params=self.cop_pars)

        ### Simple readers ###
        self.coup_capacitor_len = 20e3
        self.simp_coup_cap_gap = 24e3
        self.cap_plate_par = CPWParameters(26e3, 12e3)
        self.cap_line_len = 600e3
        self.simp_read_cap_pars = CapacitorParams(self.cap_plate_par, self.coup_capacitor_len, self.simp_coup_cap_gap)

        # resonator params
        self.cop_rad = 100e3
        self.res_width = 420e3
        self.res_height = 820e3
        self.res_turn_rad = 100e3
        self.res_line_gap = 50e3
        self.fork_shift = 400e3
        self.cross_fork_y_len = 70e3
        self.simp_reader_lens = [228, 190, 154, 120]

        ### Purcell filters ###
        self.coupling_ratio = 0.5
        self.line_capacitor_len = 420e3
        self.coup_cap_gap = 60e3
        self.line_cap_gap = 2*196e3
        self.filter_line_cap_pars = CapacitorParams(self.cap_plate_par, self.line_capacitor_len, self.cap_plate_par.gap)
        self.filter_reader_cap_pars = CapacitorParams(self.cap_plate_par, self.coup_capacitor_len, self.coup_cap_gap, True)
        self.filter_lens = [204, 168, 134, 102]
        self.resonator_lens = [230, 194, 160, 128]

        ### Transmons ###
        self.xmons: list[XmonCross] = []

        self.xmon_dy_Cg_coupling = 12e3
        self.xmon_subfeedline_gap = 800e3
        self.xmon_feedline_gap = 200e3

        self.cross_len_x = 180e3
        self.cross_width_x = 60e3
        self.cross_gnd_gap_x = 20e3
        self.cross_len_y = 154e3
        self.cross_width_y = 60e3
        self.cross_gnd_gap_y = 20e3

        # bandages
        # bandages width and height scales such that area
        # scales as `(i+1)*2`, `i` starts from 0
        self.bandages_width_list = [
            1e3 * x for x in [2.50, 3.53, 5.00, 7.07, 10, 14.14]
        ]
        self.bandages_height_list = [
            1e3 * x for x in [5.00, 7.07, 10.00, 14.14, 20, 28.28]
        ]
        self.bandage_width = 5e3
        self.bandage_height = 10e3
        self.bandage_r_outer = 2e3
        self.bandage_r_inner = 2e3
        self.bandage_curve_pts_n = 40
        self.bandages_regs_list = []

        # squids
        self.squids: list[AsymSquid] = []
        self.test_squids: list[AsymSquid] = []
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
        self.dc_cont_ph_clearance = 0.75e3
        # required extension into photo region over the hole cutted
        self.dc_cont_ph_ext = 10e3

        ### Setting simple readers ###
        self.simple_reader_starts = []
        self.simple_readers = []
        self.simple_xmons_centers = []
        self.simple_xmons: list[XmonCross] = []
        self.simple_feedlines: list[CPWRLPath] = []
        for i in range(4):
            self.simple_reader_starts.append(
                self.center + DPoint(-self.res_x_step*3/2 + self.simp_res_shift + i*self.res_x_step, self.cop_pars.width/2)
            )
            self.simple_readers.append(
                EMResonatorCapCoupLineXmonFork(
                    self.simple_reader_starts[-1],
                    self.cop_pars, self.cap_line_len, self.simp_read_cap_pars,
                    self.res_width,
                    self.res_height,
                    self.simp_reader_lens[i]*1e3,
                    self.res_turn_rad,
                    1,
                    'L',
                    [],
                    [self.fork_shift],
                    [],
                    self.cross_width_x + 2 * self.cross_gnd_gap_x + self.cop_pars.width, self.cross_fork_y_len,
                    self.cop_pars.width / 2, self.cop_pars.gap,
                    tail_trans_in=Trans.R270, trans_in=Trans.R180
                )
            )
            xmon_center = \
                (
                        self.simple_readers[-1].fork_x_cpw.start + self.simple_readers[-1].fork_x_cpw.end
                ) / 2 + \
                -DVector(
                    0,
                    -self.xmon_dy_Cg_coupling - self.simple_readers[-1].fork_metal_width / 2
                )
            # changes start #
            xmon_center += DPoint(
                0, self.cross_len_y + self.cross_width_x / 2 + self.cross_gnd_gap_y
            )
            self.simple_xmons_centers.append(xmon_center)
            self.simple_xmons.append(
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
            feed_point = self.simple_xmons[i].cpw_rempt.end + DPoint(0, self.xmon_feedline_gap)
            self.set_feedline(feed_point, self.contact_pads[self.feedline_pads_idxs[-i-1]].end, self.simple_feedlines,
                              DTrans.R270, 'l')

        ### Setting purcell filters ###
        self.filter_starts = []
        self.filter_line_caps = []
        self.filters = []
        self.filter_reader_caps = []
        self.readers = []
        self.xmons_centers = []
        self.xmons = []
        self.feedlines: list = []
        for i in range(4):
            self.filter_starts.append(
                self.center + DPoint(-self.res_x_step * 3 / 2 + i * self.res_x_step,
                                     -self.cop_pars.width / 2)
            )
            self.filter_line_caps.append(
                LinearCapacitor(self.filter_starts[-1], self.cop_pars, self.line_cap_gap, self.filter_line_cap_pars)
            )

            self.filters.append(
                EMResonatorTL3QbitWormRLTail(
                    self.cop_pars,
                    self.filter_line_caps[-1].connections[1] - DPoint(self.res_width / 2 - self.res_turn_rad - self.cop_pars.width/2, self.res_height + self.res_width / 2 + self.res_turn_rad),
                    self.res_width,
                    self.res_height,
                    self.filter_lens[i]*1e3,
                    self.res_turn_rad,
                    1,
                    'L',
                    [],
                    [self.res_width / 2],
                    [],
                    tail_trans_in=Trans.R270,
                    trans_in=Trans.R180
                )
            )
            self.filter_reader_caps.append(
                LinearCapacitor(self.filters[-1].origin - DPoint(self.coupling_ratio*self.res_width, self.cop_pars.width/2), self.cop_pars, self.cap_line_len, self.filter_reader_cap_pars)
            )
            self.readers.append(
                EMResonatorTL3QbitWormRLTailXmonFork(
                    self.cop_pars,
                    self.filter_reader_caps[-1].connections[1] - DPoint(self.coupling_ratio*self.res_width, self.cop_pars.width/2),
                    self.res_width,
                    self.res_height,
                    self.resonator_lens[i]*1e3,
                    self.res_turn_rad,
                    1,
                    'L',
                    [],
                    [self.fork_shift],
                    [],
                    self.cross_width_x + 2 * self.cross_gnd_gap_x + self.cop_pars.width, self.cross_fork_y_len,
                    self.cop_pars.width / 2, self.cop_pars.gap,
                    tail_trans_in=Trans.R270
                )
            )
            xmon_center = \
                (
                    self.readers[-1].fork_x_cpw.start + self.readers[-1].fork_x_cpw.end
                ) / 2 + \
                DVector(
                    0,
                    -self.xmon_dy_Cg_coupling - self.readers[-1].fork_metal_width / 2
                )
            # changes start #
            xmon_center += DPoint(
                0,
                -(
                    self.cross_len_y + self.cross_width_x / 2 +
                    self.cross_gnd_gap_y
                )
            )
            self.xmons_centers.append(xmon_center)
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
            feed_point = self.xmons[i].cpw_lempt.end - DPoint(0, self.xmon_feedline_gap)
            self.set_feedline(feed_point, self.contact_pads[self.feedline_pads_idxs[i]].end, self.feedlines, DTrans.R90, 'l')

        # marks
        self.marks: list[MarkBolgar] = []


    def set_feedline(self, feedpoint, in_point, feedlist, in_trans, orientation='r'):
        if in_trans is None:
            in_trans = 1
        subfeedpoint = feedpoint + in_trans * DPoint(-self.xmon_subfeedline_gap, 0)
        cpw2cpw_point = subfeedpoint + in_trans * DPoint(self.tiny_flux_cpw2cpw_len, 0)
        mid_vector = DVector(subfeedpoint) - DVector(in_point)
        mid_len = min(abs(mid_vector.x), abs(mid_vector.y))
        direct_len = max(abs(mid_vector.x), abs(mid_vector.y))

        if orientation == 'r':
            angles = [np.pi / 2, -np.pi / 2]
        else:
            angles = [-np.pi / 2, np.pi / 2]
        feedlist.append(
            CPWRLPath(in_point, 'LRLRL', self.cop_pars,
                self.res_turn_rad,
                [direct_len / 2,
                mid_len,
                direct_len / 2],
                angles, trans_in=in_trans
            )
        )
        feedlist.append(
            CPW2CPW(self.cop_pars, self.tiny_flux_cop_pars, subfeedpoint, cpw2cpw_point)
        )
        feedlist.append(
            CPW(start=cpw2cpw_point, end=feedpoint, cpw_params=self.tiny_flux_cop_pars)
        )
        feedlist.append(
            CPWRLPath(feedpoint, 'L', CPWParameters(0, self.tiny_flux_cop_pars.b / 2), 0,
                      [self.tiny_flux_cop_pars.gap], [], trans_in=in_trans)
        )

    def draw(self):
        self.draw_chip()

        self.draw_readout_waveguide()
        self.draw_simple_resonators()
        self.draw_purcell_filters()
        self.draw_feedlines()
        self.draw_josephson_loops()

        self.draw_test_structures()
        self.draw_express_test_structures_pads()
        self.draw_bandages()
        self.draw_recess()
        self.region_el.merge()
        self.draw_el_protection()

        self.draw_photo_el_marks()
        self.draw_bridges()

        self.draw_pinning_holes()

        for i, contact_pad in enumerate(self.contact_pads):
            if (i in self.readline_pads_idxs) or (i in self.feedline_pads_idxs):
                contact_pad.place(self.region_ph)

        self.region_ph.merge()
        self.extend_photo_overetching()
        self.inverse_destination(self.region_ph)
        self.draw_cut_marks()
        self.resolve_holes()  # convert to gds acceptable polygons (without inner holes)
        self.split_polygons_in_layers(max_pts=180)

        self.draw_additional_boxes()

    def simplified_draw(self):
        self.draw_chip()
        self.draw_readout_waveguide()
        self.draw_simple_resonators()
        self.draw_purcell_filters()
        self.draw_feedlines()

        for i, contact_pad in enumerate(self.contact_pads):
            if (i in self.readline_pads_idxs) or (i in self.feedline_pads_idxs):
                contact_pad.place(self.region_ph)

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

    def draw_chip(self, ph_only=False):
        if not ph_only:
            self.region_bridges2.insert(self.chip_box)

        self.region_ph.insert(self.chip_box)
        for i, contact_pad in enumerate(self.contact_pads):
            if (i in self.readline_pads_idxs) or (i in self.feedline_pads_idxs):
                contact_pad.place(self.region_ph)

    def draw_readout_waveguide(self):
        self.readline.place(self.region_ph)

    def draw_simple_resonators(self):
        for res, xmon in zip(self.simple_readers, self.simple_xmons):
            res.place(self.region_ph)
            xmon.place(self.region_ph)

    def draw_purcell_filters(self):
        for cap, fil, capr, read, xmon in zip(
                self.filter_line_caps, self.filters, self.filter_reader_caps, self.readers, self.xmons
        ):
            fil.place(self.region_ph)
            cap.place(self.region_ph)
            read.place(self.region_ph)
            capr.place(self.region_ph)
            xmon.place(self.region_ph)

    def draw_feedlines(self):
        for feed in self.feedlines:
            feed.place(self.region_ph)
        for feed in self.simple_feedlines:
            feed.place(self.region_ph)

    def draw_josephson_loops(self):
        # place left squid
        dx = SQUID_PARS.SQB_dx / 2 - SQUID_PARS.SQLBT_dx / 2
        pars_local = deepcopy(SQUID_PARS)
        pars_local.bot_wire_x = [-dx, dx]
        pars_local.SQB_dy = 0
        for res_idx, xmon_cross in enumerate(self.xmons):
            pars_local.BC_dx = [self.bandages_width_list[res_idx]]*len(
                pars_local.bot_wire_x)
            pars_local.BCW_dx = [pars_local.BCW_dx[0]]*len(
                pars_local.bot_wire_x)
            pars_local.BC_dy = (self.bandages_height_list[res_idx])/2 + 3.5e3 - 1e3
            pars_local.TC_dx = pars_local.BC_dx[0]
            pars_local.TC_dy = (self.bandages_height_list[res_idx])/2 + 3e3 - 1e3
            # below RO line
            m = 1
            squid_center = (xmon_cross.cpw_bempt.end +
                            xmon_cross.cpw_bempt.start) / 2
            trans = DTrans.R0

            squid = AsymSquid(
                squid_center + m*DVector(0, -self.squid_vertical_shift),
                pars_local,
                trans_in=trans
            )
            self.squids.append(squid)
            squid.place(self.region_el)

        for res_idx, xmon_cross in enumerate(self.simple_xmons):
            pars_local.BC_dx = [self.bandages_width_list[res_idx]]*len(
                pars_local.bot_wire_x)
            pars_local.BCW_dx = [pars_local.BCW_dx[0]]*len(
                pars_local.bot_wire_x)
            pars_local.BC_dy = (self.bandages_height_list[res_idx])/2 + \
                               3.5e3 - 1e3
            pars_local.TC_dx = pars_local.BC_dx[0]
            pars_local.TC_dy = (self.bandages_height_list[res_idx])/2 + \
                               3e3 - 1e3
            # above RO line
            m = -1
            squid_center = (xmon_cross.cpw_tempt.end +
                            xmon_cross.cpw_tempt.start) / 2
            trans = DTrans.M0

            squid = AsymSquid(
                squid_center + m*DVector(0, -self.squid_vertical_shift),
                pars_local,
                trans_in=trans
            )
            self.squids.append(squid)
            squid.place(self.region_el)

    def draw_test_structures(self):
        # DRAW CONCTACT FOR BANDAGES WITH 5um CLEARANCE

        struct_centers = [DPoint(4.5e6, 11.0e6), DPoint(9.5e6, 11.0e6),
                          DPoint(12.0e6, 3.0e6)]
        self.test_squids_pads = []
        for struct_center in struct_centers:
            ## JJ test structures ##
            dx = SQUID_PARS.SQB_dx / 2 - SQUID_PARS.SQLBT_dx / 2

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

            pars_local = deepcopy(SQUID_PARS)
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

            pars_local = deepcopy(SQUID_PARS)
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
                bridge.place(self.region_bridges1, region_id="bridges_1")
                bridge.place(self.region_bridges2, region_id="bridges_2")

            # bandages test structures
        test_dc_el2_centers = [
            DPoint(1.5e6, 5.5e6),
            DPoint(7.3e6, 11.0e6),
            DPoint(12.3e6, 5.0e6)
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

    def draw_express_test_structures_pads(self):
        for squid, test_pad in zip(
                self.test_squids[:-2],
                self.test_squids_pads[:-2]
        ):
            if squid.squid_params.SQRBJJ_dy == 0:
                # only left JJ is present

                # test pad expanded to the left
                p1 = DPoint(squid.SQRTT.start.x, test_pad.center.y)
                p2 = p1 + DVector(10e3, 0)
                etc1 = CPW(
                    start=p1, end=p2,
                    width=1e3,
                    gap=0
                )
                etc1.place(self.region_el)

                p3 = DPoint(test_pad.top_rec.p2.x, p2.y)
                etc2 = CPW(
                    start=p2, end=p3,
                    width=test_pad.gnd_gap-4e3,
                    gap=0
                )
                etc2.place(self.region_el)

                # test pad expanded to the left
                p1 = squid.BCW_list[0].end
                p2 = p1 - DVector(10e3, 0)
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
                    width=test_pad.gnd_gap - 4e3,
                    gap=0
                )
                etc4.place(self.region_el)

            elif squid.squid_params.SQLBJJ_dy == 0:
                # only right leg is present
                p1 = DPoint(squid.SQLTT.start.x, test_pad.center.y)
                p2 = p1 + DVector(-10e3, 0)
                # test pad expanded to the left
                etc1 = CPW(
                    start=p1, end=p2,
                    width=1e3,
                    gap=0
                )
                etc1.place(self.region_el)

                p3 = DPoint(test_pad.top_rec.p1.x, p2.y)
                etc2 = CPW(
                    start=p2, end=p3,
                    width=test_pad.gnd_gap - 4e3,
                    gap=0
                )
                etc2.place(self.region_el)

                # test pad expanded to the right
                p1 = squid.BCW_list[0].end
                p2 = p1 + DVector(10e3, 0)
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
                    width=test_pad.gnd_gap - 4e3,
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
            jjLoop_idx = None
            if squid in self.squids:
                jjLoop_idx = self.squids.index(squid)
            self.bandages_regs_list += \
                self.draw_squid_bandage(squid, jjLoop_idx=jjLoop_idx)

    def draw_squid_bandage(self, test_jj: AsymSquid = None,
                           jjLoop_idx=None):
        bandages_regs_list: list[Region] = []

        import re
        top_bandage_reg = self._get_bandage_reg(test_jj.TC.start, jjLoop_idx)
        bandages_regs_list.append(top_bandage_reg)
        self.dc_bandage_reg += top_bandage_reg

        # collect all bottom contacts
        for i, _ in enumerate(test_jj.squid_params.bot_wire_x):
            BC = test_jj.BC_list[0]
            bot_bandage_reg = self._get_bandage_reg(BC.end, jjLoop_idx)
            bandages_regs_list.append(bot_bandage_reg)
            self.dc_bandage_reg += bot_bandage_reg
        return bandages_regs_list

    def _get_bandage_reg(self, center, i=None):
        if i == None:
            bandage_width = self.bandage_width
            bandage_height = self.bandage_height
        else:
            bandage_width = self.bandages_width_list[i]
            bandage_height = self.bandages_height_list[i]

        rect_lb = center +\
                  DVector(
                      -bandage_width/2,
                      -bandage_height/2
                  )
        bandage_reg = Rectangle(
            origin=rect_lb,
            width=bandage_width,
            height=bandage_height
        ).metal_region
        bandage_reg.round_corners(
            self.bandage_r_inner,
            self.bandage_r_outer,
            self.bandage_curve_pts_n
        )

        return bandage_reg

    def draw_recess(self):
        for squid in itertools.chain(self.squids, self.test_squids):
            recess_reg = squid.TC.metal_region.dup().size(-self.dc_cont_ph_clearance)
            self.region_ph -= recess_reg

            for i, _ in enumerate(squid.squid_params.bot_wire_x):
                BC = squid.BC_list[0]
                recess_reg = BC.metal_region.dup().size(-self.dc_cont_ph_clearance)
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
            DPoint(1.9e6, 12.1e6), DPoint(12.5e6, 11.0e6),
            DPoint(4.3e6, 8.0e6), DPoint(9.2e6, 5.8e6),
            DPoint(1.5e6, 3.0e6), DPoint(12.1e6, 1.9e6)
        ]
        for mark_center in marks_centers:
            self.marks.append(
                MarkBolgar(mark_center)
            )
            self.marks[-1].place(self.region_ph)
            self.marks[-1].place(self.region_bridges1)

    def draw_bridges(self):
        bridges_step = 130e3
        fl_bridges_step = 130e3

        self.resonators = self.simple_readers + self.readers + self.filters
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

        self.cpw_fl_lines = self.simple_feedlines + self.feedlines
        for i, cpw_fl in enumerate(self.cpw_fl_lines):
            Bridge1.bridgify_CPW(
                cpw_fl, bridges_step=bridges_step,
                dest=self.region_bridges1,
                dest2=self.region_bridges2,
                avoid_points=[cpw_fl.end],
                avoid_distances=130e3
            )

        # for filter line capacitors
        self.cpw_caps_lines = self.filter_reader_caps + self.filter_line_caps
        for cpw_line in self.cpw_caps_lines:
            for name, res_primitive in cpw_line.primitives.items():
                if "coil" in name:
                    Bridge1.bridgify_CPW(
                        res_primitive, bridges_step,
                        dest=self.region_bridges1,
                        dest2=self.region_bridges2
                    )

        # for readout waveguide
        avoid_resonator_points = self.simple_reader_starts + self.filter_starts

        Bridge1.bridgify_CPW(
            self.readline, bridges_step,
            dest=self.region_bridges1, dest2=self.region_bridges2,
            avoid_points=avoid_resonator_points,
            avoid_distances=self.res_turn_rad
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

    def draw_cut_marks(self):
        chip_box_poly = DPolygon(self.chip_box)
        for point in chip_box_poly.each_point_hull():
            CutMark(origin=point).place(self.region_ph)

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

    def draw_additional_boxes(self):
        abox_top_ph = pya.Box(Point(self.chip.dx/2,self.chip.dy/2) + Point(-self.chip.dx * 0.3, self.chip.dx * 0.54),
                Point(self.chip.dx/2,self.chip.dy/2) + Point(self.chip.dx * 0.3, self.chip.dx * 0.64))
        abox_bot_ph = pya.Box(Point(self.chip.dx/2,self.chip.dy/2) - Point(-self.chip.dx * 0.3, self.chip.dx * 0.54),
                           Point(self.chip.dx/2,self.chip.dy/2) - Point(self.chip.dx * 0.3, self.chip.dx * 0.64))
        self.region_ph.insert(abox_top_ph)
        self.region_ph.insert(abox_bot_ph)

        abox_top_el = pya.Box(
            Point(self.chip.dx / 2, self.chip.dy / 2) + Point(-self.chip.dx * 0.4, self.chip.dx * 0.56),
            Point(self.chip.dx / 2, self.chip.dy / 2) + Point(self.chip.dx * 0.4, self.chip.dx * 0.62))
        abox_bot_el = pya.Box(
            Point(self.chip.dx / 2, self.chip.dy / 2) - Point(-self.chip.dx * 0.4, self.chip.dx * 0.56),
            Point(self.chip.dx / 2, self.chip.dy / 2) - Point(self.chip.dx * 0.4, self.chip.dx * 0.62))
        self.region_bridges1.insert(abox_top_el)
        self.region_bridges1.insert(abox_bot_el)

        ext_chip_box = self.chip_box.dup()
        ext_chip_box.left -= 2e6
        ext_chip_box.bottom -= 2e6
        ext_chip_box.top += 2e6
        ext_chip_box.right += 2e6
        ext_chip_box = Region(ext_chip_box)
        ext_chip_box -= Region(self.chip_box)
        self.region_bridges2 += ext_chip_box

    def draw_for_simp_reader_sim(self):
        self.draw_chip(True)
        self.draw_readout_waveguide()
        self.draw_simple_resonators()

    def save_logdata(self, filename=PROJECT_DIR / 'purcell_filter.xml', comment=''):

        root = Element('Design')
        root.set('version', 'gamma')
        root.set('class', str(ChipDesign))
        generalComment = Comment(f'Purcell filter design log.')
        root.append(generalComment)
        head = SubElement(root, 'head')
        date = SubElement(head, 'creationDate')
        date.text = f'{ctime()}'
        seParts = SubElement(head, 'parts')
        strlist = ''
        for prt in self.last_draw:
            strlist += prt + ' '
        seParts.text = strlist
        seComment = SubElement(head, 'comment')
        seComment.text = comment

        body = SubElement(root, 'body')

        for prt in self.last_draw:
            detail = self.design_dict[prt]
            se = SubElement(body, prt)
            se.set('class', str(type(detail)))
            sePar = SubElement(se, 'params')
            if type(detail) == list:
                i = 0
                for ln in detail:
                    seLn = SubElement(se, f'line_{i}')
                    seLn.set('class', str(type(ln)))
                    sePar = SubElement(seLn, 'params')
                    for key, value in ln.get_geometry_params_dict().items():
                        sePar.set(key.replace(', um', ''), str(value))
                    i += 1
            else:
                for key, value in detail.get_geometry_params_dict().items():
                    sePar.set(key.replace(', um', ''), str(value))

        parsed = tostring(root, 'utf-8')
        reparsed = minidom.parseString(parsed)
        with open(filename, 'w') as f:
            f.write(reparsed.toprettyxml(indent='   '))


def simlate_simple_readers_s_pars(ind, filename='simp_reader_S12.csv', min_freq=7.0, max_freq=7.5):
    ### DRAWING SECTION START ###
    design = Design8QTest("testScript")

    design.draw_for_simp_reader_sim()

    worm = design.readline
    center = design.simple_readers[ind].start + DPoint(-design.simple_xmons[ind].sideX_length, design.res_height/2 + design.cap_line_len/2)

    box_side_x = 2 * design.res_width + design.simple_xmons[ind].sideX_length * 2
    box_side_y = design.res_height + design.fork_shift * 2 + design.cap_line_len + design.simple_xmons[ind].sideY_length * 4
    dv = DPoint(box_side_x / 2, box_side_y / 2)

    crop_box = pya.Box().from_dbox(pya.Box(
        center + dv,
        center + (-1) * dv
    ))
    design.crop(crop_box)
    dr = DPoint(0, 0) - crop_box.p1

    # finding the furthest edge of cropped resonator`s central line polygon
    # sonnet port will be attached to this edge
    reg1 = worm.metal_region & Region(crop_box)
    reg1.merge()
    max_distance = center.x
    port_pt = None
    for poly in reg1.each():
        for edge in poly.each_edge():
            edge_center = (edge.p1 + edge.p2) / 2
            dp = edge_center - center
            d = max(abs(dp.x), abs(dp.y))
            if edge_center.x > max_distance:
                port_pt = edge_center
                max_distance = edge_center.x
    design.sonnet_ports.append(port_pt)

    reg1 = worm.metal_region & Region(crop_box)
    reg1.merge()
    max_distance = center.x
    port_pt = None
    for poly in reg1.each():
        for edge in poly.each_edge():
            edge_center = (edge.p1 + edge.p2) / 2
            dp = edge_center - design.center
            d = max(abs(dp.x), abs(dp.y))
            if edge_center.x < max_distance:
                port_pt = edge_center
                max_distance = edge_center.x
    design.sonnet_ports.append(port_pt)

    design.transform_region(design.region_ph, DTrans(dr.x, dr.y),
                            trans_ports=True)

    design.show()
    design.lv.zoom_fit()
    ### DRAWING SECTION END ###

    for prt in design.sonnet_ports:
        print(prt)

    simulate_S_pars(design, crop_box, filename, min_freq=min_freq, max_freq=max_freq)

def simulate_C12(crop_box, design, filename='c12.csv', resolution_dx=2e3, resolution_dy=2e3):
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
    ml_terminal.set_linspace_sweep(7, 7, 1)
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
        # print(data_row)
        for i in range(0, 2):
            for j in range(0, 2):
                s[i][j] = complex(float(data_row[1 + 2 * (i * 2 + j)]),
                                  float(data_row[
                                            1 + 2 * (i * 2 + j) + 1]))
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
        C1 -= C12
        C2 -= C12

        ### CALCULATE C_QR CAPACITANCE SECTION START ###

        ### SAVING REUSLTS SECTION START ###
        design.layout.write(str(PROJECT_DIR / "purcell_filter" / "capacitance" / (filename[:-4] + '.gds')))
        output_filepath = PROJECT_DIR / "purcell_filter" / "capacitance" / filename
        if os.path.exists(str(output_filepath)):
            # append data to file
            with open(str(output_filepath), "a", newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(
                    [C12, C1, C2]
                )
        else:
            # create file, add header, append data
            with open(str(output_filepath), "w", newline='') as csv_file:
                writer = csv.writer(csv_file)
                # create header of the file
                writer.writerow(
                    ["C12, fF", "C1, fF", "C2, fF"])
                writer.writerow(
                    [C12, C1, C2]
                )

        # design.save_logdata(PROJECT_DIR / "purcell_filter" / "capacitance" / (filename[:-4] + '.xml'),
        #                     comment=f'Capacitance simulation results saved at {filename}.')

        ### SAVING RESULTS SECTION END ###

def simulate_S_pars(design, crop_box, filename, min_freq=6.5, max_freq=7.5, resolution_dx=2e3, resolution_dy=2e3):
    ### SIMULATION SECTION START ###
    ml_terminal = SonnetLab()
    from sonnetSim.cMD import CMD

    ml_terminal._send(CMD.SAY_HELLO)
    ml_terminal.clear()
    simBox = SimulationBox(
        crop_box.width(), crop_box.height(),
        crop_box.width() / resolution_dx,
        crop_box.height() / resolution_dy
    )

    ml_terminal.set_boxProps(simBox)
    from sonnetSim.pORT_TYPES import PORT_TYPES

    ports = [
        SonnetPort(prt, PORT_TYPES.BOX_WALL) for prt in design.sonnet_ports
    ]
    ml_terminal.set_ports(ports)
    ml_terminal.send_polygons(design.cell, design.layer_ph)
    ml_terminal.set_ABS_sweep(min_freq, max_freq)
    # print(f"simulating...{resonator_idx}")
    result_path = ml_terminal.start_simulation(wait=True)
    ml_terminal.release()

    all_params = design.get_geometry_parameters()

    # creating directory with simulation results
    results_dirpath = PROJECT_DIR / "purcell_filter"/ "final_design" / "resonators_S12"

    shutil.copy(
        result_path.decode("ascii"),
        str(results_dirpath / filename)
    )

    design.layout.write(str(results_dirpath / (filename[:-4] + '.gds')))
    # design.save_logdata(results_dirpath / (filename[:-4] + '.xml'),
    #                     comment=f'S-parameters simulation results saved at {filename}.')

    ### RESULT SAVING SECTION END ###


if __name__ == "__main__":
    design = Design8QTest("testScript")
    # design.simplified_draw()
    design.draw()
    design.show()
    # simp_freqs = [7.2, 7.4, 7.6, 7.8]
    # for i, freq in enumerate(simp_freqs):
    #     simlate_simple_readers_s_pars(i, filename=f'simp_reader_{freq:.01f}_S12.csv', min_freq=freq - 0.05, max_freq=freq+0.05)
