# Enter your Python code here
from math import cos, sin, tan, atan2, pi, degrees
import itertools

import pya
from pya import Cell
from pya import Point, Vector, \
    DPoint, DVector, DSimplePolygon, SimplePolygon, DPolygon, Polygon, Region
from pya import Trans, DTrans, CplxTrans, DCplxTrans, ICplxTrans, Path

from importlib import reload
import classLib

reload(classLib)

from classLib.baseClasses import ElementBase, ComplexBase
from classLib.coplanars import CPWParameters, CPW_RL_Path, CPW2CPW, Bridge1
from classLib.shapes import XmonCross, Circle, Rectangle
from classLib.resonators import EMResonatorTL3QbitWormRLTailXmonFork
from classLib.josJ import AsymSquidParams, AsymSquid
from classLib.chipTemplates import CHIP_10x10_12pads, FABRICATION
from classLib.chipDesign import ChipDesign
from classLib.marks import MarkBolgar
from classLib.shapes import Ring

# imports for docstrings generation
from classLib.coplanars import CPW, CPW_arc

from typing import List, Dict, Union, Optional
from classLib.contactPads import ContactPad

from classLib.helpers import fill_holes, split_polygons

import copy


FABRICATION.OVERETCHING = 0.8e3  # nm


class FluxLineEnd(ElementBase):
    def __init__(self, origin, fc_cpw_params, width, trans_in=None):  # width = 5e3

        self._fc_cpw_params: Union[CPW, CPWParameters, CPW_arc] = fc_cpw_params
        self._width = width

        super().__init__(origin, trans_in)

        self.start = self.connections[0]
        self.end = self.connections[1]

    def init_regions(self):
        # flux CPW width and gap
        f_cpw = self._fc_cpw_params
        p1 = DPoint(-f_cpw.b / 2, -f_cpw.width / 2)
        p2 = p1 + DPoint(0, f_cpw.width)
        p3 = p2 + DPoint(f_cpw.gap, 0)
        p4 = p3 + DPoint(0, -3 / 2 * f_cpw.width)
        metal_points = [p1, p2, p3, p4]
        self.metal_region.insert(Region(DSimplePolygon(metal_points)))

        empty_points = [DPoint(f_cpw.width / 2, 0),
                        DPoint(f_cpw.width / 2, f_cpw.width + FABRICATION.OVERETCHING),
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
        z0_cpw : Union[CPW, CPW_arc]
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
        self.z0_cpw: Union[CPW, CPW_arc] = z0_cpw
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
        self.cpwrl_ro_line: CPW_RL_Path = None
        self.Z0: CPWParameters = CHIP_10x10_12pads.chip_Z

        # resonators objects list
        self.resonators: List[EMResonatorTL3QbitWormRLTailXmonFork] = []
        # distance between nearest resonators central conductors centers
        # constant step between resonators origin points along x-axis.
        self.resonators_dx: float = 900e3
        # resonator parameters
        self.L_coupling_list: list[float] = [1e3 * x for x in [310, 320, 320, 310, 300]]
        # corresponding to resonanse freq is linspaced in interval [6,9) GHz
        self.L0 = 1150e3
        self.L1_list = [1e3 * x for x in [50.7218, 96.3339, 138.001, 142.77, 84.9156]]
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
        self.fork_y_spans = [x * 1e3 for x in [8.73781, 78.3046, 26.2982, 84.8277, 35.3751]]

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

        # squids
        self.squids: List[AsymSquid] = []
        self.test_squids: List[AsymSquid] = []
        self.el_dc_contacts: List[List[ElementBase, ElementBase]] = []

        # md and flux lines attributes
        self.shift_fl_y = self.cross_len_y + 60e3
        self.shift_md_x = 60e3
        self.shift_md_y = 510e3

        self.cpwrl_md1: CPW_RL_Path = None
        self.cpwrl_md1_end: MDriveLineEnd = None
        self.cpwrl_fl1: CPW_RL_Path = None
        self.cpwrl_fl1_end: FluxLineEnd = None

        self.cpwrl_md2: CPW_RL_Path = None
        self.cpwrl_md2_end: MDriveLineEnd = None
        self.cpwrl_fl2: CPW_RL_Path = None
        self.cpwrl_fl2_end: FluxLineEnd = None

        self.cpwrl_md3: CPW_RL_Path = None
        self.cpwrl_md3_end: MDriveLineEnd = None
        self.cpwrl_fl3: CPW_RL_Path = None
        self.cpwrl_fl3_end: FluxLineEnd = None

        self.cpwrl_md4: CPW_RL_Path = None
        self.cpwrl_md4_end: MDriveLineEnd = None
        self.cpwrl_fl4: CPW_RL_Path = None
        self.cpwrl_fl4_end: FluxLineEnd = None

        self.cpwrl_md5: CPW_RL_Path = None
        self.cpwrl_md5_end: MDriveLineEnd = None
        self.cpwrl_fl5: CPW_RL_Path = None
        self.cpwrl_fl5_end: FluxLineEnd = None

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
        self.draw_md_and_flux_lines()
        self.draw_test_structures()
        self.draw_josephson_loops()
        self.draw_el_dc_contacts()
        self.draw_el_protection()

        self.draw_photo_el_marks()
        self.draw_bridges()
        self.draw_pinning_holes()
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
        self.cell.shapes(self.layer_el_protection).insert(self.region_el_protection)
        self.lv.zoom_fit()

    def draw_chip(self):
        self.region_bridges2.insert(self.chip_box)

        self.region_ph.insert(self.chip_box)
        for contact_pad in self.contact_pads:
            contact_pad.place(self.region_ph)

    def create_resonator_objects(self):
        # fork at the end of resonator parameters
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

        pars = list(
            zip(
                self.L1_list, self.to_line_list, self.L_coupling_list,
                self.fork_y_spans,
                tail_segment_lengths_list, tail_turn_angles_list, tail_trans_in_list,
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
            worm_x = self.contact_pads[-1].end.x + (res_idx + 1 / 2) * self.resonators_dx
            worm_y = self.contact_pads[-1].end.y - self.ro_line_dy - to_line

            resonator_cpw = CPWParameters(self.Z_res.width + 2 * FABRICATION.OVERETCHING,
                                          self.Z_res.gap - 2 * FABRICATION.OVERETCHING)
            self.resonators.append(
                EMResonatorTL3QbitWormRLTailXmonFork(
                    resonator_cpw, DPoint(worm_x, worm_y), L_coupling, L0, L1, self.r, n_coils,
                    tail_shape=res_tail_shape, tail_turn_radiuses=tail_turn_radiuses,
                    tail_segment_lengths=tail_segment_lengths,
                    tail_turn_angles=tail_turn_angles, tail_trans_in=tail_trans_in,
                    fork_x_span=fork_x_span + 2 * FABRICATION.OVERETCHING, fork_y_span=fork_y_span,
                    fork_metal_width=self.fork_metal_width + 2 * FABRICATION.OVERETCHING,
                    fork_gnd_gap=self.fork_gnd_gap - 2 * FABRICATION.OVERETCHING
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
        ro_line_turn_radius = self.ro_line_turn_radius
        ro_line_dy = self.ro_line_dy

        ## calculating segment lengths of subdivided coupling part of ro coplanar ##

        # value that need to be added to `L_coupling` to get width of resonators bbox.
        def get_res_extension(resonator: EMResonatorTL3QbitWormRLTailXmonFork):
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
            next_resonator_extension = get_res_extension(self.resonators[i + 1])
            # order of adding is from left to right (imagine chip geometry in your head to follow)
            res_line_segments_lengths.extend(
                [
                    resonator_width,
                    # `resonator_extension` accounts for the next resonator extension
                    # in this case all resonator's extensions are equal
                    self.resonators_dx - (resonator_width - resonator_extension / 2) - next_resonator_extension / 2
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

        self.cpwrl_ro_line = CPW_RL_Path(
            self.contact_pads[-1].end, shape="LR" + ''.join(['L'] * len(res_line_segments_lengths)) + "RLRLRL",
            cpw_parameters=CPWParameters(self.Z0.width + 2 * FABRICATION.OVERETCHING,
                                         self.Z0.gap - 2 * FABRICATION.OVERETCHING),
            turn_radiuses=[ro_line_turn_radius] * 4,
            segment_lengths=segment_lengths,
            turn_angles=[pi / 2, pi / 2, pi / 2, -pi / 2], trans_in=Trans.R270
        )
        self.cpwrl_ro_line.place(self.region_ph)

    def draw_xmons_and_resonators(self):
        for resonator, fork_y_span, xmon_dy_Cg_coupling in \
                list(zip(
                    self.resonators,
                    self.fork_y_spans,
                    self.xmon_dys_Cg_coupling
                )):
            xmon_center = (resonator.fork_x_cpw.start + resonator.fork_x_cpw.end) / 2 + \
                          DVector(0, -xmon_dy_Cg_coupling - resonator.fork_metal_width / 2)
            # changes start #
            xmon_center += DPoint(
                0,
                -(self.cross_len_y + self.cross_width_x / 2 + min(self.cross_gnd_gap_y,
                                                                  self.xmon_fork_gnd_gap)) + FABRICATION.OVERETCHING
            )
            self.xmons.append(
                XmonCross(xmon_center, self.cross_len_x,
                          self.cross_width_x + 2 * FABRICATION.OVERETCHING,
                          self.cross_gnd_gap_x - 2 * FABRICATION.OVERETCHING,
                          sideY_length=self.cross_len_y,
                          sideY_width=self.cross_width_y + 2 * FABRICATION.OVERETCHING,
                          sideY_gnd_gap=self.cross_gnd_gap_y - 2 * FABRICATION.OVERETCHING)
            )
            self.xmons[-1].place(self.region_ph)
            resonator.place(self.region_ph)
            xmonCross_corrected = XmonCross(
                xmon_center,
                sideX_length=self.cross_len_x,
                sideX_width=self.cross_width_x + 2 * FABRICATION.OVERETCHING,
                sideX_gnd_gap=self.cross_gnd_gap_x - 2 * FABRICATION.OVERETCHING,
                sideY_length=self.cross_len_y,
                sideY_width=self.cross_width_y + 2 * FABRICATION.OVERETCHING,
                sideY_gnd_gap=min(self.cross_gnd_gap_y, self.xmon_fork_gnd_gap) - 2 * FABRICATION.OVERETCHING)
            xmonCross_corrected.place(self.region_ph)

    def draw_md_and_flux_lines(self):
        """
        Drawing of md (microwave drive) and flux tuning lines for 5 qubits
        Returns
        -------

        """
        # TODO: MDriveLineEnd - working bad with overetching
        contact_pads = self.contact_pads
        ctr_line_turn_radius = 100e3

        xmon_center = self.xmons[-1].center
        xmon_x_distance = self.xmon_x_distance
        cross_width_y = self.cross_width_y
        cross_width_x = self.cross_width_x
        cross_len_x = self.cross_len_x
        cross_len_y = self.cross_len_y
        cross_gnd_gap_y = self.cross_gnd_gap_y
        cross_gnd_gap_x = self.cross_gnd_gap_x

        width_res = self.Z_res.width

        tmp_reg = self.region_ph

        z_md_fl_corrected = CPWParameters(
            self.z_md_fl.width + 2 * FABRICATION.OVERETCHING,
            self.z_md_fl.gap - 2 * FABRICATION.OVERETCHING
        )

        shift_fl_y = self.shift_fl_y
        shift_md_x = self.shift_md_x
        shift_md_y = self.shift_md_y
        md0_md5_gnd = 5e3
        flux_end_width = self.cross_width_x + 2 * self.cross_gnd_gap_x - 2 * FABRICATION.OVERETCHING

        md_transition = 25e3
        md_z1_params = z_md_fl_corrected
        md234_z1_length = 245e3
        md15_z1_length = md234_z1_length - 30e3
        shift_md_x_side = md234_z1_length + md_transition + md_z1_params.b / 2 + cross_len_x + cross_width_x / 2 + cross_gnd_gap_x

        # place caplanar line 1md
        self.cpwrl_md1 = CPW_RL_Path(
            contact_pads[0].end, shape="LRLRL", cpw_parameters=z_md_fl_corrected,
            turn_radiuses=[ctr_line_turn_radius] * 2,
            segment_lengths=[
                2 * (-contact_pads[0].end.x + xmon_center.x - 4 * xmon_x_distance - shift_md_x_side) / 16,
                (
                        (contact_pads[0].end.y - xmon_center.y) ** 2 +
                        (9 * (-contact_pads[0].end.x + xmon_center.x - 4 * xmon_x_distance - shift_md_x_side) / 16)
                        ** 2
                ) ** 0.5,
                5 * (-contact_pads[
                    0].end.x + xmon_center.x - 4 * xmon_x_distance - shift_md_x_side) / 16 - md0_md5_gnd],
            turn_angles=[-atan2(contact_pads[0].end.y - xmon_center.y,
                                9 * (-contact_pads[
                                    0].end.x + xmon_center.x - 4 * xmon_x_distance - shift_md_x_side) / 16),
                         atan2(contact_pads[0].end.y - xmon_center.y,
                               9 * (-contact_pads[
                                   0].end.x + xmon_center.x - 4 * xmon_x_distance - shift_md_x_side) / 16)],
            trans_in=Trans.R0
        )
        self.cpwrl_md1.place(tmp_reg)

        self.cpwrl_md1_end = MDriveLineEnd(list(self.cpwrl_md1.primitives.values())[-1], md_z1_params, md_transition,
                                           md15_z1_length)
        self.cpwrl_md1_end.place(tmp_reg)

        # place caplanar line 1 fl
        self.cpwrl_fl1 = CPW_RL_Path(
            contact_pads[1].end, shape="LRL", cpw_parameters=z_md_fl_corrected,
            turn_radiuses=[ctr_line_turn_radius],
            segment_lengths=[
                -contact_pads[1].end.x + xmon_center.x - 4 * xmon_x_distance - cross_len_x,
                xmon_center.y - contact_pads[1].end.y -
                cross_width_x / 2 - cross_gnd_gap_x - z_md_fl_corrected.width + FABRICATION.OVERETCHING
            ],
            turn_angles=[pi / 2],
            trans_in=Trans.R0
        )
        self.cpwrl_fl1.place(tmp_reg)

        self.cpwrl_fl1_end = FluxLineEnd(self.cpwrl_fl1.end, z_md_fl_corrected, width=flux_end_width, trans_in=Trans.R0)
        self.cpwrl_fl1_end.place(tmp_reg)

        # place caplanar line 2md
        self.cpwrl_md2 = CPW_RL_Path(
            contact_pads[3].end, shape="LRLRLRL", cpw_parameters=z_md_fl_corrected,
            turn_radiuses=[ctr_line_turn_radius] * 3,
            segment_lengths=[
                (-contact_pads[3].end.y + xmon_center.y - shift_md_y) / 4,
                (-contact_pads[3].end.x + xmon_center.x + shift_md_x - 3 * xmon_x_distance) / 2,
                (
                        (
                                (-contact_pads[3].end.x + xmon_center.x + shift_md_x - 3 * xmon_x_distance) / 2
                        ) ** 2 +
                        (
                                5 * (-contact_pads[3].end.y + xmon_center.y - shift_md_y) / 8
                        ) ** 2
                ) ** 0.5,
                (-contact_pads[3].end.y + xmon_center.y - shift_md_y) / 8
            ],
            turn_angles=[-pi / 2, atan2(5 * (-contact_pads[3].end.y + xmon_center.y - shift_md_y) / 8, (
                    -contact_pads[3].end.x + xmon_center.x + shift_md_x - 3 * xmon_x_distance) / 2),
                         pi / 2 - atan2(5 * (-contact_pads[3].end.y + xmon_center.y - shift_md_y) / 8, (
                                 -contact_pads[3].end.x + xmon_center.x + shift_md_x - 3 * xmon_x_distance) / 2)],
            trans_in=Trans.R90
        )
        self.cpwrl_md2.place(tmp_reg)

        self.cpwrl_md2_end = MDriveLineEnd(
            list(self.cpwrl_md2.primitives.values())[-1],
            md_z1_params, md_transition, md234_z1_length
        )
        self.cpwrl_md2_end.place(tmp_reg)

        # place caplanar line 2 fl

        self.cpwrl_fl2 = CPW_RL_Path(
            contact_pads[2].end, shape="LRLRL", cpw_parameters=z_md_fl_corrected,
            turn_radiuses=[ctr_line_turn_radius] * 2,
            segment_lengths=[(-contact_pads[2].end.x + xmon_center.x - 3 * xmon_x_distance) / 4, (
                    (3 * (-contact_pads[2].end.x + xmon_center.x - 3 * xmon_x_distance) / 4) ** 2 + (
                    7 * (-contact_pads[2].end.y + xmon_center.y - shift_fl_y) / 8) ** 2) ** 0.5,
                             (-contact_pads[2].end.y + xmon_center.y - shift_fl_y) / 8],
            turn_angles=[atan2(7 * (-contact_pads[2].end.y + xmon_center.y - shift_fl_y) / 8,
                               3 * (-contact_pads[2].end.x + xmon_center.x - 3 * xmon_x_distance) / 4),
                         pi / 2 - atan2(7 * (-contact_pads[2].end.y + xmon_center.y - shift_fl_y) / 8,
                                        3 * (-contact_pads[2].end.x + xmon_center.x - 3 * xmon_x_distance) / 4)],
            trans_in=Trans.R0
        )
        self.cpwrl_fl2.place(tmp_reg)

        self.cpwrl_fl2_end = FluxLineEnd(self.cpwrl_fl2.end, z_md_fl_corrected, width=flux_end_width, trans_in=Trans.R0)
        self.cpwrl_fl2_end.place(tmp_reg)

        # place caplanar line 3md
        self.cpwrl_md3_1 = CPW_RL_Path(
            contact_pads[5].end, shape="LRLRLRL", cpw_parameters=z_md_fl_corrected,
            turn_radiuses=[ctr_line_turn_radius] * 3,
            segment_lengths=[
                (-contact_pads[5].end.y + xmon_center.y - shift_md_y) / 4,
                (contact_pads[5].end.x - xmon_center.x - shift_md_x + 2 * xmon_x_distance) / 2,
                (
                        (
                                (contact_pads[5].end.x - xmon_center.x - shift_md_x + 2 * xmon_x_distance) / 2
                        ) ** 2 +
                        (
                                5 * (-contact_pads[5].end.y + xmon_center.y - shift_md_y) / 8
                        ) ** 2
                ) ** 0.5 - 400e3,
                (-contact_pads[5].end.y + xmon_center.y - shift_md_y) / 8],
            turn_angles=[pi / 2, -atan2(5 * (-contact_pads[5].end.y + xmon_center.y - shift_md_y) / 8,
                                        (contact_pads[5].end.x - xmon_center.x - shift_md_x + 2 * xmon_x_distance) / 2),
                         -pi / 2 + atan2(5 * (-contact_pads[5].end.y + xmon_center.y - shift_md_y) / 8, (
                                 contact_pads[5].end.x - xmon_center.x - shift_md_x + 2 * xmon_x_distance) / 2)],
            trans_in=Trans.R90
        )
        self.cpwrl_md3_1.place(tmp_reg)

        dy = self.cpwrl_md2.end.y - self.cpwrl_md3_1.end.y
        dx = self.cpwrl_md3_1.end.x - (self.xmons[2].center.x + shift_md_x)
        md3_2_radius = ctr_line_turn_radius - 50e3
        self.cpwrl_md3_2 = CPW_RL_Path(
            self.cpwrl_md3_1.end, shape="LRLR", cpw_parameters=z_md_fl_corrected,
            turn_radiuses=[md3_2_radius] * 2,
            segment_lengths=[dy - md3_2_radius, dx],
            turn_angles=[pi / 2, -pi / 2],
            trans_in=Trans.R90
        )
        self.cpwrl_md3_2.place(tmp_reg)

        self.cpwrl_md3_end = MDriveLineEnd(
            list(self.cpwrl_md3_2.primitives.values())[-1], md_z1_params, md_transition, md234_z1_length
        )
        self.cpwrl_md3_end.place(tmp_reg)

        # place caplanar line 3 fl
        fl_l1 = (self.xmons[2].cpw_bempt.end.y - contact_pads[4].end.y) / 4
        fl_l3 = fl_l1
        dr = self.xmons[2].cpw_bempt.end - contact_pads[4].end
        dr.y = dr.y - fl_l1 - fl_l3 - z_md_fl_corrected.width
        turn_angle = atan2(-dr.x, dr.y)
        fl_l2 = dr.abs()
        self.cpwrl_fl3 = CPW_RL_Path(
            self.contact_pads[4].end, shape="LRLRL", cpw_parameters=z_md_fl_corrected,
            turn_radiuses=[ctr_line_turn_radius] * 2,
            segment_lengths=[fl_l1, fl_l2, fl_l3],
            turn_angles=[turn_angle, -turn_angle], trans_in=Trans.R90
        )
        self.cpwrl_fl3.place(tmp_reg)

        self.cpwrl_fl3_end = FluxLineEnd(self.cpwrl_fl3.end, z_md_fl_corrected, width=flux_end_width, trans_in=Trans.R0)
        self.cpwrl_fl3_end.place(tmp_reg)

        # place caplanar line 4 md
        md4_len_y = -contact_pads[7].end.y + xmon_center.y - shift_md_y
        md4_fl4_dx = 100e3
        self.cpwrl_md4 = CPW_RL_Path(
            contact_pads[7].end, shape="LRLRLRL", cpw_parameters=z_md_fl_corrected,
            turn_radiuses=ctr_line_turn_radius / 2,
            segment_lengths=[
                contact_pads[7].end.x - xmon_center.x + xmon_x_distance - shift_md_x - md4_fl4_dx,
                md4_len_y - ctr_line_turn_radius / 2, md4_fl4_dx, ctr_line_turn_radius / 2
            ],
            turn_angles=[-pi / 2, pi / 2, -pi / 2], trans_in=Trans.R180
        )
        self.cpwrl_md4.place(tmp_reg)

        self.cpwrl_md4_end = MDriveLineEnd(
            list(self.cpwrl_md4.primitives.values())[-1],
            md_z1_params, md_transition, md234_z1_length
        )
        self.cpwrl_md4_end.place(tmp_reg)

        # place caplanar line 4 fl
        self.cpwrl_fl4 = CPW_RL_Path(
            contact_pads[6].end, shape="LRLRL", cpw_parameters=z_md_fl_corrected,
            turn_radiuses=[ctr_line_turn_radius] * 2,
            segment_lengths=[(contact_pads[6].end.x - xmon_center.x + xmon_x_distance) / 4,
                             ((6 * (-contact_pads[6].end.y + xmon_center.y - shift_fl_y) / 8) ** 2 +
                              (3 * (contact_pads[6].end.x - xmon_center.x + xmon_x_distance) / 4) ** 2) ** 0.5,
                             2 * (-contact_pads[6].end.y + xmon_center.y - shift_fl_y) / 8],
            turn_angles=[-atan2(6 * (-contact_pads[6].end.y + xmon_center.y - shift_fl_y) / 8,
                                3 * (contact_pads[6].end.x - xmon_center.x + xmon_x_distance) / 4),
                         -pi / 2 + atan2(6 * (-contact_pads[6].end.y + xmon_center.y - shift_fl_y) / 8,
                                         3 * (contact_pads[6].end.x - xmon_center.x + xmon_x_distance) / 4)],
            trans_in=Trans.R180
        )
        self.cpwrl_fl4.place(tmp_reg)

        self.cpwrl_fl4_end = FluxLineEnd(self.cpwrl_fl4.end, z_md_fl_corrected, width=flux_end_width, trans_in=Trans.R0)
        self.cpwrl_fl4_end.place(tmp_reg)

        # place caplanar line 5 md
        md5_l1 = 2 * (contact_pads[9].end.y - xmon_center.y) / 3
        md5_dy = 400e3
        md5_dx = 400e3
        md5_angle = atan2(
            2 * (contact_pads[9].end.x - xmon_center.x - shift_md_x_side) / 3,
            (contact_pads[9].end.y - xmon_center.y) / 3
        )

        self.cpwrl_md5 = CPW_RL_Path(
            contact_pads[9].end, shape="LRLRLRLRL", cpw_parameters=z_md_fl_corrected,
            turn_radiuses=[ctr_line_turn_radius] * 4,
            segment_lengths=[
                md5_dy, md5_dx / sin(pi / 4),
                        md5_l1 - md5_dy - md5_dx / tan(pi / 4) - md5_dx / tan(md5_angle),
                        md5_dx / sin(md5_angle) + (
                                (
                                        (contact_pads[9].end.y - xmon_center.y) / 3
                                ) ** 2 +
                                (
                                        2 * (contact_pads[9].end.x - xmon_center.x - shift_md_x_side) / 3
                                ) ** 2
                        ) ** (0.5),
                        (contact_pads[9].end.x - xmon_center.x - shift_md_x_side) / 3 - md0_md5_gnd
            ],
            turn_angles=[
                pi / 4, -pi / 4,
                -md5_angle,
                -(pi / 2 - md5_angle)
            ],
            trans_in=Trans.R270
        )

        # self.cpwrl_md5 = CPW_RL_Path(
        #     contact_pads[9].end, shape="LRLRL", cpw_parameters=z_md_fl_corrected,
        #     turn_radiuses=[ctr_line_turn_radius] * 2,
        #     segment_lengths=[
        #         md5_l1,
        #         (
        #                 (
        #                         (contact_pads[9].end.y - xmon_center.y) / 3
        #                 ) ** 2 +
        #                 (
        #                         2 * (contact_pads[9].end.x - xmon_center.x - shift_md_x_side) / 3
        #                 ) ** 2
        #         ) ** (0.5),
        #         (contact_pads[9].end.x - xmon_center.x - shift_md_x_side) / 3 - md0_md5_gnd
        #     ],
        #     turn_angles=[
        #         md5_angle,
        #         -md5_angle - pi / 2
        #     ],
        #     trans_in=Trans.R270
        # )
        self.cpwrl_md5.place(tmp_reg)

        self.cpwrl_md5_end = MDriveLineEnd(
            list(self.cpwrl_md5.primitives.values())[-1],
            md_z1_params, md_transition, md15_z1_length
        )
        self.cpwrl_md5_end.place(tmp_reg)

        # place caplanar line 5 fl
        self.cpwrl_fl5_1 = CPW_RL_Path(
            contact_pads[8].end, shape="LRLRLR", cpw_parameters=z_md_fl_corrected,
            turn_radiuses=[ctr_line_turn_radius] * 3,
            segment_lengths=[
                (contact_pads[8].end.x - xmon_center.x) / 4,
                ((contact_pads[8].end.y - xmon_center.y) + 250e3) / 3 + 50e3,
                (
                        (2 * ((contact_pads[8].end.y - xmon_center.y) + 250e3) / 3) ** 2 +
                        ((contact_pads[8].end.x - xmon_center.x) / 2) ** 2
                ) ** 0.5
            ],
            turn_angles=[
                pi / 2,
                -atan2((contact_pads[8].end.x - xmon_center.x) / 2,
                       2 * ((contact_pads[8].end.y - xmon_center.y) + 250e3) / 3),
                - pi / 2 + atan2((contact_pads[8].end.x - xmon_center.x) / 2,
                                 2 * ((contact_pads[8].end.y - xmon_center.y) + 250e3) / 3)
            ],
            trans_in=Trans.R180
        )
        self.cpwrl_fl5_1.place(tmp_reg)
        self.cpwrl_fl5_2 = CPW_RL_Path(
            self.cpwrl_fl5_1.end,
            shape="LRL", cpw_parameters=z_md_fl_corrected,
            turn_radiuses=[ctr_line_turn_radius],
            segment_lengths=[
                self.cpwrl_fl5_1.end.x - (self.xmons[4].center.x +
                                          cross_width_y / 2 + cross_len_x + cross_gnd_gap_x - flux_end_width / 2),
                self.xmons[4].center.y - self.cpwrl_fl5_1.end.y -
                cross_width_x / 2 - cross_gnd_gap_x - z_md_fl_corrected.width + FABRICATION.OVERETCHING
            ],
            turn_angles=[-pi / 2],
            trans_in=Trans.R180
        )
        self.cpwrl_fl5_2.place(tmp_reg)

        self.cpwrl_fl5_end = FluxLineEnd(self.cpwrl_fl5_2.end, z_md_fl_corrected, width=flux_end_width,
                                         trans_in=Trans.R0)
        self.cpwrl_fl5_end.place(tmp_reg)

    def draw_josephson_loops(self):
        new_pars_squid = AsymSquidParams(
            pad_r=5e3, pads_distance=30e3,
            p_ext_width=10e3, p_ext_r=200,
            sq_len=15e3, sq_area=200e6,
            j_width_1=95, j_width_2=348,
            intermediate_width=500, b_ext=1e3, j_length=94, n=20,
            bridge=180, j_length_2=250
        )
        # place left squid
        xmon0 = self.xmons[0]
        center1 = DPoint(
            self.cpwrl_fl1_end.end.x,
            xmon0.center.y - (xmon0.sideX_width + xmon0.sideX_gnd_gap) / 2
        )
        squid = AsymSquid(center1, new_pars_squid, 0)
        self.squids.append(squid)
        squid.place(self.region_el)

        # place intermediate squids
        for xmon_cross in self.xmons[1:-1]:
            squid_center = (xmon_cross.cpw_bempt.start + xmon_cross.cpw_bempt.end) / 2
            squid = AsymSquid(squid_center, new_pars_squid, 0)
            self.squids.append(squid)
            squid.place(self.region_el)

        # place right squid
        xmon5 = self.xmons[4]
        center5 = DPoint(
            self.cpwrl_fl5_end.end.x,
            xmon5.center.y - (xmon5.sideX_width + xmon5.sideX_gnd_gap) / 2
        )
        squid = AsymSquid(center5, new_pars_squid, 0)
        self.squids.append(squid)
        squid.place(self.region_el)

    def draw_el_dc_contacts(self, squids: AsymSquid = None):
        # if argument is omitted, use all squids stored in `self.squids`
        if squids is None:
            squids = self.squids + self.test_squids

        for squid in squids:
            r_pad = squid.params.pad_r
            center_up = squid.pad_up.center + DPoint(0, r_pad)
            center_down = squid.pad_down.center + DPoint(0, -r_pad)
            big_circle_r = 1.5 * r_pad
            self.el_dc_contacts.append(
                [Circle(center_up, big_circle_r), Circle(center_down, big_circle_r)]
            )
            for contact in self.el_dc_contacts[-1]:
                contact.place(self.region_el2)

            # DC contacts has to have intersection with empty layer in photo litography
            # to ensure that first e-beam layer does not broke at the step between
            # substrate and photolytography polygons.
            # Following rectangle pads are cutted from photo region to ensure
            # DC contacts are covering aforementioned level step.
            squid_pad_r = squid.params.pad_r
            squid_pads_d = squid.params.pads_distance - 2 * squid_pad_r
            rec_width = 1.6 * squid_pad_r - 2 * FABRICATION.OVERETCHING
            rec_height = 1.6 * squid_pad_r - FABRICATION.OVERETCHING
            # Rectangle for top DC contact pad
            p1 = squid.origin + DVector(-rec_width / 2, squid_pads_d / 2) + \
                 DVector(0, -FABRICATION.OVERETCHING)
            rec_top = Rectangle(p1, rec_width, rec_height, inverse=True)
            rec_top.place(self.region_ph)

            # Rectangle for bottom DC contact pad
            p2 = squid.origin + DVector(-rec_width / 2, -squid_pads_d / 2) + \
                 DVector(0, FABRICATION.OVERETCHING)
            rec_bot = Rectangle(p2, rec_width, -rec_height, inverse=True)
            rec_bot.place(self.region_ph)

    def draw_test_structures(self):
        new_pars_squid = AsymSquidParams(
            pad_r=5e3, pads_distance=30e3,
            p_ext_width=10e3, p_ext_r=200,
            sq_len=15e3, sq_area=200e6,
            j_width_1=94, j_width_2=347,
            intermediate_width=500, b_ext=1e3, j_length=94, n=20,
            bridge=180, j_length_2=250
        )

        struct_centers = [DPoint(1e6, 4e6), DPoint(8.7e6, 5.7e6), DPoint(6.5e6, 2.7e6)]
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
            DPoint(4.2e6, 1.6e6),
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
            DPoint(8e6, 4e6), DPoint(1e6, 6e6)
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
        for key, val in self.__dict__.items():
            if "cpwrl_md" in key:
                Bridge1.bridgify_CPW(
                    val, bridges_step,
                    dest=self.region_bridges1, dest2=self.region_bridges2
                )
            elif "cpwrl_fl" in key:
                Bridge1.bridgify_CPW(
                    val, bridges_step,
                    dest=self.region_bridges1, dest2=self.region_bridges2,
                    avoid_points=[squid.origin for squid in self.squids],
                    avoid_distance=500e3
                )
        # for readout waveguide
        bridgified_primitives_idxs = list(range(2))
        bridgified_primitives_idxs += list(range(2, 2 * (len(self.resonators) + 1) + 1, 2))
        bridgified_primitives_idxs += list(range(
            2 * (len(self.resonators) + 1) + 1,
            len(self.cpwrl_ro_line.primitives.values()))
        )
        for idx, primitive in enumerate(self.cpwrl_ro_line.primitives.values()):
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
    design = Design5Q("testScript")
    design.draw()
    design.show()
