import pya
from math import sqrt, cos, sin, atan2, pi, copysign
from pya import Point,DPoint,DSimplePolygon,SimplePolygon, DPolygon, Polygon,  Region
from pya import Trans, DTrans, CplxTrans, DCplxTrans, ICplxTrans

import math

from classLib.baseClasses import ComplexBase, ElementBase
from classLib.coplanars import CPWParameters, CPW, CPW2CPW


class ContactPad(ComplexBase):
    def __init__(self, origin,
                 pcb_cpw_params=CPWParameters(width=200e3, gap=120e3),
                 chip_cpw_params=CPWParameters(width=24.1e3, gap=12.95e3),
                 pad_length=300e3, back_metal_width=0, back_metal_gap=None,
                 transition_len=150e3,
                 trans_in=None):
        '''

        Parameters
        ----------
        origin : DPoint
            center of the thick pad's outer side
        pcb_cpw_params
        chip_cpw_params
        pad_length
        back_metal_width
        back_metal_gap
        transition_len
        trans_in
        '''

        self.pcb_cpw_params: CPWParameters = pcb_cpw_params
        self.chip_cpw_params: CPWParameters = chip_cpw_params
        self.pad_length = pad_length
        self.back_metal_length = back_metal_width
        if back_metal_gap is not None:
            self.back_metal_gap = back_metal_gap
        else:
            self.back_metal_gap = self.pcb_cpw_params.gap
        self.transition_length = transition_len

        super().__init__(origin, trans_in)

        self.start = self.connections[0]
        self.end = self.connections[1]

    def init_primitives(self):
        origin = DPoint(0, 0)

        # place metal between pad and PCB-chip edge
        self.cpw_back_metal = CPW(
            self.pcb_cpw_params.b, 0,
            start=origin, end=origin + DPoint(self.back_metal_length, 0)
        )
        self.primitives["cpw_back_metal"] = self.cpw_back_metal

        # erase metal between previous metal structure and pad-pcb face
        self.cpw_back_gap = CPW(
            0, self.pcb_cpw_params.b / 2,
            start=self.cpw_back_metal.end,
            end=self.cpw_back_metal.end + DPoint(self.back_metal_gap, 0)
        )
        self.primitives["cpw_back_gap"] = self.cpw_back_gap

        # place PCB-matching pad
        self.cpw_pcb_match = CPW(
            start=self.cpw_back_gap.end,
            end=self.cpw_back_gap.end + DPoint(self.pad_length, 0),
            cpw_params=self.pcb_cpw_params
        )
        self.primitives["cpw_pcb_match"] = self.cpw_pcb_match

        # PCB-match to on-chip waveguide trapeziod adapter
        self.cpw2cpw = CPW2CPW(
            self.pcb_cpw_params,
            self.chip_cpw_params,
            start=self.cpw_pcb_match.end,
            end=self.cpw_pcb_match.end + DPoint(self.transition_length, 0)
        )
        self.primitives["cpw2cpw"] = self.cpw2cpw

        self.connections = [origin, self.cpw2cpw.end]
        self.angle_connections = [0, self.cpw2cpw.alpha_end]


class Test_frame(ElementBase):

    def __init__(self, origin, w_JJ, h_JJ, asymmetry, JJ_site_span, frame_cpw_params=CPWParameters(200e3, 8e3),
                 frame_length=200e3, trans_in=None, use_cell=False):

        self._frame_cpw_params = frame_cpw_params
        self._frame_length = frame_length
        self._h_JJ, self._w_JJ = h_JJ, w_JJ
        self._asymmetry = asymmetry
        self._JJ_site_span = JJ_site_span
        self._use_cell = use_cell
        self._origin = origin

        super().__init__(origin, trans_in)

        self.start = self.connections[0]
        self.end = self.connections[1]

    def init_regions(self):

        self.metal_regions["photo"] = Region()
        self.empty_regions["photo"] = Region()
        self.metal_regions["ebeam"] = Region()
        self.empty_regions["ebeam"] = Region()

        w_frame, g_frame = self._frame_cpw_params.width, self._frame_cpw_params.gap
        l_frame = self._frame_length

        metal_points1 = [DPoint(g_frame, g_frame),
                         DPoint(g_frame + w_frame, g_frame),
                         DPoint(g_frame + w_frame, g_frame + l_frame),
                         DPoint(g_frame, g_frame + l_frame)]
        metal_poly1 = DSimplePolygon(metal_points1)

        self.metal_regions["photo"].insert(metal_poly1)

        metal_points2 = [DPoint(g_frame, 2 * g_frame + l_frame),
                         DPoint(g_frame + w_frame, 2 * g_frame + l_frame),
                         DPoint(g_frame + w_frame, 2 * g_frame + 2 * l_frame),
                         DPoint(g_frame, 2 * g_frame + 2 * l_frame)]
        metal_poly2 = DSimplePolygon(metal_points2)

        self.metal_regions["photo"].insert(metal_poly2)

        protect_points = [DPoint(0, 0),
                          DPoint(2 * g_frame + w_frame, 0),
                          DPoint(2 * g_frame + w_frame, 3 * g_frame + 2 * l_frame),
                          DPoint(0, 3 * g_frame + 2 * l_frame)]

        protect_poly = DSimplePolygon(protect_points)
        protect_region = Region(protect_poly)

        empty_region = protect_region - self.metal_regions["photo"]

        self.empty_regions["photo"].insert(empty_region)

        self.connections = [DPoint(0, 0), DPoint(g_frame + 0.5 * w_frame, 1.5 * g_frame + l_frame)]

        if not self._use_cell:

            squid = SQUIDManhattan(self.connections[1],
                                   self._w_JJ, self._h_JJ, \
                                   self._asymmetry, 150, self._JJ_site_span * 1.5, \
                                   squid_width_top=6.2e3, squid_width_bottom=4e3)

            self.metal_regions["ebeam"].insert(squid.metal_region)
        else:
            squid = SQUIDManhattan(DPoint(0, 0),
                                   self._w_JJ, self._h_JJ, \
                                   self._asymmetry, 150, self._JJ_site_span * 1.5, \
                                   squid_width_top=6.2e3, squid_width_bottom=4e3)
            # instance = Instance()
            # print(squid._cell.cell_index())
            trans = Trans(int(self._origin.x + g_frame + 0.5 * w_frame),
                          int(self._origin.y + 1.5 * g_frame + l_frame))
            arr = CellInstArray(squid._cell.cell_index(), trans)
            # print(arr.size())
            # instance.cell_inst = arr
            app = pya.Application.instance()
            cur_cell = app.main_window().current_view().active_cellview().cell
            cur_cell.insert(arr)
