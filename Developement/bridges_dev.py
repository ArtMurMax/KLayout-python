import pya
from pya import Point, DPoint, Vector, DVector, DSimplePolygon, SimplePolygon, DPolygon, Polygon, Region, Box, DBox
from pya import Trans, DTrans, CplxTrans, DCplxTrans, ICplxTrans

from math import atan2, pi
from importlib import reload
import itertools
from collections import OrderedDict
from typing import Union

import classLib

reload(classLib)
from classLib.baseClasses import ElementBase
from classLib.coplanars import CPWParameters, CPW, CPWArc, CPWRLPath
from classLib.contactPads import ContactPad


class CHIP:
    """
    10x10 mm chip
    PCB design located here:
    https://drive.google.com/drive/folders/1TGjD5wwC28ZiLln_W8M6gFJpl6MoqZWF?usp=sharing
    """
    dx = 5e6
    dy = 5e6

    origin = DPoint(0, 0)
    box = pya.Box().from_dbox(pya.DBox(origin, origin + DPoint(dx, dy)))

    pcb_width = 260e3  # 0.26 mm
    pcb_gap = 190e3  # (0.64 - 0.26) / 2 = 0.19 mm
    pcb_feedline_d = 2500e3  # 2.5 mm
    pcb_Z = CPWParameters(pcb_width, pcb_gap)

    cpw_width = 24.1e3
    cpw_gap = 12.95e3
    chip_Z = CPWParameters(cpw_width, cpw_gap)

    @staticmethod
    def get_contact_pads():
        dx = CHIP.dx
        dy = CHIP.dy
        pcb_feedline_d = CHIP.pcb_feedline_d
        pcb_Z = CHIP.pcb_Z
        chip_Z = CHIP.chip_Z

        contact_pads_left = [
            ContactPad(
                DPoint(0, dy - pcb_feedline_d * (i + 1)), pcb_Z, chip_Z, back_metal_width=50e3,
                back_metal_gap=100e3
            ) for i in range(3)
        ]

        contact_pads_bottom = [
            ContactPad(
                DPoint(pcb_feedline_d * (i + 1), 0), pcb_Z, chip_Z, back_metal_width=50e3,
                back_metal_gap=100e3,
                trans_in=Trans.R90
            ) for i in range(3)
        ]

        contact_pads_right = [
            ContactPad(
                DPoint(dx, pcb_feedline_d * (i + 1)), pcb_Z, chip_Z, back_metal_width=50e3,
                back_metal_gap=100e3,
                trans_in=Trans.R180
            ) for i in range(3)
        ]

        contact_pads_top = [
            ContactPad(
                DPoint(dx - pcb_feedline_d * (i + 1), dy), pcb_Z, chip_Z, back_metal_width=50e3,
                back_metal_gap=100e3,
                trans_in=Trans.R270
            ) for i in range(3)
        ]

        # contact pads are ordered starting with top-left corner in counter-clockwise direction
        contact_pads = itertools.chain(
            contact_pads_left, contact_pads_bottom,
            contact_pads_right, contact_pads_top
        )

        return list(contact_pads)

    @staticmethod
    def get_geometry_params_dict(prefix="", postfix=""):
        from collections import OrderedDict
        geometry_params = OrderedDict(
            [
                ("dx, um", CHIP.dx / 1e3),
                ("dy, um", CHIP.dy / 1e3),
                ("nX", CHIP.nX),
                ("nY", CHIP.nY)
            ]
        )
        modified_dict = OrderedDict()
        for key, val in geometry_params.items():
            modified_dict[prefix + key + postfix] = val
        return modified_dict


class Bridge1(ElementBase):
    """
        Class implements bridges that are used to suppress
        non-TEM modes in coplanar or other types of waveguides.
    """

    def __init__(self, center, gnd_touch_dx=20e3, gnd_touch_dy=10e3,
                 gnd2gnd_dy=70e3, bridge_width=20e3,
                 surround_gap=8e3, transition_len=12e3,
                 trans_in=None):
        self.gnd_touch_dx = gnd_touch_dx
        self.gnd_touch_dy = gnd_touch_dy
        self.gnd2gnd_dy = gnd2gnd_dy
        self.bridge_width = bridge_width
        self.surround_gap = surround_gap
        self.transition_len = transition_len

        self.center = center
        self.angle = 0
        super().__init__(center, trans_in)

        self._geometry_parameters = OrderedDict(
            [
                # TODO: add other members
                ("gnd_touch_dx, um", self.gnd_touch_dx / 1e3)
            ]
        )

    def init_regions(self):
        self.metal_regions["bridges_1"] = Region()  # region with ground contacts
        self.empty_regions["bridges_1"] = Region()  # remains empty

        self.metal_regions["bridges_2"] = Region()  # remains empty
        self.empty_regions["bridges_2"] = Region()  # region with erased bridge area

        center = DPoint(0, 0)
        self.connections = [center]
        self.angle_connections = [0]

        # init metal region of ground touching layer
        top_gnd_center = center + DPoint(0, self.gnd2gnd_dy / 2 + self.gnd_touch_dy/2)
        p1 = top_gnd_center + DPoint(-self.gnd_touch_dx / 2, -self.gnd_touch_dy / 2)
        p2 = p1 + DVector(self.gnd_touch_dx, self.gnd_touch_dy)
        top_gnd_touch_box = pya.DBox(p1, p2)
        self.metal_regions["bridges_1"].insert(pya.Box().from_dbox(top_gnd_touch_box))

        bot_gnd_center = center + DPoint(0, -(self.gnd2gnd_dy / 2 + self.gnd_touch_dy/2))
        p1 = bot_gnd_center + DPoint(-self.gnd_touch_dx / 2, -self.gnd_touch_dy / 2)
        p2 = p1 + DVector(self.gnd_touch_dx, self.gnd_touch_dy)
        bot_gnd_touch_box = pya.DBox(p1, p2)
        self.metal_regions["bridges_1"].insert(pya.Box().from_dbox(bot_gnd_touch_box))

        # init empty region for second layout layer
        # points start from left-bottom corner and goes in clockwise direction
        p1 = bot_gnd_touch_box.p1 + DPoint(-self.surround_gap, -self.surround_gap)
        p2 = p1 + DPoint(0, self.surround_gap + self.gnd_touch_dy +
                            self.transition_len - self.surround_gap)
        # top left corner + `surrounding_gap` + `transition_length`
        p3 = bot_gnd_touch_box.p1 + DPoint(0, bot_gnd_touch_box.b()) + \
             DPoint(0, self.transition_len)
        bl_pts_list = [p1, p2, p3]  # bl stands for bottom-left
        ''' exploiting symmetry of reflection at x and y axes. '''
        # reflecting at x-axis
        tl_pts_list = list(map(lambda x: DTrans.M0 * x, bl_pts_list))  # tl stands for top-left
        # preserving order
        tl_pts_list = reversed(list(tl_pts_list))  # preserving clockwise points order
        # converting iterator to list
        l_pts_list = list(itertools.chain(bl_pts_list, tl_pts_list))  # l stands for left

        # reflecting all points at y-axis
        r_pts_list = list(map(lambda x: DTrans.M90 * x, l_pts_list))
        r_pts_list = list(reversed(r_pts_list))  # preserving clockwise points order

        # gathering points
        pts_list = l_pts_list + r_pts_list  # concatenating proper ordered lists

        empty_polygon = DSimplePolygon(pts_list)
        self.empty_regions["bridges_2"].insert(SimplePolygon(empty_polygon))

    def _refresh_named_connections(self):
        self.center = self.connections[0]

    def _refresh_named_angles(self):
        self.angle = self.angle_connections[0]

    @staticmethod
    def bridgify_CPW(cpw, bridges_step, cell=None, bridge_layer1=-1, bridge_layer2=-1):
        """
            Function puts bridge patterns to fabricate bridges on coplanar waveguide
        `cpw` with bridges having period of `bridges_step` along coplanar's wave
        propagation direction.
            Bridges are distributed over coplanar starting with its center.

        Parameters
        ----------
        cpw : Union[CPW, CPWArc, CPWDPath]
            instance of coplanar class to be bridged during fabrication
        bridges_step : float
            distance between centers of bridges in nm
        cell : pya.Cell
            cell to place bridge polygons at
        bridge_layer1 : int
            index of the layer in the `cell` with ground touching polygons
        bridge_layer2 : int
            index of the layer in the `cell` with empty polygons

        Returns
        -------
        None
        """
        if isinstance(cpw, CPW):
            cpw_center = cpw.start + cpw.end
            alpha = atan2(cpw.dr.y, cpw.dr.x)
            cpw_len = cpw.dr.abs()
            cpw_dir_unit_vector = cpw.dr/cpw.dr.abs()

            # bridge with some initial dimensions
            tmp_bridge = Bridge1(DPoint(0, 0))
            bridge_width = tmp_bridge.gnd_touch_dx + 2*tmp_bridge.surround_gap
        
            # number of additional bridges on either side of center
            additional_bridges_n = int((cpw_len/2 - bridge_width/2) // bridges_step)
            bridge_centers = []
            for i in range(-additional_bridges_n, additional_bridges_n + 1):
                bridge_centers.append(
                    cpw.start + (cpw_len/2 + i*bridges_step)*cpw_dir_unit_vector
                )

            bridges = []
            for center in bridge_centers:
                bridges.append(
                    Bridge1(
                        center,
                        trans_in=DCplxTrans(1, alpha/pi*180, False, 0, 0)
                    )
                )
            for bridge in bridges:
                bridge.place(dest=cell, layer_i=bridge_layer1, region_name="bridges_1")
                bridge.place(dest=cell, layer_i=bridge_layer2, region_name="bridges_2")
        elif isinstance(cpw, CPWArc):
            # to be implemented
            pass
        elif isinstance(cpw, CPWRLPath):
            for primitive in cpw.primitives.values():
                if isinstance(primitive, CPW):
                    Bridge1.bridgify_CPW(
                        primitive, bridges_step,
                        cell, bridge_layer1, bridge_layer2
                    )
        else:
            # do nothing for other shapes
            return


if __name__ == "__main__":
    # getting main references of the application
    # getting main references of the application
    app = pya.Application.instance()
    mw = app.main_window()
    lv = mw.current_view()
    cv = None

    # this insures that lv and cv are valid objects
    if lv is None:
        cv = mw.create_layout(1)
        lv = mw.current_view()
    else:
        cv = lv.active_cellview()

    # find or create the desired by programmer cell and layer
    layout = cv.layout()
    layout.dbu = 0.001
    if (layout.has_cell("testScript")):
        pass
    else:
        cell = layout.create_cell("testScript")
    cell.clear()
    layer_info_photo1_bridges = pya.LayerInfo(12, 0)
    layer_photo1_bridges = layout.layer(layer_info_photo1_bridges)
    layer_info_photo2_bridges = pya.LayerInfo(13, 0)
    layer_photo2_bridges = layout.layer(layer_info_photo2_bridges)
    layer_photo_info = pya.LayerInfo(11, 0)
    layer_photo = layout.layer(layer_photo_info)

    # setting layout view
    lv.select_cell(cell.cell_index(), 0)
    lv.add_missing_layers()

    ### DRAWING SECTION START ###
    # drawing chip
    cell.shapes(layer_photo2_bridges).insert(CHIP.box)
    cell.shapes(layer_photo).insert(CHIP.box)

    origin = DPoint(CHIP.dx / 2, CHIP.dy / 2)
    bridge = Bridge1(origin)
    bridge.place(cell, layer_photo1_bridges, "bridges_1")
    bridge.place(cell, layer_photo2_bridges, "bridges_2")

    p1 = DPoint(1/4*CHIP.dx, CHIP.dy/5)
    p2 = DPoint(3/4*CHIP.dx, CHIP.dy/4)
    width = 20e3
    gap = 10e3
    cpw = CPW(width, gap, p1, p2)
    cpw.place(cell, layer_photo)
    bridges_step = 50e3
    Bridge1.bridgify_CPW(
        cpw, bridges_step,
        cell=cell, bridge_layer1=layer_photo1_bridges, bridge_layer2=layer_photo2_bridges
    )
    z2 = CPWParameters(width, gap)
    cpw2 = CPWRLPath(DPoint(400e3, 400e3), "LRLRL", z2, 60e3, 100e3, [pi / 4, pi / 3])
    cpw2.place(cell, layer_photo)
    Bridge1.bridgify_CPW(
        cpw2, bridges_step,
        cell=cell, bridge_layer1=layer_photo1_bridges, bridge_layer2=layer_photo2_bridges
    )

    lv.zoom_fit()
    ### DRAWING SECTION END ###
