# import native python libraries
from importlib import reload

# import well-supported third party libraries

# import project specific 3rd party libraries
from pya import Point, Vector, DPoint, DVector, DEdge, \
    DSimplePolygon, SimplePolygon, DPolygon, DBox, Polygon, Region

# import project libraries
import classLib
reload(classLib)

from classLib.baseClasses import ComplexBase
from classLib.shapes import Rectangle

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