import pya
from math import sqrt, cos, sin, atan2, pi, copysign
from pya import Point, DPoint, DSimplePolygon, SimplePolygon, DPolygon, Polygon, Region, Vector, DVector
from pya import Trans, DTrans, CplxTrans, DCplxTrans, ICplxTrans

from classLib.baseClasses import ComplexBase
from classLib.shapes import Ring, Circle, Rectangle, Cross, IsoTrapezoid, Cross2
from classLib.coplanars import CPW, CPWParameters, CPWArc


class Mark1(ComplexBase):
    def __init__(self, origin, trans_in=None):
        self.cross_in_a = 20e3
        self.cross_out_a = 40e3
        self.leaf_inner = 50e3
        self.leaf_outer = 150e3
        self.leaf_angle = 70
        self.leafs_N = 4
        self.empty_rings_N = 2
        self.empty_rings_width = 10e3

        self.empty_leaf_angle = (360 - self.leaf_angle * self.leafs_N) / self.leafs_N
        self.avg_r = (self.leaf_inner + self.leaf_outer) / 2

        if (self.cross_out_a < self.cross_in_a):
            print("cross inner square must be larger than outer")
        super(self).__init__(origin, trans_in)

    def init_primitives(self):
        origin = DPoint(0, 0)

        # square box that clears out the area of the mark
        self.primitives["empty_box"] = Rectangle(DPoint(-self.leaf_outer, -self.leaf_outer), 2 * self.leaf_outer,
                                                 2 * self.leaf_outer, inverse=True)

        self.primitives["cross"] = Cross(DPoint(-self.cross_out_a / 2, -self.cross_out_a / 2), self.cross_in_a,
                                         self.cross_out_a)

        Z = CPWParameters(self.leaf_outer - self.leaf_inner, 0)

        for i in range(self.leafs_N):
            start_angle = i * (self.leaf_angle + self.empty_leaf_angle) - self.leaf_angle / 2
            trans = DCplxTrans(1, start_angle + 90, False, 0, -self.avg_r)
            self.primitives["leaf_" + str(i)] = CPWArc(Z, origin, self.avg_r, self.leaf_angle * pi / 180,
                                                       trans_in=trans)

        Z_empty = CPWParameters(0, self.empty_rings_width / 2)
        for i in range(1, self.empty_rings_N + 1):
            r = self.leaf_inner + (self.leaf_outer - self.leaf_inner) / (self.empty_rings_N + 1) * i
            self.primitives["empty_ring_" + str(i)] = CPWArc(Z_empty, DVector(0, -r), r, 2 * pi)


class Mark2(ComplexBase):
    def __init__(self, origin, cross_thickness=1e3, cross_size=3e3, trans_in=None):
        self.ring1_outer_r = 300e3
        self.ring1_thickness = 30e3
        self.ring2_outer_r = 250e3
        self.ring2_thickness = 30e3
        self.inner_circle_radius = 200e3
        self.trap_h = 150e3
        self.trap_t = 100e3
        self.cross_thickness = cross_thickness
        self.cross_size = cross_size
        self.trap_b = self.cross_thickness
        self.trap_dist = 2*self.cross_thickness  # distance from the center
        super().__init__(origin, trans_in)

    def init_primitives(self):
        origin = DPoint(0, 0)
        self.primitives["empty_ring1"] = Ring(origin, self.ring1_outer_r, self.ring1_thickness, inverse=True)
        self.primitives["empty_ring2"] = Ring(origin, self.ring2_outer_r, self.ring2_thickness, inverse=True)
        self.primitives["inner_circle"] = Circle(origin, self.inner_circle_radius, inverse=True)
        self.primitives["trap_top"] = IsoTrapezoid(
            origin + DPoint(-self.trap_b / 2, self.trap_dist),
            self.trap_h, self.trap_b, self.trap_t
        )
        self.primitives["trap_left"] = IsoTrapezoid(
            origin + DPoint(-self.trap_dist, -self.trap_b / 2),
            self.trap_h, self.trap_b, self.trap_t, trans_in=Trans.R90
        )
        self.primitives["trap_bottom"] = IsoTrapezoid(
            origin + DPoint(self.trap_b / 2, -self.trap_dist),
            self.trap_h, self.trap_b, self.trap_t, trans_in=Trans.R180
        )
        self.primitives["trap_right"] = IsoTrapezoid(
            origin + DPoint(self.trap_dist, self.trap_b / 2),
            self.trap_h, self.trap_b, self.trap_t, trans_in=Trans.R270
        )
        self.primitives["cross"] = Cross2(origin, self.cross_thickness, self.cross_size)


class MarkBolgar(ComplexBase):
    def __init__(self, origin, lines_thickness=3e3, ring1_thickness=15e3, ring2_thickness=15e3,
                 overetching=0.0, trans_in=None):
        self.ring1_outer_r = 200e3 - overetching
        self.ring1_thickness = ring1_thickness + 2*overetching
        self.ring2_outer_r = 100e3 - overetching
        self.ring2_thickness = ring2_thickness + 2*overetching
        self.aim_lines_width = lines_thickness + 2*overetching
        self.overetching = overetching
        self.center = None
        super().__init__(origin, trans_in)

    def init_primitives(self):
        center = DPoint(0, 0)

        self.empty_circle = Circle(center, self.ring1_outer_r + self.ring1_thickness, inverse=True)
        self.primitives["empty_circle"] = self.empty_circle

        # outer ring
        self.ring1 = Ring(center, self.ring1_outer_r, self.ring1_thickness)
        self.primitives["ring1"] = self.ring1

        # inner ring
        self.ring2 = Ring(center, self.ring2_outer_r, self.ring2_thickness)
        self.primitives["ring2"] = self.ring2

        ## four aim lines ##
        center_shift = self.aim_lines_width/3
        line_length = self.ring1_outer_r - self.ring1_thickness/2 - center_shift
        # left horizontal line
        p1 = center + DPoint(-center_shift, 0)
        p2 = p1 + DPoint(-line_length, 0)
        self.left_aim_line = CPW(self.aim_lines_width, 0, p1, p2)
        self.primitives["left_aim_line"] = self.left_aim_line

        # bottom vertical line
        p1 = center + DPoint(0, -center_shift)
        p2 = p1 + DPoint(0, -line_length)
        self.bottom_aim_line = CPW(self.aim_lines_width, 0, p1, p2)
        self.primitives["bottom_aim_line"] = self.bottom_aim_line

        # right horizontal line
        p1 = center + DPoint(center_shift, 0)
        p2 = p1 + DPoint(line_length, 0)
        self.right_aim_line = CPW(self.aim_lines_width, 0, p1, p2)
        self.primitives["right_aim_line"] = self.right_aim_line

        # top vertical line
        p1 = center + DPoint(0, center_shift)
        p2 = p1 + DPoint(0, line_length)
        self.top_aim_line = CPW(self.aim_lines_width, 0, p1, p2)
        self.primitives["top_aim_line"] = self.top_aim_line

        # center romb for better aiming
        self.center_romb = Rectangle(
            center,
            self.aim_lines_width, self.aim_lines_width,
            trans_in=DCplxTrans(1, 45, False, -self.aim_lines_width/2, -self.aim_lines_width/2),
            inverse=True
        )
        self.primitives["center_romb"] = self.center_romb

        self.connections = [center]

    def _refresh_named_connections(self):
        self.center = self.connections[0]
