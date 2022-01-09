import pya
from math import cos, sin, atan2, pi
from numpy import sign
from pya import Point, DPoint, DSimplePolygon, SimplePolygon, DPolygon, Polygon, Region
from pya import Trans, DTrans, CplxTrans, DCplxTrans, ICplxTrans

from classLib.baseClasses import ElementBase
from classLib.airbridge import Airbridge


class BridgedCPW(ElementBase):
    """@brief: class represents single coplanar waveguide
          @params:   float width - represents width of the central conductor
                            float gap - spacing between central conductor and ground planes
                            float gndWidth - width of ground plane to be drawed
                            DPoint start - center aligned point, determines the start point of the coplanar segment
                            DPoint end - center aligned point, determines the end point of the coplanar segment
        """

    def __init__(self, width, gap, bridge_interval=100e3, start=DPoint(0, 0), end=DPoint(0, 0), gndWidth=-1,
                 trans_in=None):
        self.width = width
        self.gap = gap
        self._bridge_interval = bridge_interval
        self.b = 2 * gap + width
        self.gndWidth = gndWidth
        self.end = end
        self.start = start
        self.dr = end - start
        super().__init__(start, trans_in)
        self.start = self.connections[0]
        self.end = self.connections[1]
        self.dr = self.end - self.start
        self.alpha_start = self.angle_connections[0]
        self.alpha_end = self.angle_connections[1]

    def init_regions(self):

        self.metal_regions["photo"] = Region()
        self.empty_regions["photo"] = Region()

        self.metal_regions["bridges"] = Region()
        self.empty_regions["bridges"] = Region()

        self.metal_regions["bridge_patches"] = Region()
        self.empty_regions["bridge_patches"] = Region()

        self.connections = [DPoint(0, 0), self.dr]
        self.start = DPoint(0, 0)
        self.end = self.start + self.dr
        alpha = atan2(self.dr.y, self.dr.x)
        self.angle_connections = [alpha, alpha]
        alpha_trans = ICplxTrans().from_dtrans(DCplxTrans(1, alpha * 180 / pi, False, self.start))
        self.metal_regions["photo"].insert(pya.Box(Point().from_dpoint(DPoint(0, -self.width / 2)),
                                                   Point().from_dpoint(DPoint(self.dr.abs(), self.width / 2))))
        self.empty_regions["photo"].insert(pya.Box(Point().from_dpoint(DPoint(0, self.width / 2)),
                                                   Point().from_dpoint(
                                                       DPoint(self.dr.abs(), self.width / 2 + self.gap))))
        self.empty_regions["photo"].insert(pya.Box(Point().from_dpoint(DPoint(0, -self.width / 2 - self.gap)),
                                                   Point().from_dpoint(DPoint(self.dr.abs(), -self.width / 2))))
        self.metal_regions["photo"].transform(alpha_trans)
        self.empty_regions["photo"].transform(alpha_trans)

        if self.dr.x == 0:
            N_bridges = int((self.dr.y - self._bridge_interval) // self._bridge_interval + 1)
            for i in range(N_bridges):
                ab = Airbridge(
                    self.start + DPoint(0, (self._bridge_interval / 2 + i * self._bridge_interval) * sign(self.dr.y)),
                    trans_in=DTrans.R90)
                self.metal_regions["bridges"] += ab.metal_regions["bridges"]
                self.metal_regions["bridges"] += ab.metal_regions["bridge_patches"]
        elif self.dr.y == 0:
            N_bridges = int((self.dr.x - self._bridge_interval) // self._bridge_interval + 1)
            print("N_bridges", N_bridges)
            for i in range(N_bridges):
                bridge_pos = self.start + DPoint(
                    (self._bridge_interval / 2 + i * self._bridge_interval) * sign(self.dr.x), 0)
                ab = Airbridge(bridge_pos, trans_in=None)
                self.metal_regions["bridges"] += ab.metal_regions["bridges"]
                self.metal_regions["bridge_patches"] += ab.metal_regions["bridge_patches"]


class BridgedCPWArc(ElementBase):
    def __init__(self, Z0, start, R, delta_alpha, bridge_interval=100e3, trans_in=None):
        self.R = R
        self.start = start
        self.center = start + DPoint(0, self.R)
        self.end = self.center + DPoint(sin(delta_alpha), -cos(delta_alpha)) * self.R
        self.dr = self.end - self.start

        self.width = Z0.width
        self.gap = Z0.gap

        self.delta_alpha = delta_alpha
        self.alpha_start = 0
        self.alpha_end = self.delta_alpha
        self._bridge_interval = bridge_interval

        super().__init__(start, trans_in)
        self.start = self.connections[0]
        self.end = self.connections[1]
        self.center = self.connections[2]

        # print("End coords:", self.dr, self.end)

        self.alpha_start = self.angle_connections[0]
        self.alpha_end = self.angle_connections[1]

    def _get_solid_arc(self, center, R, width, alpha_start, alpha_end, n_inner, n_outer):
        pts = []
        #        print(alpha_start/pi, alpha_end/pi, cos( alpha_start ), cos( alpha_end ),
        #                         sin(alpha_start), sin(alpha_end))

        if alpha_end > alpha_start:
            alpha_start = alpha_start - 1e-3
            alpha_end = alpha_end + 1e-3
        else:
            alpha_start = alpha_start + 1e-3
            alpha_end = alpha_end - 1e-3

        d_alpha_inner = (alpha_end - alpha_start) / (n_inner - 1)
        d_alpha_outer = -(alpha_end - alpha_start) / (n_outer - 1)

        #        print("Center:", center)

        for i in range(0, n_inner):
            alpha = alpha_start + d_alpha_inner * i
            pts.append(center + DPoint(cos(alpha), sin(alpha)) * (R - width / 2))
        for i in range(0, n_outer):
            alpha = alpha_end + d_alpha_outer * i
            pts.append(center + DPoint(cos(alpha), sin(alpha)) * (R + width / 2))
        #        print("Points:", pts[:n_inner],"\n       ", pts[n_inner:], "\n")
        return DSimplePolygon(pts)

    def init_regions(self):

        self.metal_regions["photo"] = Region()
        self.empty_regions["photo"] = Region()

        self.metal_regions["bridges"] = Region()
        self.empty_regions["bridges"] = Region()

        self.metal_regions["bridge_patches"] = Region()
        self.empty_regions["bridge_patches"] = Region()

        self.connections = [DPoint(0, 0), self.dr, DPoint(0, self.R)]
        self.angle_connections = [self.alpha_start, self.alpha_end]
        self.start = DPoint(0, 0)
        self.end = self.dr
        self.center = DPoint(0, self.R)

        n_inner = 10
        n_outer = 10

        metal_arc = self._get_solid_arc(self.center, self.R, self.width,
                                        self.alpha_start - pi / 2, self.alpha_end - pi / 2, n_inner, n_outer)

        empty_arc1 = self._get_solid_arc(self.center, self.R - (self.width + self.gap) / 2,
                                         self.gap, self.alpha_start - pi / 2, self.alpha_end - pi / 2, n_inner, n_outer)

        empty_arc2 = self._get_solid_arc(self.center, self.R + (self.width + self.gap) / 2,
                                         self.gap, self.alpha_start - pi / 2, self.alpha_end - pi / 2, n_inner, n_outer)

        self.metal_regions["photo"].insert(SimplePolygon(metal_arc))
        self.empty_regions["photo"].insert(SimplePolygon(empty_arc1))
        self.empty_regions["photo"].insert(SimplePolygon(empty_arc2))

        bridge_pos = self.center + DPoint(sin(self.delta_alpha / 2), -cos(self.delta_alpha / 2)) * self.R

        ab = Airbridge(bridge_pos, trans_in=DCplxTrans(1, self.delta_alpha / 2 * 180 / pi, False, 0, 0))
        self.metal_regions["bridges"] += ab.metal_regions["bridges"]
        self.metal_regions["bridge_patches"] += ab.metal_regions["bridge_patches"]
