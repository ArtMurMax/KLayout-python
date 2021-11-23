from math import sqrt, cos, sin, atan2, pi, copysign, tan

import pya
import numpy as np
from pya import Point, DPoint, DVector, DSimplePolygon, SimplePolygon, DPolygon, Polygon, Region
from pya import Trans, DTrans, CplxTrans, DCplxTrans, ICplxTrans

from typing import Union, List
from collections import OrderedDict
import itertools
import copy

from classLib.baseClasses import ElementBase, ComplexBase
from classLib.bridgedCoplanars import BridgedCPW, BridgedCPWArc


class CPWParameters(ElementBase):
    def __init__(self, width=0, gap=0, smoothing=False):
        # smoothing `True` value means that this CPW is designated
        # to continuously connect two coplanars. None of the other
        # parameters matter in this case.
        self.smoothing = smoothing
        self.width = width
        self.gap = gap
        self.b = 2 * gap + width

        self._geometry_parameters = {"cpw width, um": self.width, "cpw_gap, um": self.gap}


class CPW(ElementBase):
    """@brief: class represents single coplanar waveguide
        @params:  float width - represents width of the central conductor
                        float gap - spacing between central conductor and ground planes
                        float gndWidth - width of ground plane to be drawed
                        DPoint start - center aligned point, determines the start point of the coplanar segment
                        DPoint end - center aligned point, determines the end point of the coplanar segment
    """

    def __init__(self, width=None, gap=None, start=DPoint(0, 0), end=DPoint(0, 0), gndWidth=-1, trans_in=None,
                 cpw_params=None):
        if (cpw_params is None):
            self.width = width
            self.gap = gap
            self.b = 2 * gap + width
        else:
            self.width = cpw_params.width
            self.gap = cpw_params.gap
            self.b = 2 * self.gap + self.width
        self.gndWidth = gndWidth
        self.end = end
        self.start = start
        self.dr = end - start
        super().__init__(start, trans_in)

        self._geometry_parameters = OrderedDict(
            [
                ("width, um", self.width / 1e3),
                ("gap, um", self.gap / 1e3),
                ("start.x, um", self.start.x / 1e3),
                ("start.y, um", self.start.y / 1e3),
                ("end.x", self.end.x / 1e3),
                ("end.y", self.end.y / 1e3)
            ]
        )

    def init_regions(self):
        self.connections = [DPoint(0, 0), self.dr]
        self.start = DPoint(0, 0)
        self.end = self.start + self.dr
        alpha = atan2(self.dr.y, self.dr.x)
        self.angle_connections = [alpha, alpha]
        alpha_trans = ICplxTrans().from_dtrans(DCplxTrans(1, alpha * 180 / pi, False, self.start))
        metal_poly = DSimplePolygon([DPoint(0, -self.width / 2),
                                     DPoint(self.dr.abs(), -self.width / 2),
                                     DPoint(self.dr.abs(), self.width / 2),
                                     DPoint(0, self.width / 2)])
        self.connection_edges = [3, 1]
        self.metal_region.insert(pya.SimplePolygon().from_dpoly(metal_poly))
        if (self.gap != 0):
            self.empty_region.insert(
                pya.Box(
                    Point().from_dpoint(DPoint(0, self.width / 2)),
                    Point().from_dpoint(
                        DPoint(self.dr.abs(), self.width / 2 + self.gap)
                    )
                )
            )
            self.empty_region.insert(
                pya.Box(
                    Point().from_dpoint(DPoint(0, -self.width / 2 - self.gap)),
                    Point().from_dpoint(DPoint(self.dr.abs(), -self.width / 2))
                )
            )
        self.metal_region.transform(alpha_trans)
        self.empty_region.transform(alpha_trans)

    def _refresh_named_connections(self):
        self.end = self.connections[1]
        self.start = self.connections[0]
        self.dr = self.end - self.start

    def _refresh_named_angles(self):
        self.alpha_start = self.angle_connections[0]
        self.alpha_end = self.angle_connections[1]


class CPWArc(ElementBase):
    def __init__(self, z0=CPWParameters(width=20e3, gap=10e3),
                 start=DPoint(0, 0), R=2e3,
                 delta_alpha=pi/4, trans_in=None):
        # TODO: make constructor parametrical
        #  i.e. request center of the arc and angle interval in
        #  radians (\alpha_1, \alpha_2) such that \alpha_1 <= \alpha_2 and
        #  \alpha_1, \alpha_2 lies in [0,2\pi]
        #  HEADACHE NOTE: all the classes used older notation has to be
        #  updated
        self.R = R
        self.start = start
        self.center = start + DPoint(0, self.R)
        self.end = self.center + DPoint(sin(delta_alpha), -cos(delta_alpha)) * self.R
        self.dr = self.end - self.start

        self.width = z0.width
        self.gap = z0.gap
        self.b = self.width + 2*self.gap

        self.delta_alpha = delta_alpha
        self.alpha_start = 0
        self.alpha_end = self.delta_alpha

        super().__init__(start, trans_in)
        self.start = self.connections[0]
        self.end = self.connections[1]
        self.center = self.connections[2]

        self.alpha_start = self.angle_connections[0]
        self.alpha_end = self.angle_connections[1]

    def _get_solid_arc(self, center, R, width,
                       alpha_start, alpha_end, n_inner, n_outer):
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
        d_alpha_outer = -d_alpha_inner

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
        self.connections = [DPoint(0, 0), self.dr, DPoint(0, self.R)]
        self.angle_connections = [self.alpha_start, self.alpha_end]
        self.start = DPoint(0, 0)
        self.end = self.dr
        self.center = DPoint(0, self.R)

        from ._PROG_SETTINGS import PROGRAM
        n_inner = PROGRAM.ARC_PTS_N
        n_outer = PROGRAM.ARC_PTS_N

        metal_arc = self._get_solid_arc(self.center, self.R, self.width,
                                        self.alpha_start - pi / 2, self.alpha_end - pi / 2, n_inner, n_outer)
        self.connection_edges = [n_inner + n_outer, n_inner]
        empty_arc1 = self._get_solid_arc(self.center, self.R - (self.width + self.gap) / 2,
                                         self.gap, self.alpha_start - pi / 2, self.alpha_end - pi / 2, n_inner, n_outer)

        empty_arc2 = self._get_solid_arc(self.center, self.R + (self.width + self.gap) / 2,
                                         self.gap, self.alpha_start - pi / 2, self.alpha_end - pi / 2, n_inner, n_outer)
        self.metal_region.insert(SimplePolygon().from_dpoly(metal_arc))
        self.empty_region.insert(SimplePolygon().from_dpoly(empty_arc1))
        self.empty_region.insert(SimplePolygon().from_dpoly(empty_arc2))

    def _refresh_named_connections(self):
        self.start = self.connections[0]
        self.end = self.connections[1]
        self.center = self.connections[2]

    def _refresh_named_angles(self):
        self.alpha_start = self.angle_connections[0]
        self.alpha_end = self.angle_connections[1]


class CPW2CPW(ElementBase):
    def __init__(self, Z0, Z1, start, end, trans_in=None):
        self.Z0 = Z0
        self.Z1 = Z1
        self.start = start
        self.end = end
        self.dr = self.end - self.start
        super().__init__(start, trans_in)
        self.start = self.connections[0]
        self.end = self.connections[1]
        self.alpha_start = self.angle_connections[0]
        self.alpha_end = self.angle_connections[1]

    def init_regions(self):
        self.connections = [DPoint(0, 0), DPoint(self.dr.abs(), 0)]
        self.angle_connections = [0, 0]
        alpha = atan2(self.dr.y, self.dr.x)
        self.angle_connections = [alpha, alpha]
        alpha_trans = DCplxTrans(1, alpha * 180 / pi, False, 0, 0)

        m_poly = DSimplePolygon([DPoint(0, -self.Z0.width / 2), DPoint(self.dr.abs(), -self.Z1.width / 2),
                                 DPoint(self.dr.abs(), self.Z1.width / 2), DPoint(0, self.Z0.width / 2)])
        e_poly1 = DSimplePolygon([DPoint(0, -self.Z0.b / 2), DPoint(self.dr.abs(), -self.Z1.b / 2),
                                  DPoint(self.dr.abs(), -self.Z1.width / 2), DPoint(0, -self.Z0.width / 2)])
        e_poly2 = DSimplePolygon([DPoint(0, self.Z0.b / 2), DPoint(self.dr.abs(), self.Z1.b / 2),
                                  DPoint(self.dr.abs(), self.Z1.width / 2), DPoint(0, self.Z0.width / 2)])

        m_poly.transform(alpha_trans)
        e_poly1.transform(alpha_trans)
        e_poly2.transform(alpha_trans)

        self.metal_region.insert(SimplePolygon.from_dpoly(m_poly))
        self.empty_region.insert(SimplePolygon.from_dpoly(e_poly1))
        self.empty_region.insert(SimplePolygon.from_dpoly(e_poly2))

    def _refresh_named_connections(self):
        self.start = self.connections[0]
        self.end = self.connections[1]

    def _refresh_named_angles(self):
        self.alpha_start = self.angle_connections[0]
        self.alpha_end = self.angle_connections[1]


class CPW2CPWArc(ElementBase):
    def __init__(self, origin=DPoint(0, 0), r=10, start_angle=0,
                 end_angle=np.pi / 4,
                 cpw1_params=None, cpw2_params=None,
                 trans_in=None, inverse=False):
        """
        Parameters
        ----------
        origin : DPoint
            Point where reference frame point (0,0) will be transfered
            at designs reference frame.
        r : float
            arc central radius
        start_angle : float
            starting angle in radians lying in interval (0, 2pi)
        end_angle : float
            ending angle in radians lying in interval (0, 2pi)
        cpw1_params : CPWParameters
            parameters of the starting CPW
        cpw2_params : Optional[CPWParameters]
            parameters of the ending CPW
            If `None` - equals to `cpw1`
        trans_in : Union[DcplxTrans, DTrans]
            Transformation in object`s reference frame.
        inverse : bool
            Empty and metal polygons are interchanged.
            Metal polygon will be erased from background and then empty
            polygon will be filled with metal during call of `place()`
            function.
        """
        self.cpw1_params: CPWParameters = cpw1_params
        self.cpw2_params: CPWParameters = cpw2_params
        self.r: float = r
        self.start_angle: float = start_angle
        self.end_angle: float = end_angle

        self.center = DPoint(0, 0)
        self.start = DPoint(
            self.r * np.cos(self.start_angle),
            self.r * np.sin(self.start_angle)
        )
        self.end = DPoint(
            self.r * np.cos(self.end_angle),
            self.r * np.sin(self.end_angle)
        )

        self.width = self.cpw2_params.width
        self.gap = self.cpw2_params.gap
        self.b = self.width + 2*self.gap

        from classLib._PROG_SETTINGS import PROGRAM
        self.n_arc_pts = PROGRAM.ARC_PTS_N

        super().__init__(
            origin=origin, trans_in=trans_in, inverse=inverse
        )
        # print("end: ", self.end)
        # print("start: ", self.start)
        # print("center: ", self.center)
        # print()

    def init_regions(self):
        metal_arc = self._get_cpw_arc(
            center=self.center, r=self.r,
            segment="center_metal"
        )
        empty_arc1 = self._get_cpw_arc(
            center=self.center, r=self.r,
            segment="outer_gap"
        )
        empty_arc2 = self._get_cpw_arc(
            center=self.center, r=self.r,
            segment="inner_gap"
        )
        self.metal_region.insert(SimplePolygon().from_dpoly(metal_arc))
        self.empty_region.insert(SimplePolygon().from_dpoly(empty_arc1))
        self.empty_region.insert(SimplePolygon().from_dpoly(empty_arc2))

        self.connections = [self.start.dup(), self.center.dup(),
                            self.end.dup()]
        self.angle_connections = [self.start_angle + np.pi/2,
                                  self.end_angle + np.pi/2]

    def _get_cpw_arc(self, center, r, n_arc_pts=200, method="linear",
                     segment="center_metal"):
        """
        TODO: add description

        Parameters
        ----------
        center : Union[DPoint, List[DPoint]]
        r : float
        n_arc_pts : int
            number of points along width guide line (including both ends).
        method : str
            "linear" - width of polygons are scaled linearly from one
            end to another
        segment : str
            "inner_gap" - inner gap of CPW
            "center_metal" - center metal polygon (defualt)
            "outer_gap" - outer gap of CPW polygon

        Returns
        -------

        """
        pts = []

        if self.end_angle > self.start_angle:
            start_angle = self.start_angle - 1e-3
            end_angle = self.end_angle + 1e-3
        else:
            start_angle = self.start_angle + 1e-3
            end_angle = self.end_angle - 1e-3

        if method == "linear":
            alphas = np.linspace(
                start_angle, end_angle, n_arc_pts
            )

            self._gaps = np.linspace(
                self.cpw1_params.gap, self.cpw2_params.gap, n_arc_pts
            )
            self._widths = np.linspace(
                self.cpw1_params.width, self.cpw2_params.width, n_arc_pts
            )

            if segment == "center_metal":
                dr_arr = self._widths / 2
                r_arr = np.repeat(r, n_arc_pts)
            elif segment == "inner_gap":
                dr_arr = self._gaps / 2
                r_arr = r - (self._widths + self._gaps) / 2
            elif segment == "outer_gap":
                dr_arr = self._gaps / 2
                r_arr = r + (self._widths + self._gaps) / 2
            else:
                raise ValueError(
                    "`segment` argument has invalid value.\n"
                    "See docstring for details"
                )

            # inner points
            for i in range(0, n_arc_pts):
                alpha = alphas[i]
                dr = dr_arr[i]
                r = r_arr[i]
                pts.append(
                    center + DPoint(np.cos(alpha), np.sin(alpha)) *
                    (r - dr)
                )
            # outer points
            for i in range(0, n_arc_pts):
                alpha = alphas[-i - 1]
                dr = dr_arr[-i - 1]
                r = r_arr[-i - 1]
                pts.append(
                    center + DPoint(np.cos(alpha), np.sin(alpha)) *
                    (r + dr)
                )
        return DSimplePolygon(pts)

    def _refresh_named_connections(self):
        self.start = self.connections[0]
        self.center = self.connections[1]
        self.end = self.connections[2]

    def _refresh_named_angles(self):
        self.alpha_start = self.angle_connections[0]
        self.alpha_end = self.angle_connections[1]


class Coil_type_1(ComplexBase):
    def __init__(self, Z0, start, L1, r, L2, trans_in=None):
        self.Z0 = Z0
        self.L1 = L1
        self.r = r
        self.L2 = L2
        super().__init__(start, trans_in)
        self.start = self.connections[0]
        self.end = self.connections[-1]
        self.dr = self.end - self.start
        self.alpha_start = self.angle_connections[0]
        self.alpha_end = self.angle_connections[1]

    def init_primitives(self):
        self.cop1 = CPW(self.Z0.width, self.Z0.gap, DPoint(0, 0), DPoint(self.L1, 0))
        self.arc1 = CPWArc(self.Z0, self.cop1.end, -self.r, -pi)
        self.cop2 = CPW(self.Z0.width, self.Z0.gap, self.arc1.end, self.arc1.end - DPoint(self.L2, 0))
        self.arc2 = CPWArc(self.Z0, self.cop2.end, -self.r, pi)

        self.connections = [self.cop1.start, self.arc2.end]
        self.angle_connections = [self.cop1.alpha_start, self.arc2.alpha_end]
        self.primitives = {"cop1": self.cop1, "arc1": self.arc1, "cop2": self.cop2, "arc2": self.arc2}


from collections import Counter


class CPWRLPath(ComplexBase):
    __version__ = "0.1"

    def __init__(self, origin, shape, cpw_parameters, turn_radiuses,
                 segment_lengths, turn_angles, trans_in=None,
                 bridged=False):
        """
        A piecewise-linear coplanar waveguide with rounded turns.

        Segment lengths are treated as the lengths of the segments of
        width line with turn_raduises = 0. Changing turning raduises
        will not alter the position of the end of the line.

        TODO: Rewrite based on to be rewritten DPathCPW lines construction
            algorithm
            construction.
         CPW2CPW straight and circular transitions has to be implemented
         for `CPWDPath` to be continuous coplanar (may be even with
         continuous parametric derivative)

        Parameters
        ----------
        origin : DPoint
            The point where the line should start
        shape : str
            String in format "RLLRL" where an R means width turn
            and an L means width straight part
        cpw_parameters : Union[CPWParameters, List[CPWParameters]]
            Parameters of the CPW or an array-like with parameters
            for each peace (R or L)
        turn_radiuses : Union[float, List[float]]
            Radius of the turns or an array-like with radiuses for
            each turn
        segment_lengths: list[float]
            Lengths of the straight parts of the equivalent
            piecewise-linear line with no corner rounding
        turn_angles: list[float]
            Angles for each turn of the line in radians
            !!! 180 turns are not yet supported !!!
        trans_in: DTrans
            Transformation of the line as width whole

        Returns
        -------

        """
        self._shape_string = shape
        self._N_elements = len(shape)
        self._shape_string_counter = Counter(shape)
        self._bridged = bridged

        self._N_turns = self._shape_string_counter['R']
        self._N_straights = self._shape_string_counter['L']
        if hasattr(cpw_parameters, "__len__"):
            if len(cpw_parameters) != self._N_elements:
                raise ValueError("CPW parameters dimension mismatch")
            else:
                self._cpw_parameters = copy.deepcopy(cpw_parameters)
        else:
            self._cpw_parameters: List[CPWParameters] = \
                [cpw_parameters] * self._N_elements

        if hasattr(turn_radiuses, "__len__"):
            if len(turn_radiuses) != self._N_turns:
                raise ValueError("Turn raduises dimension mismatch")
            else:
                self._turn_radiuses = copy.deepcopy(turn_radiuses)
        else:
            self._turn_radiuses = [turn_radiuses] * self._N_turns

        self._segment_lengths: List[float] = [0.0]
        if hasattr(segment_lengths, "__len__"):
            if len(segment_lengths) != self._N_straights:
                raise ValueError("Straight segments dimension mismatch")
            else:
                self._segment_lengths = copy.deepcopy(segment_lengths)
        else:
            self._segment_lengths = [segment_lengths] * self._N_straights

        if hasattr(turn_angles, "__len__"):
            if len(turn_angles) != self._N_turns:
                raise ValueError("Turn angles dimension mismatch")
            self._turn_angles = turn_angles
        else:
            self._turn_angles = [turn_angles] * self._N_turns
        super().__init__(origin, trans_in)
        self.start = self.connections[0]
        self.end = self.connections[1]
        self.alpha_start = self.angle_connections[0]
        self.alpha_end = self.angle_connections[1]

    def init_primitives(self):
        idx_r = 0
        idx_l = 0
        origin = DPoint(0, 0)

        prev_primitive_end = origin
        prev_primitive_end_angle = 0

        for i, symbol in enumerate(self._shape_string):
            if symbol == 'R':
                turn_radius = self._turn_radiuses[idx_r]
                turn_angle = self._turn_angles[idx_r]

                if abs(turn_radius) < self._cpw_parameters[i].b / 2:
                    raise Warning(
                        f"for round segment with index {idx_r}:\n"
                        "turn radius may be depicted incorrectly due to "
                        "the fact that curvature radius is lesser that "
                        "CPW metal width")

                if turn_angle < 0:
                    turn_radius *= -1

                arc_center = prev_primitive_end + DPoint(
                    -turn_radius * np.sin(prev_primitive_end_angle),
                    turn_radius * np.cos(prev_primitive_end_angle)
                )
                if self._cpw_parameters[i].smoothing:
                    if i > 0:
                        cpw1_params = self._cpw_parameters[i - 1]
                    else:
                        raise ValueError(
                            "No previous segment to smooth into"
                        )

                    if i < self._N_elements:
                        cpw2_params = self._cpw_parameters[i + 1]
                        self._cpw_parameters[i] = cpw2_params

                    cpw_arc = CPW2CPWArc(
                        origin=arc_center, r=turn_radius,
                        start_angle=-np.pi / 2,
                        end_angle=turn_angle - np.pi / 2,
                        cpw1_params=cpw1_params,
                        cpw2_params=cpw2_params,
                        trans_in=DCplxTrans(1, prev_primitive_end_angle
                                            * 180 / np.pi, False, 0, 0)
                    )
                else:
                        cpw_arc = CPW2CPWArc(
                            origin=arc_center, r=turn_radius,
                            start_angle=-np.pi / 2,
                            end_angle=turn_angle - np.pi / 2,
                            cpw1_params=self._cpw_parameters[i],
                            cpw2_params=self._cpw_parameters[i],
                            trans_in=DCplxTrans(1, prev_primitive_end_angle
                                                * 180 / np.pi, False, 0, 0)
                        )

                self.primitives["arc_" + str(idx_r)] = cpw_arc
                idx_r += 1
            elif symbol == 'L':
                # Turns are reducing segments' lengths so as if there were no roundings at all
                # next 'R' segment if exists
                if (i + 1 < self._N_elements
                        and self._shape_string[i + 1] == 'R'
                        and abs(self._turn_angles[idx_r]) < np.pi):
                    coeff = abs(np.tan(self._turn_angles[idx_r] / 2))
                    # print(self._segment_lengths)
                    # print(coeff)
                    # print(self._turn_radiuses)
                    self._segment_lengths[idx_l] -= \
                        self._turn_radiuses[idx_r] * coeff
                # previous 'R' segment if exists
                if (i - 1 > 0
                        and self._shape_string[i - 1] == 'R'
                        and abs(self._turn_angles[idx_r - 1]) < np.pi):
                    coeff = abs(np.tan(self._turn_angles[idx_r - 1] / 2))
                    self._segment_lengths[idx_l] -= \
                        self._turn_radiuses[idx_r - 1] * coeff

                if(self._segment_lengths[idx_l] < 0):
                    raise Warning(
                        f"{self.__class__.__name__} warning: segment №"
                        f"{idx_l} length is less than zero\n\n"
                        f"{self._segment_lengths[idx_l]}, {coeff}"
                        f"{self._turn_radiuses[idx_r - 1]}"
                    )

                cpw = CPW(self._cpw_parameters[i].width,
                          self._cpw_parameters[i].gap,
                          prev_primitive_end, prev_primitive_end + DPoint(
                        self._segment_lengths[idx_l], 0),
                          trans_in=DCplxTrans(1,
                                              prev_primitive_end_angle *
                                              180 / np.pi, False, 0, 0))

                self.primitives["cpw_" + str(idx_l)] = cpw
                idx_l += 1

            prev_primitive = list(self.primitives.values())[-1]
            prev_primitive_end = prev_primitive.end
            prev_primitive_end_angle = prev_primitive.alpha_end

        self.connections = [list(self.primitives.values())[0].start,
                            list(self.primitives.values())[-1].end]
        self.angle_connections = [
            list(self.primitives.values())[0].alpha_start,
            list(self.primitives.values())[-1].alpha_end]

    def _refresh_named_connections(self):
        self.start = self.connections[0]
        self.end = self.connections[1]

    def _refresh_named_angles(self):
        self.alpha_start = self.angle_connections[0]
        self.alpha_end = self.angle_connections[1]

    def get_total_length(self):
        return sum(self._segment_lengths) + \
               sum([abs(R * alpha) for R, alpha in
                    zip(self._turn_radiuses, self._turn_angles)])


class DPathCPW(ComplexBase):

    def __init__(self, points, cpw_parameters,
                 turn_radiuses, trans_in=None):
        """
        A piecewise-linear coplanar waveguide with rounded turns.

        Segment lengths are treated as the lengths of the segments of
        width line with turn_raduises = 0. Changing turning raduises
        will not alter the position of the end of the line.

        Parameters
        ----------
        points : List[DPoint]
            list of anchor points of width Path
        cpw_parameters : Union[CPWParameters, List[CPWParameters]]
            Parameters of the CPW or an array-like with parameters
            for each peace (R or L)
        turn_radiuses : Union[float, List[float]]
            Radius of the turns or an array-like with radiuses for
            each turn
        trans_in: DTrans
            Transformation of the line as width whole

        Returns
        -------
        ComplexBase
            `ComplexBase` object representing extended DPath in this
            library.
        Notes
        -------
        May have some bugs. Not fully tested and/or refactored since
        14.08.2021 (Shamil).
        """
        # TODO: remove shape strings
        if len(points) < 3:
            raise Warning("DPathCPW received < 3 points. Use `CPW` class "
                          "or increase anchor points number ")
        self.points: List[DPoint] = points
        # number of `CPW` segments + number of `CPW2CPWarc`s
        self._shape_string = []
        for p1, p2, p3 in zip(points, points[1:], points[2:]):
            a = (p2 - p1)
            b = (p3 - p2)
            max_len = max(a.length(), b.length())
            # collinearity testing
            if np.arccos(a.sprod(b)/a.length()/b.length())*max_len > 1:
                # if vectors discrepancy is less bigger than 1 nm
                self._shape_string += ["L", "R"]
            else:
                if(
                        len(self._shape_string) > 0
                        and self._shape_string[-1] == "L"
                ):
                    self._shape_string += ["L"]
                else:
                    self._shape_string += ["L", "L"]
        if self._shape_string[-1] == "R":
            self._shape_string += ["L"]

        self._shape_string = "".join(self._shape_string)
        _ctr = Counter(self._shape_string)
        self._N_straights = _ctr["L"]
        self._N_turns = _ctr["R"]
        self._N_elements = self._N_straights + self._N_turns

        self._cpw_parameters: List[CPWParameters] = []
        # multi-type parameters parsing
        if hasattr(cpw_parameters, "__len__"):
            if len(cpw_parameters) != self._N_elements:
                raise ValueError("CPW parameters dimension mismatch")
            else:
                self._cpw_parameters = copy.deepcopy(cpw_parameters)
        else:
            self._cpw_parameters = [cpw_parameters] * self._N_elements

        if hasattr(turn_radiuses, "__len__"):
            if len(turn_radiuses) != self._N_turns:
                raise ValueError("Turn raduises dimension mismatch")
            else:
                self._turn_radiuses = copy.deepcopy(turn_radiuses)
        else:
            self._turn_radiuses = [turn_radiuses] * self._N_turns

        segment_lengths = []
        for p1, p2 in zip(points, points[1:]):
            segment_lengths.append(p1.distance(p2))
        # print(segment_lengths)
        if hasattr(segment_lengths, "__len__"):
            if len(segment_lengths) != self._N_straights:
                raise ValueError("Straight segments dimension mismatch")
            else:
                self._segment_lengths = copy.deepcopy(segment_lengths)
        else:
            self._segment_lengths: List[float] = [segment_lengths] * \
                                       self._N_straights

        self._turn_angles = []
        for i, (p1, p2, p3) in enumerate(
                zip(points, points[1:], points[2:])
        ):
            a = p2 - p1
            b = p3 - p2
            self._turn_angles.append(
                a.vprod_sign(b) *
                np.abs(np.arccos(a.sprod(b)/a.length()/b.length()))
            )

        super().__init__(self.points[0], trans_in)
        self.start = self.connections[0]
        self.end = self.connections[1]
        self.alpha_start = self.angle_connections[0]
        self.alpha_end = self.angle_connections[1]

    def init_primitives(self):
        pass

        idx_r = 0
        idx_l = 0
        origin = DPoint(0, 0)

        prev_primitive_end = origin
        prev_primitive_end_angle = 0

        for i, symbol in enumerate(self._shape_string):
            if symbol == 'R':
                turn_radius = self._turn_radiuses[idx_r]
                turn_angle = self._turn_angles[idx_r]

                if abs(turn_radius) < self._cpw_parameters[i].b / 2:
                    raise Warning(
                        f"for round segment with index {idx_r}:\n"
                        "turn radius may be depicted incorrectly due to "
                        "the fact that curvature radius is lesser that "
                        "CPW metal width")

                if turn_angle < 0:
                    turn_radius *= -1

                arc_center = prev_primitive_end + DPoint(
                    -turn_radius * np.sin(prev_primitive_end_angle),
                    turn_radius * np.cos(prev_primitive_end_angle)
                )
                if self._cpw_parameters[i].smoothing:
                    if i > 0:
                        cpw1_params = self._cpw_parameters[i-1]
                    else:
                        raise ValueError(
                            "No previous segment to smooth into"
                        )

                    if i < self._N_elements:
                        cpw2_params = self._cpw_parameters[i+1]
                        self._cpw_parameters[i] = cpw2_params

                    cpw_arc = CPW2CPWArc(
                        origin=arc_center, r=turn_radius,
                        start_angle=-np.pi / 2,
                        end_angle=turn_angle - np.pi / 2,
                        cpw1_params=cpw1_params,
                        cpw2_params=cpw2_params,
                        trans_in=DCplxTrans(1, prev_primitive_end_angle
                                            * 180 / np.pi, False, 0, 0)
                    )
                else:
                    cpw_arc = CPW2CPWArc(
                        origin=arc_center, r=turn_radius,
                        start_angle=-np.pi / 2,
                        end_angle=turn_angle - np.pi / 2,
                        cpw1_params=self._cpw_parameters[i],
                        cpw2_params=self._cpw_parameters[i],
                        trans_in=DCplxTrans(1, prev_primitive_end_angle
                                            * 180 / np.pi, False, 0, 0)
                    )

                self.primitives["arc_" + str(idx_r)] = cpw_arc
                idx_r += 1
            elif symbol == 'L':
                # Turns are reducing segments' lengths so as if there were no roundings at all
                # next 'R' segment if exists
                if (i + 1 < self._N_elements
                        and self._shape_string[i + 1] == 'R'
                        and abs(self._turn_angles[idx_r]) < np.pi):
                    coeff = abs(np.tan(self._turn_angles[idx_r] / 2))
                    self._segment_lengths[idx_l] -= self._turn_radiuses[
                                                          idx_r] * coeff
                # previous 'R' segment if exists
                if (i - 1 > 0
                        and self._shape_string[i - 1] == 'R'
                        and abs(self._turn_angles[idx_r - 1]) < np.pi):
                    coeff = abs(np.tan(self._turn_angles[idx_r - 1] / 2))
                    self._segment_lengths[idx_l] -= self._turn_radiuses[
                                                          idx_r - 1] * coeff

                if (self._segment_lengths[idx_l] < 0):
                    raise Warning(
                        "CPWDPath warning: segment length "
                        "is less than zero"
                    )
                cpw = CPW(self._cpw_parameters[i].width,
                          self._cpw_parameters[i].gap,
                          prev_primitive_end, prev_primitive_end + DPoint(
                        self._segment_lengths[idx_l], 0),
                          trans_in=DCplxTrans(1,
                                              prev_primitive_end_angle *
                                              180 / np.pi,
                                              False, 0, 0))

                self.primitives["cpw_" + str(idx_l)] = cpw
                idx_l += 1

            prev_primitive = list(self.primitives.values())[-1]
            prev_primitive_end = prev_primitive.end
            prev_primitive_end_angle = prev_primitive.alpha_end

        first_vec = self.points[1] - self.points[0]
        firts_angle = np.arctan2(first_vec.y, first_vec.x)
        self.connections = [list(self.primitives.values())[0].start,
                            list(self.primitives.values())[-1].end]
        self.angle_connections = [
            list(self.primitives.values())[0].alpha_start,
            list(self.primitives.values())[-1].alpha_end]
        self.make_trans(
            DCplxTrans(1, firts_angle * 180 / np.pi, False, 0, 0)
        )

    def _refresh_named_connections(self):
        self.start = self.connections[0]
        self.end = self.connections[1]

    def _refresh_named_angles(self):
        self.alpha_start = self.angle_connections[0]
        self.alpha_end = self.angle_connections[1]

    def get_total_length(self):
        return sum(self._segment_lengths) + \
               sum([abs(R * alpha) for R, alpha in
                    zip(self._turn_radiuses, self._turn_angles)])


class Bridge1(ElementBase):
    """
        Class implements bridges that are used to suppress
        non-TEM modes in coplanar or other types of waveguides.
        based on this design:
        https://drive.google.com/file/d/1nHM9lJNT9sBIWH9isRc_zKL6hUPwhhnP/view?usp=sharing
    """
    bridge_width = 20e3
    surround_gap = 8e3
    gnd_touch_dx = 20e3
    gnd_touch_dy = 10e3
    transition_len = 12e3
    gnd2gnd_dy = 70e3

    def __init__(self, center, gnd_touch_dx=20e3, gnd2gnd_dy=70e3, trans_in=None):
        self.center = center
        self.gnd_touch_dx = gnd_touch_dx
        self.angle = 0
        self.gnd2gnd_dy=gnd2gnd_dy
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
        top_gnd_center = center + DPoint(0, self.gnd2gnd_dy / 2 + self.gnd_touch_dy / 2)
        p1 = top_gnd_center + DPoint(-self.gnd_touch_dx / 2, -self.gnd_touch_dy / 2)
        p2 = p1 + DVector(self.gnd_touch_dx, self.gnd_touch_dy)
        top_gnd_touch_box = pya.DBox(p1, p2)
        self.metal_regions["bridges_1"].insert(pya.Box().from_dbox(top_gnd_touch_box))

        bot_gnd_center = center + DPoint(0, -(self.gnd2gnd_dy / 2 + self.gnd_touch_dy / 2))
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
        p3 = bot_gnd_touch_box.p1 + DPoint(0, bot_gnd_touch_box.height()) + \
             DPoint(-(20e3-self.gnd_touch_dx)/2, self.transition_len)
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
        self.empty_regions["bridges_2"].insert(SimplePolygon.from_dpoly(empty_polygon))

    def _refresh_named_connections(self):
        self.center = self.connections[0]

    def _refresh_named_angles(self):
        self.angle = self.angle_connections[0]

    @staticmethod
    def bridgify_CPW(cpw, bridges_step, dest=None, bridge_layer1=-1,
                     bridge_layer2=-1, dest2=None,
                     avoid_points=[], avoid_distance=0):
        """
            Function puts bridge patterns to fabricate bridges on coplanar waveguide
        `cpw` with bridges having period of `bridges_step` along coplanar's wave
        propagation direction.
            Bridges are distributed over coplanar starting with its center.

        Parameters
        ----------
        cpw : Union[CPW, CPWArc, DPathCPW]
            instance of coplanar class to be bridged during fabrication
        bridges_step : float
            distance between centers of bridges in nm
        dest : pya.Cell
            cell to place bridge polygons at
        bridge_layer1 : int
            index of the layer in the `cell` with ground touching polygons
        bridge_layer2 : int
            index of the layer in the `cell` with empty polygons
        avoid_points : list[Union[DPoint,Point,Vector, DVector]]
            list points that you wish to keep bridges away
        avoid_distance : float
            distance in nm where there will be no bridges
            near the `avoid_points`
        Returns
        -------
        None
        """
        bridge_tmp = Bridge1(DPoint(0, 0))
        bridge_tmp.__bridgify_CPW(
            cpw, bridges_step,
            dest=dest, bridge_layer1=bridge_layer1,
            bridge_layer2=bridge_layer2, dest2=dest2,
            avoid_points=avoid_points, avoid_distance=avoid_distance
        )

    def __bridgify_CPW(self, cpw, bridges_step, dest=None,
                       bridge_layer1=-1, bridge_layer2=-1, dest2=None,
                       avoid_points=[], avoid_distance=0):
        """
            Function puts bridge patterns to fabricate bridges on coplanar waveguide
        `cpw` with bridges having period of `bridges_step` along coplanar's wave
        propagation direction.
            Bridges are distributed over coplanar starting with its center.

        Parameters
        ----------
        cpw : Union[CPW, CPWArc, DPathCPW]
            instance of coplanar class to be bridged during fabrication
        bridges_step : float
            distance between centers of bridges in nm
        dest : pya.Cell
            cell to place bridge polygons at
        bridge_layer1 : int
            index of the layer in the `cell` with ground touching polygons
        bridge_layer2 : int
            index of the layer in the `cell` with empty polygons
        avoid_points : list[Union[DPoint,Point,Vector, DVector]]
            list points that you wish to keep bridges away
        avoid_distance : float
            distance in nm where there will be no bridges
            near the `avoid_points`
        Returns
        -------
        None
        """
        if isinstance(cpw, CPW):
            # recursion base
            alpha = atan2(cpw.dr.y, cpw.dr.x)
            cpw_len = cpw.dr.abs()
            if cpw_len < (self.bridge_width + self.surround_gap):
                return

            cpw_dir_unit_vector = cpw.dr / cpw.dr.abs()

            # bridge with some initial dimensions
            tmp_bridge = Bridge1(DPoint(0, 0))
            bridge_width = tmp_bridge.gnd_touch_dx + 2 * tmp_bridge.surround_gap

            # number of additional bridges on either side of center
            additional_bridges_n = int((cpw_len / 2 - bridge_width / 2) // bridges_step)
            bridge_centers = []
            for i in range(-additional_bridges_n, additional_bridges_n + 1):
                bridge_center = cpw.start + (cpw_len / 2 + i * bridges_step) * cpw_dir_unit_vector

                avoid = False
                for avoid_point in avoid_points:
                    if (avoid_point - bridge_center).abs() < avoid_distance:
                        avoid = True
                        break

                if not avoid:
                    bridge_centers.append(bridge_center)

            bridges = []
            for center in bridge_centers:
                bridges.append(
                    Bridge1(
                        center,
                        trans_in=DCplxTrans(1, alpha / pi * 180, False, 0, 0)
                    )
                )
            for bridge in bridges:
                bridge.place(dest=dest, layer_i=bridge_layer1, region_name="bridges_1")
                if dest2 is not None:
                    bridge.place(dest=dest2, layer_i=bridge_layer2, region_name="bridges_2")
                else:
                    bridge.place(dest=dest, layer_i=bridge_layer2, region_name="bridges_2")
        elif isinstance(cpw, CPWArc):
            # recursion base
            # to be implemented
            pass
        elif isinstance(cpw, CPWRLPath) or isinstance(cpw, Coil_type_1)\
                or isinstance(cpw, DPathCPW):
            for name, primitive in cpw.primitives.items():
                if isinstance(primitive, CPW):
                    Bridge1.bridgify_CPW(
                        primitive, bridges_step,
                        dest, bridge_layer1, bridge_layer2, dest2=dest2,
                        avoid_points=avoid_points, avoid_distance=avoid_distance
                    )
        else:
            # do nothing for other shapes
            return
