import pya
from math import sqrt, cos, sin, atan2, pi, copysign
from pya import Point,DPoint,DSimplePolygon,SimplePolygon, DPolygon, Polygon,  Region
from pya import Trans, DTrans, CplxTrans, DCplxTrans, ICplxTrans

from importlib import reload

from classLib.baseClasses import *
from classLib.coplanars import *
from .baseClasses import ComplexBase
from .coplanars import CPWRLPath, CPW
from .coplanars import Coil_type_1, CPWArc # backward compatibility TODO: delete classes


class CPWResonator():

    _c = 299792458.0

    def __init__(self, origin, cpw_parameters, turn_radius, frequency, ɛ,
        wavelength_fraction = 1/4, coupling_length = 200e3, offset_length = 200e3,
        meander_periods = 4, neck_length = 100e3, end_primitive = None, trans_in = None):
        '''
        A CPW resonator of wavelength fraction with automatic length calculation
        from given frequency.

        It's also possible to attach width primitive to the end of the resonator which
        should provide width get_phase_shift(   ) method to calculate it's effective length
        as if it was just width straight CPW piece.

        Parameters:
            origin: DPoint
                The point where the resonator's coupling tail ends
                and the meander starts
            cpw_parameters: CPWParameters
                CPW parameters for the resonator
            turn_radius: float
                Raduis of the resonator's meander turns
            frequency: float, GHz
                The frequency that the resonator will have
            ɛ: float
                Substrate relative permittivity
            wavelength_fraction: float
                The fraction of the wavelength the resonator's
                fundamental mode has at resonance. This parameter
                is not checked for consistensy with the resonator's
                boundaries
            coupling_length: float
                The length of the segment parallel to the feedline
            offset_length: float
                The distance away from the feedline before meandering
            meander_periods: int
                The number of periods in the meandering part
            neck_length: float
                The length of the straight part before the uncoupled end
                of the resonator
            end_primitive: ElementBase object
                Element that will be attached to the end of the resonator
        '''
        
        self._origin = origin
        self._cpw_parameters = cpw_parameters
        self._turn_radius = turn_radius
        self._coupling_length = coupling_length
        self._offset_length = offset_length
        self._meander_periods = meander_periods
        self._neck_length = neck_length
        self._frequency = frequency
        self._wavelength_fraction = wavelength_fraction
        self._ɛ = ɛ
        self._end_primitive = end_primitive
        self._length = self._calculate_total_length()
        self._trans_in = trans_in

        self._init_primitives()


    def _init_primitives(self):
        N, R = self._meander_periods, self._turn_radius

        
        meander_length = (self._length - self._coupling_length \
            - self._offset_length - 2*2*pi*R/4 \
            -N*2*pi*R - 2*pi*R/2 + R - 2*pi*R/4 - self._neck_length)/(2*N+3/2)
#        print(meander_length)

        shape = "LRLRL"+N*"RLRL"+"RLRL"


        segment_lengths = [self._coupling_length + R, self._offset_length + 2*R]\
             + [meander_length+R] + 2*N*[meander_length] \
             + [meander_length/2, self._neck_length+R]

        turn_angles = [pi/2, pi/2] + N*[-pi,pi]+[-pi, pi/2]

#        print(sum([abs(turn_angle) for turn_angle in turn_angles])/pi)
#        print(sum(segment_lengths) +\
#             R*sum([abs(turn_angle) for turn_angle in turn_angles])-6*R)
#        print(self._length)
#        print(segment_lengths)

        self._line = CPWRLPath(DPoint(0, 0), shape,
                               self._cpw_parameters, self._turn_radius, segment_lengths, turn_angles,
                               DTrans(DPoint(-self._coupling_length+meander_length/2, 0)), bridged=False)
        # print("Connections[0] 1:", self._line.connections[0])
        # print("Primitive_start:", list(self._line.primitives.values())[0].connections[0])
        self._line.make_trans(self._trans_in)
        self._line.make_trans(DTrans(self._origin))
        # print("Primitive_start:", )
        # print("Connections[0] 2:", self._line.connections[0])

        self.start = list(self._line.primitives.values())[0].connections[0]
        self.end = list(self._line.primitives.values())[-1].connections[-1]
        self.alpha_start =\
                    list(self._line.primitives.values())[0].angle_connections[0]
        self.alpha_end =\
                    list(self._line.primitives.values())[-1].angle_connections[-1]

#        print(line.get_total_length())
#        print(line._segment_lengths)

    def place(self, cell, layer = None):


        self._line.place(cell, layer)


    def _calculate_total_length(self):

        length = self._c/self._frequency/(sqrt(self._ɛ/2+0.5))/1e9*self._wavelength_fraction

       # if self._end_primitive is not None:
        #     claw_phase_shift = self._claw.get_phase_shift(self._frequency)
        #     claw_effective_length = 1/180*claw_phase_shift/self._frequency/1e9*self._c/2/sqrt(self._ɛ/2+0.5)/2
        #    length -= claw_effective_length

        return length*1e9 # in nm


class CPWResonator2(ComplexBase):
    # neck doesn't stand out and instead spreads along the resonator
    _c = 299792458.0

    def __init__(self, origin, cpw_parameters, turn_radius, frequency, ɛ,
        wavelength_fraction = 1/4, coupling_length = 200e3,
        meander_periods = 4, neck_length = 100e3, no_neck=False, extra_neck_length=0, end_primitive=None, trans_in = None):
        """
        A CPW resonator of wavelength fraction with automatic length calculation
        from given frequency.

        Modification: with width neck along the resonator

        It is also possible to attach width primitive to the end of the resonator which
        should provide width get_phase_shift(   ) method to calculate its effective length
        as if it was just width straight CPW piece.

        Parameters:
            origin: DPoint
                The point where the resonator's coupling tail starts
            cpw_parameters: CPWParameters
                CPW parameters for the resonator
            turn_radius: float
                Raduis of the resonator's meander turns
            frequency: float, GHz
                The frequency that the resonator will have
            ɛ: float
                Substrate relative permittivity
            wavelength_fraction: float
                The fraction of the wavelength the resonator's
                fundamental mode has at resonance. This parameter
                is not checked for consistency with the resonator's
                boundaries
            coupling_length: float
                The length of the segment parallel to the feedline
            meander_periods: int
                The number of periods in the meandering part
            neck_length: float
                The length of the straight part before the uncoupled end
                of the resonator
            no_neck: bool
                If no_neck is true then the neck is absent
            extra_neck_length: float
                The length of the segment perpendicular to the neck
        """
        self.Z = cpw_parameters
        self.R = turn_radius
        self.cpl_L = coupling_length
        self.N = meander_periods
        self.neck = neck_length
        self.freq = frequency
        self.wavelength_fraction = wavelength_fraction
        self.eps = ɛ
        self.no_neck = no_neck
        self.extra_neck_length = extra_neck_length
        self.len = self._calculate_total_length()

        self.line = None # RL path that represents meandr
        self.open_end = end_primitive # open-ended coplanar at the and of the resonator

        super().__init__(origin, trans_in)

        self.start = self.line.connections[0]
        self.end = self.line.connections[1]
        self.alpha_start = self.line.angle_connections[0]
        self.alpha_end = self.line.angle_connections[1]

    def init_primitives(self):
        origin = DPoint(0, 0)
        total_neck_length =  pi * self.R / 2 + self.neck if not self.no_neck else 0
        meander_length = (self.len - self.cpl_L - 2 * (1 + self.N) * pi * self.R - 3 * self.R - total_neck_length - self.extra_neck_length) / (2 * self.N + 3 / 2)
        shape = "LRL" + self.N * "RLRL" + "RL" + ("RL" if not self.no_neck else "")
        segment_lengths = [self.cpl_L, meander_length] + 2 * self.N * [meander_length] + [meander_length + 3 * self.R + self.extra_neck_length] + ([self.neck] if not self.no_neck else [])
        turn_angles = [pi] + self.N * [-pi, pi]+[-pi] + ([-pi/2] if not self.no_neck else [])

        self.line = CPWRLPath(origin, shape, self.Z, self.R, segment_lengths, turn_angles)
        self.primitives['line'] = self.line

        if self.open_end == None:
            self.open_end = CPW(0, self.Z.b / 2, self.line.end, self.line.end + (DPoint(0, -self.Z.b) if not self.no_neck else DPoint(-self.Z.b, 0)))
        self.primitives['open_end'] = self.open_end

    def _calculate_total_length(self):
        length = self._c / self.freq / sqrt(self.eps/2 + 0.5) / 1e9 * self.wavelength_fraction
        return length * 1e9 # in nm

class EMResonator_TL2Qbit_worm(ComplexBase):
    def __init__( self, Z0, start, L_coupling, L1, r, L2, N, trans_in=None ):
        self.Z0 = Z0
        self.L_coupling = L_coupling
        self.L1 = L1
        self.r = r
        self.L2 = L2

        self.N = N
        self.primitives_gnd = {}
        super().__init__( start, trans_in )

        self.start = self.connections[0]
        self.end = self.connections[-1]
        self.dr = self.end - self.start
        self.alpha_start = self.angle_connections[0]
        self.alpha_end = self.angle_connections[1]

    def init_primitives( self ):
        name = "coil0"
        setattr(self, name, Coil_type_1(self.Z0, DPoint(0,0), self.L_coupling, self.r, self.L1) )
        self.primitives[name] = getattr( self, name )
        # coils filling
        for i in range(self.N):
            name = "coil" + str(i+1)
            setattr( self, name, Coil_type_1( self.Z0, DPoint( -self.L1 + self.L_coupling, -(i+1)*(4*self.r) ), self.L1, self.r, self.L1 ) )
            self.primitives[name] = getattr( self, name )

        # draw the "tail"
        self.arc_tail = CPWArc(self.Z0, self.primitives["coil" + str(self.N)].end, -self.L1 / 2, -pi / 2)
        self.cop_tail = CPW( self.Z0.width, self.Z0.gap, self.arc_tail.end, self.arc_tail.end - DPoint( 0,self.L2 ) )
        self.cop_open_end = CPW(0, self.Z0.b / 2, self.cop_tail.end, self.cop_tail.end - DPoint(0, self.Z0.b))
        self.primitives["arc_tail"] = self.arc_tail
        self.primitives["cop_tail"] = self.cop_tail
        self.primitives["cop_open_end"] = self.cop_open_end

        self.connections = [DPoint(0,0), self.cop_tail.end]
        self.angle_connections = [0,self.cop_tail.alpha_end]


class EMResonator_TL2Qbit_worm2(ComplexBase):
    def __init__( self, Z0, start, L_coupling, L1, r, L2, N, trans_in=None ):
        self.Z0 = Z0
        self.L_coupling = L_coupling
        self.L1 = L1
        self.r = r
        self.L2 = L2

        self.N = N
        self.primitives_gnd = {}
        super().__init__( start, trans_in )

        self.start = self.connections[0]
        self.end = self.connections[-1]
        self.dr = self.end - self.start
        self.alpha_start = self.angle_connections[0]
        self.alpha_end = self.angle_connections[1]

    def init_primitives( self ):
        self.arc1 = CPWArc(self.Z0, DPoint(0, 0), -self.r, pi / 2)
        self.primitives["arc1"] = self.arc1

        # making coil
        name = "coil0"
        setattr(self,name, Coil_type_1(self.Z0, DPoint(0,0), self.L_coupling, self.r, self.L1) )
        self.primitives[name] = getattr( self, name )
        # coils filling
        for i in range(self.N):
            name = "coil" + str(i+1)
            setattr( self, name, Coil_type_1( self.Z0, DPoint( -self.L1 + self.L_coupling, -(i+1)*(4*self.r) ), self.L1, self.r, self.L1 ) )
            self.primitives[name] = getattr( self, name )

        # draw the "tail"
        self.arc_tail = CPWArc(self.Z0, self.primitives["coil" + str(self.N)].end, -self.L1 / 2, -pi / 2)
        self.cop_tail = CPW( self.Z0.width, self.Z0.gap, self.arc_tail.end, self.arc_tail.end - DPoint( 0,self.L2 ) )
        self.cop_open_end = CPW(0, self.Z0.b / 2, self.cop_tail.end, self.cop_tail.end - DPoint(0, self.Z0.b))
        self.primitives["arc_tail"] = self.arc_tail
        self.primitives["cop_tail"] = self.cop_tail
        self.primitives["cop_open_end"] = self.cop_open_end

        self.connections = [DPoint(0,0), self.cop_tail.end]
        self.angle_connections = [0,self.cop_tail.alpha_end]


class EMResonator_TL2Qbit_worm2_XmonFork(EMResonator_TL2Qbit_worm2):
    def __init__(self, Z0, start, L_coupling, L1, r, L2, N,
                 fork_x_span, fork_y_span, fork_metal_width, fork_gnd_gap,
                 trans_in=None):
        self.fork_x_span = fork_x_span
        self.fork_y_span = fork_y_span
        self.fork_metal_width = fork_metal_width
        self.fork_gnd_gap = fork_gnd_gap

        super().__init__(Z0, start, L_coupling, L1, r, L2, N, trans_in)

        self.start = self.connections[0]
        self.end = self.connections[-1]
        self.dr = self.end - self.start
        self.alpha_start = self.angle_connections[0]
        self.alpha_end = self.angle_connections[1]

    def init_primitives(self):
        super().init_primitives()

        """ add fork to the end of the resonator """
        # remove open end from the resonator
        del self.primitives["cop_open_end"]
        del self.cop_open_end

        # adding fork horizontal part
        self.draw_fork_along_x()
        self.draw_fork_along_y()

    def draw_fork_along_x(self):
        forkZ = CPW(self.fork_metal_width, self.fork_gnd_gap)
        p1 = self.cop_tail.end + DPoint(-self.fork_x_span / 2, - forkZ.b / 2)
        p2 = p1 + DPoint(self.fork_x_span, 0)
        self.fork_x_cpw = CPW(forkZ.width, forkZ.gap, p1, p2)
        self.primitives["fork_x_cpw"] = self.fork_x_cpw

        # return waveguide that was erased during ground drawing
        p3 = self.cop_tail.end + DPoint(0, -forkZ.gap)
        erased_cpw = CPW(self.Z0.width, self.Z0.gap, self.cop_tail.end, p3)
        self.primitives["erased_cpw"] = erased_cpw

        # erase additional spaces at the ends of fork_x_cpw
        p1 = self.fork_x_cpw.start + DPoint(-forkZ.gap, 0)
        self.primitives["erased_fork_left"] = CPW(0, forkZ.b / 2, self.fork_x_cpw.start, p1)
        p1 = self.fork_x_cpw.end + DPoint(forkZ.gap, 0)
        self.primitives["erased_fork_right"] = CPW(0, forkZ.b / 2, self.fork_x_cpw.end, p1)

    def draw_fork_along_y(self):
        forkZ = CPW(self.fork_metal_width, self.fork_gnd_gap)

        # draw left part
        p1 = self.fork_x_cpw.start + DPoint(forkZ.width / 2, -forkZ.width / 2)
        p2 = p1 + DPoint(0, -self.fork_y_span)
        self.fork_y_cpw1 = CPW(forkZ.width, forkZ.gap, p1, p2)
        self.primitives["fork_y_cpw1"] = self.fork_y_cpw1

        # draw right part
        p1 = self.fork_x_cpw.end + DPoint(-forkZ.width / 2, -forkZ.width / 2)
        p2 = p1 + DPoint(0, -self.fork_y_span)
        self.fork_y_cpw2 = CPW(forkZ.width, forkZ.gap, p1, p2)
        self.primitives["fork_y_cpw2"] = self.fork_y_cpw2

        # erase gap at the ends of `y` fork parts
        p1 = self.fork_y_cpw1.end + DPoint(0, -forkZ.gap)
        self.primitives["erased_fork_left_cpw_end"] = CPW(0, forkZ.b / 2, self.fork_y_cpw1.end, p1)
        p1 = self.fork_y_cpw2.end + DPoint(0, -forkZ.gap)
        self.primitives["erased_fork_right_cpw_end"] = CPW(0, forkZ.b / 2, self.fork_y_cpw2.end, p1)


class EMResonator_TL2Qbit_worm3(ComplexBase):
    def __init__(self, Z0, start, L_coupling, L0, L1, r, L2, N, trans_in=None):
        self.Z0 = Z0
        self.L_coupling = L_coupling
        self.L0 = L0
        self.L1 = L1
        self.r = r
        self.L2 = L2
        self.N = N

        super().__init__(start, trans_in)

        self._geometry_parameters["cpw_width, um"] = Z0.width
        self._geometry_parameters["cpw_gap, um"] = Z0.gap
        self._geometry_parameters["L_coupling, um"] = L_coupling
        self._geometry_parameters["L0, um"] = L0
        self._geometry_parameters["L1, um"] = L1
        self._geometry_parameters["r, um"] = r
        self._geometry_parameters["L2, um"] = L2
        self._geometry_parameters["N_coils"] = N

        self.start = self.connections[0]
        self.end = self.connections[-1]
        self.dr = self.end - self.start
        self.alpha_start = self.angle_connections[0]
        self.alpha_end = self.angle_connections[1]

    def init_primitives(self):
        self.arc1 = CPWArc(self.Z0, DPoint(0, 0), -self.r, pi / 2)
        self.primitives["arc1"] = self.arc1

        p1 = self.arc1.end
        p2 = self.arc1.end + DPoint(0, -self.L0)
        self.L0_cpw = CPW(start=p1, end=p2, cpw_params=self.Z0)
        self.primitives["L0_cpw"] = self.L0_cpw

        # making coil
        name = "coil0"
        setattr(self, name, Coil_type_1(self.Z0, DPoint(0, 0), self.L_coupling, self.r, self.L1))
        self.primitives[name] = getattr(self, name)
        # coils filling
        for i in range(self.N):
            name = "coil" + str(i + 1)
            setattr(self, name,
                    Coil_type_1(self.Z0, DPoint(-self.L1 + self.L_coupling, -(i + 1) * (4 * self.r)), self.L1, self.r,
                                self.L1))
            self.primitives[name] = getattr(self, name)

        # draw the "tail"
        self.arc_tail = CPWArc(self.Z0, self.primitives["coil" + str(self.N)].end, -self.L1 / 2, -pi / 2)
        self.cop_tail = CPW(self.Z0.width, self.Z0.gap, self.arc_tail.end, self.arc_tail.end - DPoint(0, self.L2))

        # tail face is separated from ground by `b = width + 2*gap`
        self.cop_open_end = CPW(0, self.Z0.b / 2, self.cop_tail.end, self.cop_tail.end - DPoint(0, self.Z0.b))
        self.primitives["arc_tail"] = self.arc_tail
        self.primitives["cop_tail"] = self.cop_tail
        self.primitives["cop_open_end"] = self.cop_open_end

        self.connections = [DPoint(0, 0), self.cop_tail.end]
        self.angle_connections = [0, self.cop_tail.alpha_end]


class EMResonator_TL2Qbit_worm4(EMResonator_TL2Qbit_worm3):
    def __init__(self, Z0, start, L_coupling, L0, L1, r, L2, N,
                 stab_width, stab_gnd_gap, stab_len, stab_end_gnd_gap, trans_in=None):
        """

        Parameters
        ----------
        Z0
        start
        L_coupling
        L0
        L1
        r
        L2
        N
        Z_stab : CPWParams
        trans_in
        """
        self.stab_width = stab_width
        self.stab_gnd_gap = stab_gnd_gap
        self.stab_len = stab_len
        self.stab_end_gnd_gap = stab_end_gnd_gap

        # primitives introduced in this class
        self.Z_stab_cpw: CPW = None
        self.Z_stab_end_empty: CPW = None

        super().__init__(Z0, start, L_coupling, L0, L1, r, L2, N, trans_in)

        self._geometry_parameters.update(
            {**self.Z_stab_cpw.get_geometry_params_dict(prefix="resCoupleStab_"),
             "resCoupleStab_stab_end_gnd_gap, um": self.stab_end_gnd_gap/1e3}
        )

    def init_primitives(self):
        super().init_primitives()
        origin = DPoint(0,0)
        p1 = origin + DPoint(self.L_coupling/2, self.Z0.width/2)
        p2 = p1 + DPoint(0, self.stab_len)
        self.Z_stab_cpw = CPW(self.stab_width, self.stab_gnd_gap, p1, p2)
        self.primitives["Z_stab_cpw"] = self.Z_stab_cpw

        p1 = self.Z_stab_cpw.end
        p2 = p1 + DPoint(0, self.stab_end_gnd_gap)
        self.Z_stab_end_empty = CPW(0, self.Z_stab_cpw.b/2, p1, p2)
        self.primitives["Z_stab_end_empty"] = self.Z_stab_end_empty


class EMResonator_TL2Qbit_worm4_XmonFork(EMResonator_TL2Qbit_worm4):
    def __init__(self, Z0, start, L_coupling, L0, L1, r, L2, N,
                 stab_width, stab_gnd_gap, stab_len, stab_end_gnd_gap,
                 fork_x_span, fork_y_span, fork_metal_width, fork_gnd_gap,
                 trans_in=None):
        self.fork_x_span = fork_x_span
        self.fork_y_span = fork_y_span
        self.fork_metal_width = fork_metal_width
        self.fork_gnd_gap = fork_gnd_gap

        super().__init__(
            Z0, start, L_coupling, L0, L1, r, L2, N,
            stab_width, stab_gnd_gap, stab_len, stab_end_gnd_gap, trans_in
        )

        self._geometry_parameters["fork_x_span, um"] = fork_x_span / 1e3
        self._geometry_parameters["fork_y_span, um"] = fork_y_span / 1e3
        self._geometry_parameters["fork_metal_width, um"] = fork_metal_width / 1e3
        self._geometry_parameters["fork_gnd_gap, um"] = fork_gnd_gap / 1e3

    def init_primitives(self):
        super().init_primitives()

        """ add fork to the end of the resonator """
        # remove open end from the resonator
        del self.primitives["cop_open_end"]
        del self.cop_open_end

        # adding fork horizontal part
        self.draw_fork_along_x()
        self.draw_fork_along_y()

    def draw_fork_along_x(self):
        forkZ = CPW(self.fork_metal_width, self.fork_gnd_gap)
        p1 = self.cop_tail.end + DPoint(-self.fork_x_span / 2, - forkZ.b / 2)
        p2 = p1 + DPoint(self.fork_x_span, 0)
        self.fork_x_cpw = CPW(forkZ.width, forkZ.gap, p1, p2)
        self.primitives["fork_x_cpw"] = self.fork_x_cpw

        # return waveguide that was erased during ground drawing
        p3 = self.cop_tail.end + DPoint(0, -forkZ.gap)
        erased_cpw = CPW(self.Z0.width, self.Z0.gap, self.cop_tail.end, p3)
        self.primitives["erased_cpw"] = erased_cpw

        # erase additional spaces at the ends of fork_x_cpw
        p1 = self.fork_x_cpw.start + DPoint(-forkZ.gap, 0)
        self.primitives["erased_fork_left"] = CPW(0, forkZ.b / 2, self.fork_x_cpw.start, p1)
        p1 = self.fork_x_cpw.end + DPoint(forkZ.gap, 0)
        self.primitives["erased_fork_right"] = CPW(0, forkZ.b / 2, self.fork_x_cpw.end, p1)

    def draw_fork_along_y(self):
        forkZ = CPW(self.fork_metal_width, self.fork_gnd_gap)

        # draw left part
        p1 = self.fork_x_cpw.start + DPoint(forkZ.width / 2, -forkZ.width / 2)
        p2 = p1 + DPoint(0, -self.fork_y_span)
        self.fork_y_cpw1 = CPW(forkZ.width, forkZ.gap, p1, p2)
        self.primitives["fork_y_cpw1"] = self.fork_y_cpw1

        # draw right part
        p1 = self.fork_x_cpw.end + DPoint(-forkZ.width / 2, -forkZ.width / 2)
        p2 = p1 + DPoint(0, -self.fork_y_span)
        self.fork_y_cpw2 = CPW(forkZ.width, forkZ.gap, p1, p2)
        self.primitives["fork_y_cpw2"] = self.fork_y_cpw2

        # erase gap at the ends of `y` fork parts
        p1 = self.fork_y_cpw1.end + DPoint(0, -forkZ.gap)
        self.primitives["erased_fork_left_cpw_end"] = CPW(0, forkZ.b / 2, self.fork_y_cpw1.end, p1)
        p1 = self.fork_y_cpw2.end + DPoint(0, -forkZ.gap)
        self.primitives["erased_fork_right_cpw_end"] = CPW(0, forkZ.b / 2, self.fork_y_cpw2.end, p1)


class EMResonator_TL2Qbit_worm3_2(ComplexBase):
    """
    same as `EMResonator_TL2Qbit_worm3` but shorted and open ends are
    interchanged their places. In addition, width few primitives had been renamed.
    """
    def __init__(self, Z0, start, L_coupling, L0, L1, r, L2, N, L3=0, trans_in=None):
        self.Z0 = Z0
        self.L_coupling = L_coupling
        self.L0 = L0
        self.L1 = L1
        self.r = r
        self.L2 = L2
        self.L3 = 0
        self.N = N

        super().__init__(start, trans_in)

        self._geometry_parameters["cpw_width, um"] = Z0.width
        self._geometry_parameters["cpw_gap, um"] = Z0.gap
        self._geometry_parameters["L_coupling, um"] = L_coupling
        self._geometry_parameters["L0, um"] = L0
        self._geometry_parameters["L1, um"] = L1
        self._geometry_parameters["r, um"] = r
        self._geometry_parameters["L2, um"] = L2
        self._geometry_parameters["N_coils"] = N

        self.start = self.connections[0]
        self.end = self.connections[-1]
        self.dr = self.end - self.start
        self.alpha_start = self.angle_connections[0]
        self.alpha_end = self.angle_connections[1]

    def init_primitives(self):
        self.arc1 = CPWArc(self.Z0, DPoint(0, 0), -self.r, pi / 2)
        self.primitives["arc1"] = self.arc1

        p1 = self.arc1.end
        p2 = self.arc1.end + DPoint(0, -self.L0)
        self.cop_vertical = CPW(start=p1, end=p2, cpw_params=self.Z0)
        # open end tail face is separated from ground by `b = width + 2*gap`

        self.primitives["cop_vertical"] = self.cop_vertical

        # draw the open-circuited "tail"
        p1 = self.cop_vertical
        self.arc1_end_open = CPWArc(
            self.Z0, self.cop_vertical.end,
            (self.L_coupling + 2*self.r) / 4, pi / 2,
            trans_in=DTrans.R270
        )
        self.cpw_hor_end_open = CPW(
            self.Z0.width, self.Z0.gap,
            self.arc1_end_open.end, self.arc1_end_open.end + DPoint(self.L2, 0)
        )
        self.arc2_end_open = CPWArc(
            self.Z0, self.cpw_hor_end_open.end,
            -(self.L_coupling + 2 * self.r) / 4, -pi / 2
        )
        self.primitives["arc1_end_open"] = self.arc1_end_open
        self.primitives["cpw_hor_end_open"] = self.cpw_hor_end_open
        self.primitives["arc2_end_open"] = self.arc2_end_open

        if self.L3 != 0:
            self.cpw_vert_end_open = CPW(
                self.Z0.width, self.Z0.gap,
                self.arc2_end_open.end, self.arc2_end_open.end + DPoint(0, -self.L3)
            )
            self.primitives["cpw_hor_end_open"] = self.cpw_hor_end_open
            self.cpw_end_open_gap = CPW(
                0, self.Z0.b / 2,
                self.cpw_vert_end_open.end,
                   self.cpw_vert_end_open.end - DPoint(0, self.Z0.b)
            )
        else:
            self.cpw_end_open_gap = CPW(
                0, self.Z0.b / 2,
                self.arc2_end_open.end,
                   self.arc2_end_open.end - DPoint(0, self.Z0.b)
            )

        self.primitives["cpw_end_open_gap"] = self.cpw_end_open_gap

        # making coil
        name = "coil0"
        setattr(self, name, Coil_type_1(self.Z0, DPoint(0, 0), self.L_coupling, self.r, self.L1))
        self.primitives[name] = getattr(self, name)
        # coils filling
        for i in range(self.N):
            name = "coil" + str(i + 1)
            setattr(self, name,
                    Coil_type_1(self.Z0, DPoint(-self.L1 + self.L_coupling, -(i + 1) * (4 * self.r)), self.L1, self.r,
                                self.L1))
            self.primitives[name] = getattr(self, name)

        self.connections = [DPoint(0, 0), self.cpw_hor_end_open.end]
        self.angle_connections = [0, self.cpw_hor_end_open.alpha_end]


class EMResonator_TL2Qbit_worm3_2_XmonFork(EMResonator_TL2Qbit_worm3_2):
    def __init__(self, Z0, start, L_coupling, L0, L1, r, L2, N,
                 fork_x_span, fork_y_span, fork_metal_width, fork_gnd_gap,
                 L3=0, trans_in=None):
        self.fork_x_span = fork_x_span
        self.fork_y_span = fork_y_span
        self.fork_metal_width = fork_metal_width
        self.fork_gnd_gap = fork_gnd_gap

        super().__init__(Z0, start, L_coupling, L0, L1, r, L2, N, L3=L3, trans_in=trans_in)

        self._geometry_parameters["fork_x_span, um"] = fork_x_span / 1e3
        self._geometry_parameters["fork_y_span, um"] = fork_y_span / 1e3
        self._geometry_parameters["fork_metal_width, um"] = fork_metal_width / 1e3
        self._geometry_parameters["fork_gnd_gap, um"] = fork_gnd_gap / 1e3

    def init_primitives(self):
        super().init_primitives()

        """ add fork to the end of the resonator """
        # adding fork horizontal part
        self.draw_fork_along_x()
        self.draw_fork_along_y()

        # remove open end from the resonator
        del self.primitives["cpw_end_open_gap"]
        del self.cpw_end_open_gap

    def draw_fork_along_x(self):
        forkZ = CPW(self.fork_metal_width, self.fork_gnd_gap)
        p1 = self.cpw_end_open_gap.start + DPoint(-self.fork_x_span / 2, -forkZ.b / 2)
        p2 = p1 + DPoint(self.fork_x_span, 0)
        self.fork_x_cpw = CPW(forkZ.width, forkZ.gap, p1, p2)
        self.primitives["fork_x_cpw"] = self.fork_x_cpw

        # draw waveguide that was erased during `fork_x_cpw` ground erasing
        p1 = self.cpw_end_open_gap.start
        p2 = self.cpw_end_open_gap.start + DPoint(0, -forkZ.gap)
        erased_cpw = CPW(self.Z0.width, self.Z0.gap, p1, p2)
        self.primitives["erased_cpw"] = erased_cpw

        # erase additional spaces at the ends of fork_x_cpw
        p1 = self.fork_x_cpw.start
        p2 = self.fork_x_cpw.start + DPoint(-forkZ.gap, 0)
        self.primitives["erased_fork_left"] = CPW(0, forkZ.b / 2, p1, p2)

        p1 = self.fork_x_cpw.end
        p2 = self.fork_x_cpw.end + DPoint(forkZ.gap, 0)
        self.primitives["erased_fork_right"] = CPW(0, forkZ.b / 2, p1, p2)

    def draw_fork_along_y(self):
        forkZ = CPW(self.fork_metal_width, self.fork_gnd_gap)

        # draw left part
        p1 = self.fork_x_cpw.start + DPoint(forkZ.width / 2, -forkZ.width / 2)
        p2 = p1 + DPoint(0, -self.fork_y_span)
        self.fork_y_cpw1 = CPW(forkZ.width, forkZ.gap, p1, p2)
        self.primitives["fork_y_cpw1"] = self.fork_y_cpw1

        # draw right part
        p1 = self.fork_x_cpw.end + DPoint(-forkZ.width / 2, -forkZ.width / 2)
        p2 = p1 + DPoint(0, -self.fork_y_span)
        self.fork_y_cpw2 = CPW(forkZ.width, forkZ.gap, p1, p2)
        self.primitives["fork_y_cpw2"] = self.fork_y_cpw2

        # erase gap at the ends of `y` fork parts
        p1 = self.fork_y_cpw1.end
        p2 = self.fork_y_cpw1.end + DPoint(0, -forkZ.gap)
        self.primitives["erased_fork_left_cpw_end"] = CPW(0, forkZ.b / 2, p1, p2)
        p1 = self.fork_y_cpw2.end
        p2 = self.fork_y_cpw2.end + DPoint(0, -forkZ.gap)
        self.primitives["erased_fork_right_cpw_end"] = CPW(0, forkZ.b / 2, p1, p2)


class   EMResonatorTL3QbitWormRLTail(ComplexBase):
    """
    same as `EMResonator_TL3Qbit_worm3` but shorted and open ends are
    interchanged their places. In addition, width few primitives had been renamed.
    """

    def __init__(self, Z0, start, L_coupling, L0, L1, r, N,
                 tail_shape, tail_turn_radiuses,
                 tail_segment_lengths, tail_turn_angles,
                 tail_trans_in=None, trans_in=None):
        """

        Parameters
        ----------
        Z0
        start
        L_coupling
        L0
        L1
        r
        L3
        N
        L4
        tail_direction : str
            "left" - end of the resonator will be located to
            the left from long vertical line.
            "right" - end of the resonator will be located to
            the right from long vertical line.
            Try to change this parameter to get the difference.
        trans_in
        """
        self.Z0 = Z0
        self.L_coupling = L_coupling
        self.L0 = L0
        self.L1 = L1
        self.r = r
        self.N = N
        self.tail_shape = tail_shape
        self.tail_turn_radiuses = tail_turn_radiuses
        self.tail_segment_lengths = tail_segment_lengths
        self.tail_turn_angles = tail_turn_angles
        self.tail_trans_in = tail_trans_in

        super().__init__(start, trans_in)

        self._geometry_parameters["cpw_width, um"] = Z0.width / 1e3
        self._geometry_parameters["cpw_gap, um"] = Z0.gap / 1e3
        self._geometry_parameters["L_coupling, um"] = L_coupling / 1e3
        self._geometry_parameters["L0, um"] = L0 / 1e3
        self._geometry_parameters["L1, um"] = L1 / 1e3
        self._geometry_parameters["r, um"] = r / 1e3
        self._geometry_parameters["N_coils"] = N

        self.start = self.connections[0]
        self.end = self.connections[-1]
        self.dr = self.end - self.start
        self.alpha_start = self.angle_connections[0]
        self.alpha_end = self.angle_connections[1]

    def init_primitives(self):
        self.arc1 = CPWArc(self.Z0, DPoint(0, 0), -self.r, pi / 2)
        self.primitives["arc1"] = self.arc1

        p1 = self.arc1.end
        p2 = self.arc1.end + DPoint(0, -self.L0)
        self.cop_vertical = CPW(start=p1, end=p2, cpw_params=self.Z0)
        # open end tail face is separated from ground by `b = width + 2*gap`

        self.primitives["cop_vertical"] = self.cop_vertical

        # draw the open-circuited "tail"

        self.cpw_end_open_RLPath = CPWRLPath(
            origin=self.cop_vertical.end,
            shape=self.tail_shape,
            cpw_parameters=self.Z0,
            turn_radiuses=self.tail_turn_radiuses,
            segment_lengths=self.tail_segment_lengths,
            turn_angles=self.tail_turn_angles,
            trans_in=self.tail_trans_in
        )
        self.primitives["cpw_end_open_RLPath"] = self.cpw_end_open_RLPath

        self.cpw_end_open_gap = CPW(
            0, self.Z0.b / 2,
            self.cpw_end_open_RLPath.end,
            self.cpw_end_open_RLPath.end - DPoint(0, self.Z0.b)
        )
        self.primitives["cpw_end_open_gap"] = self.cpw_end_open_gap

        # making coil
        name = "coil0"
        setattr(self, name, Coil_type_1(self.Z0, DPoint(0, 0), self.L_coupling, self.r, self.L1))
        self.primitives[name] = getattr(self, name)
        # coils filling
        for i in range(self.N):
            name = "coil" + str(i + 1)
            setattr(
                self, name, Coil_type_1(
                        self.Z0, DPoint(-self.L1 + self.L_coupling, -(i + 1) * (4 * self.r)),
                        self.L1, self.r, self.L1
                    )
            )
            self.primitives[name] = getattr(self, name)

        self.connections = [DPoint(0, 0), self.cpw_end_open_RLPath.end]
        self.angle_connections = [0, self.cpw_end_open_RLPath.alpha_end]


class EMResonatorTL3QbitWormRLTailXmonFork(EMResonatorTL3QbitWormRLTail):
    def __init__(
            self, Z0, start, L_coupling, L0, L1, r, N,
            tail_shape, tail_turn_radiuses,
            tail_segment_lengths, tail_turn_angles,
            fork_x_span, fork_y_span, fork_metal_width, fork_gnd_gap,
            tail_trans_in=None, trans_in=None
    ):
        self.fork_x_span = fork_x_span
        self.fork_y_span = fork_y_span
        self.fork_metal_width = fork_metal_width
        self.fork_gnd_gap = fork_gnd_gap

        super().__init__(
            Z0, start, L_coupling, L0, L1, r, N,
            tail_shape, tail_turn_radiuses,
            tail_segment_lengths, tail_turn_angles,
            tail_trans_in=tail_trans_in,
            trans_in=trans_in)

        self._geometry_parameters["fork_x_span, um"] = fork_x_span / 1e3
        self._geometry_parameters["fork_y_span, um"] = fork_y_span / 1e3
        self._geometry_parameters["fork_metal_width, um"] = fork_metal_width / 1e3
        self._geometry_parameters["fork_gnd_gap, um"] = fork_gnd_gap / 1e3

    def init_primitives(self):
        super().init_primitives()

        """ add fork to the end of the base class resonator """
        # adding fork horizontal part
        self.draw_fork_along_x()
        self.draw_fork_along_y()

        # remove open end from the base class resonator
        del self.primitives["cpw_end_open_gap"]
        del self.cpw_end_open_gap

    def draw_fork_along_x(self):
        forkZ = CPW(self.fork_metal_width, self.fork_gnd_gap)
        p1 = self.cpw_end_open_gap.start + DPoint(-self.fork_x_span / 2, -forkZ.b / 2)
        p2 = p1 + DPoint(self.fork_x_span, 0)
        self.fork_x_cpw = CPW(forkZ.width, forkZ.gap, p1, p2)
        self.primitives["fork_x_cpw"] = self.fork_x_cpw

        # draw waveguide that was erased during `fork_x_cpw` ground erasing
        p1 = self.cpw_end_open_gap.start
        p2 = self.cpw_end_open_gap.start + DPoint(0, -forkZ.gap)
        erased_cpw = CPW(self.Z0.width, self.Z0.gap, p1, p2)
        self.primitives["erased_fork_cpw"] = erased_cpw

        # erase additional spaces at the ends of fork_x_cpw
        p1 = self.fork_x_cpw.start
        p2 = self.fork_x_cpw.start + DPoint(-forkZ.gap, 0)
        self.primitives["erased_fork_left"] = CPW(0, forkZ.b / 2, p1, p2)

        p1 = self.fork_x_cpw.end
        p2 = self.fork_x_cpw.end + DPoint(forkZ.gap, 0)
        self.primitives["erased_fork_right"] = CPW(0, forkZ.b / 2, p1, p2)

    def draw_fork_along_y(self):
        forkZ = CPW(self.fork_metal_width, self.fork_gnd_gap)

        # draw left part
        p1 = self.fork_x_cpw.start + DPoint(forkZ.width / 2, -forkZ.width / 2)
        p2 = p1 + DPoint(0, -self.fork_y_span)
        self.fork_y_cpw1 = CPW(forkZ.width, forkZ.gap, p1, p2)
        self.primitives["fork_y_cpw1"] = self.fork_y_cpw1

        # draw right part
        p1 = self.fork_x_cpw.end + DPoint(-forkZ.width / 2, -forkZ.width / 2)
        p2 = p1 + DPoint(0, -self.fork_y_span)
        self.fork_y_cpw2 = CPW(forkZ.width, forkZ.gap, p1, p2)
        self.primitives["fork_y_cpw2"] = self.fork_y_cpw2

        # erase gap at the ends of `y` fork parts
        p1 = self.fork_y_cpw1.end
        p2 = self.fork_y_cpw1.end + DPoint(0, -forkZ.gap)
        self.primitives["erased_fork_left_cpw_end"] = CPW(0, forkZ.b / 2, p1, p2)
        p1 = self.fork_y_cpw2.end
        p2 = self.fork_y_cpw2.end + DPoint(0, -forkZ.gap)
        self.primitives["erased_fork_right_cpw_end"] = CPW(0, forkZ.b / 2, p1, p2)

    def get_approx_frequency(self, refractive_index, n=0):
        """

        Parameters
        ----------
        refractive_index : float
            absolute refractive index of CPW waveguide.
        n : int
            number of frequency mode to return

        Returns
        -------
        freq : float
            Estimated frequency
        """
        length = self.length(exception="fork")  # nm
        light_speed = 299792458/refractive_index  # m/s
        # print("resonator length: ", length)
        freq = light_speed/(4*length)  # GHz
        res_freq = freq*(2*n + 1)

        return res_freq

    def _refresh_named_connections(self):
        self.start = self.connections[0]
        self.end = self.connections[1]






