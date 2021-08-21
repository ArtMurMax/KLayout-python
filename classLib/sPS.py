import pya
from math import sqrt, cos, sin, atan2, pi, copysign
from pya import Point, DPoint, DSimplePolygon, SimplePolygon, DPolygon, Polygon, Region
from pya import Trans, DTrans, CplxTrans, DCplxTrans, ICplxTrans
from collections import namedtuple

from classLib.baseClasses import ComplexBase
from classLib.capacitors import CWave, CWave2CPW
from classLib.coplanars import CPW
from classLib.josJ import Squid, AsymSquid


SPSparams = namedtuple("SPSparams", "r_out dr n_semiwaves s alpha r_curve n_pts_cwave Z1 d_alpha1 width1 gap1 Z2 d_alpha2 width2 gap2 n_pts_arcs")

class SFS_Csh_emb(ComplexBase):
    """@brief: class represents width qubit for width single photon source
        @params:  DPoint origin - position of the center of width qubit
                        params - width dict or width list of parameters (len = 16),
                                 width list is not recommended as it's difficult to use
                        squid_params - width list of parameters for width dc-SQUID (can included in params
                                as width list but not recommended)
                        int squid_pos - position of width dc-squid in relation to width capacitor (-1 - left, 0 - center, 1 - right), is 0 by default
                        Trans trans_in - initial transformation (None by default)
    """

    def __init__(self, origin, params, squid_params=None, squid_pos=0, trans_in=None):
        if squid_params is not None and isinstance(params, dict):
            self.params = params
            self.r_out = params['r_out']
            self.dr = params['dr']
            self.r_in = self.r_out - self.dr
            self.n_semiwaves = params['n_semiwaves']
            self.s = params['s']
            self.alpha = params['alpha']
            self.r_curve = params['r_curve']
            self.n_pts_cwave = params['n_pts_cwave']
            self.Z1 = params['Z1']
            self.d_alpha1 = params['d_alpha1']
            self.width1 = params['width1']
            self.gap1 = params['gap1']
            self.Z2 = params['Z2']
            self.d_alpha2 = params['d_alpha2']
            self.width2 = params['width2']
            self.gap2 = params['gap2']
            self.n_pts_arcs = params['n_pts_arcs']
            self.squid_params = squid_params
        else:
            # not recommended, because dict is more convinient
            self.params = params
            self.r_out = params[0]
            self.dr = params[1]
            self.r_in = self.r_out - self.dr
            self.n_semiwaves = params[2]
            self.s = params[3]
            self.alpha = params[4]
            self.r_curve = params[5]
            self.n_pts_cwave = params[6]
            self.Z1 = params[7]
            self.d_alpha1 = params[8]
            self.width1 = params[9]
            self.gap1 = params[10]
            self.Z2 = params[11]
            self.d_alpha2 = params[12]
            self.width2 = params[13]
            self.gap2 = params[14]
            self.n_pts_arcs = params[15]
            self.squid_params = self.params[16:]
        self.squid_pos = squid_pos
        super().__init__(origin, trans_in)
        '''
        self.excitation_port = self.connections[0]
        self.output_port = self.connections[1]
        self.excitation_angle = self.angle_connections[0]
        self.output_angle = self.angle_connections[1]
        '''

    def init_primitives(self):

        origin = DPoint(0, 0)

        self.c_wave = CWave(origin, self.r_out, self.dr, self.n_semiwaves, self.s, self.alpha, self.r_curve,
                            n_pts=self.n_pts_cwave)
        self.primitives["c_wave"] = self.c_wave

        Z1_start = origin + DPoint(0, self.r_in + self.gap1 + self.width1 / 2)
        Z1_end = Z1_start + DPoint(0, -self.gap1 - self.width1 / 2 + self.dr)
        self.cpw1 = CPW(self.Z1.width, self.Z1.gap, Z1_start, Z1_end)
        self.primitives["cpw1"] = self.cpw1

        Z2_start = origin - DPoint(0, self.r_in + self.gap2 + self.width2 / 2)
        Z2_end = Z2_start - DPoint(0, -self.gap2 - self.width2 / 2 + self.dr)
        self.cpw2 = CPW(self.Z2.width, self.Z2.gap, Z2_start, Z2_end)
        self.primitives["cpw2"] = self.cpw2

        if isinstance(self.params, dict):
            self.c_wave_2_cpw_adapter = CWave2CPW(self.c_wave, self.params, n_pts=self.n_pts_arcs)
        else:
            # not recommended
            self.c_wave_2_cpw_adapter = CWave2CPW(self.c_wave, self.params[7:15], n_pts=self.n_pts_arcs)
        self.primitives["c_wave_2_cpw_adapter"] = self.c_wave_2_cpw_adapter

        p_squid = None
        squid_trans_in = None

        if not self.squid_pos:
            if (self.c_wave.n_segments % 2 == 1):
                squid_trans_in = DCplxTrans(1, -self.c_wave.alpha * 180 / pi, False, 0, 0)
                p_squid = origin
            else:
                squid_trans_in = None
                second_parity = self.c_wave.n_segments / 2
                y_shift = self.c_wave.L0 * sin(self.c_wave.alpha) - self.c_wave.r_curve * (
                            1 / cos(self.c_wave.alpha) - 1)
                if (second_parity % 2 == 0):
                    p_squid = origin + DPoint(0, y_shift)
                else:
                    p_squid = origin + DPoint(0, -y_shift)
        else:
            squid_trans_in = None
            p_squid = origin - DPoint(self.squid_pos * (self.r_out - 1.8 * self.dr), 0)
        
        self.squid = AsymSquid( p_squid, self.squid_params, trans_in=squid_trans_in )
        self.primitives["qubit"] = self.squid

        self.connections = [Z1_end, Z2_end]
        self.angle_connections = [pi / 2, 3 / 2 * pi]

    def place(self, dest, layer_ph=-1, layer_el=-1):
        if layer_el != -1:
            for prim_name in list(self.primitives.keys())[:-1]:
                self.primitives[prim_name].place(dest, layer_ph)
            self.squid.place(dest, layer_el)
        else:  # dest is region_ph and layer_ph is actually region_el
            reg_ph = dest
            reg_el = layer_ph  # this is redefinition of the input parameter
            for prim_name in list(self.primitives.keys())[:-1]:
                self.primitives[prim_name].place(reg_ph)
            self.squid.place(reg_el)
