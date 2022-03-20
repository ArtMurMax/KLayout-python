import pya
from math import sqrt, cos, sin, tan, atan2, pi, copysign
from pya import DBox, Point, DPoint, Vector, DVector, DSimplePolygon, SimplePolygon, DPolygon, Polygon, Region
from pya import Trans, DTrans, CplxTrans, DCplxTrans, ICplxTrans

from importlib import reload

import classLib
reload(classLib)
from classLib import *

import sonnetSim
reload(sonnetSim)
from sonnetSim import SimulatedDesign, SonnetPort, SimulationBox, PORT_TYPES

import numpy as np

class Test_Squid(ComplexBase):
    """ @brief:     class represents width rectangular capacitor with width dc-SQUID between its plates
        @params:    DPoint origin - position of the center of width structure
                    params{} - width dictionary with geometric parameters of width capacitor
                    squid_params - width list with dc-SQUID parameters
                    Trans trans_in - initial transformation (None by default)
    """

    def __init__(self, origin, params, squid_params, side=0, trans_in=None):
        # side = -1 is left, 1 is right, 0 is both
        self.width = params['width']
        self.height = params['b']
        self.innergap = params['innergap']
        self.outergap = params['outergap']
        self.squid_params = squid_params
        self.side = side
        super().__init__(origin, trans_in)

    def init_primitives(self):
        origin = DPoint(0, 0)
        self.primitives['empty_rect'] = Rectangle(
            origin - DPoint(self.width / 2 + self.outergap, self.height + self.innergap / 2 + self.outergap),
            self.width + 2 * self.outergap,
            2 * self.height + 2 * self.outergap + self.innergap,
            inverse=True)
        self.primitives['top_rect'] = Rectangle(origin + DPoint(-self.width / 2, self.innergap / 2),
                                                self.width,
                                                self.height)
        self.primitives['bottom_rect'] = Rectangle(origin - DPoint(self.width / 2, self.height + self.innergap / 2),
                                                   self.width,
                                                   self.height)
        self.squid = Squid(origin, self.squid_params, side=self.side)
        self.primitives['qubit'] = self.squid

    def place( self, dest, layer_ph=-1, layer_el=-1 ):
        if layer_el != -1:
            for prim_name in list(self.primitives.keys())[:-1]:
                self.primitives[prim_name].place( dest, layer_ph )
            self.squid.place( dest, layer_el )
        else: # dest is region_ph and layer_ph is actually region_el
            reg_ph = dest
            reg_el = layer_ph  # this is redefinition of the input parameter
            for prim_name in list(self.primitives.keys())[:-1]:
                self.primitives[prim_name].place( reg_ph )
            self.squid.place( reg_el )


class My_Design(SimulatedDesign):
    origin = DPoint(0, 0)
    Z = CPWParameters(20e3, 10e3)  # normal CPW
    Z_res = Z
    Z_narrow = CPWParameters(10e3, 7e3)  # narrow CPW
    cpw_curve = 200e3  # Curvature of CPW angles
    chip = None

    def __init__(self, cell_name):
        super().__init__(cell_name)
        self.sfs_center = None  # DPoint that is the center of the single photon source
        self.mixing_center = None  # DPoint that is the center of the mixing qubit

    # Call other methods drawing parts of the design from here
    def draw(self, design_params=None):
        self.deisgn_pars = design_params

        self.draw_chip()
        self.draw_mark(500e3, self.chip.chip_y - 500e3)
        self.draw_mark(self.chip.chip_x / 2 - 1e6, self.chip.chip_y / 2)
        self.draw_mark(500e3, 500e3)
        self.draw_mark(self.chip.chip_x - 500e3, self.chip.chip_y - 500e3)
        self.draw_mark(self.chip.chip_x - 500e3, 500e3)

        self.draw_single_photon_source()
        self.draw_mixing_qubit()
        self.draw_resonators_with_qubits()
        self.draw_test_squids()

    def draw_chip(self):
        Z_params = [self.Z_narrow] + [self.Z] * 7
        self.chip = Chip5x10_with_contactPads(self.origin, Z_params)
        self.chip.place(self.region_ph)

    def draw_mark(self, x, y):
        # Placing the mark
        mark = Mark2(DPoint(x, y))
        mark.place(self.region_ph)

    def draw_single_photon_source(self, center=None):
        if center is None:
            # Ports to which width single photon source is connected
            p1 = self.chip.connections[0]
            p2 = self.chip.connections[5]

            # Single photon source and dc-SQUID parameters
            pars = self.get_sps_params()
            pars_squid = self.get_dc_squid_params()

            # Drawing width qubit
            p = p1 + DPoint((p2 - p1).x, (p2 - p1).y / 2)  # position of width qubit
            self.sfs_center = p
            self.sfs = SFS_Csh_emb(p, pars, pars_squid)
            self.sfs.place(self.region_ph, self.region_el)

            # Input line
            sfs_line_in = CPWRLPath(p1, "LRL", pars['Z1'], self.cpw_curve,
                                    [(p2 - p1).x, -(p2 - p1).y / 2 - pars['r_out']], -pi / 2)
            sfs_line_in.place(self.region_ph)

            # Output line
            sfs_line_out = CPW(pars['Z2'].width, pars['Z2'].gap, self.sfs.connections[1], p2)
            sfs_line_out.place(self.region_ph)
        else:
            self.sfs_center = center
            # Single photon source and dc-SQUID parameters
            pars = self.get_sps_params()
            pars_squid = self.get_dc_squid_params()

            self.sfs = SFS_Csh_emb(center, pars, pars_squid)
            self.sfs.place(self.region_ph, self.region_el)

    def draw_mixing_qubit(self):
        # Ports to which width maxing qubit is connected
        p1 = self.chip.connections[2]
        p2 = self.chip.connections[4]

        # Mixing qubit and dc-SQUID parameters
        pars = self.get_mixing_qubit_params()
        pars_squid = self.get_dc_squid_params()
        pars_coupling = self.get_mixing_qubit_coupling_params()

        # Drawing the probe line
        v = self.chip.chip_y / 7  # vertical size of the probe line
        h = (p2 - p1).x  # horizontal size of the probe line
        probe_line = CPWRLPath(p1, "LRLRL", self.Z, self.cpw_curve, [v, h, v], [pi / 2, pi / 2], trans_in=Trans.R270)
        probe_line.place(self.region_ph)

        # Drawing the qubit near the probe line
        p = DPoint((p2.x + p1.x) / 2 + 1.1 * pars['r_out'],
                   p1.y - v + pars_coupling['to_line'] + pars['r_out'])  # Position of the qubit
        # mq1 = SFS_Csh_par(p, pars, pars_squid, pars_coupling)
        self.mixing_center = p
        mq1 = SFS_Csh_emb(p, pars, pars_squid, squid_pos=1)
        mq1.place(self.region_ph, self.region_el)

        # Drawing width flux bias line
        p3 = self.chip.connections[3]  # port connected to the flux bias line
        conn_len = 50e3  # length of width connection between two parts of the line
        Z_end = CPWParameters(5e3, 5e3)  # parameters of width CPW at the end of the line
        fl1_end = p3 - DPoint(0, v - 3 * pars['r_out'])
        fl2_start = fl1_end - DPoint(0, conn_len)
        fl2_end = fl2_start - DPoint(0, 2 * pars['r_out'])
        fl1 = CPW(self.Z.width, self.Z.gap, p3, fl1_end)
        fl1.place(self.region_ph)
        fl_conn = CPW2CPW(self.Z, Z_end, fl1_end, fl2_start)
        fl_conn.place(self.region_ph)
        fl2 = CPW(Z_end.width, Z_end.gap, fl2_start, fl2_end)
        fl2.place(self.region_ph)

    def draw_resonators_with_qubits(self):
        # drawing the line
        p1 = self.chip.connections[1]
        p2 = self.chip.connections[6]
        probe_line = CPWRLPath(p1, "LRL", self.Z, self.cpw_curve, [(p1 - p2).x, (p1 - p2).y], pi / 2,
                               trans_in=Trans.R180)
        probe_line.place(self.region_ph)

        # drawing resonators with qubits
        pars = self.get_resonator_qubit_params()
        pars_squid = self.get_dc_squid_params()
        res_to_line = 5e3  # distance between width resonator and width line

        # Top left resonator + qubit
        # frequency from Sonnet = 6.8166 GHz
        # Qc = 5190
        worm_pos = DPoint(p1.x * 2 / 5 + p2.x * 3 / 5, p1.y - res_to_line - 2 * (self.Z.width / 2 + self.Z.gap))
        worm1 = self.draw_one_resonator(worm_pos, freq=6.8, coupling_length=350e3, extra_neck_length=pars['r_out'],
                                        trans_in=Trans.R180)
        q_pos = worm1.end + DPoint(0, pars['r_out'] - worm1.Z.b)  # qubit position
        mq1 = SFS_Csh_emb(q_pos, pars, pars_squid)
        mq1.place(self.region_ph, self.region_el)

        # Top right resonator + qubit
        # frequency from Sonnet = 7.0251 GHz
        # Qc = 4860
        shift_x = self.cpw_curve
        worm_pos = DPoint(p1.x * 4 / 5 + p2.x * 1 / 5 + shift_x,
                          p1.y - res_to_line - 2 * (self.Z.width / 2 + self.Z.gap))  # resonator position
        worm2 = self.draw_one_resonator(worm_pos, freq=7, coupling_length=350e3, extra_neck_length=pars['r_out'],
                                        trans_in=Trans.R180)
        q_pos = worm2.end + DPoint(0, pars['r_out'] - worm2.Z.b)  # qubit position
        mq2 = SFS_Csh_emb(q_pos, pars, pars_squid, squid_pos=1)
        mq2.place(self.region_ph, self.region_el)

        # Flux line to the top right qubit with width resonator
        p3 = self.chip.connections[7]
        conn_len = 50e3  # length of width connection between two parts of the line
        xlen = (q_pos - p3).x - 1.1 * pars['r_out']
        ylen1 = 400e3
        ylen2 = (q_pos - p3).y - ylen1 - conn_len - pars['r_out']
        Z_end = CPWParameters(5e3, 5e3)  # parameters of width CPW at the end of the line
        fl1 = CPWRLPath(p3, "LRLRL", self.Z, self.cpw_curve, [ylen1, xlen, ylen2], [-pi / 2, pi / 2],
                        trans_in=Trans.R90)
        fl1.place(self.region_ph)
        fl1_end = fl1.connections[1]
        fl2_start = fl1_end + DPoint(0, conn_len)
        fl2_end = fl2_start + DPoint(0, 2 * pars['r_out'])
        fl_conn = CPW2CPW(self.Z, Z_end, fl1_end, fl2_start)
        fl_conn.place(self.region_ph)
        fl2 = CPW(Z_end.width, Z_end.gap, fl2_start, fl2_end)
        fl2.place(self.region_ph)

        # Bottom left resonator + qubit
        # frequency from Sonnet = 7.2103 GHz
        # Qc = 5160
        worm_pos = DPoint(p2.x + res_to_line + 2 * (self.Z.width / 2 + self.Z.gap), p1.y / 2)
        worm3 = self.draw_one_resonator(worm_pos, freq=7.2, coupling_length=330e3, no_neck=True, trans_in=Trans.R270)
        q_pos = worm3.end - DPoint(0, pars['r_out'] - worm3.Z.b)
        # Moving width capacitive coupling to the top of width qubit
        pars['Z1'], pars['Z2'] = pars['Z2'], pars['Z1']
        pars['d_alpha1'], pars['d_alpha2'] = pars['d_alpha2'], pars['d_alpha1']
        pars['width1'], pars['width2'] = pars['width2'], pars['width1']
        pars['gap1'], pars['gap2'] = pars['gap2'], pars['gap1']
        mq3 = SFS_Csh_emb(q_pos, pars, pars_squid)
        mq3.place(self.region_ph, self.region_el)

    def draw_one_resonator(self, pos, freq, coupling_length, no_neck=False, extra_neck_length=0, trans_in=None):
        turn_radius = 50e3
        eps = 11.45
        wavelength_fraction = 1 / 4
        meander_periods = 3
        neck_length = 200e3
        worm = CPWResonator2(pos, self.Z, turn_radius, freq, eps, wavelength_fraction, coupling_length, meander_periods,
                             neck_length,
                             no_neck=no_neck, extra_neck_length=extra_neck_length, trans_in=trans_in)
        worm.place(self.region_ph)
        return worm

    def draw_test_squids(self):
        pars_probe = {'width': 300e3, 'b': 200e3, 'innergap': 30e3, 'outergap': 30e3}
        pars_squid = self.get_dc_squid_params()
        pars_squid[2] = pars_probe['innergap'] + 3 * pars_squid[0]
        Test_Squid(DPoint(1.5e6, 1e6), pars_probe, pars_squid, side=1).place(self.region_ph, self.region_el)
        Test_Squid(DPoint(1.5e6, 4e6), pars_probe, pars_squid, side=-1).place(self.region_ph, self.region_el)
        Test_Squid(DPoint(4e6, 1e6), pars_probe, pars_squid, side=1).place(self.region_ph, self.region_el)
        Test_Squid(DPoint(8.5e6, 3.5e6), pars_probe, pars_squid, side=-1).place(self.region_ph, self.region_el)

    def get_sps_params(self):
        pars = {'r_out': 175e3,  # Radius of an outer ring including the empty region
                'dr': 25e3,  # Gap in the outer ring
                'n_semiwaves': 2,
                's': 10e3,  # Gap between two pads of width central capacitor
                'alpha': pi / 4,  # period of width gap zigzag
                'r_curve': 30e3,  # curvature of the roundings at the edges of width zigzag
                'n_pts_cwave': 200,  # number of points for drawing width wave gap between to conductors
                'Z1': self.Z_narrow,  # Parameters of width top CPW
                'd_alpha1': 0,  # width of width tip  of width central conductor of the top CPW
                'width1': 0,  # width of width conductor in the top semiring
                'gap1': 25e3 - 1.33e3,  # gap between the top semiring and the central capacitor
                'Z2': self.Z,  # Parameters of width bottom CPW
                'd_alpha2': 2 / 9 * pi,  # length of width circumference covered by the bottom semiring
                'width2': 25e3 / 3,  # width of width conductor in the bottom semiring
                'gap2': 25e3 / 3,  # gap between the bottom semiring and the central capacitor
                'n_pts_arcs': 50,  # number of points for drawing width circle
                }
        return pars

    def get_mixing_qubit_params(self):
        pars = {'r_out': 175e3,  # Radius of an outer ring including the empty region
                'dr': 25e3,  # Gap in the outer ring
                'n_semiwaves': 2,
                's': 10e3,  # Gap between two pads of width central capacitor
                'alpha': pi / 4,  # period of width gap zigzag
                'r_curve': 30e3,  # curvature of the rotundings at the edges of width zigzag
                'n_pts_cwave': 200,  # number of points for drawing width wave gap between to conductors
                'Z1': self.Z_narrow,  # Parameters of width top CPW
                'd_alpha1': 0,  # width of width tip  of width central conductor of the top CPW
                'width1': 0,  # width of width conductor in the top semi-ring
                'gap1': 25e3,  # gap between the top semi-ring and the central capacitor
                'Z2': self.Z,  # Parameters of width bottom CPW
                'd_alpha2': 0,  # length of width circumference covered by the bottom semiring
                'width2': 0,  # width of width conductor in the bottom semi-ring
                'gap2': 25e3,  # gap between the bottom semi-ring and the central capacitor
                'n_pts_arcs': 50,  # number of points for drawing width circle
                }
        return pars

    def get_resonator_qubit_params(self):
        pars = {'r_out': 175e3,  # Radius of an outer ring including the empty region
                'dr': 25e3,  # Gap in the outer ring
                'n_semiwaves': 2,
                's': 10e3,  # Gap between two pads of width central capacitor
                'alpha': pi / 4,  # period of width gap zigzag
                'r_curve': 30e3,  # curvature of the roundings at the edges of width zigzag
                'n_pts_cwave': 200,  # number of points for drawing width wave gap between to conductors
                'Z1': self.Z_narrow,  # Parameters of width top CPW
                'd_alpha1': 0,  # width of width tip  of width central conductor of the top CPW
                'width1': 0,  # width of width conductor in the top semiring
                'gap1': 25e3 - 1.33e3,  # gap between the top semiring and the central capacitor
                'Z2': self.Z,  # Parameters of width bottom CPW
                'd_alpha2': 4 / 9 * pi,  # length of width circumference covered by the bottom semiring
                'width2': 15e3,  # width of width conductor in the bottom semiring
                'gap2': 5e3,  # gap between the bottom semiring and the central capacitor
                'n_pts_arcs': 50,  # number of points for drawing width circle
                }
        return pars

    def get_mixing_qubit_coupling_params(self):
        pars = {"to_line": 34.4e3,  # length between outer circle and the center of the coplanar
                "cpw_params": self.Z_res,
                "width": 10e3,
                "overlap": 10e3
                }
        return pars

    def get_dc_squid_params(self):
        pad_side = 5e3  # A length of the side of triangle pad
        pad_r = 1e3  # The outer_r of round angle of the contact pad
        pads_distance = 30e3  # The distance between triangle contact pads
        p_ext_width = 3e3  # The width of curved rectangle leads which connect triangle contact pads and junctions
        p_ext_r = 0.5e3  # The angle outer_r of the pad extension
        sq_dy = 7e3  # The length of the squid, along leads
        sq_area = 15e6  # The total area of the squid
        j_width = 0.3e3  # The width of the upper small leads (straight) and also width width of the junction
        intermediate_width = 0.5e3  # The width of the lower small bended leads before bending
        b_ext = 0.9e3  # The extension of bended leads after bending
        j_length = 0.2e3  # The length of the jj and the width of bended parts of the lower leads
        n = 7  # The number of angle in regular polygon which serves as width large contact pad
        bridge = 0.3e3  # The value of the gap between two parts of junction in the design
        return [pad_side, pad_r, pads_distance, p_ext_width,
                p_ext_r, sq_dy, sq_area, j_width, intermediate_width,
                b_ext, j_length, n, bridge]

    def calculate_ports(self, design_params=None):
        box = self._reg_from_layer(self.simulated_layer).bbox()

        port1 = SonnetPort(DPoint(box.width()/2, 0), PORT_TYPES.BOX_WALL)
        port2 = SonnetPort(DPoint(box.width() / 2, box.b()), PORT_TYPES.BOX_WALL)
        self.ports = [port1, port2]

    def draw_simulation(self, iter_params_dict):
        pass


### MAIN FUNCTION ###
if __name__ == "__main__":
    my_design = My_Design('testScript')
    origin = DPoint(0, 0)

    # width = 1.0e6
    # b = 1.0e6
    # box = Rectangle(origin, width, b)
    # box.place(my_design.region_ph)
    # my_design.draw_single_photon_source(DPoint(width/2, b/2))
    # p1 = my_design.sfs.connections[0]
    # p2 = my_design.sfs.connections[1]
    #
    # CPW(start=p1, end=p1 + DPoint(0, b - p1.y), cpw_params=My_Design.Z_narrow).place(my_design.region_ph)
    # CPW(start=p2, end=p2 + DPoint(0, -p2.y), cpw_params=My_Design.Z).place(my_design.region_ph)
    # my_design.show()
    #
    # freqs = np.linspace(1e9, 5e9, 300)
    # my_design.set_fixed_parameters(freqs)
    # my_design.set_swept_parameters( {"simBox": [SimulationBox(width, b, 100+20*k, 100+20*k) for k in range(11)]} )
    # my_design.simulate_sweep()
    # my_design.save()

    my_design.draw()
    
    alpha = 2.0
    dr = DVector(my_design.sfs.r_out, my_design.sfs.r_out)
    p1 = my_design.sfs_center - dr
    p2 = p1 + dr*alpha
    box = pya.Box().from_dbox(DBox(p1, p2))
    my_design.crop( pya.Box().from_dbox(DBox(p1,p2)) )
    my_design.region_ph.trans

    freqs = np.linspace(1e9, 5e9, 300)
    my_design.set_fixed_parameters(freqs)
    my_design.set_swept_parameters(
        {"simBox": [SimulationBox(box.width(),
                                  box.b(),
                                  200 + 2 * i,
                                  100 + 1 * i) for i in range(6)]}
    )
    my_design.simulate_sweep()
    my_design.set_measurement_name("helooo")
    my_design.save()