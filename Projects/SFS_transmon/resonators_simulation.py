from pya import DPoint, Trans
from math import pi

from importlib import reload
import classLib
reload(classLib)
from classLib import ChipDesign, CPWParameters, CPW, CPWResonator2, SFS_Csh_emb, Rectangle

import sonnetSim
reload(sonnetSim)
from sonnetSim import SonnetLab, PORT_TYPES

class ResonatorSimulator(ChipDesign):

    origin = DPoint(0, 0)
    Z = CPWParameters(20e3, 10e3) # normal CPW
    Z_res = Z
    Z_narrow = CPWParameters(10e3,7e3) # narrow CPW
    cpw_curve = 200e3 # Curvature of CPW angles
    chip_x = 1.5e6
    chip_y = 1.5e6
    ncells_x = 400
    ncells_y = 400
    cpw = None

    # Call other methods drawing parts of the design from here
    def draw(self):
        self.draw_chip()
        self.cpw = self.draw_line(vert=True)
        self.draw_bottom_left_resonator(330e3)
    
    def draw_chip(self):
        origin = DPoint(0, 0)
        chip = Rectangle(origin, self.chip_x, self.chip_y)
        chip.place(self.cell, self.layer_ph)
    
    def draw_line(self, vert=False):
        if vert:
          start = DPoint(self.chip_x * 0.2, 0)
          end = DPoint(self.chip_x * 0.2, self.chip_y)
        else:
          start = DPoint(0, self.chip_y * 0.8)
          end = DPoint(self.chip_y, self.chip_y * 0.8)
        cpw = CPW(self.Z.width, self.Z.gap, start, end)
        cpw.place(self.cell, self.layer_ph)
        return cpw
    
    def draw_top_left_resonator(self, coupling_length=350e3):
        # frequency from Sonnet = 6.8166 GHz
        # Qc = 5190
        coupling_length = 350e3
        pars = self.get_sps_params()
        pars_squid = self.get_dc_squid_params()
        res_to_line = 5e3 # distance between width resonator and width line

        worm_pos = DPoint(self.chip_x / 2, self.cpw.start.y - res_to_line - 2 * (self.Z.width/2 + self.Z.gap))
        worm1 = self.draw_one_resonator(worm_pos, 6.8, coupling_length, extra_neck_length=pars['r_out'], trans_in=Trans.R180)
        q_pos = worm1.end + DPoint(0, pars['r_out'] - worm1.Z.b) # qubit position
        mq1 = SFS_Csh_emb(q_pos, pars, pars_squid)
        mq1.place(self.cell, self.layer_ph, self.layer_el)

    def draw_top_right_resonator(self, coupling_length=350e3):
        # frequency from Sonnet = 7.0251 GHz
        # Qc = 4860
        coupling_length = 350e3
        pars = self.get_sps_params()
        pars_squid = self.get_dc_squid_params()
        res_to_line = 5e3 # distance between width resonator and width line

        worm_pos = DPoint(self.chip_x / 2, self.cpw.start.y - res_to_line - 2 * (self.Z.width/2 + self.Z.gap))
        worm2 = self.draw_one_resonator(worm_pos, 7, coupling_length, extra_neck_length=pars['r_out'], trans_in=Trans.R180)
        q_pos = worm2.end + DPoint(0, pars['r_out'] - worm2.Z.b) # qubit position
        mq2 = SFS_Csh_emb(q_pos, pars, pars_squid)
        mq2.place(self.cell, self.layer_ph, self.layer_el)

    def draw_bottom_left_resonator(self, coupling_length=330e3):
        # frequency from Sonnet = 7.2103 GHz
        # Qc = 5160
        coupling_length = 330e3
        pars = self.get_sps_params()
        pars_squid = self.get_dc_squid_params()
        res_to_line = 5e3 # distance between width resonator and width line
        # Moving width capacitive coupling to the top of width qubit
        pars['Z1'],       pars['Z2']       = pars['Z2'],       pars['Z1']
        pars['d_alpha1'], pars['d_alpha2'] = pars['d_alpha2'], pars['d_alpha1']
        pars['width1'],   pars['width2']   = pars['width2'],   pars['width1']
        pars['gap1'],     pars['gap2']     = pars['gap2'],     pars['gap1']

        worm_pos = DPoint(self.cpw.start.x + res_to_line + 2 * (self.Z.width/2 + self.Z.gap), self.chip_y / 2)
        worm3 = self.draw_one_resonator(worm_pos, 7.4, coupling_length, no_neck=True, vert=False, trans_in=Trans.R270)
        q_pos = worm3.end - DPoint(0, pars['r_out'] - worm3.Z.b)
        mq3 = SFS_Csh_emb(q_pos, pars, pars_squid)
        mq3.place(self.cell, self.layer_ph, self.layer_el)
    
    def simulate(self):
        SL = SonnetLab()
        SL.clear()

        SL.set_boxProps(self.chip_x, self.chip_y,
                        self.ncells_x, self.ncells_y)
        SL.set_ABS_sweep(7.2, 7.22)
        SL.set_ports(self.cpw.connections, [0, 0])
        SL.send_polygons(self.cell, self.layer_ph)
        SL.start_simulation(wait=True)
        SL.visualize_sever()
        SL.release()

    def draw_one_resonator(self, pos, freq, coupling_length, no_neck=False, extra_neck_length=0, vert=True, trans_in=None):
        turn_radius = 50e3
        eps = 11.45
        wavelength_fraction = 1/4
        meander_periods = 3
        neck_length = 200e3
        if vert:
            pos = pos + DPoint((coupling_length + extra_neck_length)/2 + 2 * turn_radius, 0)
        else:
            pos = pos + DPoint(0, (coupling_length + extra_neck_length)/2 + 2 * turn_radius)
        worm = CPWResonator2(pos, self.Z, turn_radius, freq, eps, wavelength_fraction, coupling_length, meander_periods, neck_length,
                            no_neck=no_neck, extra_neck_length=extra_neck_length, trans_in=trans_in)
        worm.place(self.cell, self.layer_ph)
        print(worm._calculate_total_length())
        return worm

    def get_sps_params(self):
        pars = {'r_out'	:	175e3, # Radius of an outer ring including the empty region
                'dr'	:	25e3, # Gap in the outer ring
                'n_semiwaves'	:	2,
                's'	:	10e3, # Gap between two pads of width central capacitor
                'alpha'	:	pi/4, # period of width gap zigzag
                'r_curve'	:	30e3, # curvature of the roundings at the edges of width zigzag
                'n_pts_cwave'	:	200, # number of points for drawing width wave gap between to conductors
                'Z1'	:	self.Z_narrow, # Parameters of width top CPW
                'd_alpha1'	:	0, # width of width tip  of width central conductor of the top CPW
                'width1'	:	0, # width of width conductor in the top semiring
                'gap1'	:	25e3 - 1.33e3, # gap between the top semiring and the central capacitor
                'Z2'	:	self.Z, # Parameters of width bottom CPW
                'd_alpha2'	:	2 / 9 * pi, # length of width circumference covered by the bottom semiring
                'width2'	:	25e3/3, # width of width conductor in the bottom semiring
                'gap2'	:	25e3/3, # gap between the bottom semiring and the central capacitor
                'n_pts_arcs'	:	 50, # number of points for drawing width circle
                }
        return pars

    def get_dc_squid_params(self):
        pad_side = 5e3 # A length of the side of triangle pad
        pad_r = 1e3 # The outer_r of round angle of the contact pad
        pads_distance = 30e3 # The distance between triangle contact pads
        p_ext_width = 3e3 # The width of curved rectangle leads which connect triangle contact pads and junctions
        p_ext_r = 0.5e3 # The angle outer_r of the pad extension
        sq_len = 7e3 # The length of the squid, along leads
        sq_area = 15e6 # The total area of the squid
        j_width = 0.2e3 # The width of the upper small leads (straight) and also width width of the junction
        intermediate_width = 0.5e3 # The width g the lower small bended leads before bending
        b_ext =   0.9e3 # The extension of bended leads after bending
        j_length =  0.1e3 # The length of the jj and the width of bended parts of the lower leads
        n = 7 # The number of angle in regular polygon which serves as width large contact pad
        bridge = 0.2e3 # The value of the gap between two parts of junction in the design
        return [pad_side, pad_r, pads_distance, p_ext_width,
                p_ext_r, sq_len, sq_area, j_width, intermediate_width,
                b_ext, j_length, n,bridge]

### MAIN FUNCTION ###
if __name__ == "__main__":
    resSim = ResonatorSimulator('resonatorsSim')
    resSim.show()
    resSim.simulate()
    # resSim.save_as_gds2(r'C:\Users\andre\Documents\chip_designs\chip_design.gds2')