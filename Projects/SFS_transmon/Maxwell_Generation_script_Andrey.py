from pya import DPoint, Trans
from math import pi

from importlib import reload
import classLib
reload(classLib)
from classLib import ChipDesign, CPWParameters, CPW, CPWResonator2, SFS_Csh_emb, Rectangle

class ResonatorQubitSimulator(ChipDesign):

    origin = DPoint(0, 0)
    Z = CPWParameters(20e3, 10e3) # normal CPW
    Z_res = Z
    Z_narrow = CPWParameters(10e3,7e3) # narrow CPW
    chip_x = 2e6
    chip_y = 2e6
    cpw = None

    # Call other methods drawing parts of the design from here
    def draw(self):
        self.draw_chip()
        self.draw_qubit()
    
    def draw_chip(self):
        origin = DPoint(0, 0)
        chip = Rectangle(origin, self.chip_x, self.chip_y)
        chip.place(self.cell, self.layer_ph)
    
    def draw_qubit(self):
        # Mixing qubit and dc-SQUID parameters
        pars = self.get_resonator_qubit_params()
        pars_squid = self.get_dc_squid_params()
        
        # Drawing the qubit near the probe line
        pos = DPoint(self.chip_x/2, self.chip_y/2)  # Position of the qubit
        rq1 = SFS_Csh_emb(pos, pars, pars_squid)
        rq1.place(self.cell, self.layer_ph, self.layer_el)
        
        # Draw line at the bottom
        start = rq1.connections[1] #DPoint(self.chip_x/2, self.chip_y/2 - pars['r_out'])
        end = DPoint(self.chip_x/2, 0)
        cpw = CPW(self.Z.width, self.Z.gap, start, end)
        cpw.place(self.cell, self.layer_ph)
    
    def get_resonator_qubit_params(self):
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
                'd_alpha2'	:	4 / 9 * pi, # length of width circumference covered by the bottom semiring
                'width2'	:	15e3, # width of width conductor in the bottom semiring
                'gap2'	:	5e3, # gap between the bottom semiring and the central capacitor
                'n_pts_arcs'	:	 50, # number of points for drawing width circle
                }
        return pars

    def get_dc_squid_params(self):
        pad_side = 5e3 # A length of the side of triangle pad
        pad_r = 1e3 # The outer_r of round angle of the contact pad
        pads_distance = 30e3 # The distance between triangle contact pads
        p_ext_width = 3e3 # The width of curved rectangle leads which connect triangle contact pads and junctions
        p_ext_r = 0.5e3 # The angle outer_r of the pad extension
        sq_dy = 7e3 # The length of the squid, along leads
        sq_area = 15e6 # The total area of the squid
        j_width = 100 # The width of the upper small leads (straight) and also width width of the junction
        intermediate_width = 0.5e3 # The width of the lower small bended leads before bending
        b_ext = 0.9e3 # The extension of bended leads after bending
        j_length_1 =  114 # The length of the LEFT jj and the width of bended parts of the lower leads
        j_length_2 = 342 # The length of the RIGHT jj and the width of bended parts of the lower leads
        n = 7 # The number of angle in regular polygon which serves as width large contact pad
        bridge = 0.3e3 # The value of the gap between two parts of junction in the design
        return [pad_side, pad_r, pads_distance, p_ext_width,
                p_ext_r, sq_dy, sq_area, j_width, intermediate_width,
                b_ext, j_length_1, j_length_2, n,bridge]

### MAIN FUNCTION ###
if __name__ == "__main__":
    rqSim = ResonatorQubitSimulator('resQubitSim')
    rqSim.show()