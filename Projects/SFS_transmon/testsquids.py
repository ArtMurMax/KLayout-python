from pya import DPoint, Trans
from math import pi

from importlib import reload
import classLib
reload(classLib)
from classLib import ChipDesign, Rectangle, AsymSquid

class Test_Squid(Complex_Base):
    """ @brief:     class represents width rectangular capacitor with width dc-SQUID between its plates
        @params:    DPoint origin - position of the center of width structure
                    params{} - width dictionary with geometric parameters of width capacitor
                    squid_params - width list with dc-SQUID parameters
                    Trans trans_in - initial transformation (None by default)
    """
    def __init__(self, origin, params, squid_params, trans_in=None):
        self.width = params['width']
        self.height = params['b']
        self.innergap = params['innergap']
        self.outergap = params['outergap']
        self.squid_params = squid_params
        super().__init__(origin, trans_in)

    def init_primitives(self):
        origin = DPoint(0, 0)
        self.primitives['empty_rect'] = Rectangle(origin - DPoint(self.width/2 + self.outergap, self.height + self.innergap / 2 + self.outergap),
                                    self.width + 2 * self.outergap,
                                    2 * self.height + 2 * self.outergap + self.innergap,
                                    inverse=True)  
        self.primitives['top_rect'] = Rectangle(origin + DPoint(-self.width/2, self.innergap/2),
                                    self.width,
                                    self.height)
        self.primitives['bottom_rect'] = Rectangle(origin - DPoint(self.width/2, self.height + self.innergap / 2),
                                    self.width,
                                    self.height)  
        self.squid = AsymSquid(origin, self.squid_params)
        self.primitives['qubit'] = self.squid

    def place(self, dest, layer_ph=-1, layer_el=-1):
        for prim_name in list(self.primitives.keys())[:-1]:
            self.primitives[prim_name].place(dest, layer_ph)
        self.squid.place(dest, layer_el)  

class SquidModel(ChipDesign):

    origin = DPoint(0, 0)
    chip_x = 1e6
    chip_y = 1e6
    cpw = None

    # Call other methods drawing parts of the design from here
    def draw(self):
        self.draw_chip()
        self.draw_test_squids()
    
    def draw_chip(self):
        origin = DPoint(0, 0)
        chip = Rectangle(origin, self.chip_x, self.chip_y)
        chip.place(self.cell, self.layer_ph)
        
    def draw_test_squids(self):
        pars_probe = {'width': 300e3, 'b': 200e3, 'innergap': 30e3, 'outergap': 30e3}
        pad_side = 5e3 # A length of the side of triangle pad
        pad_r = 1e3 # The outer_r of round angle of the contact pad
        pads_distance = pars_probe['innergap'] + 3 * pad_side # The distance between triangle contact pads
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
        pars_squid = AsymSquidParams(pad_side, pad_r, pads_distance, p_ext_width,
                p_ext_r, sq_dy, sq_area, j_width, intermediate_width,
                b_ext, j_length_1, j_length_2, n,bridge)
        Test_Squid(DPoint(self.chip_x/2, self.chip_y/2), pars_probe, pars_squid).place(self.cell, self.layer_ph, self.layer_el)

### MAIN FUNCTION ###
if __name__ == "__main__":
    sq = SquidModel('testSquids')
    sq.show()