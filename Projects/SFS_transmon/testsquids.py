from pya import DPoint, Trans
from math import pi

from importlib import reload
import ClassLib
reload(ClassLib)
from ClassLib import Chip_Design, Rectangle, Squid

class Test_Squid(Complex_Base):
    """ @brief:     class represents a rectangular capacitor with a dc-SQUID between its plates
        @params:    DPoint origin - position of the center of a structure
                    params{} - a dictionary with geometric parameters of a capacitor
                    squid_params - a list with dc-SQUID parameters
                    Trans trans_in - initial transformation (None by default)
    """
    def __init__(self, origin, params, squid_params, trans_in=None):
        self.width = params['width']
        self.height = params['height']
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
        self.squid = Squid(origin, self.squid_params)
        self.primitives['qubit'] = self.squid

    def place(self, dest, layer_ph=-1, layer_el=-1):
        for prim_name in list(self.primitives.keys())[:-1]:
            self.primitives[prim_name].place(dest, layer_ph)
        self.squid.place(dest, layer_el)  

class SquidModel(Chip_Design):

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
        pars_probe = {'width': 300e3, 'height': 200e3, 'innergap': 30e3, 'outergap': 30e3}
        pars_squid = self.get_dc_squid_params()
        Test_Squid(DPoint(self.chip_x/2, self.chip_y/2), pars_probe, pars_squid).place(self.cell, self.layer_ph, self.layer_el)

    def get_dc_squid_params(self):
        pad_side = 5e3 # A length of the side of triangle pad
        pad_r = 1e3 # The radius of round angle of the contact pad
        pads_distance = 30e3 # The distance between triangle contact pads
        p_ext_width = 3e3 # The width of curved rectangle leads which connect triangle contact pads and junctions
        p_ext_r = 0.5e3 # The angle radius of the pad extension
        sq_len = 7e3 # The length of the squid, along leads
        sq_area = 15e6 # The total area of the squid
        j_width = 0.2e3 # The width of the upper small leads (straight) and also a width of the junction
        low_lead_w = 0.5e3 # The width g the lower small bended leads before bending
        b_ext =   0.9e3 # The extension of bended leads after bending
        j_length =  0.1e3 # The length of the jj and the width of bended parts of the lower leads
        n = 7 # The number of angle in regular polygon which serves as a large contact pad
        bridge = 0.2e3 # The value of the gap between two parts of junction in the design
        return [pad_side, pad_r, pads_distance, p_ext_width,
                p_ext_r, sq_len, sq_area, j_width, low_lead_w,
                b_ext, j_length, n,bridge]

### MAIN FUNCTION ###
if __name__ == "__main__":
    sq = SquidModel('testSquids')
    sq.show()