import pya
from math import sqrt, cos, sin, tan, atan2, pi, copysign
from pya import Point, DPoint, Vector, DVector, DSimplePolygon, SimplePolygon, DPolygon, Polygon, Region, DBox, Box
from pya import Trans, DTrans, CplxTrans, DCplxTrans, ICplxTrans

from importlib import reload
import classLib
from classLib import *
reload(classLib)
from classLib import *

class CHIP:
    dx = 10e6
    dy = 5e6

### START classes to be delegated to different file ###

class SFS_Csh_emb(ComplexBase):
    def __init__( self, origin, params, trans_in=None ):
        self.params = params
        self.r_out = params[0]
        self.dr = params[1]
        self.n_segments = params[2]
        self.s = params[3]
        self.alpha = params[4]
        self.r_curve = params[5]
        self.n_pts_cwave = params[6]
        self.L0 = params[7]
        self.Z1 = params[8]
        self.d_alpha1 = params[9]
        self.width1 = params[10]
        self.gap1 = params[11]
        self.Z2 = params[12]
        self.d_alpha2 = params[13]
        self.width2 = params[14]
        self.gap2 = params[15]
        self.n_pts_arcs = params[16]
        super().__init__( origin, trans_in )
        '''
        self.excitation_port = self.connections[0]
        self.output_port = self.connections[1]
        self.excitation_angle = self.angle_connections[0]
        self.output_angle = self.angle_connections[1]
        '''

    def init_primitives( self ):
        origin = DPoint(0,0)
        chip_center = DPoint(CHIP.dx/2, CHIP.dy/2)
        self.c_wave = CWave(origin, self.r_out, self.dr, self.n_segments,
                            self.s, self.alpha, self.r_curve,
                            n_pts=self.n_pts_cwave)
        self.primitives["c_wave"] = self.c_wave

        Z1_start = origin + DPoint( 0,self.r_out + self.gap1 + self.width1/2 )
        Z1_end = Z1_start + DPoint( 0, -self.gap1 - self.width1/2 + self.dr )
        self.cpw1 = CPW( self.Z1.width, self.Z1.gap, Z1_start, Z1_end )
        self.primitives["cpw1"] = self.cpw1

        Z2_start = origin - DPoint( 0,self.r_out + self.gap1 + self.width1/2 )
        Z2_end = Z2_start - DPoint( 0, -self.gap1 - self.width1/2 + self.dr )
        self.cpw2 = CPW( self.Z2.width, self.Z2.gap, Z2_start, Z2_end )
        self.primitives["cpw2"] = self.cpw2

        self.c_wave_2_cpw_adapter = CWave2CPW( self.c_wave, self.params[8:16], n_pts=self.n_pts_arcs )
        self.primitives["c_wave_2_cpw_adapter"] = self.c_wave_2_cpw_adapter



        self.connections = [Z1_end, Z2_end]
        self.angle_connections = [pi/2, 3/2*pi]

class StickQubit_Cap(ElementBase):

    def __init__(self, origin, s_w, s_l, gap_w, gap_l, trans_in=None):
        self._s_w = s_w
        self._s_l = s_l
        self._gap_w = gap_w
        self._gap_l = gap_l
        super().__init__(origin, trans_in)

    def init_regions(self):
        ll = DPoint(-self._s_l/2, -self._s_w/2)
        ur = DPoint(self._s_l/2, self._s_w/2)
        self.metal_region.insert(Box(DBox(ll,ur)))
        ll = DPoint(-(self._s_l+self._gap_l)/2, -(self._s_w+self._gap_w)/2)
        ur = DPoint((self._s_l+self._gap_l)/2, (self._s_w+self._gap_w)/2)
        protect_reg = Region(Box(DBox(ll,ur)))
        empty_region = protect_reg - self.metal_region
        self.empty_region.insert(empty_region)
        self.connections = [DPoint(0,0)]






# END classes to be delegated to different file ###

# Enter your Python code here
### MAIN FUNCTION ###
if __name__ == "__main__":
# getting main references of the application
    app = pya.Application.instance()
    mw = app.main_window()
    lv = mw.current_view()
    cv = None

    #this insures that lv and cv are valid objects
    if( lv == None ):
        cv = mw.create_layout(1)
        lv = mw.current_view()
    else:
        cv = lv.active_cellview()

# find or create the desired by programmer cell and layer
    layout = cv.layout()
    layout.dbu = 0.001
    if( layout.has_cell( "testScript") ):
        pass
    else:
        cell = layout.create_cell( "testScript" )

    info = pya.LayerInfo(1,0)
    info2 = pya.LayerInfo(2,0)
    layer_photo = layout.layer( info )
    layer_el = layout.layer( info2 )

    # clear this cell and layer
    cell.clear()

    # setting layout view
    lv.select_cell(cell.cell_index(), 0)
    lv.add_missing_layers()

    ### DRAW SECTION START ###
    origin = DPoint(0,0)
    # Chip drwaing START #
    chip = pya.DBox(DPoint(-CHIP.dx/2, -CHIP.dy/2),DPoint(CHIP.dx/2, CHIP.dy/2))
    cell.shapes(layer_photo).insert(pya.Box(chip))
    # Chip drawing END #
    cp_coords = [DPoint(-CHIP.dx/4, -CHIP.dy/2),
                 DPoint(0, -CHIP.dy/2),
                 DPoint(CHIP.dx/4, -CHIP.dy/2),
                 DPoint(-CHIP.dx/4, +CHIP.dy/2),
                 DPoint(0, +CHIP.dy/2),
                 DPoint(CHIP.dx/4, +CHIP.dy/2),
                 DPoint(-CHIP.dx/2, 0),
                 DPoint(CHIP.dx/2, 0),]
    cp_trans = [DTrans.R90,
                DTrans.R90,
                DTrans.R90,
                DTrans.R270,
                DTrans.R270,
                DTrans.R270,
                None,
                DTrans.R180,]
    cp_list = [Contact_Pad(cp_pos, {"w":20e3, "g":10e3}, trans_in = cp_tr) \
                    for cp_pos, cp_tr in zip(cp_coords, cp_trans)]
    [cp.place(cell, layer_photo) for cp in cp_list]


    # Single photon source photo layer drawing START #
    Z = CPWParameters(20e3, 10e3)
    Zl = CPWParameters(10e3,7e3)
    r_out = 150e3
    r_gap = 25e3
    n_segments = 3
    s = 7e3
    alpha = pi/2.3
    r_curve = 20e3
    n_pts_cwave = 200
    L0 = 20e3
    Z1 = Z
    d_alpha1 = pi/26
    width1 = r_gap/3
    gap1 = width1
    Z2 = Z
    d_alpha2 = 2/3*pi
    width2 = width1
    gap2 = width2
    n_pts_arcs = 50
    params = [r_out, r_gap, n_segments, s, alpha, r_curve, n_pts_cwave, L0,
              Z1, d_alpha1, width1, gap1, Z2, d_alpha2, width2, gap2 , n_pts_arcs]
    p = DPoint( -CHIP.dx/4, CHIP.dy/4 )
    sfs = SFS_Csh_emb( p, params )
    sfs.place( cell, layer_photo)

    f_l = sfs.connections[0] - cp_list[3].end
    feedline = CPWRLPath(cp_list[3].end, "L", Z, 100e3, [f_l.y - 17e3], [], trans_in = DTrans.R90)
    feedline.place(cell,layer_photo)


    out_dist = cp_list[4].end - sfs.connections[1]
    out_dist2 = cp_list[3].end - sfs.connections[1]
    print(sfs.connections[1])
    r_turn = 200e3
    constr = 200e3
    out_l_segments = [ 300e3,out_dist.x/2-constr/2, constr, out_dist.x/2-constr/2,300e3+out_dist2.y]
    out_l_segments = [ 300e3,out_dist.x,300e3+out_dist2.y]
    outline = CPWRLPath(sfs.connections[1], "LRLRL", Z, r_turn,
                        out_l_segments, [pi/2,pi/2], trans_in = DTrans.R270)
    outline.place(cell,layer_photo)

    # drawing stick qubit (sq)
    
    bb = outline.primitives['cpw_1'].empty_region.bbox()
    sq_y = bb.bottom
    sq_x = bb.center().x
    chip_center = DPoint( chip.chip_x/2, chip.chip_y/2)
    jj = LineNJJ(chip_center)

    sq_c = StickQubit_Cap(DPoint(sq_x,sq_y-10e3),20e3,200e3,6e3,8e3)
    sq_c.place(cell,layer_photo)

    zonator = EMResonator_TL2Qbit_worm(Zl, sq_c.connections[0]-DPoint(0,5e4), 50e3, 20e4, 50e3, 5e4, 2)
    zonator.place(cell, layer_photo)
    ### DRAW SECTION END ###

    lv.zoom_fit()
