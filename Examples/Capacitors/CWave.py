import pya
from math import sqrt, cos, sin, tan, atan2, pi, copysign
from pya import Point, DPoint, Vector, DVector, DSimplePolygon, SimplePolygon, DPolygon, Polygon, Region
from pya import Trans, DTrans, CplxTrans, DCplxTrans, ICplxTrans

from importlib import reload
from classLib import *

### START classes to be delegated to different file ###
class CWave(ComplexBase):
    def __init__(self, center, r_out, dr, n_semiwaves, s, alpha, r_curve, n_pts=50,solid=True, L0=None, trans_in=None ):
        self.r_out = r_out
        self.dr = dr
        self.n_semiwaves = n_semiwaves
        self.s = s
        self.alpha = alpha
        self.r_curve = r_curve
        self.n_pts = n_pts
        
        ## MAGIC VARIABLE ##
        self.delta = 2e3 # 2 mkm from each side of circle will be erased along the diameter
        
        self.L_x = 2*self.r_out - 2*self.delta
        # calculating parameters of the CPWDPath #
        L_x = self.L_x
        r = self.r_curve
        n = self.n_semiwaves
        alpha = self.alpha
        if( self.alpha == pi/2 or self.alpha == -pi/2 ):
            self.L0 = L0
            self.r_curve = L_x/(2*n+2)
            r = self.r_curve
        else:
            self.L0 = ( ( L_x + 2*(n-1)*r*tan(alpha/2) ) / ( 2*n ) - 2*r*sin(alpha) ) / cos(alpha)
        self.L1 = 2*r*tan(alpha/2) + 2*self.L0
        if( self.L0 < 0 or self.L1 < 0 ):
            print( "CPWDPath: impossible parameters combination" )
        
        super( CWave,self ). __init__( center,trans_in )
            
    def init_primitives(self):
        origin = DPoint(0,0)
        #erased line params
        Z = CPWParameters( 0, self.s/2 )
        
        # placing circle r_out with dr clearance from ground polygon
        self.empt_circle = Circle(origin, self.r_out + self.dr, n_pts=self.n_pts, inverse=True)
        self.in_circle = Circle(origin, self.r_out, n_pts=self.n_pts, inverse=False)
        self.empt_circle.empty_region -= self.in_circle.metal_region
        self.primitives["empt_circle"] = self.empt_circle
        self.primitives["in_circle"] = self.in_circle
        
        # pads on left and right sides        
        left = DPoint( -self.r_out, 0 )
        self.pad_left = CPW( Z.width, Z.gap, left, left + DPoint( self.delta, 0 ) )
        self.primitives["pad_left"] = self.pad_left
        
        right = DPoint( self.r_out,0 )
        self.pad_right = CPW( Z.width, Z.gap, right, right + DPoint( -self.delta,0 ) )
        self.primitives["pad_right"] = self.pad_right
        
        # starting RLR
        self.RL_start = origin + DPoint( -self.in_circle.r + self.delta,0 )
        rl_path_start = CPWRLPath(self.RL_start, "RLR", Z, self.r_curve, [self.L0], [self.alpha, -self.alpha])
        self.primitives["rl_start"] = rl_path_start
        
        # ending RLR
        if( self.n_semiwaves%2 == 0 ):
            m_x = False
        else:
            m_x = True
        self.RL_end = origin + DPoint( self.in_circle.r - self.delta,0 )
        rl_path_end = CPWRLPath(self.RL_end, "RLR", Z, self.r_curve, [self.L0], [self.alpha, -self.alpha], trans_in = DCplxTrans(1, 180, m_x, 0, 0))

        
        # intermidiate RLRs
        for i in range(self.n_semiwaves - 1):
            if( i%2 == 0 ):
                m_x = False
            else:
                m_x = True 
                
            prev_path = list(self.primitives.values())[-1]
            rl_path_p = CPWRLPath(prev_path.end, "RLR", [Z], [self.r_curve], [self.L1], [-self.alpha, self.alpha], trans_in=DCplxTrans(1, 0, m_x, 0, 0))
            self.primitives["rl_path_" + str(i)] = rl_path_p
        
        self.primitives["rl_path_end"] = rl_path_end


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
    layer_ph = layout.layer( info )
    layer_el = layout.layer( info2 )

    # clear this cell and layer
    cell.clear()

    # setting layout view  
    lv.select_cell(cell.cell_index(), 0)
    lv.add_missing_layers()

    ### DRAW SECTION START ###
    origin = DPoint(0,0)
    
    # Chip drwaing START #    
    chip = pya.DBox( origin, DPoint( CHIP.dx, CHIP.dy ) )
    cell.shapes( layer_ph ).insert( pya.Box().from_dbox(chip) )
    # Chip drawing END #
    
    # CWave capacitor drawing START #
    r_out = 200e3
    r_gap = 25e3
    r_curve = 2e4
    s = 5e3  
    alpha = pi/2
    circle_center = DPoint( CHIP.dx/2, CHIP.dy/2 )
    
    crc = CWave(circle_center,r_out, r_gap, 8, s, alpha, r_curve, L0 = r_curve)
    crc.place( cell,layer_ph )
    # CWave capacitor drawing END #
    
    ### DRAW SECTION END ###
    
    lv.zoom_fit()