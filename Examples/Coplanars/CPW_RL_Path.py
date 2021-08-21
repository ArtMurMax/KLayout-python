import pya
from math import sqrt, cos, sin, atan2, pi, copysign
from pya import Point, DPoint, Vector, DVector, DSimplePolygon, SimplePolygon, DPolygon, Polygon, Region
from pya import Trans, DTrans, CplxTrans, DCplxTrans, ICplxTrans

from importlib import reload
from classLib import *

### START classes to be delegated to different file ###
from collections import Counter

class CPW_RL_Path(ComplexBase):
    def __init__( self, start, RL_str, Z_list, R_list, L_list, delta_alpha_list, trans_in = None ):
        self.RL_str = RL_str
        self.N_elements = len(RL_str)
        self.RL_counter = Counter( RL_str )
        
        # periodically filling  Z,R,L lists to make their length equal to the RL_str
        self.Z_list = [Z_list[i%len(Z_list)] for i in range( self.N_elements )]
        self.R_list = [R_list[i%len(R_list)] for i in range( self.RL_counter['R'] )]
        self.L_list = [L_list[i%len(L_list)] for i in range( self.RL_counter['L'] )]
        self.delta_alpha_list = delta_alpha_list
        
        self.start = start
        super(CPWRLPath, self).__init__(start, trans_in)
        '''self.start = self.connections[0]
        self.end = self.connections[1]
        self.alpha_start = self.angle_connections[0]
        self.alpha_end = self.angle_connections[1]
        '''
        
    def init_primitives( self ):
        R_index = 0
        L_index = 0
        origin = DPoint(0,0)
        # placing the first element
        symbol_0 = self.RL_str[0]
        if( symbol_0 == 'R' ):
            self.primitives["arc_0"] = CPWArc(self.Z_list[0], origin, self.R_list[0], self.delta_alpha_list[0])
            R_index += 1
        elif( symbol_0 == 'L' ):
            self.primitives["cpw_0"] = CPW( self.Z_list[0].width, self.Z_list[0].gap, self.start, self.start + DPoint( self.L_list[0],0 ) )
            L_index += 1
        
        for i,symbol in enumerate(self.RL_str):
            if( i == 0 ):
                continue

            prev_primitive = list(self.primitives.values())[i-1]     
                       
            if( symbol == 'R' ):
                if( self.delta_alpha_list[R_index] > 0 ):
                    sgn = 1
                else:
                    sgn = -1
                    
                R = self.R_list[R_index]*sgn
                delta_alpha = self.delta_alpha_list[R_index]
            
                cpw_arc = CPWArc(self.Z_list[i], prev_primitive.end, R, delta_alpha,
                                 trans_in=DCplxTrans(1,prev_primitive.alpha_end*180/pi, False,0,0))
                self.primitives["arc_"+str(R_index)] = cpw_arc
                R_index += 1
                
            elif( symbol == 'L' ):
                cpw = CPW( self.Z_list[i].width, self.Z_list[i].gap,
                                        prev_primitive.end, prev_primitive.end + DPoint(self.L_list[L_index],0),
                                        trans_in=DCplxTrans(1,prev_primitive.alpha_end*180/pi,False,0,0) )
                self.primitives["cpw_"+str(L_index)] = cpw
                L_index += 1 

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
    
    # Path test drawing START #
    RL_str = "RLRLR"
    Z0 = CPW( 12.2e3, 6.2e3 )
    Z_list = [Z0]
    R_list = [0.5e6]
    L_list = [0.2e6]
    delta_alpha_list = [pi/4,-pi/4,-pi]
    start = origin
    path = CPW_RL_Path(start, RL_str, Z_list, R_list, L_list, delta_alpha_list, trans_in = DCplxTrans(1,45,False,0,0) )
    path.place( cell, layer_ph )
    # Path test drawing END #
    
    ### DRAW SECTION END ###
    
    lv.zoom_fit()