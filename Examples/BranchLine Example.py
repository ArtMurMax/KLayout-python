#TODO don't work in KLayout
import pya
from math import cos, sin, atan2, pi
from pya import Point,DPoint,DSimplePolygon,SimplePolygon, DPolygon, Polygon,  Region
from pya import Trans, DTrans, CplxTrans, DCplxTrans, ICplxTrans

from classLib import *

### START classes to be delegated to different file ###

### END classes to be delegated to different file ###

### MAIN FUNCTION ###
if( __name__ == "__main__" ):
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

    info = pya.LayerInfo(10,0)
    layer_i = layout.layer( info )

    # clear this cell and layer
    cell.clear()

    # setting layout view
    lv.select_cell(cell.cell_index(), 0)
    lv.add_missing_layers()

    ### DRAW SECTION START ###
    #parameters
    width1 = 48e3
    gap1 = 26e3
    width2 = 74e3
    gap2 = 13e3
    gndWidth = 20     #not used
    L1 = 1e6
    L2 = 0.46538e6
    L3 = 1e6
    L4 = 0.108427e6
    L5 = 1e6
    R1 = 0.3e6
    R2 = 0.3e6
    R3 = 0.3e6
    R4 = 0.3e6
    gamma1 = pi/4
    gamma2 = 2.35619

    origin = DPoint(1e6,0)
    #cell.shapes( layer_i ).insert( pya.Box( Point(0,0), Point( CHIP_10x10_12pads.dx, CHIP_10x10_12pads.dy ) ) )
    Z0 = CPW( width1, gap1, origin, origin )
    Z1 = CPW( width2, gap2, origin, origin )
    '''Z1 = CPW( width1, gap1, origin, origin + DPoint( 1e6,0 ), trans_in=DCplxTrans( 1,90,False,1e6,0 ) )
    Z1.place( cell, layer_i )
    for val in Z1.connections:
        print( val )
    print( Z0.origin )'''

    '''Z1 = CPWArc( Z0, origin + DPoint(1e6,0), R1, gamma1, trans_in=DCplxTrans( 1,90,False,1e6,0 ) )
    Z1.place( cell, layer_i )
    for val in Z1.connections:
        print( val )
    print( Z0.origin )'''

    '''Z1 = CPW( width2,gap2, DPoint(0,0), DPoint(0,0) )
    tj = TJunction_112( Z0,Z1, origin + DPoint(1e6,1e6), DCplxTrans( 1,30,False,1e6,1e6 ) )
    tj.place( cell, layer_i )'''

    '''bl = BranchLine_finger( Z0, origin, L1,L2,L3,R1,R2,gamma1, trans_in=DTrans( DTrans.R90 ) )
    bl.place( cell, layer_i )
    for val in bl.connections:
        print( val )
    print( Z0.origin )'''

    bl1_params = [L1,L2,L3,R1,R2,gamma1]
    bl2_params = [L1,L2,L3,L4,L5,R1,R2,R3,R4,gamma1,gamma2]

    bl_hor = BranchLine_finger2( Z1, origin, bl2_params )
    bl_vert = BranchLine_finger2( Z0, origin, bl2_params )
    tj = TJunction_112( Z0,Z1, origin )
    coupler = Coupler_BranchLine( origin, bl_hor, bl_vert, tj )
    coupler.place( cell, layer_i )
    for conn in coupler.connections:
        print( conn )
    print()
    for alpha in coupler.angle_connections:
        print( alpha )


    lv.zoom_fit()
    ### MAIN FUNCTION END ###
