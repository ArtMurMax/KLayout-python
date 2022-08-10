#TODO don't work in KLayout
import pya
from math import cos, sin, atan2, pi
from pya import Point,DPoint,DSimplePolygon,SimplePolygon, DPolygon, Polygon,  Region
from pya import Trans, DTrans, CplxTrans, DCplxTrans, ICplxTrans    
from classLib import *
        
### END classes to be delegated to different file ###

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
    
    info = pya.LayerInfo(10,0)
    layer_i = layout.layer( info )

    # clear this cell and layer
    cell.clear()

    # setting layout view  
    lv.select_cell(cell.cell_index(), 0)
    lv.add_missing_layers()
    
    ### DRAW SECTION START ###
    origin = DPoint(1e6,0)
    a = 4.6e3
    b = 4.6e3
    jos1_b = 0.32e3
    jos1_a = 0.1e3
    f1 = 0.13e3
    d1 = 0.05e3
    jos2_b = 0.8e3
    jos2_a = 0.09e3
    f2 = 0.25e3
    d2 = d1
    w = 0.2e3
    B1_width = 2.405e3
    B1_height = 7.325e3
    B2_height = 0.6e3
    B5_height = 0.4e3
    B6_width = 10e3
    B6_height = 2.5e3
    B7_width = 4.076e3
    B7_height = 5.051e3
    dCap = 0
    gap = 25e3
    #cell.shapes( layer_i ).insert( pya.Box( Point(0,0), Point( 40e3, 1e3 ) ) )
    qbit_params = [a,b,
                            jos1_b,jos1_a,f1,d1,
                            jos2_b,jos2_a,f2,d2,
                            w,B1_width,B1_height,
                            B2_height,B5_height,
                            B6_width,B6_height,
                            B7_width,B7_height,
                            dCap,gap]
    qbit = QBit_Flux_1( DPoint(dCap,0), qbit_params )
    qbit.place( cell, layer_i )
    
    ### DRAW SECTION END ###
    lv.zoom_fit()
