# $description: test_script  
# $version: 1.0.0  
# $show-in-menu
import pya 
from importlib import reload

from math import cos, sin, atan2, pi
from pya import Point,DPoint,DSimplePolygon,SimplePolygon, DPolygon, Polygon,  Region
from pya import Trans, DTrans, CplxTrans, DCplxTrans, ICplxTrans


import classLib
from classLib import *
reload(classLib)
from classLib import *

from classLib.coplanars import *
from classLib.resonators import *
from classLib.qbits import *
            
class CHIP:
    dx = 10.1e6
    dy = 5.1e6
    L1 = 2.5e6
    gap = 150.e3
    width = 260.e3
    b = 2*gap + width
    origin = DPoint( 0,0 )
    box = pya.DBox( origin, origin + DPoint( dx,dy ) )
    # only 4 connections programmed by now
    connections = [box.p1 + DPoint( L1 + b/2,0 ), box.p1 + DPoint( dx - (L1+b/2),0 ), box.p2 - DPoint( L1 + b/2,0 ),  box.p1 + DPoint( L1 + b/2, dy )]
    



if ( __name__ ==  "__main__" ):
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
    

    
    ## DRAWING SECTION START ##
    cell.shapes( layer_photo ).insert( pya.Box( Point(0,0), Point( CHIP.dx, CHIP.dy ) ) )
    origin = DPoint( 0,0 )
    
    contact_L = 1e6
    
    # main drive line coplanar
    width = 24.1e3
    gap = 12.95e3
    p1 = DPoint( 0 + contact_L, CHIP.dy/2 )
    p2 = DPoint( CHIP.dx - contact_L, CHIP.dy/2 )
    Z0 = CPW( width, gap, p1, p2 )
    #Z0.place( cell, layer_photo )
    
    # left contact pad
    width1 = CHIP.width*2
    gap1 = CHIP.gap*2
    p3 = DPoint( 0, CHIP.dy/2 )
    p4 = DPoint( contact_L/2, CHIP.dy/2 )
    Z1 = CPW( width1, gap1, p3, p4 )
    #Z1.place( cell, layer_photo )
    
    adapter1 = CPW2CPW( Z1, Z0, p4, p1 )
    #adapter1.place( cell, layer_photo )
    
    # right contact pad
    width1 = CHIP.width*2
    gap1 = CHIP.gap*2
    p5 = DPoint( CHIP.dx - contact_L/2, CHIP.dy/2 )
    p6 = DPoint( CHIP.dx, CHIP.dy/2 )
    Z2 = CPW( width1, gap1, p5, p6 )
    #Z2.place( cell, layer_photo )
    
    adapter1 = CPW2CPW( Z2, Z0, p5, p2 )
    #adapter1.place( cell, layer_photo )
    
    # resonator
    L_coupling = 300e3
    # corresponding to resonanse freq is linspaced in interval [6,9) GHz
    L1_list = [261653., 239550., 219456., 201109., 184291., 168819., 154536., 141312., 129032., 117600.]
    r = 50e3
    L2 = 270e3
    gnd_width = 35e3
    N = 4
    width_res = 9.6e3
    gap_res = 5.2e3
    Z_res = CPW( width_res, gap_res, origin, origin )
    to_line = Z_res.gap + Z0.gap
    worm = EMResonator_TL2Qbit_worm( Z_res, DPoint(CHIP.dx/2, CHIP.dy/2 - Z0.b) , L_coupling, L1_list[0], r, L2, N )
#    worm.place( cell, layer_photo )

    # qBit 
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
    dCap = 0
    dSquares = 30e3
    gap = 25e3
    square_a = 150e3
    alum_over = 20e3
    B1_width = 3e3

    qbit_params = [a,b,
                            jos1_b,jos1_a,f1,d1,
                            jos2_b,jos2_a,f2,d2,
                            w, dCap,gap, square_a, dSquares, alum_over, B1_width]
    toLine = 25e3
    dy = (worm.cop_tail.dr.abs() - square_a)/2
    qbit = QBit_Flux_Сshunted( worm.end + DPoint( toLine, dy ) , qbit_params, DCplxTrans( 1,90,False,0,0 ) )
    
    #empty polygon for qBit
    qbit_bbox = pya.DBox().from_ibox( qbit.metal_regions["photo"].bbox() )
    #print(qbit.metal_regions)
    #qbit_bbox = qbit.metal_region.bbox() 
    print(0, qbit_bbox.p2, qbit_bbox.b(), qbit_bbox.width())
    empty = CPW(0, qbit_bbox.b() / 2 + 2 * qbit.a,
                qbit_bbox.p1 + DPoint(0, qbit_bbox.b() / 2),
                qbit_bbox.p2 - DPoint(qbit.B4.width() + qbit.B3.width() / 2, qbit_bbox.b() / 2))
  
    empty.place( cell, layer_photo )
    qbit.place( cell, layer_photo, layer_el )

    print(empty.start, empty.end)
    ## DRAWING SECTION END ##
    lv.zoom_fit()
