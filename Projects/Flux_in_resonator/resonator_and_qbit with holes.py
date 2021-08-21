# $description: test_script  
# $version: 1.0.0  
# $show-in-menu
import pya
from math import cos, sin, atan2, pi
from pya import Point,DPoint,DSimplePolygon,SimplePolygon, DPolygon, Polygon,  Region
from pya import Trans, DTrans, CplxTrans, DCplxTrans, ICplxTrans
from classLib import *

def fill_holes( poly, dx=10e3, dy=8e3, width=5e3, height=5e3, d=0 ):
    bbox = poly.bbox()
    poly_reg = Region( poly )
    t_reg = Region()
    
    y = bbox.p1.y + height
    while( y < bbox.p2.y - height ):
        x = bbox.p1.x + width
        while( x < bbox.p2.x -  width ):
            box = pya.Box().from_dbox( pya.DBox(DPoint(x,y), DPoint(x + width,y + height)) )
            x += dx
            t_reg.clear()
            t_reg.insert( box )
            qwe = t_reg.select_inside( poly_reg )
            if( qwe.is_empty() ):
                continue
            edge_pairs = qwe.inside_check( poly_reg, d )
            if( edge_pairs.is_empty() ):
                poly.insert_hole( box )
        y += dy       
    return poly

            
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
    
    layer_info_photo = pya.LayerInfo(10,0)
    layer_info_el = pya.LayerInfo(1,0)    
    layer_photo = layout.layer( layer_info_photo )
    layer_el = layout.layer( layer_info_el )

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
    Z0.place( cell, layer_photo )
    
    # left contact pad
    width1 = CHIP.width*2
    gap1 = CHIP.gap*2
    p3 = DPoint( 0, CHIP.dy/2 )
    p4 = DPoint( contact_L/2, CHIP.dy/2 )
    Z1 = CPW( width1, gap1, p3, p4 )
    Z1.place( cell, layer_photo )
    
    adapter1 = CPW2CPW( Z1, Z0, p4, p1 )
    adapter1.place( cell, layer_photo )
    
    # right contact pad
    width1 = CHIP.width*2
    gap1 = CHIP.gap*2
    p5 = DPoint( CHIP.dx - contact_L/2, CHIP.dy/2 )
    p6 = DPoint( CHIP.dx, CHIP.dy/2 )
    Z2 = CPW( width1, gap1, p5, p6 )
    Z2.place( cell, layer_photo )
    
    adapter1 = CPW2CPW( Z2, Z0, p5, p2 )
    adapter1.place( cell, layer_photo )
    
    # resonator
    L_coupling = 100e3
    L1_list = [280573., 256143., 234156., 214263., 196178., 179666., 164530., 150605., 137751.]
    r = 50e3
    L2 = 270e3
    gnd_width = 35e3
    N = 4
    width_res = 4.8e3
    gap_res = 2.6e3
    Z_res = CPW( width_res, gap_res, origin, origin )
    
    # qBit 
    origin = DPoint(1e6,0)
    a_list = [4.84883, 5.08549, 5.31162, 5.52851, 5.73721, 5.93857, 6.13333, 6.3221, 6.50538]
    for i,val in enumerate(a_list):
        a_list[i] = val*1e3
    a = 4.6e3
    b = 4.6e3
    jos1_b = 0.32e3
    jos1_a = 0.1e3
    f1 = 0.13e3
    d1 = 0.05e3
    jos2_b = 0.8e3
    jos2_a = 0.1e3
    f2 = 0.25e3
    d2 = d1
    w = 0.2e3
    B1_width = 7.325e3
    B1_height = 2.405e3
    B2_width = 0.6e3
    B5_width = 0.4e3
    B6_width = 2.5e3
    B6_height = 10e3
    B7_width = 5.051e3
    B7_height = 4.076e3
    dCap = 1.e3
    gap = 25e3
    #cell.shapes( layer_i ).insert( pya.Box( Point(0,0), Point( 40e3, 1e3 ) ) )
    qbit_params = [a,b,
                            jos1_b,jos1_a,f1,d1,
                            jos2_b,jos2_a,f2,d2,
                            w,B1_width,B1_height,
                            B2_width,B5_width,
                            B6_width,B6_height,
                            B7_width,B7_height,
                            dCap,gap]

    
    # place bottom array of resonators with qBits
    N_bottom = 5
    delta = 1e6
    step_bot = (CHIP.dx - 2*contact_L - 2*delta)/(N_bottom-1)
    resonators = []
    qbits = []
    empties = []
    for i in range( N_bottom ):
        point = DPoint( contact_L + delta + i*step_bot, CHIP.dy/2 - Z0.b/3 )
        L_coupling = 100e3
        L1 = 280e3
        r = 50e3
        L2 = 270e3
        N = 4
        width_res = 4.8e3
        gap_res = 2.6e3
        Z_res = CPW( width_res, gap_res, origin, origin )
        worm = EMResonator_TL2Qbit_worm( Z_res, point, L_coupling, L1_list[i], r, L2, N, gnd_width )
        worm.place( cell, layer_photo )
        resonators.append( worm )    
        
        qbit_bbox = pya.Box().from_dbox( qbit_bbox )
        h = qbit_bbox.b()
        reg = worm.gnd_reg + ( Region( pya.Box( qbit_bbox.p1 - Point(2*h,2*h), qbit_bbox.p2 + Point( 2*h,dCap ) ) ) - Region( qbit.metal_region.bbox() ))
        new_reg = Region()
      #  reg.merged_semantics=False
        r_cell = Region( cell.begin_shapes_rec( layer_photo ) )    
        reg = reg & r_cell
        for poly in reg:
            poly_temp = fill_holes( poly )
            new_reg.insert( poly_temp )
            
    #    r_cell.merged_semantics=False   
        r_cell = r_cell - reg
        r_cell = r_cell + new_reg
        temp_i = cell.layout().layer( pya.LayerInfo(PROGRAM.LAYER1_NUM,0) ) 
        cell.shapes( temp_i ).insert( r_cell )
        cell.layout().clear_layer( layer_photo )
        cell.layout().move_layer( temp_i, layer_photo )
        cell.layout().delete_layer( temp_i )
    # place top array of resonators with qBits
    N_top = N_bottom - 1
    delta = 1e6
    step_top = step_bot
    resonators = []
    qbits = []
    empties = []
    for i in range( N_top ):
        point = DPoint( contact_L + delta + step_bot/2 + i*step_top, CHIP.dy/2 + Z0.b/3 )
        L_coupling = 100e3
        L1 = 280e3
        r = 50e3
        L2 = 270e3
        N = 4
        width_res = 4.8e3
        gap_res = 2.6e3
        Z_res = CPW( width_res, gap_res, origin, origin )
        worm = EMResonator_TL2Qbit_worm( Z_res, point, L_coupling, L1_list[i + N_bottom], r, L2, N, gnd_width, Trans( Trans.M0 ) )
        worm.place( cell, layer_photo )
        resonators.append( worm )    
        
        qbit_params = [a_list[N_bottom + i],a_list[N_bottom + i],
                            jos1_b,jos1_a,f1,d1,
                            jos2_b,jos2_a,f2,d2,
                            w,B1_width,B1_height,
                            B2_width,B5_width,
                            B6_width,B6_height,
                            B7_width,B7_height,
                            dCap,gap]
        qbit_start = worm.end + DPoint( B1_width/2,0 )
        qbit = QBit_Flux_2( qbit_start, qbit_params, DCplxTrans( 1,180,False,0,0 ) )
        qbit.place( cell, layer_el )
        qbits.append(qbit)
        
        qbit_bbox = pya.DBox().from_ibox( qbit.metal_region.bbox() )
        empty = CPW(0, qbit_bbox.width() / 2 + 2 * qbit.width, worm.end, worm.end - DPoint(0, (qbit.p11 + DPoint(qbit.B6_width / 2, qbit.B6_height / 2)).y))
        empty.place( cell, layer_photo )
        empties.append( empty )
        
        qbit_bbox = pya.Box().from_dbox( qbit_bbox )
        h = qbit_bbox.b()
        reg = worm.gnd_reg + ( Region( pya.Box( qbit_bbox.p1 - Point(2*h,dCap), qbit_bbox.p2 + Point( 2*h,2*h ) ) ) - Region( qbit.metal_region.bbox() ))
        new_reg = Region()
      #  reg.merged_semantics=False
        r_cell = Region( cell.begin_shapes_rec( layer_photo ) )    
        reg = reg & r_cell
        for poly in reg:
            poly_temp = fill_holes( poly )
            new_reg.insert( poly_temp )
            
    #    r_cell.merged_semantics=False   
        r_cell = r_cell - reg
        r_cell = r_cell + new_reg
        temp_i = cell.layout().layer( pya.LayerInfo(PROGRAM.LAYER1_NUM,0) ) 
        cell.shapes( temp_i ).insert( r_cell )
        cell.layout().clear_layer( layer_photo )
        cell.layout().move_layer( temp_i, layer_photo )
        cell.layout().delete_layer( temp_i )
    ## DRAWING SECTION END ##
    lv.zoom_fit()