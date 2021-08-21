import sys
import os

import pya
from math import cos, sin, atan2, pi
from pya import Point,DPoint,DSimplePolygon,SimplePolygon, DPolygon, Polygon,  Region
from pya import Trans, DTrans, CplxTrans, DCplxTrans, ICplxTrans
from classLib.baseClasses import *
from classLib.qbits import *
from classLib.coplanars import *

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

class EMResonator_TL2Qbit_worm(ComplexBase):
    def __init__( self, Z0, start, L_coupling, L1, r, L2, N, gnd_width, trans_in=None ):
        self.Z0 = Z0
        self.gnd_width = gnd_width
        self.L_coupling = L_coupling
        self.L1 = L1
        self.r = r
        self.L2 = L2

        self.N = N
        self.primitives_gnd = {}
        super( EMResonator_TL2Qbit_worm,self ).__init__( start, trans_in )
        self.init_primitives_gnd_trans()
        self.gnd_reg = Region()
        for elem in self.primitives_gnd.values():
            self.gnd_reg = self.gnd_reg + elem.metal_region
        for i in range( self.N ):
            for elem in self.primitives["coil" + str(i+1)].primitives_gnd.values():
                elem.make_trans( self.DCplxTrans_init )
                elem.make_trans( DCplxTrans( 1,0,False,self.origin ) )
                self.gnd_reg = self.gnd_reg + elem.metal_region
        self.start = self.connections[0]
        self.end = self.connections[-1]
        self.dr = self.end - self.start
        self.alpha_start = self.angle_connections[0]
        self.alpha_end = self.angle_connections[1]
        
    def init_primitives( self ):
        # coils filling
        self.cop1_coupling = CPW( self.Z0.width, self.Z0.gap, DPoint(0,0), DPoint( self.L_coupling,0 ) )
        self.arc1_coupling = CPWArc(self.Z0, self.cop1_coupling.end, -self.r, -pi)
        self.cop2_coupling = CPW( self.Z0.width, self.Z0.gap, self.arc1_coupling.end, self.arc1_coupling.end - DPoint( self.L1,0 ) )
        self.arc2_coupling = CPWArc(self.Z0, self.cop2_coupling.end, -self.r, pi)
        self.primitives["cop1_coupling"] = self.cop1_coupling
        self.primitives["arc1_coupling"] = self.arc1_coupling
        self.primitives["cop2_coupling"] = self.cop2_coupling
        self.primitives["arc2_coupling"] = self.arc2_coupling
        
        self.r_outer = self.r + self.Z0.b / 2 + self.gnd_width / 2
        self.r_inner = self.r - self.Z0.b / 2 - self.gnd_width / 2
        self.cop1_gnd_1 = CPW(self.gnd_width, 0, DPoint(0, self.Z0.b / 2 + self.gnd_width / 2), DPoint(self.L_coupling, self.Z0.b / 2 + self.gnd_width / 2))
        self.primitives_gnd["cop1_gnd_1"] = self.cop1_gnd_1
        self.arc1_gnd_1 = CPWArc(CPW(self.gnd_width, 0, DPoint(0, 0), DPoint(0, 0)), self.cop1_gnd_1.end, -self.r_outer, -pi)
        self.primitives_gnd["arc1_gnd_1"] = self.arc1_gnd_1
        self.cop2_gnd_1 = CPW( self.gnd_width, 0, self.arc1_gnd_1.end, self.arc1_gnd_1.end - DPoint( self.L1,0 ) )
        self.primitives_gnd["cop2_gnd_1"] = self.cop2_gnd_1
        self.arc2_gnd_1 = CPWArc(CPW(self.gnd_width, 0, DPoint(0, 0), DPoint(0, 0)), self.cop2_gnd_1.end, -self.r_inner, pi)
        self.primitives_gnd["arc2_gnd_1"] = self.arc2_gnd_1
        
        self.cop1_gnd_2 = CPW(self.gnd_width, 0, -DPoint(0, self.Z0.b / 2 + self.gnd_width / 2), DPoint(self.L_coupling, -(self.Z0.b / 2 + self.gnd_width / 2)))
        self.primitives_gnd["cop1_gnd_2"] = self.cop1_gnd_2
        self.arc1_gnd_2 = CPWArc(CPW(self.gnd_width, 0, DPoint(0, 0), DPoint(0, 0)), self.cop1_gnd_2.end, -self.r_inner, -pi)
        self.primitives_gnd["arc1_gnd_2"] = self.arc1_gnd_2
        self.cop2_gnd_2 = CPW( self.gnd_width, 0, self.arc1_gnd_2.end, self.arc1_gnd_2.end - DPoint( self.L1,0 ) )
        self.primitives_gnd["cop2_gnd_2"] = self.cop2_gnd_2
        self.arc2_gnd_2 = CPWArc(CPW(self.gnd_width, 0, DPoint(0, 0), DPoint(0, 0)), self.cop2_gnd_2.end, -self.r_outer, pi)
        self.primitives_gnd["arc2_gnd_2"] = self.arc2_gnd_2

        
        for i in range(self.N):
            name = "coil" + str(i+1)
            setattr( self, name, Coil_type_1( self.Z0, DPoint( -self.L1 + self.L_coupling, -(i+1)*(4*self.r) ), self.L1, self.r, self.L2, self.gnd_width ) )
            self.primitives[name] = getattr( self, name )
            
        # draw the "tail"
        self.arc_tail = CPWArc(self.Z0, self.primitives["coil" + str(self.N)].end, -self.L1 / 2, -pi / 2)
        self.cop_tail = CPW( self.Z0.width, self.Z0.gap, self.arc_tail.end, self.arc_tail.end - DPoint( 0,self.L2 ) )
        self.cop_open_end = CPW(0, self.Z0.b / 2, self.cop_tail.end, self.cop_tail.end - DPoint(0, self.Z0.b))
        self.primitives["arc_tail"] = self.arc_tail
        self.primitives["cop_tail"] = self.cop_tail
        self.primitives["cop_open_end"] = self.cop_open_end
        
        self.arc_tail_gnd_1 = CPWArc(CPW(self.gnd_width, 0, DPoint(0, 0), DPoint(0, 0)), self.primitives["coil" + str(self.N)].end + DPoint(0, self.Z0.b / 2 + self.gnd_width / 2), -self.L1 / 2 - self.Z0.b / 2 - self.gnd_width / 2, -pi / 2)
        self.cop_tail_gnd_1 = CPW( self.gnd_width, self.Z0.gap, self.arc_tail_gnd_1.end, self.arc_tail_gnd_1.end - DPoint( 0,self.L2 ) )
        self.cop_open_end_gnd_1 = CPW(0, self.Z0.b / 2, self.cop_tail_gnd_1.end, self.cop_tail_gnd_1.end - DPoint(0, self.Z0.b))
        
        self.primitives_gnd["arc_tail_gnd_1"] = self.arc_tail_gnd_1
        self.primitives_gnd["cop_tail_gnd_1"] = self.cop_tail_gnd_1
        self.primitives_gnd["cop_open_end_gnd_1"] = self.cop_open_end_gnd_1
        
        self.arc_tail_gnd_2 = CPWArc(CPW(self.gnd_width, 0, DPoint(0, 0), DPoint(0, 0)), self.primitives["coil" + str(self.N)].end - DPoint(0, self.Z0.b / 2 + self.gnd_width / 2), -self.L1 / 2 + self.Z0.b / 2 + self.gnd_width / 2, -pi / 2)
        self.cop_tail_gnd_2 = CPW( self.gnd_width, self.Z0.gap, self.arc_tail_gnd_2.end, self.arc_tail_gnd_2.end - DPoint( 0,self.L2 ) )
        self.cop_open_end_gnd_2 = CPW(0, self.Z0.b / 2, self.cop_tail_gnd_2.end, self.cop_tail_gnd_2.end - DPoint(0, self.Z0.b))
        
        self.primitives_gnd["arc_tail_gnd_2"] = self.arc_tail_gnd_2
        self.primitives_gnd["cop_tail_gnd_2"] = self.cop_tail_gnd_2
        self.primitives_gnd["cop_open_end_gnd_2"] = self.cop_open_end_gnd_2
        
        self.connections = [self.cop1_coupling.start, self.cop_tail.end]
        self.angle_connections = [self.arc2_coupling.alpha_start,self.cop_tail.alpha_end]
        
    def init_primitives_gnd_trans( self ):
        dr_origin = DSimplePolygon( [DPoint(0,0)] )
        if( self.DCplxTrans_init is not None ):
            # constructor trans displacement
            dCplxTrans_temp = DCplxTrans( 1,0,False, self.DCplxTrans_init.disp )
            for element in self.primitives_gnd.values():
                element.make_trans( dCplxTrans_temp )
            dr_origin.transform( dCplxTrans_temp )
            
            # rest of the constructor trans functions
            dCplxTrans_temp = self.DCplxTrans_init.dup()
            dCplxTrans_temp.disp = DPoint(0,0)
            for element in self.primitives_gnd.values():
                element.make_trans( dCplxTrans_temp )
            dr_origin.transform( dCplxTrans_temp )
        
        dCplxTrans_temp = DCplxTrans( 1,0,False, self.origin )
        for element in self.primitives_gnd.values():    
            element.make_trans( dCplxTrans_temp ) # move to the origin

class Coil_type_1(ComplexBase):
    def __init__( self, Z0, start, L1, r, L2, gnd_width, trans_in=None ):
        self.gnd_width = gnd_width
        self.Z0 = Z0
        self.L1 = L1
        self.r = r
        self.L2 = L2
        self.primitives_gnd = {}
        super( Coil_type_1,self ).__init__( start, trans_in )
        self.init_primitives_gnd_trans()
        self.start = self.connections[0]
        self.end = self.connections[-1]
        self.dr = self.end - self.start
        self.alpha_start = self.angle_connections[0]
        self.alpha_end = self.angle_connections[1]
        
    def init_primitives( self ):
            self.cop1 = CPW( self.Z0.width, self.Z0.gap, DPoint(0,0), DPoint( self.L1,0 ) )
            self.arc1 = CPWArc(self.Z0, self.cop1.end, -self.r, -pi)
            self.cop2 = CPW( self.Z0.width, self.Z0.gap, self.arc1.end, self.arc1.end - DPoint( self.L1,0 ) )
            self.arc2 = CPWArc(self.Z0, self.cop2.end, -self.r, pi)
            
            self.r_outer = self.r + self.Z0.b / 2 + self.gnd_width / 2
            self.r_inner = self.r - self.Z0.b / 2 - self.gnd_width / 2
            self.cop1_gnd_1 = CPW(self.gnd_width, 0, DPoint(0, self.Z0.b / 2 + self.gnd_width / 2), DPoint(self.L1, self.Z0.b / 2 + self.gnd_width / 2))
            self.primitives_gnd["cop1_gnd_1"] = self.cop1_gnd_1
            self.arc1_gnd_1 = CPWArc(CPW(self.gnd_width, 0, DPoint(0, 0), DPoint(0, 0)), self.cop1_gnd_1.end, -self.r_outer, -pi)
            self.primitives_gnd["arc1_gnd_1"] = self.arc1_gnd_1
            self.cop2_gnd_1 = CPW( self.gnd_width, 0, self.arc1_gnd_1.end, self.arc1_gnd_1.end - DPoint( self.L1,0 ) )
            self.primitives_gnd["cop2_gnd_1"] = self.cop2_gnd_1
            self.arc2_gnd_1 = CPWArc(CPW(self.gnd_width, 0, DPoint(0, 0), DPoint(0, 0)), self.cop2_gnd_1.end, -self.r_inner, pi)
            self.primitives_gnd["arc2_gnd_1"] = self.arc2_gnd_1
            
        
            self.cop1_gnd_2 = CPW(self.gnd_width, 0, -DPoint(0, self.Z0.b / 2 + self.gnd_width / 2), DPoint(self.L1, -(self.Z0.b / 2 + self.gnd_width / 2)))
            self.primitives_gnd["cop1_gnd_2"] = self.cop1_gnd_2
            self.arc1_gnd_2 = CPWArc(CPW(self.gnd_width, 0, DPoint(0, 0), DPoint(0, 0)), self.cop1_gnd_2.end, -self.r_inner, -pi)
            self.primitives_gnd["arc1_gnd_2"] = self.arc1_gnd_2
            self.cop2_gnd_2 = CPW( self.gnd_width, 0, self.arc1_gnd_2.end, self.arc1_gnd_2.end - DPoint( self.L1,0 ) )
            self.primitives_gnd["cop2_gnd_2"] = self.cop2_gnd_2
            self.arc2_gnd_2 = CPWArc(CPW(self.gnd_width, 0, DPoint(0, 0), DPoint(0, 0)), self.cop2_gnd_2.end, -self.r_outer, pi)
            self.primitives_gnd["arc2_gnd_2"] = self.arc2_gnd_2
            
            self.connections = [self.cop1.start,self.arc2.end]
            self.angle_connections = [self.cop1.alpha_start,self.arc2.alpha_end]
            self.primitives = {"cop1":self.cop1,"arc1":self.arc1,"cop2":self.cop2,"arc2":self.arc2}

    def init_primitives_gnd_trans( self ):
        dr_origin = DSimplePolygon( [DPoint(0,0)] )
        if( self.DCplxTrans_init is not None ):
            # constructor trans displacement
            dCplxTrans_temp = DCplxTrans( 1,0,False, self.DCplxTrans_init.disp )
            for element in self.primitives_gnd.values():
                element.make_trans( dCplxTrans_temp )
            dr_origin.transform( dCplxTrans_temp )
            
            # rest of the constructor trans functions
            dCplxTrans_temp = self.DCplxTrans_init.dup()
            dCplxTrans_temp.disp = DPoint(0,0)
            for element in self.primitives_gnd.values():
                element.make_trans( dCplxTrans_temp )
            dr_origin.transform( dCplxTrans_temp )
        
        dCplxTrans_temp = DCplxTrans( 1,0,False, self.origin )
        for element in self.primitives_gnd.values():    
            element.make_trans( dCplxTrans_temp ) # move to the origin
 
class CHIP:
    dx = 0.8e6
    dy = 1.8e6
    L1 = 1e6
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
    y = 0.2*CHIP.dy
    p1 = DPoint( 0, y )
    p2 = DPoint( CHIP.dx, y )
    Z0 = CPW( width, gap, p1, p2 )
    Z0.place( cell, layer_photo )
    
    
    # resonator
    L_coupling = 300e3
    L1 = 257.7e3
    r = 50e3
    L2 = 270e3
    N = 4
    width_res = 9.6e3
    gap_res = 5.2e3
    gnd_width = 25e3
    Z_res = CPW( width_res, gap_res, origin, origin )
    worm = EMResonator_TL2Qbit_worm( Z_res, DPoint(3*CHIP.dx/4, y - Z0.b) , L_coupling, L1, r, L2, N, gnd_width, DCplxTrans(1,180,False,0,0) )
    worm.place( cell, layer_photo )   
    
     # qBit 
    origin = DPoint(1e6,0)
    a_list = []
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
    qbit_params = [a,b,
                        jos1_b,jos1_a,f1,d1,
                        jos2_b,jos2_a,f2,d2,
                        w,B1_width,B1_height,
                        B2_width,B5_width,
                        B6_width,B6_height,
                        B7_width,B7_height,
                        dCap,gap]
    qbit_start = worm.end + DPoint( -B1_width/2,0 )
    qbit = QBit_Flux_2( worm.end + DPoint( -B1_width/2,0 ), qbit_params, DCplxTrans( 1,0,False,0,0 ) )
    qbit.place( cell, layer_el )
    
   
    qbit_bbox = pya.DBox().from_ibox( qbit.metal_region.bbox() )
    empty = CPW( 0, qbit_bbox.width()/2 + 2*qbit.a, worm.end, worm.end - DPoint( 0, (qbit.p11 + DPoint( qbit.B6_width/2,qbit.B6_height/2) ).y ) )
    empty.place( cell, layer_photo )
#    r_cell.merged_semantics=False   
    qbit_bbox = pya.Box().from_dbox( qbit_bbox )
    h = qbit_bbox.b()
    reg = worm.gnd_reg + ( Region( pya.Box( qbit_bbox.p1 - Point(2*h,dCap), qbit_bbox.p2 + Point( 2*h,2*h ) ) ) - Region( qbit.metal_region.bbox() ))
    new_reg = Region()
    r_cell = Region( cell.begin_shapes_rec( layer_photo ) )    
    reg = reg & r_cell
    for poly in reg:
        poly_temp = fill_holes( poly )
        new_reg.insert( poly_temp )
    
    r_cell = r_cell - reg
    r_cell = r_cell + new_reg
    temp_i = cell.layout().layer( pya.LayerInfo(PROGRAM.LAYER1_NUM,0) ) 
    cell.shapes( temp_i ).insert( r_cell )
    cell.layout().clear_layer( layer_photo )
    cell.layout().move_layer( temp_i, layer_photo )
    cell.layout().delete_layer( temp_i )
       

    ## DRAWING SECTION END ##
    lv.zoom_fit()