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

class EMResonator_TL2Qbit_worm2(ComplexBase):
    def __init__( self, Z0, start, L_coupling, L1, r, L2, N, trans_in=None ):
        self.Z0 = Z0
        self.L_coupling = L_coupling
        self.L1 = L1
        self.r = r
        self.L2 = L2

        self.N = N
        self.primitives_gnd = {}
        super( EMResonator_TL2Qbit_worm2,self ).__init__( start, trans_in )

        self.start = self.connections[0]
        self.end = self.connections[-1]
        self.dr = self.end - self.start
        self.alpha_start = self.angle_connections[0]
        self.alpha_end = self.angle_connections[1]
        
    def init_primitives( self ):
        name = "coil0"
        setattr(self,name, Coil_type_1(self.Z0, DPoint(0,0), self.L_coupling, self.r, self.L1) )
        self.primitives[name] = getattr( self, name )      
        # coils filling        
        for i in range(self.N):
            name = "coil" + str(i+1)
            setattr( self, name, Coil_type_1( self.Z0, DPoint( -self.L1 + self.L_coupling, -(i+1)*(4*self.r) ), self.L1, self.r, self.L1 ) )
            self.primitives[name] = getattr( self, name )
            
        # draw the "tail"
        self.arc_tail = CPWArc(self.Z0, self.primitives["coil" + str(self.N)].end, -self.L1 / 2, -pi / 2)
        self.cop_tail = CPW( self.Z0.width, self.Z0.gap, self.arc_tail.end, self.arc_tail.end - DPoint( 0,self.L2 ) )
        self.cop_open_end = CPW(0, self.Z0.b / 2, self.cop_tail.end, self.cop_tail.end - DPoint(0, self.Z0.b))
        self.primitives["arc_tail"] = self.arc_tail
        self.primitives["cop_tail"] = self.cop_tail
        self.primitives["cop_open_end"] = self.cop_open_end
        
        self.connections = [DPoint(0,0), self.cop_tail.end]
        self.angle_connections = [0,self.cop_tail.alpha_end]


class EMResonator_TL2Qbit_worm2(ComplexBase):
    def __init__( self, Z0, start, L_coupling, L1, r, L2, N, trans_in=None ):
        self.Z0 = Z0
        self.L_coupling = L_coupling
        self.L1 = L1
        self.r = r
        self.L2 = L2

        self.N = N
        self.primitives_gnd = {}
        super( EMResonator_TL2Qbit_worm2,self ).__init__( start, trans_in )

        self.start = self.connections[0]
        self.end = self.connections[-1]
        self.dr = self.end - self.start
        self.alpha_start = self.angle_connections[0]
        self.alpha_end = self.angle_connections[1]
        
    def init_primitives( self ):
        name = "coil0"
        setattr(self,name, Coil_type_1(self.Z0, DPoint(0,0), self.L_coupling, self.r, self.L1) )
        self.primitives[name] = getattr( self, name )      
        # coils filling        
        for i in range(self.N):
            name = "coil" + str(i+1)
            setattr( self, name, Coil_type_1( self.Z0, DPoint( -self.L1 + self.L_coupling, -(i+1)*(4*self.r) ), self.L1, self.r, self.L1 ) )
            self.primitives[name] = getattr( self, name )
            
        # draw the "tail"
        self.arc_tail = CPWArc(self.Z0, self.primitives["coil" + str(self.N)].end, -self.L1 / 2, -pi / 2)
        self.cop_tail = CPW( self.Z0.width, self.Z0.gap, self.arc_tail.end, self.arc_tail.end - DPoint( 0,self.L2 ) )
        self.cop_open_end = CPW(0, self.Z0.b / 2, self.cop_tail.end, self.cop_tail.end - DPoint(0, self.Z0.b))
        self.primitives["arc_tail"] = self.arc_tail
        self.primitives["cop_tail"] = self.cop_tail
        self.primitives["cop_open_end"] = self.cop_open_end
        
        self.connections = [DPoint(0,0), self.cop_tail.end]
        self.angle_connections = [0,self.cop_tail.alpha_end]

class Coil_type_1(ComplexBase):
    def __init__( self, Z0, start, L1, r, L2, trans_in=None ):
        self.Z0 = Z0
        self.L1 = L1
        self.r = r
        self.L2 = L2
        super( Coil_type_1,self ).__init__( start, trans_in )
        self.start = self.connections[0]
        self.end = self.connections[-1]
        self.dr = self.end - self.start
        self.alpha_start = self.angle_connections[0]
        self.alpha_end = self.angle_connections[1]
        
    def init_primitives( self ):
            self.cop1 = CPW( self.Z0.width, self.Z0.gap, DPoint(0,0), DPoint( self.L1,0 ) )
            self.arc1 = CPWArc(self.Z0, self.cop1.end, -self.r, -pi)
            self.cop2 = CPW( self.Z0.width, self.Z0.gap, self.arc1.end, self.arc1.end - DPoint( self.L2,0 ) )
            self.arc2 = CPWArc(self.Z0, self.cop2.end, -self.r, pi)
            
            self.connections = [self.cop1.start,self.arc2.end]
            self.angle_connections = [self.cop1.alpha_start,self.arc2.alpha_end]
            self.primitives = {"cop1":self.cop1,"arc1":self.arc1,"cop2":self.cop2,"arc2":self.arc2}
            
            
class CHIP:
    dx = 0.6e6
    dy = 1.8e6
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
    x = 0.25*CHIP.dx
    y = 0.9*CHIP.dy
    p1 = DPoint( 0, y )
    p2 = DPoint( CHIP.dx, y )
    Z0 = CPW( width, gap, p1, p2 )
    Z0.place( cell, layer_photo )

    
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
    to_line = 25e3
    Z_res = CPW( width_res, gap_res, origin, origin )
    worm = EMResonator_TL2Qbit_worm2( Z_res, DPoint(x, y - to_line) , L_coupling, L1_list[0], r, L2, N )
    worm.place( cell, layer_photo )
    
    # placing holes around the resonator
    '''
    reg = worm.gnd_reg
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
    '''
    ## DRAWING SECTION END ##
    lv.zoom_fit()