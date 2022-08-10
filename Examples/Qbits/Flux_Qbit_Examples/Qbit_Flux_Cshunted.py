#OK
import pya
from math import sqrt, cos, sin, atan2, pi, copysign
from pya import Point, DPoint, DSimplePolygon, SimplePolygon, DPolygon, Polygon, Region
from pya import Trans, DTrans, CplxTrans, DCplxTrans, ICplxTrans

from classLib import *

### START classes to be delegated to different file ###
class Element_Base():
    def __init__(self, origin, trans_in=None):
        ## MUST BE IMPLEMENTED ##
        self.connections = []       # DPoint list with possible connection points
        self.angle_connections = [] #list with angle of connecting elements
        self.connection_ptrs = [] # pointers to connected structures represented by their class instances
        ## MUST BE IMLPEMENTED END ##
        self.origin = origin
        self.metal_region = Region()
        self.empty_region = Region()
        self.metal_regions = {}
        self.empty_regions = {}        
        self.metal_regions["default"] = self.metal_region
        self.empty_regions["default"] = self.empty_region
        
        self.metal_region.merged_semantics = False
        self.empty_region.merged_semantics = False
        self.DCplxTrans_init = None
        self.ICplxTrans_init = None
        
        if( trans_in is not None ):
        # if( isinstance( trans_in, ICplxTrans ) ): <==== FORBIDDEN
            if( isinstance( trans_in, DCplxTrans ) ):
                self.DCplxTrans_init = trans_in
                self.ICplxTrans_init = ICplxTrans().from_dtrans( trans_in )
            elif( isinstance( trans_in, CplxTrans ) ):
                self.DCplxTrans_init = DCplxTrans().from_itrans( trans_in )
                self.ICplxTrans_init = ICplxTrans().from_trans( trans_in )
            elif( isinstance( trans_in, DTrans ) ):
                self.DCplxTrans_init = DCplxTrans( trans_in, 1 )
                self.ICplxTrans_init = ICplxTrans( Trans().from_dtrans( trans_in ), 1 )
            elif( isinstance( trans_in, Trans ) ):
                self.DCplxTrans_init = DCplxTrans( DTrans().from_itrans( trans_in ), 1 )
                self.ICplxTrans_init = ICplxTrans( trans_in, 1 )
                
        self._init_regions_trans()
    
    def init_regions( self ):
        raise NotImplementedError
    
    # first it makes trans_init displacement
    # then the rest of the trans_init
    # then displacement of the current state to the origin
    # after all, origin should be updated
    def _init_regions_trans( self ):
        self.init_regions()         # must be implemented in every subclass
        dr_origin = DSimplePolygon( [DPoint(0,0)] )
        if( self.DCplxTrans_init is not None ):
            # constructor trans displacement
            dCplxTrans_temp = DCplxTrans( 1,0,False, self.DCplxTrans_init.disp )
            self.make_trans( dCplxTrans_temp )
            dr_origin.transform( dCplxTrans_temp )
            
            # rest of the constructor trans functions
            dCplxTrans_temp = self.DCplxTrans_init.dup()
            dCplxTrans_temp.disp = DPoint(0,0)
            self.make_trans( dCplxTrans_temp )
            dr_origin.transform( dCplxTrans_temp )                  
            
        # translation to the old origin (self.connections are alredy contain proper values)
        self.make_trans( DCplxTrans( 1,0,False, self.origin ) ) # move to the origin
        self.origin += dr_origin.point( 0 )
        
    def make_trans( self, dCplxTrans ):
        if( dCplxTrans is not None ):
            iCplxTrans = ICplxTrans().from_dtrans( dCplxTrans )
            for metal_region, empty_region in zip(self.metal_regions.values(), self.empty_regions.values()):            
                metal_region.transform( iCplxTrans )
                empty_region.transform( iCplxTrans )
            self._update_connections( dCplxTrans )
            self._update_alpha( dCplxTrans )
    
    def _update_connections( self, dCplxTrans ):       
        if( dCplxTrans is not None ):
            # the problem is, if k construct polygon with multiple points
            # their order in poly_temp.each_point() doesn't coinside with the 
            # order of the list that was passed to the polygon constructor
            # so, when k perform transformation and try to read new values through poly_temp.each_point()
            # they values are rearranged
            # solution is: k need to create polygon for each point personally, and the initial order presists
            for i,pt in enumerate(self.connections):
                poly_temp = DSimplePolygon( [pt] )
                poly_temp.transform( dCplxTrans )
                self.connections[i] = poly_temp.point( 0 )
    
    def _update_alpha( self, dCplxTrans ):
        if( dCplxTrans is not None ):
            dCplxTrans_temp = dCplxTrans.dup()
            dCplxTrans_temp.disp = DPoint(0,0)
            
            for i,alpha in enumerate(self.angle_connections):
                poly_temp = DSimplePolygon( [DPoint( cos(alpha), sin(alpha) )] )
                poly_temp.transform( dCplxTrans_temp )
                pt = poly_temp.point( 0 )
                self.angle_connections[i] = atan2( pt.y, pt.x )
    
    def _update_origin( self, dCplxTrans ):
        if( dCplxTrans is not None ):     
            poly_temp = DSimplePolygon( [self.origin] )
            poly_temp.transform( dCplxTrans )
            self.origin = poly_temp.point( 0 )
    
    def place( self, dest, layer_i=-1, region_name=None ):
        r_cell = None
        if( layer_i != -1 ): 
            metal_region = None
            empty_region = None
            if( region_name == None ):
                metal_region = self.metal_region
                empty_region = self.empty_region
            else:
                metal_region = self.metal_regions[region_name]
                empty_region = self.empty_regions[region_name]

            r_cell = Region( dest.begin_shapes_rec( layer_i ) )        
            temp_i = dest.layout().layer( pya.LayerInfo(PROGRAM.LAYER1_NUM,0) ) 
            dest.shapes( temp_i ).insert( r_cell + metal_region  - empty_region )
            dest.layout().clear_layer( layer_i )
            dest.layout().move_layer( temp_i, layer_i )
            dest.layout().delete_layer( temp_i )
        if( layer_i == -1 ): # dest is interpreted as instance of Region() class
            for metal_region,empty_region in zip(self.metal_regions.values(),self.empty_regions.values()):
                dest += metal_region
                dest -= empty_region
            


class QBit_Flux_Сshuted( Element_Base ):
    def __init__( self, origin, params, trans_in=None ):
        self.params = params
        self.a = params[0]
        self.b = params[1]
        self.jos1_b = params[2]
        self.jos1_a = params[3]
        self.f1 = params[4]
        self.d1 = params[5]
        self.jos2_b = params[6]
        self.jos2_a = params[7]
        self.f2 = params[8]
        self.d2 = params[9]
        self.w = params[10]
        self.dCap = params[11]
        self.gap = params[12]        
        self.square_a = params[13]
        self.dSquares = params[14]
        self.alum_over = params[15]
        self.B1_width = params[16]
        
        # calculated parameters
        self.B2_width = self.a + self.jos1_a/2 - self.jos2_a/2 - self.w
        self.qbit_width = 2*self.w + self.B2_width
        self.B3_width = self.a + self.jos2_a/2 - self.jos1_a/2 - self.w
        self.B1_height = (self.dSquares - 2*self.w - self.b)/2
        self._alpha_1 = 0.5
        self._length1 = (self.b + 2*self.w + self.jos1_b)
        self._alpha_2 = 0.5
        self._length2 = (self.b + 2*self.w - 2*self.f2 - self.jos2_b) #length
        
        self.p0 = DPoint(0,0)
        self.p1 = self.p0 - DPoint( 0, self.dCap + self.square_a )
        self.p2 = self.p1 + DPoint( self.square_a/2 - self.qbit_width/2, -(self.dCap + self.B1_height) )
        self.p3 = self.p2 + DPoint( self.qbit_width - self.w , 0 )
        self.p4 = self.p2 + DPoint( self.w, -self.w )
        self.p5 = self.p3 - DPoint( 0,self.f2 + self._length2*self._alpha_2 ) 
        self.p6 = self.p5 + DPoint( 2*self.w + self.jos2_a, -self.jos2_b )
        self.p7 = self.p6 - DPoint( 0,self.f2 + (1-self._alpha_2)*self._length2 )
        self.p8 = self.p7 - DPoint( self.w + self.B3_width,0 ) 
        self.p9 = self.p7 - DPoint( self.w + (self.qbit_width + self.B3_width)/2,self.B1_height )
        self.p10 = self.p9 + DPoint( -self.square_a/2 + self.B1_width/2, -self.square_a )
        
        self.SQ1 = pya.DBox( self.p1, self.p1 + DPoint( self.square_a, self.square_a ) )
        self._B1p1 = self.p2 + DPoint( self.qbit_width/2 - self.B1_width/2,0 )
        self.B1 = pya.DBox( self._B1p1, self._B1p1+ DPoint( self.B1_width,self.B1_height + self.alum_over ) )
        self.B2 = pya.DBox( self.p4, self.p3 )
        self.B3 = pya.DBox( self.p8, self.p8 + DPoint( self.B3_width, self.w ) )
        self._B4p1  = self.p9 + DPoint( self.qbit_width/2 - self.B1_width/2,-self.alum_over )
        self.B4 = pya.DBox( self._B4p1, self._B4p1 + DPoint( self.B1_width, self.B1_height + self.alum_over ) )
        self.SQ2 = pya.DBox( self.p10, self.p10 + DPoint( self.square_a, self.square_a ) )
    
        self.poly_1 = self._make_polygon( self._length1*self._alpha_1, self.w, self.d1, self.f1, self.jos1_b )
        self.poly_1.transform( DCplxTrans( 1.0, 270, False, self.p2 ) )
        self.poly_2 = self._make_polygon( self._length1*(1-self._alpha_1), self.w, self.d1, self.f1, self.jos1_b )
        self.poly_2.transform( DCplxTrans( 1.0, 90, False, self.p8 ) )
        self.poly_3 = self._make_polygon( self._length2*self._alpha_2, self.w, self.d2, self.f2, self.jos2_b )
        self.poly_3.transform( DCplxTrans( 1.0, 270, False, self.p3 ) )
        self.poly_4 = self._make_polygon( 2*self.jos2_b + self.f2, self.w, self.d2, self.f2, self.jos2_b )
        self.poly_4.transform( DCplxTrans( 1.0, 90, False, self.p6 ) ) 
        self.poly_5 = self._make_polygon( 2*self.jos2_b + self.f2, self.w, self.d2, self.f2, self.jos2_b )
        self.poly_5.transform( DCplxTrans( 1.0, 270, False, self.p5 ) )           
        self.poly_6 = self._make_polygon( (1-self._alpha_2)*self._length2, self.w, self.d2, self.f2, self.jos2_b )
        self.poly_6.transform( DCplxTrans( 1.0, 90, False, self.p7 ) )               
        super( QBit_Flux_Сshuted,self ).__init__( origin, trans_in )
        
    def _make_polygon( self, length, w, d, f, overlapping ):
        polygon = DSimplePolygon
        p1 = DPoint(0,0)
        p2 = p1 + DPoint( length,0 )
        p3 = p2 + DPoint( 0, w )
        p4 = p3 - DPoint( overlapping,0 )
        p5 = p4 - DPoint( 0, d )
        p6 = p5 - DPoint( f,0 )
        p7 = p6 + DPoint( 0, d )
        p8 = p1 + DPoint( 0,w )
        
        polygon = DSimplePolygon( [p1,p2,p3,p4,p5,p6,p7,p8] )
        return polygon
        
    def init_regions( self ):
        # photolitography regions 
        self.metal_regions["photo"] = Region()
        self.empty_regions["el"] = Region()
        # electron-beam litography regions
        self.metal_regions["el"] = Region() 
        self.empty_regions["photo"] = Region()
    
        self.metal_regions["photo"].insert( pya.Box().from_dbox(self.SQ1) ) 
        self.metal_regions["el"].insert( pya.Box().from_dbox(self.B1) ) 
        self.metal_regions["el"].insert( pya.Box().from_dbox(self.B2) )
        self.metal_regions["el"].insert( pya.Box().from_dbox(self.B3) )
        self.metal_regions["el"].insert( pya.Box().from_dbox(self.B4) )
        self.metal_regions["photo"].insert( pya.Box().from_dbox(self.SQ2) )
        
        self.metal_regions["el"].insert( SimplePolygon(self.poly_1) )
        self.metal_regions["el"].insert( SimplePolygon(self.poly_2) )
        self.metal_regions["el"].insert( SimplePolygon(self.poly_3) )
        self.metal_regions["el"].insert( SimplePolygon(self.poly_4) )
        self.metal_regions["el"].insert( SimplePolygon(self.poly_5) )
        self.metal_regions["el"].insert( SimplePolygon(self.poly_6) )
    
    # overwritig parent method "place" to be able to draw on 2 different layers simultaneously
    def place( self, cell, layer_photo, layer_el ):
        # placing photolitography
        super( QBit_Flux_Сshuted,self ).place( cell, layer_photo, "photo" )
        
        # placing electron-beam litography
        super( QBit_Flux_Сshuted,self ).place( cell, layer_el, "el" )
# END classes to be delegated to different file ###


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
    qbit = QBit_Flux_Сshuted( DPoint(1e6,1e6), qbit_params, DCplxTrans( 1,0,False,0,0 ) )
    print( qbit.origin )
    qbit.place( cell, layer_photo, layer_el )
    ### DRAW SECTION END ###
    
    lv.zoom_fit()