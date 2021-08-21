import pya
from math import sqrt, cos, sin, atan2, pi, copysign
from pya import Point, DPoint, DSimplePolygon, SimplePolygon, DPolygon, Polygon, Region
from pya import Trans, DTrans, CplxTrans, DCplxTrans, ICplxTrans

import numpy as np

from classLib import *
         
### START classes to be delegated to different file ###

# END classes to be delegated to different file ###

class Path_finder():
    def __init__( self, *argv, **karg ):
        self.objects = argv
        self.bbox = karg["bbox"]
        
        self.wall = None
        self.field = None
        self.start_last = None
        self.end_last = None
        self.width = None
        self.height = None
        self.Nx = 15
        self.Ny = 15
    
    def _prepare_wall( self ):
        self.wall = []
        for Z0 in self.objects:
            if( isinstance( Z0,CPW ) and Z0.dr.abs() != 0 ):
                for ny in range( 0,self.Ny ):
                    for nx in range( 0,self.Nx ):
                        x = nx*self.width
                        y = ny*self.height
                        w = Z0.width
                        drLeft = DPoint( -Z0.dr.y, Z0.dr.x )*(1/Z0.dr.abs())
                        box = pya.DBox( DPoint(x,y), DPoint( x + self.width, y + self.height ) )
                        if( (box.left - abs(drLeft.x)*2*Z0.b/2) > Z0.end.x or (box.right + abs(drLeft.x)*2*Z0.b/2) < Z0.start.x ):
                            continue
                        if( (box.bottom - abs(drLeft.y)*2*Z0.b/2) > Z0.end.y or (box.top + abs(drLeft.y)*2*Z0.b/2) < Z0.start.y ):
                            continue
                        self.wall.append( Point( nx,ny ) )
        for pt in self.wall:
            self.field[pt.x,pt.y] = -2
            
    def find_path_mur( self, start, end, Nx=15, Ny=15 ):
        self.start_last = start
        self.end_last = end
        self.Nx = Nx
        self.Ny = Ny
        self.width = self.bbox.width()/Nx
        self.height = self.bbox.b() / Ny
        self.field = np.full( (self.Nx,self.Ny), -1, dtype=np.int32 )
        field_start = Point( int(start.x/self.width), int(start.y/self.height) )
        field_end = Point( int(end.x/self.width), int(end.y/self.height) )
        
        self._prepare_wall()
        
        in_propogation = True
        self.field[ field_start.x,field_start.y ] = 0
        field_back_buffer = np.copy( self.field )
        print( self.field, "\n\n" )
        
        while( self.field[ field_end.x, field_end.y] == -1 and in_propogation):
            for x in range( 0,Nx ):
                for y in range( 0,Ny ):
                    if( self.field[x,y] >= 0 ):
                        for dx in [-1,0,1]:
                            for dy in [-1,0,1]:
                                if( (x + dx) >= Nx or (x + dx < 0) or (y + dy) >= Ny or (y + dy < 0) ):
                                    continue
                                if( dy == 0 and dx == 0 ):
                                    continue
                                elif( self.field[x + dx,y + dy] == -1 ):
                                    field_back_buffer[x+dx,y+dy] = self.field[x,y] + 1
            if( np.array_equal( self.field, field_back_buffer ) ):
                in_propogation = False
            np.copyto( self.field, field_back_buffer )
            print( self.field, "\n\n" )    
            
        path = []
        if( self.field[ field_end.x, field_end.y ] > 0 ):
            current = Point( field_end.x, field_end.y )
            while( True ):
                for dx in [-1,0,1]:
                    isNext = False
                    for dy in [-1,0,1]:
                        if( (current.x + dx) >= Nx or (current.x + dx < 0) or (current.y + dy) >= Ny or (current.y + dy < 0) ):
                            continue
                        elif( current == Point(0,0) ):
                            return path
                        else:
                            if( self.field[ current.x + dx, current.y + dy ] == (self.field[current.x,current.y] - 1) ):
                                path.append(current.dup())
                                current.x += dx
                                current.y += dy
                                isNext = True
                                break
                    if( isNext ):
                        break
        else:
            print( "No path is founded" )
            return None


    
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
    layer_i = layout.layer( info )
    layer_j = layout.layer( info2 )

    # clear this cell and layer
    cell.clear()

    # setting layout view  
    lv.select_cell(cell.cell_index(), 0)
    lv.add_missing_layers()

    origin = DPoint( 0,0 )
    width = 20e3
    gap = 25e3

    Z0 = CPW( width, gap, origin + DPoint(0.5e6,0.5e6), origin + DPoint(1e6,0.5e6) )
    Z0.place( cell, layer_i )
    start = DPoint( 0,0 )
    end = DPoint( 1e6,1e6 )
    ## pathfinding START ##

    pf = Path_finder( Z0, bbox=CHIP.box  )
    
    path = pf.find_path_mur( start, end, 100, 100 )
    ## pathfinding END ##
    
    
    ### DRAW SECTION START ###
    for y in range(0,pf.Ny):
        for x in range(0,pf.Nx):
            if( pf.field[x,y] == -2 ):
                cell.shapes( layer_j ).insert( pya.Box().from_dbox(pya.DBox(DPoint( x*pf.width,y*pf.height ), DPoint( (x+1)*pf.width,(y+1)*pf.height))) )
            if( Point(x,y) not in path ):
                cell.shapes( layer_i ).insert( pya.Box().from_dbox(pya.DBox(DPoint( x*pf.width,y*pf.height ), DPoint( (x+1)*pf.width,(y+1)*pf.height))) )
        
    ### DRAW SECTION END ###
    lv.zoom_fit()
   ### MAIN FUNCTION END ###