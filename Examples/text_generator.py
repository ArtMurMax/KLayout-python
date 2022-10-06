#OK
from pya import Trans, DTrans, CplxTrans, DCplxTrans, ICplxTrans
from pya import Text

from classLib import *

import numpy as np

if( __name__ == "__main__" ):
    # getting main references of the appl

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
    X_SIZE = 200e3
    Y_SIZE = 200e3
    
    # Box drwaing START #
    box = pya.Box( 0,0, X_SIZE,Y_SIZE )
    box_reg = Region( box )
    text_reg = pya.TextGenerator.default_generator().text("A TEXT", 0.001,20,False,-0.5,5)
    p0 = Vector( X_SIZE/3, Y_SIZE/3 )
    text_reg.transform( ICplxTrans( 1.0, 45, True, p0 ) )
    box_reg -= text_reg
    cell.shapes( layer_ph ).insert( box_reg )
    ### DRAW SECTION END ###
    lv.zoom_fit()