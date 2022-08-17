
#OK

# Enter your Python code here
import pya
from importlib import reload

import classLib
from classLib import *
reload(classLib)
from classLib import *

from classLib.coplanars import *
from classLib.coplanars import CPWParameters, CPW, CPW2CPW
from classLib.contactPads import *


class CHIP:
    dx = 10e6
    dy = 10e6
    
    
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
cell.shapes( layer_photo ).insert( pya.Box( Point( -CHIP.dx/2, -CHIP.dy/2 ), Point( CHIP.dx/2, CHIP.dy/2 ) ) )

cp1 = ContactPad(origin = DPoint(-5e6, -2.5e6), chip_cpw_params = CPWParameters(width=20e3, gap=10e3))
cp1.place(cell, layer_photo)

cp2 = ContactPad(DPoint(-5e6, 2.5e6), chip_cpw_params = CPWParameters(width=20e3, gap=10e3))
cp2.place(cell, layer_photo)

cp3 = ContactPad(DPoint(-2.5e6, 5e6), chip_cpw_params = CPWParameters(width=20e3, gap=10e3), trans_in = DTrans.R270)
cp3.place(cell, layer_photo)

cp4 = ContactPad(DPoint(2.5e6, 5e6), chip_cpw_params = CPWParameters(width=20e3, gap=10e3), trans_in = DTrans.R270)
cp4.place(cell, layer_photo)

cp5 = ContactPad(DPoint(5e6, 2.5e6), chip_cpw_params = CPWParameters(width=20e3, gap=10e3), trans_in = DTrans.R180)
cp5.place(cell, layer_photo)

cp6 = ContactPad(DPoint(5e6, -2.5e6), chip_cpw_params = CPWParameters(width=20e3, gap=10e3), trans_in = DTrans.R180)
cp6.place(cell, layer_photo)

cp7 = ContactPad(DPoint(2.5e6, -5e6), chip_cpw_params = CPWParameters(width=20e3, gap=10e3), trans_in = DTrans.R90)
cp7.place(cell, layer_photo)

cp8 = ContactPad(DPoint(-2.5e6, -5e6), chip_cpw_params = CPWParameters(width=20e3, gap=10e3), trans_in = DTrans.R90)
cp8.place(cell, layer_photo)

#First feedline

cpw_params = CPWParameters(20e3, 10e3)

segment_lenghts = [2.5e6, cp7.end.x-cp8.end.x, 2.5e6]

feedline = CPWRLPath(cp8.end, "LRLRL", cpw_params, 100e3,
                     segment_lenghts, [-pi/2,-pi/2], trans_in = DTrans.R90)
feedline.place(cell, layer_photo)

#Second feedline

segment_lengths = [0.5e6, 1e6]
turn2 = cp1.end+DPoint(0.5e6, 1e6)
segment_lengths += [(cp5.end.y-turn2.y)*2/sqrt(2)]
segment_lengths += [cp5.end.x-turn2.x-segment_lengths[-1]*sqrt(2)/2]

feedline = CPWRLPath(cp1.end, "LRLRLRL", cpw_params, 200e3, segment_lengths, [pi / 2, -pi / 4, -pi / 4])
feedline.place(cell, layer_photo)


#Snake feedline


segment_lengths = [1e6]*5

feedline = CPWRLPath(cp2.end, "LRRLRRLRRLRRL", cpw_params, 200e3, segment_lengths, [pi / 2, pi / 2, -pi / 2, -pi / 2] * 2)
feedline.place(cell, layer_photo)


### DRAW SECTION END ###

lv.zoom_fit()