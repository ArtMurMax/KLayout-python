# Enter your Python code here
import pya
from importlib import reload
import classLib
from classLib import *

reload(baseClasses)
reload(capacitors)
reload(coplanars)
reload(JJ)
reload(qbits)
reload(resonators)
reload(shapes)
reload(ContactPad)
reload(Claw)
reload(Tmon)
reload(FluxCoil)
reload(_PROG_SETTINGS)
from classLib import *

from classLib.ContactPad import *
from classLib.claw import *
from classLib.resonators import *
from classLib.tmon import *
from classLib.fluxCoil import *

from time import time

class CHIP:
    dx = 10e6
    dy = 5e6


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
    
print(cv)
cell_name = "TmonCapTest"

layout = cv.layout()
layout.dbu = 0.001
if not layout.has_cell(cell_name):
  layout.create_cell(cell_name)

cv.cell_name = cell_name
cell = cv.cell

info = pya.LayerInfo(1,0)
info2 = pya.LayerInfo(2,0)
layer_photo = layout.layer( info )
layer_el = layout.layer( info2 )

# clear this cell and layer
cell.clear()

# setting layout view
#lv.select_cell(cell.cell_index(), 0)
lv.add_missing_layers()


#Constants

ground = pya.Box(Point(-CHIP.dx/2, -CHIP.dy/2), Point(CHIP.dx/2, CHIP.dy/2))
canvas = Region(ground)

ebeam = Region()

feed_cpw_params = CPWParameters(20e3, 10e3)
md_cpw_params = CPWParameters(7e3, 4e3)
fc_cpw_params = CPWParameters(7e3, 4e3)

tmon_arm_len = 280e3
tmon_JJ_arm_len = 40e3
tmon_JJ_site_span = 8e3
tmon_coupling_pads_len = 100e3
h_jj = 200
w_jj = 100
asymmetry = 0.5
chain_length = 8

for i in range(-(chain_length)//2, (chain_length)//2, 1):
  tmon_cpw_params = CPWParameters(20e3, 10e3*(i+4)/4)

  tmon = Tmon(DPoint(1e6*i, 0), tmon_cpw_params, tmon_arm_len, \
                tmon_JJ_arm_len, tmon_JJ_site_span, tmon_coupling_pads_len, \
                  h_jj, w_jj, asymmetry, None)

  tmon.place(canvas, region_id="photo")
  tmon.place(ebeam, region_id="ebeam")


  qubit_ports.append(tmon.end)
  
### DRAW SECTION END ###
ebeam = ebeam.merge()
cell.shapes( layer_photo ).insert(canvas)
cell.shapes( layer_el ).insert(ebeam)