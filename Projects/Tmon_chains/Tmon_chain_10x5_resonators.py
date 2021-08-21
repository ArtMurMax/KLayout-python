# $description: xmon_chains
# $version: 0
# $show-in-menu


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

cell_name = "Tmon_chain_10x5_resonators"
print(cell_name)

layout = cv.layout()
layout.dbu = 0.001
if not layout.has_cell(cell_name):
    layout.create_cell(cell_name)

cv.cell_name = cell_name
cell = cv.cell

info = pya.LayerInfo(1, 0)
info2 = pya.LayerInfo(2, 0)
layer_photo = layout.layer(info)
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

### DRAW SECTION START ###

cp1 = Contact_Pad(origin=DPoint(-CHIP.dx/2, -CHIP.dy/4), feedline_cpw_params=md_cpw_params)
cp1.place(canvas)

cp2 = Contact_Pad(DPoint(-CHIP.dx/2, CHIP.dy/4), fc_cpw_params)
cp2.place(canvas)

cp3 = Contact_Pad(DPoint(-2e6, CHIP.dy/2), md_cpw_params, trans_in = DTrans.R270)
cp3.place(canvas)

cp4 = Contact_Pad(DPoint(2e6, CHIP.dy/2), md_cpw_params, trans_in = DTrans.R270)
cp4.place(canvas)

cp5 = Contact_Pad(DPoint(CHIP.dx/2, CHIP.dy/4), fc_cpw_params, trans_in = DTrans.R180)
cp5.place(canvas)

cp6 = Contact_Pad(DPoint(CHIP.dx/2, -CHIP.dy/4), md_cpw_params, trans_in = DTrans.R180)
cp6.place(canvas)

cp7 = Contact_Pad(DPoint(2e6, -CHIP.dy/2), feed_cpw_params, trans_in = DTrans.R90)
cp7.place(canvas)

cp8 = Contact_Pad(DPoint(-2e6, -CHIP.dy/2), feed_cpw_params, trans_in = DTrans.R90)
cp8.place(canvas)



# ======== Main feedline =========

turn_rad = 0.24e6
feed_segment_lenghts = [turn_rad, 1.5e6, 0.5e6, 3e6+cp7.end.x-cp8.end.x, 0.5e6, 1.5e6, turn_rad]

feedline = CPWRLPath(cp8.end, "LRLRLRLRLRLRL", feed_cpw_params, turn_rad,
                     feed_segment_lenghts, [pi/2, -pi/2, -pi/2, -pi/2, -pi/2, pi/2], trans_in = DTrans.R90)
feedline.place(canvas)


# ======= Chain loop =========

resonator_offsets = 5e3
chain_length = 10

res_cpw_params = CPWParameters(7e3, 4e3)
tmon_cpw_params = CPWParameters(20e3, 20e3)

resonators_site_span = cp7.end.x - cp8.end.x
resonators_interval = 650e3

resonators_y_positions = cp8.end.y + turn_rad*3 + feed_cpw_params.b+res_cpw_params.b/2+resonator_offsets

tmon_arm_len = 280e3
tmon_JJ_arm_len = 40e3
tmon_JJ_site_span = 8e3
tmon_coupling_pads_len = 100e3
h_jj = 200
w_jj = 100
asymmetry = 0.5

qubit_ports = []


i=-6
for i in range(-(chain_length)//2, (chain_length)//2, 1):
  coupling_length = 200e3
  res_cursor = DPoint(i*resonators_interval+resonators_interval/2, resonators_y_positions)
  print(i)
  trans_in = None if i>=0 else DTrans.M90
  claw = Claw(DPoint(0,0), res_cpw_params, 100e3, w_claw = 20e3, w_claw_pad=0, l_claw_pad = 0)
  res = CPWResonator(res_cursor, res_cpw_params, 40e3, 7+(i+4)/10, 11.45, coupling_length=450e3,
                                    meander_periods = 3, trans_in = trans_in)
  claw.make_trans(DTrans(res.end))
  claw.place(canvas)
  res.place(canvas)





### DRAW SECTION END ###
ebeam = ebeam.merge()
invert_region = Region(pya.Box(Point(-CHIP.dx/2-50e3, -CHIP.dy/2-50e3), 
                        Point(CHIP.dx/2+50e3, CHIP.dy/2+50e3)))

cell.shapes( layer_photo ).insert(invert_region - canvas)
cell.shapes( layer_el ).insert(ebeam)



lv.zoom_fit()
