# $show-in-menu
# $group-chip_name: Macros
# $menu-path: Chain_bridged
# Enter your Python code here

# $description: xmon_chains
# $version: 0
# $show-in-menu


# Enter your Python code here
import sys

sys.path.append(r"C:\Users\user\Desktop\lab\Klayout-python")
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
reload(Airbridge)
reload(BridgedCoplanars)
reload(_PROG_SETTINGS)
from classLib import *

from classLib.ContactPad import *
from classLib.claw import *
from classLib.resonators import *
from classLib.tmon import *
from classLib.fluxCoil import *
from classLib.airbridge import *
from time import time
from pya import DText


class CHIP:
    dx = 10e6
    dy = 10e6


app = pya.Application.instance()
mw = app.main_window()
lv = mw.current_view()
cv = None

# this insures that lv and cv are valid objects
if (lv == None):
    cv = mw.create_layout(1)
    lv = mw.current_view()
else:
    cv = lv.active_cellview()

cell_name = "Tmon_5Q_chain_10x5"
print(cell_name)

for cell in list(cv.layout().each_cell()):
    print(cell.name)
    if "SQUID" in cell.name:
        cv.layout().delete_cell(cell.cell_index())

layout = cv.layout()
layout.dbu = 0.001
if not layout.has_cell(cell_name):
    layout.create_cell(cell_name)

cv.cell_name = cell_name
cell = cv.cell
lv.clear_layers()
cell.clear()
for layer_idx in layout.layer_indexes():
    layout.delete_layer(layer_idx)

layer_photo = layout.layer(_PROG_SETTINGS.LAYERS.photo)
layer_el = layout.layer(_PROG_SETTINGS.LAYERS.ebeam)
layer_negative = layout.layer(_PROG_SETTINGS.LAYERS.photo_neg)
layer_crystal = layout.layer(_PROG_SETTINGS.LAYERS.crystal)
layer_bridges = layout.layer(_PROG_SETTINGS.LAYERS.bridges)
layer_bridge_patches = layout.layer(_PROG_SETTINGS.LAYERS.bridge_patches)

lv.add_missing_layers()
# clear this cell and layer

# setting layout view
# lv.select_cell(cell.cell_index(), 0)


# Constants

crystal_bounds = pya.Box(Point(-12.5e6, -12.5e6), Point(12.5e6, 12.5e6))
cell.shapes(layer_crystal).insert(Region(crystal_bounds))

ground = pya.Box(Point(-CHIP.dx / 2, -CHIP.dy / 2), Point(CHIP.dx / 2, CHIP.dy / 2))
canvas = Region(ground)
negative = Region()
ebeam = Region()
bridges = Region()
bridge_patches = Region()

invert_region = Region(pya.Box(Point(-CHIP.dx / 2 - 50e3, -CHIP.dy / 2 - 50e3),
                               Point(CHIP.dx / 2 + 50e3, CHIP.dy / 2 + 50e3)))

feed_cpw_params = CPWParameters(20e3, 10e3)
md_cpw_params = CPWParameters(7e3, 4e3)
fc_cpw_params = CPWParameters(7e3, 4e3)

### DRAW SECTION START ###

cp1 = Contact_Pad(origin=DPoint(-CHIP.dx / 2, -CHIP.dy / 4), feedline_cpw_params=md_cpw_params)
cp1.place(canvas)

cp2 = Contact_Pad(DPoint(-CHIP.dx / 2, 0), fc_cpw_params)
cp2.place(canvas)

cp3 = Contact_Pad(origin=DPoint(-CHIP.dx / 2, CHIP.dy / 4), feedline_cpw_params=md_cpw_params)
cp3.place(canvas)

cp4 = Contact_Pad(DPoint(-CHIP.dx/4, CHIP.dy / 2), md_cpw_params, trans_in=DTrans.R270)
cp4.place(canvas)

cp5 = Contact_Pad(origin=DPoint(0, CHIP.dy / 2), feedline_cpw_params=md_cpw_params, trans_in=DTrans.R270)
cp5.place(canvas)

cp6 = Contact_Pad(DPoint(CHIP.dx/4, CHIP.dy / 2), md_cpw_params, trans_in=DTrans.R270)
cp6.place(canvas)

cp7 = Contact_Pad(DPoint(CHIP.dx / 2, CHIP.dy / 4), fc_cpw_params, trans_in=DTrans.R180)
cp7.place(canvas)

cp8 = Contact_Pad(DPoint(CHIP.dx / 2, 0), fc_cpw_params, trans_in=DTrans.R180)
cp8.place(canvas)

cp9 = Contact_Pad(DPoint(CHIP.dx / 2, -CHIP.dy / 4), md_cpw_params, trans_in=DTrans.R180)
cp9.place(canvas)

cp10 = Contact_Pad(DPoint(CHIP.dx/4, -CHIP.dy / 2), feed_cpw_params, trans_in=DTrans.R90)
cp10.place(canvas)

cp11 = Contact_Pad(DPoint(0, -CHIP.dy / 2), feed_cpw_params, trans_in=DTrans.R90)
cp11.place(canvas)

cp12 = Contact_Pad(DPoint(-CHIP.dx/4, -CHIP.dy / 2), feed_cpw_params, trans_in=DTrans.R90)
cp12.place(canvas)

# ======== Main feedline =========

turn_rad = 0.24e6
feed_segment_lenghts = [turn_rad, 2 * turn_rad, 0.5e6, 4 * turn_rad + cp7.end.x - cp8.end.x, 0.5e6, 2 * turn_rad,
                        turn_rad]

feedline = CPWRLPath(cp8.end, "LRLRLRLRLRLRL", feed_cpw_params, turn_rad,
                     feed_segment_lenghts, [pi / 2, -pi / 2, -pi / 2, -pi / 2, -pi / 2, pi / 2], trans_in=DTrans.R90)
feedline.place(canvas)

# ======= Chain loop =========

resonator_offsets = 5e3
chain_length = 5

res_cpw_params = CPWParameters(7e3, 4e3)
tmon_cpw_params = CPWParameters(20e3, 20e3)

resonators_site_span = cp7.end.x - cp8.end.x
resonators_interval = 650e3

resonators_y_positions = cp8.end.y + turn_rad * 3 + feed_cpw_params.b + res_cpw_params.b / 2 + resonator_offsets

tmon_arm_len = 280e3
tmon_JJ_arm_len = 40e3
tmon_JJ_site_span = 8e3
tmon_coupling_pads_len = 100e3
h_jj = 240
w_jj = 240
asymmetry = 0.7

qubit_ports = []

for i in range(-(chain_length) // 2, (chain_length) // 2, 1):
    coupling_length = 200e3
    res_cursor = DPoint(i * resonators_interval + resonators_interval,
                        resonators_y_positions)
    trans_in = None if i >= 0 else DTrans.M90
    claw = Claw(DPoint(0, 0), res_cpw_params, 100e3, w_claw=20e3, w_claw_pad=0, l_claw_pad=0)
    res = CPWResonator(res_cursor, res_cpw_params, 40e3, 7.3 + (i + 4) / 10, 11.45, coupling_length=450e3,
                       meander_periods=3, trans_in=trans_in)
    print(i, 7.3 + (i + 3) / 10)
    claw.make_trans(DTrans(res.end))
    ##resonator_length = CPWResonator._calculate_total_length(res)
    ##print(resonator_length)
    claw.place(canvas)
    res.place(canvas)

    tmon = Tmon(claw.connections[1], tmon_cpw_params, tmon_arm_len, \
                tmon_JJ_arm_len, tmon_JJ_site_span, tmon_coupling_pads_len, \
                h_jj, w_jj, asymmetry, None, True)

    tmon.place(canvas, region_name="photo")
    tmon.place(ebeam, region_name="ebeam")

    qubit_ports.append(tmon.end)

claw_test1 = Claw(DPoint(0, 0), res_cpw_params, 100e3, w_claw=20e3, w_claw_pad=0, l_claw_pad=0)
res_cursor_test1 = DPoint(-4 * resonators_interval + resonators_interval,
                          resonators_y_positions)

res_test1 = CPWResonator(res_cursor_test1, res_cpw_params, 40e3, 7.8, 11.45, coupling_length=450e3,
                         meander_periods=3, trans_in=DTrans.M90)

claw_test1.make_trans(DTrans(res_test1.end))
claw_test1.place(canvas)
res_test1.place(canvas)

claw_test2 = Claw(DPoint(0, 0), res_cpw_params, 100e3, w_claw=20e3, w_claw_pad=0, l_claw_pad=0)
res_cursor_test2 = DPoint(2 * resonators_interval + resonators_interval,
                          resonators_y_positions)

res_test2 = CPWResonator(res_cursor_test2, res_cpw_params, 40e3, 7.9, 11.45, coupling_length=450e3,
                         meander_periods=3, trans_in=None)
claw_test2.make_trans(DTrans(res_test2.end))
claw_test2.place(canvas)
res_test2.place(canvas)

# ====== Microwave drives ========


# tmon1_md_segment_lengths =\
#   [300e3, qubit_ports[0].y - cp1.end.y-tmon_JJ_arm_len - tmon_JJ_site_span-tmon_cpw_params.width/2,
#    qubit_ports[0].x - cp1.end.x-300e3 - tmon_arm_len*2]
# tmon1_md = CPWDPath(cp1.end, "LRLRL", md_cpw_params, 240e3,
#   tmon1_md_segment_lengths, [pi/2, -pi/2] ,trans_in = None)
# tmon1_md.place(canvas)

# tmon1_md_end = CPW(0, md_cpw_params.b/2, tmon1_md.end, tmon1_md.end+DPoint(4e3, 0))
# tmon1_md_end.place(canvas)

# tmon_m1_md_segment_lengths =\
#   [300e3, qubit_ports[-1].y - cp6.end.y-tmon_JJ_arm_len - tmon_JJ_site_span-tmon_cpw_params.width/2,
#    -qubit_ports[-1].x + cp6.end.x-300e3 - tmon_arm_len*2]
# tmon_m1_md = CPWDPath(cp6.end, "LRLRL", md_cpw_params, 240e3,
#   tmon_m1_md_segment_lengths, [pi/2, -pi/2] ,trans_in = DTrans.M90)
# tmon_m1_md.place(canvas)

# tmon_m1_md_end = CPW(0, md_cpw_params.b/2, tmon_m1_md.end, tmon_m1_md.end+DPoint(-4e3, 0))
# tmon_m1_md_end.place(canvas)

# ======= Flux coils =======


tmon1_fc_segment_lengths = \
    [(qubit_ports[0].x - cp1.end.x) / 3, -cp1.end.y + qubit_ports[0].y + 0.35e6,
     (qubit_ports[0].x - cp1.end.x) / 2 + 0.495e6 + 2e3, (-cp1.end.y + qubit_ports[0].y) / 3 - 120e3 + 7.5e3]
tmon1_fc = CPWRLPath(cp1.end, "LRLRLRL", fc_cpw_params, 240e3,
                     tmon1_fc_segment_lengths, [pi / 2, -pi / 2, -pi / 2], trans_in=None, bridged=True)
tmon1_fc.place(canvas, region_name="photo")
tmon1_fc.place(bridges, region_name="bridges")
tmon1_fc.place(bridge_patches, region_name="bridge_patches")

tmon1_fc_end = FluxCoil(tmon1_fc.end, fc_cpw_params, width=20e3, trans_in=DTrans.R180)
tmon1_fc_end.place(canvas)

tmon2_fc_segment_lengths = \
    [0.5e6, (qubit_ports[1].x - cp2.end.x) - 0.5e6, -(-cp2.end.y + qubit_ports[1].y) / 2 + 0.26e6 + 4.25e3]
tmon2_fc = CPWRLPath(cp2.end, "LRRLRL", fc_cpw_params, 150e3,
                     tmon2_fc_segment_lengths, [-pi / 2, pi / 2, -pi / 2], trans_in=None, bridged=True)
tmon2_fc.place(canvas, region_name="photo")
tmon2_fc.place(bridges, region_name="bridges")
tmon2_fc.place(bridge_patches, region_name="bridge_patches")

tmon1_fc_end = FluxCoil(tmon2_fc.end, fc_cpw_params, width=20e3, trans_in=DTrans.R180)
tmon1_fc_end.place(canvas)

tmon3_fc_segment_lengths = \
    [(-cp3.end.y + qubit_ports[2].y) / 50, 3 * (-cp3.end.x + qubit_ports[2].x) / 4 + 0.5e6,
     -(-cp3.end.y + qubit_ports[2].y) / 3 + 0.775e6 + 2.2e3]
tmon3_fc = CPWRLPath(cp3.end, "LRLRL", fc_cpw_params, 150e3,
                     tmon3_fc_segment_lengths, [pi / 2, -pi / 2], trans_in=DTrans.R90, bridged=True)
tmon3_fc.place(canvas, region_name="photo")
tmon3_fc.place(bridges, region_name="bridges")
tmon3_fc.place(bridge_patches, region_name="bridge_patches")

tmon1_fc_end = FluxCoil(tmon3_fc.end, fc_cpw_params, width=20e3, trans_in=DTrans.R180)
tmon1_fc_end.place(canvas)

tmon4_fc_segment_lengths = \
    [(-cp4.end.y + qubit_ports[3].y) / 50, -3 * (-cp4.end.x + qubit_ports[3].x) / 4 + 0.34e6 - 3e3,
     -(-cp4.end.y + qubit_ports[3].y) / 3 + 0.771e6 + 6.2e3]
tmon4_fc = CPWRLPath(cp4.end, "LRLRL", fc_cpw_params, 150e3,
                     tmon4_fc_segment_lengths, [-pi / 2, pi / 2], trans_in=DTrans.R90, bridged=True)
tmon4_fc.place(canvas, region_name="photo")
tmon4_fc.place(bridges, region_name="bridges")
tmon4_fc.place(bridge_patches, region_name="bridge_patches")

tmon1_fc_end = FluxCoil(tmon4_fc.end, fc_cpw_params, width=20e3, trans_in=DTrans.R180)
tmon1_fc_end.place(canvas)

tmon_m1_fc_segment_lengths = \
    [0.5e6, -qubit_ports[-1].x + cp5.end.x - 0.5e6, cp5.end.y - qubit_ports[-1].y - 0.320e6 + 4e3]
tmon_m1_fc = CPWRLPath(cp5.end, "LRRLRL", fc_cpw_params, 150e3,
                       tmon_m1_fc_segment_lengths, [-pi / 2, pi / 2, -pi / 2], trans_in=DTrans.M90, bridged=True)
tmon_m1_fc.place(canvas, region_name="photo")
tmon_m1_fc.place(bridges, region_name="bridges")
tmon_m1_fc.place(bridge_patches, region_name="bridge_patches")

tmon1_fc_end = FluxCoil(tmon_m1_fc.end, fc_cpw_params, width=20e3, trans_in=DTrans.R180)
tmon1_fc_end.place(canvas)

###TEST STRUCTURE
test_frame1 = Test_frame(DPoint(-3146e3, -1752e3), h_jj, w_jj, asymmetry, 8e3, use_cell=True)
test_frame1.place(canvas, region_name="photo")
test_frame1.place(ebeam, region_name="ebeam")

test_frame2 = Test_frame(DPoint(3146e3, -1752e3), h_jj, w_jj, asymmetry, 8e3, use_cell=True)
test_frame2.place(canvas, region_name="photo")
test_frame2.place(ebeam, region_name="ebeam")

test_frame3 = Test_frame(DPoint(-3146e3, 1500e3), h_jj, w_jj, asymmetry, 8e3, use_cell=True)
test_frame3.place(canvas, region_name="photo")
test_frame3.place(ebeam, region_name="ebeam")

test_frame4 = Test_frame(DPoint(3146e3, 1500e3), h_jj, w_jj, asymmetry, 8e3, use_cell=True)
test_frame4.place(canvas, region_name="photo")
test_frame4.place(ebeam, region_name="ebeam")

# ab = Airbridge(DPoint(0, 0), None)
# ab.place(bridges, region_name = "bridges")
# ab.place(bridge_patches, region_name = "bridge_patches")

### DRAW SECTION END ###
ebeam = ebeam.merge()

cell.shapes(layer_negative).insert(invert_region - canvas - ebeam)
cell.shapes(layer_photo).insert(canvas)

cell.shapes(layer_el).insert(ebeam)

cell.shapes(layer_bridges).insert(bridges)
cell.shapes(layer_bridge_patches).insert(bridge_patches)

for layer_idx in layout.layer_indexes():
    for shape in cell.shapes(layer_idx).each():
        shape.transform(Trans(int(12.5e6), int(12.5e6)))
for instance in cell.each_inst():
    instance.transform(Trans(int(12.5e6), int(12.5e6)))

lv.zoom_fit()
