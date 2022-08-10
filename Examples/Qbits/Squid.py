#TODO don't work in KLayout
import pya
from math import sqrt, cos, sin, tan, atan2, pi, copysign
from pya import Point, DPoint, Vector, DVector, DSimplePolygon, SimplePolygon, DPolygon, Polygon, Region, DBox, Box
from pya import Trans, DTrans, CplxTrans, DCplxTrans, ICplxTrans, DPath

from importlib import reload
# reload(classLib)
import classLib
from classLib import *

reload(classLib)
from classLib import *


class CHIP:
    dx = 10e6
    dy = 5e6


chip_center = DPoint(CHIP.dx / 2, CHIP.dy / 2)

### MAIN FUNCTION ###
if __name__ == "__main__":
    # getting main references of the application
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

    # find or create the desired by programmer cell and layer
    layout = cv.layout()
    layout.dbu = 0.001
    if (layout.has_cell("testScript")):
        pass
    else:
        cell = layout.create_cell("testScript")

    info = pya.LayerInfo(1, 0)
    info2 = pya.LayerInfo(2, 0)
    layer_photo = layout.layer(info)
    layer_el = layout.layer(info2)

    # clear this cell and layer
    cell.clear()

    # setting layout view
    lv.select_cell(cell.cell_index(), 0)
    lv.add_missing_layers()

    ### DRAW SECTION START ###
    origin = DPoint(0, 0)

    pad_side = 5e3
    pad_r = 1e3
    pads_distance = 30e3
    p_ext_width = 3e3
    p_ext_r = 0.5e3
    sq_dy = 7e3
    sq_width = 15e6
    j_width = 0.4e3
    intermediate_width = 0.5e3
    b_ext = 0.9e3
    j_length = 0.3e3
    n = 7
    bridge = 0.5e3
    trans_in = None

    pars_squid = [pad_side, pad_r, pads_distance, p_ext_width, \
                  p_ext_r, sq_dy, sq_width, j_width, intermediate_width, \
                  b_ext, j_length, n, bridge]
    sq = Squid(origin, pars_squid)
    sq.place(cell, layer_el)
    ### DRAW SECTION END ###

    lv.zoom_fit()
