# $description: test_script
# $version: 1.0.0
# $show-in-menu

# script generates files that are placed in branchline_curv v.3.0.*

import pya
from math import sqrt, cos, sin, atan2, pi, copysign
from pya import Point, DPoint, DSimplePolygon, SimplePolygon, DPolygon, Polygon, Region
from pya import Trans, DTrans, CplxTrans, DCplxTrans, ICplxTrans

import csv

from classLib import ComplexBase
from classLib.coplanars import CPW, CPWArc, CPW2CPW
from classLib.couplers import TJunction_112, BranchLine_finger2, Coupler_BranchLine


### START classes to be delegated to different file ###

class CHIP:
    dx = 10.0e6
    dy = 5.0e6
    L1 = 2.416721e6
    gap = 150e3
    width = 260e3
    Z = CPW(width, gap, DPoint(0, 0), DPoint(1, 1))
    b = 2 * gap + width
    origin = DPoint(0, 0)
    box = pya.DBox(origin, origin + DPoint(dx, dy))
    # only 4 connections programmed by now
    connections = [box.p1 + DPoint(L1 + b / 2, 0), box.p1 + DPoint(dx - (L1 + b / 2), 0),
                   box.p2 - DPoint(L1 + b / 2, 0), box.p1 + DPoint(L1 + b / 2, dy)]


# END classes to be delegated to different file ###

def import_bl2_params(file_name):
    params = []
    try:
        with open(file_name) as csv_file:
            reader = csv.reader(csv_file)
            for row in reader:
                params.append(float(row[1]))
    except:
        print("file not found")

    return params


class Path_RS(ComplexBase):
    def __init__(self, Z0, start, end, trans_in=None):
        self.Z0 = Z0
        self.end = end
        self.dr = end - start
        super().__init__(start, trans_in)
        self.start = self.connections[0]
        self.end = self.connections[1]
        self.alpha_start = self.angle_connections[0]
        self.alpha_end = self.angle_connections[1]

    def init_primitives(self):
        L = abs(abs(self.dr.y) - abs(self.dr.x))
        r = min(abs(self.dr.y), abs(self.dr.x))
        if (abs(self.dr.y) > abs(self.dr.x)):
            self.arc1 = CPWArc(self.Z0, DPoint(0, 0), copysign(
                r, self.dr.y), copysign(pi / 2, self.dr.x * self.dr.y))
            self.cop1 = CPW(self.Z0.width, self.Z0.gap, self.arc1.end,
                            self.arc1.end + DPoint(0, copysign(L, self.dr.y)))
            self.connections = [self.arc1.start, self.cop1.end]
            self.angle_connections = [self.arc1.alpha_start, self.cop1.alpha_end]
        else:
            self.cop1 = CPW(self.Z0.width, self.Z0.gap, DPoint(
                0, 0), DPoint(0, 0) + DPoint(copysign(L, self.dr.x), 0))
            self.arc1 = CPWArc(self.Z0, self.cop1.end, copysign(
                r, self.dr.y), copysign(pi / 2, self.dr.y * self.dr.x))
            self.connections = [self.cop1.start, self.arc1.end]
            self.angle_connections = [self.cop1.alpha_start, self.arc1.alpha_end]
        self.primitives = {"arc1": self.arc1, "cop1": self.cop1}


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

    info = pya.LayerInfo(10, 0)
    layer_i = layout.layer(info)

    # clear this cell and layer
    cell.clear()

    # setting layout view  
    lv.select_cell(cell.cell_index(), 0)
    lv.add_missing_layers()

    ### DRAW SECTION START ###
    # parameters
    width1 = 48e3
    gap1 = 27e3
    width2 = 74e3
    gap2 = 13e3
    gndWidth = 20  # not used

    L1 = 0
    L2 = 0.222e6
    L3 = 0
    L4 = 0.3e6
    L5 = 0.8e6
    r1 = 0.709e6
    r2 = 0.2e6
    r3 = 0.2e6
    r4 = 0.179e6
    gamma1 = 0.8042477
    gamma2 = 2.318495378

    L1_2 = 0
    L2_2 = 0.222e6
    L3_2 = 0
    L4_2 = 0.3e6
    L5_2 = 1e6
    r1_2 = 0.709e6
    r2_2 = 0.2e6
    r3_2 = 0.2e6
    r4_2 = 0.3e6
    gamma1_2 = 0.8042477
    gamma2_2 = 2.318495378

    origin = DPoint(0, 0)
    chip = pya.DBox(origin, DPoint(CHIP.dx, CHIP.dy))
    cell.shapes(layer_i).insert(pya.Box().from_dbox(chip))

    Z0 = CPW(width1, gap1, origin, origin)
    Z1 = CPW(width2, gap2, origin, origin)

    # bl2_params = import_bl2_params( "C:/Users/botan_000/Documents/BozonSampling/Calculator/layout2_opt_res.txt" )
    # bl2_params_2 = bl2_params
    # save_dir = "C:/Users/botan_000/Documents/BozonSampling/Lambda_div_4_coupler_curv/v3.0_28_03_2017/v.3.0.6/"

    bl2_params = [L1, L2, L3, L4, L5, r1, r2, r3, r4, gamma1, gamma2]
    bl2_params_2 = [L1_2, L2_2, L3_2, L4_2, L5_2, r1_2, r2_2, r3_2, r4_2, gamma1_2, gamma2_2]
    bl_hor = BranchLine_finger2(Z1, origin, bl2_params_2)
    bl_vert = BranchLine_finger2(Z0, origin, bl2_params)
    tj = TJunction_112(Z0, Z1, origin)
    coupler = Coupler_BranchLine(origin, bl_hor, bl_vert, tj)
    coupler.make_trans(DCplxTrans(1, 0, False, (CHIP.dx - coupler.connections[2].x) / 2,
                                  (CHIP.dy - (coupler.connections[2].y + coupler.tj_TR.Z1.b / 2)) / 2))
    coupler.place(cell, layer_i)

    dy = 0.5e6
    p_dy = DPoint(0, dy)
    p_BL = Path_RS(Z0, coupler.connections[0], CHIP.connections[0] + p_dy)
    p_BL.place(cell, layer_i)
    p_BR = Path_RS(Z0, coupler.connections[1], CHIP.connections[1] + p_dy)
    p_BR.place(cell, layer_i)
    p_TR = Path_RS(Z0, coupler.connections[2], CHIP.connections[2] - p_dy)
    p_TR.place(cell, layer_i)
    p_TL = Path_RS(Z0, coupler.connections[3], CHIP.connections[3] - p_dy)
    p_TL.place(cell, layer_i)

    lv.zoom_fit()

    ad_BL = CPW2CPW(CHIP.Z, Z0, CHIP.connections[0], CHIP.connections[0] + p_dy)
    ad_BL.place(cell, layer_i)
    ad_BR = CPW2CPW(CHIP.Z, Z0, CHIP.connections[1], CHIP.connections[1] + p_dy)
    ad_BR.place(cell, layer_i)
    ad_TR = CPW2CPW(CHIP.Z, Z0, CHIP.connections[2], CHIP.connections[2] - p_dy)
    ad_TR.place(cell, layer_i)
    ad_TL = CPW2CPW(CHIP.Z, Z0, CHIP.connections[3], CHIP.connections[3] - p_dy)
    ad_TL.place(cell, layer_i)

    '''layout.write(save_dir + "sixth_attempt.gds")
    if( bl2_params is not None ):
        with open( save_dir + "params.txt", "w") as csvfile:
            writer = csv.writer( csvfile )
            for val in bl2_params:
                writer.writerow( [val] )
            writer.writerow( [] )
            for val in bl2_params_2:
                writer.writerow( [val] )'''
### MAIN FUNCTION END ###
