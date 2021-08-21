# Enter your Python code here
import pya
from math import cos, sin, atan2, pi
from pya import Point, DPoint, DSimplePolygon, SimplePolygon, DPolygon, Polygon, Region
from pya import Trans, DTrans, CplxTrans, DCplxTrans, ICplxTrans
from classLib.coplanars import CPW, CPWArc
from classLib.coplanars import Coil_type_1
from classLib.baseClasses import ComplexBase, ElementBase

from importlib import reload
import sonnetSim.sonnetLab

reload(sonnetSim.sonnetLab)
from sonnetSim.sonnetLab import SonnetLab, SonnetPort, SimulationBox


class Xmon_cross(ComplexBase):
    def __init__(self, origin, cross_width, side_length, gnd_gap, trans_in=None):
        self.cross_width = cross_width
        self.side_length = side_length
        self.gnd_gap = gnd_gap
        self.center = origin
        super().__init__(origin, trans_in)
        self.center = self.connections[0]

    def init_primitives(self):
        origin = DPoint(0, 0)

        # draw central square
        from classLib.shapes import Rectangle
        lb_corner = DPoint(-self.cross_width / 2, -self.cross_width / 2)
        center_square = Rectangle(lb_corner, self.cross_width, self.cross_width)
        self.primitives["center_square"] = center_square

        """ left part of Xmon cross """
        p1 = origin + DPoint(-self.cross_width / 2, 0)
        p2 = p1 + DPoint(-self.side_length, 0)
        self.cpw_l = CPW(self.cross_width, self.gnd_gap, p1, p2)
        self.primitives["cpw_l"] = self.cpw_l
        p3 = p2 + DPoint(-self.gnd_gap, 0)
        self.cpw_lempt = CPW(0, self.cpw_l.b / 2, p2, p3)
        self.primitives["cpw_lempt"] = self.cpw_lempt

        """ right part of Xmon cross """
        p1 = origin + DPoint(self.cross_width / 2, 0)
        p2 = p1 + DPoint(self.side_length, 0)
        self.cpw_r = CPW(self.cross_width, self.gnd_gap, p1, p2)
        self.primitives["cpw_r"] = self.cpw_r
        p3 = p2 + DPoint(self.gnd_gap, 0)
        self.cpw_rempt = CPW(0, self.cpw_r.b / 2, p2, p3)
        self.primitives["cpw_rempt"] = self.cpw_rempt

        """ top part of Xmon cross """
        p1 = origin + DPoint(0, self.cross_width / 2)
        p2 = p1 + DPoint(0, self.side_length)
        self.cpw_t = CPW(self.cross_width, self.gnd_gap, p1, p2)
        self.primitives["cpw_t"] = self.cpw_t
        p3 = p2 + DPoint(0, self.gnd_gap)
        self.cpw_tempt = CPW(0, self.cpw_t.b / 2, p2, p3)
        self.primitives["cpw_tempt"] = self.cpw_tempt

        """ bottom part of Xmon cross """
        p1 = origin + DPoint(0, -self.cross_width / 2)
        p2 = p1 + DPoint(0, -self.side_length)
        self.cpw_b = CPW(self.cross_width, self.gnd_gap, p1, p2)
        self.primitives["cpw_b"] = self.cpw_b
        p3 = p2 + DPoint(0, -self.gnd_gap)
        self.cpw_bempt = CPW(0, self.cpw_l.b / 2, p2, p3)
        self.primitives["cpw_bempt"] = self.cpw_bempt

        self.connections = [origin]


class CHIP:
    dx = 0.2e6
    dy = 0.2e6
    L1 = 2.5e6
    gap = 150.e3
    width = 260.e3
    b = 2 * gap + width
    origin = DPoint(0, 0)
    center = DPoint(dx / 2, dy / 2)
    box = pya.DBox(origin, origin + DPoint(dx, dy))
    # only 4 connections programmed by now
    connections = [box.p1 + DPoint(L1 + b / 2, 0), box.p1 + DPoint(dx - (L1 + b / 2), 0),
                   box.p2 - DPoint(L1 + b / 2, 0), box.p1 + DPoint(L1 + b / 2, dy)]


if __name__ == "__main__":
    # getting main references of the application
    app = pya.Application.instance()
    mw = app.main_window()
    lv = mw.current_view()
    cv = None

    # this insures that lv and cv are valid objects
    if lv is None:
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

    layer_info_photo = pya.LayerInfo(10, 0)
    layer_info_el = pya.LayerInfo(1, 0)
    layer_photo = layout.layer(layer_info_photo)
    layer_el = layout.layer(layer_info_el)

    # setting layout view
    lv.select_cell(cell.cell_index(), 0)
    lv.add_missing_layers()

    ## DRAWING SECTION START ##
    origin = DPoint(0, 0)

    cross_widths = [60e3]
    cross_lens = [125e3]
    cross_gnd_gaps = [20e3]
    xmon_dX = 2 * cross_lens[0] + cross_widths[0] + 2 * cross_gnd_gaps[0]
    xmon_distances = [1e3 * x for x in range(393, 394, 10)]
    from itertools import product

    pars = product(cross_widths, cross_lens, cross_gnd_gaps, xmon_distances)
    for cross_width, cross_len, cross_gnd_gap, xmon_distance in pars:
        # clear this cell and layer
        cell.clear()
        xmon_dX = 2 * cross_len + cross_width + 2 * cross_gnd_gap
        CHIP.dx = 5 * xmon_dX + xmon_distance
        CHIP.dy = 5 * xmon_dX + xmon_distance
        CHIP.center = DPoint(CHIP.dx / 2, CHIP.dy / 2)

        chip_box = pya.Box(Point(0, 0), Point(CHIP.dx, CHIP.dy))
        cell.shapes(layer_photo).insert(chip_box)

        xmon_cross1 = Xmon_cross(
            CHIP.center + DPoint(-xmon_distance / 2, 0),
            cross_width, cross_len, cross_gnd_gap
        )
        xmon_cross1.place(cell, layer_photo)

        xmon_cross2 = Xmon_cross(
            CHIP.center + DPoint(xmon_distance / 2, 0),
            cross_width, cross_len, cross_gnd_gap
        )
        xmon_cross2.place(cell, layer_photo)

        ## DRAWING SECTION END ##
        # lv.zoom_fit()

        ### MATLAB COMMANDER SECTION START ###
        ml_terminal = SonnetLab()
        print("starting connection...")
        from sonnetSim.cMD import CMD

        ml_terminal._send(CMD.SAY_HELLO)
        ml_terminal.clear()
        simBox = SimulationBox(CHIP.dx, CHIP.dy, 600, 600)
        ml_terminal.set_boxProps(simBox)
        print("sending cell and layer")
        from sonnetSim.pORT_TYPES import PORT_TYPES

        ports = [
            SonnetPort(xmon_cross1.cpw_l.end, PORT_TYPES.AUTOGROUNDED),
            SonnetPort(xmon_cross2.cpw_r.end, PORT_TYPES.AUTOGROUNDED)
        ]
        ml_terminal.set_ports(ports)

        ml_terminal.send_polygons(cell, layer_photo)
        ml_terminal.set_linspace_sweep(0.01, 0.01, 1)
        print("simulating...")
        result_path = ml_terminal.start_simulation(wait=True)
        # print("visualizing...")
        # ml_terminal.visualize_sever()
        ml_terminal.release()

        # get the .csv result file and exctract capcity of island in fF
        import shutil
        import os
        import csv

        project_dir = os.path.dirname(__file__)

        C12 = None
        with open(result_path.decode("ascii"), "r") as csv_file:
            data_rows = list(csv.reader(csv_file))
            ports_imps_row = data_rows[6]
            R = float(ports_imps_row[0].split(' ')[1])
            data_row = data_rows[8]
            freq0 = float(data_row[0])

            s = [[0, 0], [0, 0]]  # s-matrix
            for i in range(0, 2):
                for j in range(0, 2):
                    s[i][j] = complex(float(data_row[1 + 2*(i*2 + j)]), float(data_row[1 + 2*(i*2 + j)+1]))
            import math
            # formula taken from https://en.wikipedia.org/wiki/Admittance_parameters#Two_port
            delta = (1+s[0][0])*(1+s[1][1]) - s[0][1]*s[1][0]
            y21 = -2 * s[1][0] / delta * 1/R
            C12 = 1e15 / (2 * math.pi * freq0 * 1e9 * (1 / y21).imag)

        print(C12)

        output_filepath = os.path.join(project_dir, "Xmon_Xmon_C12.csv")
        if os.path.exists(output_filepath):
            # append data to file
            with open(output_filepath, "width") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(
                    [cross_width / 1e3, cross_len / 1e3, cross_gnd_gap / 1e3,
                     xmon_distance/1e3,  C12]
                )
        else:
            # create file, add header, append data
            with open(output_filepath, "w") as csv_file:
                writer = csv.writer(csv_file)
                # create header of the file
                writer.writerow(
                    ["cross_width, um", "cross_len, um", "cross_gnd_gap, um", "xmon_distance, um", "C12, fF"])
                writer.writerow(
                    [cross_width / 1e3, cross_len / 1e3, cross_gnd_gap / 1e3,
                     xmon_distance / 1e3, C12]
                )