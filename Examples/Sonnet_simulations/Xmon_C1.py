#OK
# Enter your Python code here
import pya
from math import cos, sin, atan2, pi
from pya import Point, DPoint, DSimplePolygon, SimplePolygon, DPolygon, Polygon, Region
from pya import Trans, DTrans, CplxTrans, DCplxTrans, ICplxTrans
from classLib.coplanars import CPW, CPWArc
from classLib.resonators import Coil_type_1
from classLib.baseClasses import ComplexBase, ElementBase

from importlib import reload
import sonnetSim.sonnetLab

reload(sonnetSim.sonnetLab)
from sonnetSim.sonnetLab import SonnetLab, SonnetPort, SimulationBox
from classLib.shapes import XmonCross


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

    cross_widths = [1e3 * x for x in range(60, 61, 20)]
    cross_lens = [1e3 * x for x in range(125, 126, 10)]
    cross_gnd_gaps = [1e3 * x for x in range(20, 21, 1)]

    # clear this cell and layer
    cell.clear()

    from itertools import product

    pars = product(cross_widths, cross_lens, cross_gnd_gaps)
    for cross_width, cross_len, cross_gnd_gap in pars:
        xmon_dX = 2 * cross_len + cross_width + 2 * cross_gnd_gap
        CHIP.dx = 5 * xmon_dX
        CHIP.dy = 5 * xmon_dX
        CHIP.center = DPoint(CHIP.dx / 2, CHIP.dy / 2)

        chip_box = pya.Box(Point(0, 0), Point(CHIP.dx, CHIP.dy))
        cell.shapes(layer_photo).insert(chip_box)

        xmon_cross1 = XmonCross(
            CHIP.center, cross_width, cross_len, cross_gnd_gap
        )
        xmon_cross1.place(cell, layer_photo)

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
            SonnetPort(xmon_cross1.cpw_l.end, PORT_TYPES.AUTOGROUNDED)
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

        C1 = None
        with open(result_path.decode("ascii"), "r") as csv_file:
            data_rows = list(csv.reader(csv_file))
            ports_imps_row = data_rows[6]
            R = float(ports_imps_row[0].split(' ')[1])
            data_row = data_rows[8]
            freq0 = float(data_row[0])
            re_s11 = float(data_row[1])
            im_s11 = float(data_row[2])
            import math

            s11 = complex(re_s11, im_s11)
            y11 = 1 / R * (1 - s11) / (1 + s11)
            C1 = -1e15 / (2 * math.pi * freq0 * 1e9 * (1 / y11).imag)

        print(C1)

        output_filepath = os.path.join(project_dir, "Xmon_C1.csv")
        if os.path.exists(output_filepath):
            # append data to file
            with open(output_filepath, "width") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow([cross_width / 1e3, cross_len / 1e3, cross_gnd_gap / 1e3, C1])
        else:
            # create file, add header, append data
            with open(output_filepath, "w") as csv_file:
                writer = csv.writer(csv_file)
                # create header of the file
                writer.writerow(["cross_width, um", "cross_len, um", "cross_gnd_gap, um", "C12, fF"])
                writer.writerow([cross_width / 1e3, cross_len / 1e3, cross_gnd_gap / 1e3, C1])


