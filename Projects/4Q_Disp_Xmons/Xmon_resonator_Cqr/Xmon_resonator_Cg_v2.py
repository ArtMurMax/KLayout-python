# Enter your Python code here
import pya
from math import cos, sin, atan2, pi
from pya import Point, DPoint, DVector, DSimplePolygon, SimplePolygon, DPolygon, Polygon, Region
from pya import Trans, DTrans, CplxTrans, DCplxTrans, ICplxTrans

from classLib.baseClasses import ComplexBase
from classLib.coplanars import CPW, CPWArc
from classLib.resonators import Coil_type_1
from classLib.shapes import XmonCross

from sonnetSim.sonnetLab import SonnetLab, SonnetPort, SimulationBox


class EMResonator_TL2Qbit_worm2(ComplexBase):
    def __init__(self, Z0, start, L_coupling, L1, r, L2, N, trans_in=None):
        self.Z0 = Z0  # resonator's coplanar waveguide parameters
        # first horizontal coplanar length.
        # Usually located near readout waveguide, hence the name origin
        self.L_coupling = L_coupling
        self.L1 = L1  # horizontal coplanars length
        self.r = r  # curvature outer_r of all arc coplanars
        self.L2 = L2  # tail length

        self.N = N  # number of Coil_type_1 structure utilized
        super().__init__(start, trans_in)

        self.start = self.connections[0]
        self.end = self.connections[-1]
        self.dr = self.end - self.start
        self.alpha_start = self.angle_connections[0]
        self.alpha_end = self.angle_connections[1]

    def init_primitives(self):
        self.arc1 = CPWArc(self.Z0, DPoint(0, 0), -self.r, pi / 2)
        self.primitives["arc1"] = self.arc1

        # making coil
        name = "coil0"
        setattr(self, name, Coil_type_1(self.Z0, DPoint(0, 0), self.L_coupling, self.r, self.L1))
        self.primitives[name] = getattr(self, name)
        # coils filling
        for i in range(self.N):
            name = "coil" + str(i + 1)
            prev_coil_end = getattr(self, "coil" + str(i)).connections[-1]
            setattr(
                self, name,
                Coil_type_1( self.Z0, prev_coil_end, self.L1, self.r, self.L1)
            )
            self.primitives[name] = getattr(self, name)

        # draw the "tail"
        self.arc_tail = CPWArc(self.Z0, self.primitives["coil" + str(self.N)].end, -self.L1 / 2, -pi / 2)
        self.cop_tail = CPW(self.Z0.width, self.Z0.gap, self.arc_tail.end, self.arc_tail.end - DPoint(0, self.L2))

        # tail face is separated from ground by `b = width + 2*gap`
        self.cop_open_end = CPW(0, self.Z0.b / 2, self.cop_tail.end, self.cop_tail.end - DPoint(0, self.Z0.b))
        self.primitives["arc_tail"] = self.arc_tail
        self.primitives["cop_tail"] = self.cop_tail
        self.primitives["cop_open_end"] = self.cop_open_end

        self.connections = [DPoint(0, 0), self.cop_tail.end]
        self.angle_connections = [0, self.cop_tail.alpha_end]


class CHIP:
    dx = 0.6e6
    dy = 1.8e6
    L1 = 2.5e6
    gap = 150.e3
    width = 260.e3
    b = 2 * gap + width
    origin = DPoint(0, 0)
    center = DPoint(dx/2, dy/2)
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

    # resonator
    L_coupling = 300e3
    # corresponding to resonanse freq is linspaced in interval [6,9) GHz
    L1 = 100e3
    r = 100e3
    L2 = 270e3
    gnd_width = 35e3
    N = 4
    width_res = 9.6e3
    gap_res = 5.2e3
    to_line = 35e3
    Z_res = CPW(width_res, gap_res)

    # xmon cross parameters
    cross_widths = 60e3
    cross_lens = 60e3
    cross_gnd_gaps = 20e3

    cross_lens_x = 180e3
    cross_widths_x = 60e3
    cross_gnd_gaps_x = 20e3
    cross_lens_y = 60e3
    cross_widths_y = 60e3
    cross_gnd_gaps_y = 20e3

    xmon_resEnd_distances = [1e3*x for x in range(1, 2)]

    for xmon_resEnd_distance in xmon_resEnd_distances:
        # clear this cell and layer
        cell.clear()

        worm = EMResonator_TL2Qbit_worm2(
            Z_res,
            origin,
            L_coupling, L1, r, L2, N
        )
        xmon_center = worm.cop_tail.end
        xmon_center += DPoint(0, -(cross_len + cross_width / 2 + xmon_resEnd_distance))
        xmonCross = XmonCross(xmon_center, cross_width, cross_len, cross_gnd_gap)
        # cutting everything out except for the tail that will be capcitavely coupled
        # to Xmon
        for key in list(worm.primitives.keys()):
            if key != "cop_tail":
                del worm.primitives[key]

        tmp_reg = Region()
        worm.place(tmp_reg)
        xmonCross.place(tmp_reg)
        bbox = tmp_reg.bbox()

        # calculate simulation box placement
        CHIP.dx = 3*bbox.width()
        CHIP.dy = 3*bbox.b()
        chip_p1 = DPoint(bbox.center()) + DPoint(-0.5*CHIP.dx, -0.5*CHIP.dy)
        chip_p2 = DPoint(bbox.center()) + DPoint(0.5*CHIP.dx, 0.5*CHIP.dy)
        chip_box = pya.Box(DPoint(0, 0), DPoint(CHIP.dx, CHIP.dy))

        # forming resonator's tail upper side open-circuited for simulation
        p1 = worm.cop_tail.start
        p2 = p1 + DPoint(0, 20e3)
        empty_cpw_tail_start = CPW(0, Z_res.b/2, p1, p2)

        # placing all objects in proper order and translating them to origin
        cell.shapes(layer_photo).insert(chip_box)

        translation_trans = DCplxTrans(-DVector(chip_p1))
        worm.make_trans(translation_trans)
        worm.place(cell, layer_photo)

        empty_cpw_tail_start.place(cell, layer_photo)
        empty_cpw_tail_start.make_trans(translation_trans)
        empty_cpw_tail_start.place(cell, layer_photo)

        xmonCross.make_trans(translation_trans)
        xmonCross.place(cell, layer_photo)

        ## DRAWING SECTION END ##
        lv.zoom_fit()

        ### MATLAB COMMANDER SECTION START ###
        ml_terminal = SonnetLab()
        print("starting connection...")
        from sonnetSim.cMD import CMD

        ml_terminal._send(CMD.SAY_HELLO)
        ml_terminal.clear()
        simBox = SimulationBox(CHIP.dx, CHIP.dy, 200, 300)
        ml_terminal.set_boxProps(simBox)
        print("sending cell and layer")
        from sonnetSim.pORT_TYPES import PORT_TYPES

        ports = [
            SonnetPort(worm.cop_tail.start, PORT_TYPES.AUTOGROUNDED),
            SonnetPort(xmonCross.cpw_b.end, PORT_TYPES.AUTOGROUNDED)
        ]
        ml_terminal.set_ports(ports)

        ml_terminal.send_polygons(cell, layer_photo)
        ml_terminal.set_ABS_sweep(5, 7)
        print("simulating...")
        result_path = ml_terminal.start_simulation(wait=True)
        print("visualizing...")
        ml_terminal.visualize_sever()
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
                    s[i][j] = complex(float(data_row[1 + 2 * (i * 2 + j)]), float(data_row[1 + 2 * (i * 2 + j) + 1]))
            import math

            # formula taken from https://en.wikipedia.org/wiki/Admittance_parameters#Two_port
            delta = (1 + s[0][0]) * (1 + s[1][1]) - s[0][1] * s[1][0]
            y21 = -2 * s[1][0] / delta * 1 / R
            C12 = 1e15 / (2 * math.pi * freq0 * 1e9 * (1 / y21).imag)

        print(C12)

        output_filepath = os.path.join(project_dir, "Xmon_resonatov_Cg_v2.csv")
        if os.path.exists(output_filepath):
            # append data to file
            with open(output_filepath, "width") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(
                    [xmon_resEnd_distance/1e3, C12]
                )
        else:
            # create file, add header, append data
            with open(output_filepath, "w") as csv_file:
                writer = csv.writer(csv_file)
                # create header of the file
                writer.writerow(
                    ["xmon_resEnd_distance, um", "C12, fF"])
                writer.writerow(
                    [xmon_resEnd_distance/1e3, C12]
                )