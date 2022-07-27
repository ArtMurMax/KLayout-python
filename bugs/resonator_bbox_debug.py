# Enter your Python code here
import pya
from math import cos, sin, atan2, pi
from pya import Point, DPoint, DSimplePolygon, SimplePolygon, DPolygon, Polygon, Region
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
        # Usually located near readout waveguide, hence the chip_name origin
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


class EMResonator_TL2Qbit_worm2_XmonFork(EMResonator_TL2Qbit_worm2):
    def __init__(self, Z0, start, L_coupling, L1, r, L2, N,
                 fork_x_span, fork_y_span, fork_metal_width, fork_gnd_gap,
                 trans_in=None):
        self.fork_x_span = fork_x_span
        self.fork_y_span = fork_y_span
        self.fork_metal_width = fork_metal_width
        self.fork_gnd_gap = fork_gnd_gap

        super().__init__(Z0, start, L_coupling, L1, r, L2, N, trans_in)

        self.start = self.connections[0]
        self.end = self.connections[-1]
        self.dr = self.end - self.start
        self.alpha_start = self.angle_connections[0]
        self.alpha_end = self.angle_connections[1]

    def init_primitives(self):
        super().init_primitives()

        """ add fork to the end of the resonator """
        # remove open end from the resonator
        del self.primitives["cop_open_end"]
        del self.cop_open_end

        # adding fork
        self.draw_fork_along_x()
        self.draw_fork_along_y()

    def draw_fork_along_x(self):
        forkZ = CPW(self.fork_metal_width, self.fork_gnd_gap)
        p1 = self.cop_tail.end + DPoint(-self.fork_x_span / 2, - forkZ.b / 2)
        p2 = p1 + DPoint(self.fork_x_span, 0)
        self.fork_x_cpw = CPW(forkZ.width, forkZ.gap, p1, p2)
        self.primitives["fork_x_cpw"] = self.fork_x_cpw

        # return waveguide that was erased during ground drawing
        p3 = self.cop_tail.end + DPoint(0, -forkZ.gap)
        erased_cpw = CPW(self.Z0.width, self.Z0.gap, self.cop_tail.end, p3)
        self.primitives["erased_cpw"] = erased_cpw

        # erase additional spaces at the ends of fork_x_cpw
        p1 = self.fork_x_cpw.start + DPoint(-forkZ.gap, 0)
        self.primitives["erased_fork_left"] = CPW(0, forkZ.b / 2, self.fork_x_cpw.start, p1)
        p1 = self.fork_x_cpw.end + DPoint(forkZ.gap, 0)
        self.primitives["erased_fork_right"] = CPW(0, forkZ.b / 2, self.fork_x_cpw.end, p1)

    def draw_fork_along_y(self):
        forkZ = CPW(self.fork_metal_width, self.fork_gnd_gap)

        # draw left part
        p1 = self.fork_x_cpw.start + DPoint(forkZ.width / 2, -forkZ.width / 2)
        p2 = p1 + DPoint(0, -self.fork_y_span)
        self.fork_y_cpw1 = CPW(forkZ.width, forkZ.gap, p1, p2)
        self.primitives["fork_y_cpw1"] = self.fork_y_cpw1

        # draw right part
        p1 = self.fork_x_cpw.end + DPoint(-forkZ.width / 2, -forkZ.width / 2)
        p2 = p1 + DPoint(0, -self.fork_y_span)
        self.fork_y_cpw2 = CPW(forkZ.width, forkZ.gap, p1, p2)
        self.primitives["fork_y_cpw2"] = self.fork_y_cpw2

        # erase gap at the ends of `y` fork parts
        p1 = self.fork_y_cpw1.end + DPoint(0, -forkZ.gap)
        self.primitives["erased_fork_left_cpw_end"] = CPW(0, forkZ.b / 2, self.fork_y_cpw1.end, p1)
        p1 = self.fork_y_cpw2.end + DPoint(0, -forkZ.gap)
        self.primitives["erased_fork_right_cpw_end"] = CPW(0, forkZ.b / 2, self.fork_y_cpw2.end, p1)


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
    chip_box = pya.Box(Point(0, 0), Point(CHIP.dx, CHIP.dy))
    origin = DPoint(0, 0)

    contact_L = 1e6

    # main drive line coplanar
    width = 24.1e3
    gap = 12.95e3
    x = 0.25 * CHIP.dx
    y = 0.9 * CHIP.dy
    p1 = DPoint(0, y)
    p2 = DPoint(CHIP.dx, y)
    Z0 = CPW(width, gap, p1, p2)

    # resonator
    L_coupling = 300e3
    # corresponding to resonanse freq is linspaced in interval [6,9) GHz
    L1_list = [261653., 239550., 219456., 201109., 184291., 168819., 154536., 141312., 129032., 117600.]
    r = 50e3
    L2 = 270e3
    gnd_width = 35e3
    N = 4
    width_res = 9.6e3
    gap_res = 5.2e3
    to_line = 35e3
    Z_res = CPW(width_res, gap_res)

    # fork at the end of resonator parameters
    fork_x_span = 80e3
    fork_y_span = 25e3
    fork_metal_width = 15e3
    fork_gnd_gap = 10e3

    # xmon cross parameters
    cross_width = 60e3
    cross_len = 125e3
    cross_gnd_gap = 20e3

    for L1 in [L1_list[0]]:
        # clear this cell and layer
        cell.clear()
        # draw chip box
        cell.shapes(layer_photo).insert(chip_box)

        worm = EMResonator_TL2Qbit_worm2_XmonFork(
            Z_res, origin, L_coupling, L1, r, L2, N,
            fork_x_span, fork_x_span, fork_metal_width, fork_gnd_gap
        )
        worm.place(cell, layer_photo)


        res_bbox = worm.metal_region.bbox()
        print(res_bbox.p1, "  ", res_bbox.p2)
        for name, instance in worm.primitives.items():
          print(name)
          print(instance.metal_region.bbox().p1,"  ",instance.metal_region.bbox().p2)

        ## DRAWING SECTION END ##
        lv.zoom_fit()
        print(pya.Region().bbox().p1)
        ### MATLAB COMMANDER SECTION START ###
        # ml_terminal = SonnetLab()
        # print("starting connection...")
        # from sonnetSim.cMD import CMD
        #
        # ml_terminal._send(CMD.SAY_HELLO)
        # ml_terminal.clear()
        # simBox = SimulationBox(CHIP_10x10_12pads.dx, CHIP_10x10_12pads.dy, 300, 900)
        # ml_terminal.set_boxProps(simBox)
        # print("sending cell and layer")
        # from sonnetSim.pORT_TYPES import PORT_TYPES
        #
        # ports = [SonnetPort(point, PORT_TYPES.BOX_WALL) for point in [Z0.start, Z0.end]]
        # ml_terminal.set_ports(ports)
        #
        # ml_terminal.send_polygons(cell, layer_photo)
        # ml_terminal.set_ABS_sweep(5, 7)
        # print("simulating...")
        # result_path = ml_terminal.start_simulation(wait=True)
        # print("visualizing...")
        # ml_terminal.visualize_sever()
        # ml_terminal.release()
        # #
        # import shutil
        # import os
        #
        # project_dir = os.path.dirname(__file__)
        # shutil.copy(result_path.decode("ascii"), os.path.join(project_dir, "result_L1_" + str(L1) + ".csv"))
