import pya
from pya import Point, DPoint, Vector, DVector, DSimplePolygon, SimplePolygon, DPolygon, Polygon, Region, Box, DBox
from pya import Trans, DTrans, CplxTrans, DCplxTrans, ICplxTrans

from importlib import reload

import classLib
reload(classLib)
from classLib.chipDesign import ChipDesign
from classLib.shapes import TmonT, Rectangle, XmonCross
from classLib.coplanars import CPWRLPath, CPWParameters, CPW
from classLib.baseClasses import ElementBase
from classLib.chipTemplates import CHIP_16p5x16p5_20pads
from classLib.resonators import EMResonatorTL3QbitWormRLTailXmonFork, EMResonatorTL3QbitWormRLTail
from typing import List, Generic, TypeVar
import numpy as np
import pandas as pd
from time import ctime
from xml.etree import ElementTree
from xml.etree.ElementTree import Element, SubElement, Comment, tostring
from xml.dom import minidom
from collections import OrderedDict
import os
import csv
import shutil
from pathlib import Path
import sonnetSim

reload(sonnetSim)
from sonnetSim import SonnetLab, SonnetPort, SimulationBox


refractive_index = np.sqrt(6.26423)
PROJECT_DIR = Path(r'C:\Users\Artem\Desktop\books\Phystech\LAQS\tasks')


class ResonatorWithFilter(ChipDesign):

    center: DPoint
    cop_pars: CPWParameters
    resonator: EMResonatorTL3QbitWormRLTailXmonFork
    filter: EMResonatorTL3QbitWormRLTail
    coupler_capacitor: list
    capacitor_ratio: float
    transmon: XmonCross

    design_dict = OrderedDict()

    last_draw = None

    def __init__(self, cell_name="testScript"):
        super().__init__(cell_name)

        self.center = DPoint(16500e3 / 2, 16500e3 / 2)
        self.main_axe = self.center.x

        self.cop_rad = 100e3
        self.res_width = 420e3
        self.res_height = 820e3
        self.res_turn_rad = 100e3
        self.res_line_gap = 50e3
        self.res_subheight = 220e3
        self.filt_subheight = 220e3

        ### Transmons ###
        self.xmon_dy_Cg_coupling = 12e3 # 14e3
        self.xmon_subfeedline_gap = 800e3
        self.xmon_feedline_gap = 200e3

        self.cross_len_x = 180e3
        self.cross_width_x = 60e3
        self.cross_gnd_gap_x = 20e3
        self.cross_len_y = 154e3
        self.cross_width_y = 60e3
        self.cross_gnd_gap_y = 20e3

        self.fork_shift = 400e3
        self.cross_fork_y_len = 70e3

        self.filter_resonator_gap = 600e3
        self.capacitor_ratio = 0.5
        self.coup_capacitor_len = 20e3
        self.coup_cap_gap = 60e3

        self.line_capacitor_len = 420e3
        self.line_cap_gap = 164e3
        self.cap_plate_par = CPWParameters(26e3, 12e3)

        self.cop_pars = CPWParameters(20e3, 12e3)

        self.readline_len = 2600e3

        self.chip = CHIP_16p5x16p5_20pads
        self.chip_box: pya.DBox = self.chip.box

    def set_parts(self):
        self.resonator = EMResonatorTL3QbitWormRLTailXmonFork(
            self.cop_pars,
            self.center,
            self.res_width,
            self.res_height,
            self.res_subheight,
            self.res_turn_rad,
            1,
            'L',
            [],
            [self.fork_shift],
            [],
            self.cross_width_x + 2 * self.cross_gnd_gap_x + self.cop_pars.width, self.cross_fork_y_len,
            self.cop_pars.width / 2, self.cop_pars.gap,
            tail_trans_in=Trans.R270
        )
        self.design_dict['resonator'] = self.resonator

        self.filter = EMResonatorTL3QbitWormRLTail(
            self.cop_pars,
            self.center + DPoint(self.res_width, self.filter_resonator_gap),
            self.res_width,
            self.res_height,
            self.filt_subheight,
            self.res_turn_rad,
            1,
            'L',
            [],
            [self.res_width / 2],
            [],
            tail_trans_in=Trans.R270,
            trans_in=Trans.R180
        )
        self.design_dict['filter'] = self.filter

        xmon_center = \
            (
                    self.resonator.fork_x_cpw.start + self.resonator.fork_x_cpw.end
            ) / 2 + \
            DVector(
                0,
                -self.xmon_dy_Cg_coupling - self.resonator.fork_metal_width / 2
            )
        # changes start #
        xmon_center += DPoint(
            0,
            -(
                    self.cross_len_y + self.cross_width_x / 2 +
                    self.cross_gnd_gap_y
            )
        )

        self.transmon = XmonCross(
            xmon_center,
            sideX_length=self.cross_len_x,
            sideX_width=self.cross_width_x,
            sideX_gnd_gap=self.cross_gnd_gap_x,
            sideY_length=self.cross_len_y,
            sideY_width=self.cross_width_y,
            sideY_gnd_gap=self.cross_gnd_gap_y,
            sideX_face_gnd_gap=self.cross_gnd_gap_x,
            sideY_face_gnd_gap=self.cross_gnd_gap_y
        )
        self.design_dict['transmon'] = self.transmon

        ### Setting filter-resonator capacitor ###
        coup_cap_start = self.center + DPoint(self.res_width * self.capacitor_ratio,
                                              self.filter_resonator_gap - self.cop_pars.width / 2
                                              )
        coup_cap_end = self.center + DPoint(self.res_width * self.capacitor_ratio,
                                            self.cop_pars.width / 2)
        self.coup_cap_mid = (coup_cap_start + coup_cap_end) / 2

        self.coupler_capacitor = []
        # erase gaps at the capacitors corner
        self.coupler_capacitor.append(
            CPW(
                self.coup_cap_gap - 2 * self.cop_pars.gap, self.cap_plate_par.width + self.cap_plate_par.gap,
                start=self.coup_cap_mid - DPoint(self.coup_capacitor_len / 2, 0),
                end=self.coup_cap_mid - DPoint(self.coup_capacitor_len / 2 + self.cop_pars.gap, 0)
            )
        )
        self.coupler_capacitor.append(
            CPW(
                self.coup_cap_gap - 2 * self.cop_pars.gap, self.cap_plate_par.width + self.cap_plate_par.gap,
                start=self.coup_cap_mid - DPoint(-self.coup_capacitor_len / 2, 0),
                end=self.coup_cap_mid - DPoint(-self.coup_capacitor_len / 2 - self.cop_pars.gap, 0)
            )
        )

        # erase gap in the capacitor mid, if needed
        # self.coupler_capacitor.append(
        #     CPW(
        #         0, self.coup_cap_gap/2,
        #         start=self.coup_cap_mid - DPoint(-self.coup_capacitor_len / 2, 0),
        #         end=self.coup_cap_mid - DPoint(self.coup_capacitor_len / 2, 0)
        #     )
        # )

        # add capacitor's plates
        self.coupler_capacitor.append(
            CPW(
                start=self.coup_cap_mid - DPoint(self.coup_capacitor_len / 2,
                                                 self.coup_cap_gap / 2 + self.cap_plate_par.width / 2),
                end=self.coup_cap_mid - DPoint(-self.coup_capacitor_len / 2,
                                               self.coup_cap_gap / 2 + self.cap_plate_par.width / 2),
                cpw_params=self.cap_plate_par
            )
        )
        self.coupler_capacitor.append(
            CPW(
                start=self.coup_cap_mid + DPoint(self.coup_capacitor_len / 2,
                                                 self.coup_cap_gap / 2 + self.cap_plate_par.width / 2),
                end=self.coup_cap_mid + DPoint(-self.coup_capacitor_len / 2,
                                               self.coup_cap_gap / 2 + self.cap_plate_par.width / 2),
                cpw_params=self.cap_plate_par
            )
        )
        # add lines to resonators
        self.coupler_capacitor.append(
            CPW(start=coup_cap_start,
                end=self.coup_cap_mid + DPoint(0, self.coup_cap_gap / 2 + self.cap_plate_par.width),
                cpw_params=self.cop_pars)
        )
        self.coupler_capacitor.append(
            CPW(start=self.coup_cap_mid - DPoint(0, self.coup_cap_gap / 2 + self.cap_plate_par.width), end=coup_cap_end,
                cpw_params=self.cop_pars)
        )
        self.design_dict['coupler_capacitor'] = self.coupler_capacitor

        ### Setting filter-line capacitor ###
        line_cap_start = self.filter.end + DPoint(0, -2 * self.cap_plate_par.gap)
        line_cap_end = self.filter.end + DPoint(0, -self.cap_plate_par.gap + 2 * self.cap_plate_par.b)
        self.line_cap_mid = (line_cap_start + line_cap_end) / 2

        self.line_capacitor = []
        # erase gaps at the capacitors corner
        self.line_capacitor.append(
            CPW(
                0, self.cap_plate_par.width + 3 / 2 * self.cap_plate_par.gap,
                start=self.line_cap_mid - DPoint(self.line_capacitor_len / 2, 0),
                end=self.line_cap_mid - DPoint(self.line_capacitor_len / 2 + self.cap_plate_par.gap, 0)
            )
        )
        self.line_capacitor.append(
            CPW(
                0, self.cap_plate_par.width + 3 / 2 * self.cap_plate_par.gap,
                start=self.line_cap_mid - DPoint(-self.line_capacitor_len / 2, 0),
                end=self.line_cap_mid - DPoint(-self.line_capacitor_len / 2 - self.cap_plate_par.gap, 0)
            )
        )
        # add capacitor's plates
        self.line_capacitor.append(
            CPW(
                start=self.line_cap_mid - DPoint(self.line_capacitor_len / 2,
                                                 self.cap_plate_par.gap / 2 + self.cap_plate_par.width / 2),
                end=self.line_cap_mid - DPoint(-self.line_capacitor_len / 2,
                                               self.cap_plate_par.gap / 2 + self.cap_plate_par.width / 2),
                cpw_params=self.cap_plate_par
            )
        )
        self.line_capacitor.append(
            CPW(
                start=self.line_cap_mid + DPoint(self.line_capacitor_len / 2,
                                                 self.cap_plate_par.gap / 2 + self.cap_plate_par.width / 2),
                end=self.line_cap_mid + DPoint(-self.line_capacitor_len / 2,
                                               self.cap_plate_par.gap / 2 + self.cap_plate_par.width / 2),
                cpw_params=self.cap_plate_par
            )
        )
        # add lines to resonators
        self.line_capacitor.append(
            CPW(start=line_cap_start, end=line_cap_start + DPoint(0, 2 * self.cap_plate_par.gap),
                cpw_params=self.cop_pars)
        )
        self.line_capacitor.append(
            CPW(start=line_cap_end - DPoint(0, 2 * self.cap_plate_par.gap),
                end=line_cap_end + DPoint(0, self.line_cap_gap - 2 * self.cap_plate_par.gap),
                cpw_params=self.cop_pars)
        )
        self.design_dict['line_capacitor'] = self.line_capacitor

        readline_y = line_cap_end.y + self.line_cap_gap - 2 * self.cap_plate_par.gap - self.cop_pars.gap + self.cop_pars.b / 2

        # Add readline
        self.readline = CPW(start=DPoint(self.coup_cap_mid.x - self.readline_len / 2, readline_y),
                            end=DPoint(self.coup_cap_mid.x + self.readline_len / 2, readline_y),
                            cpw_params=self.cop_pars)
        self.design_dict['readline'] = self.readline

        self.design_set = self.design_dict.keys()

        # Temporal elements for S-parameters callculation
        # Add readline for resonator simulation

        self.res_readline = CPWRLPath(self.center - DPoint(self.res_width + 300e3, self.res_height),
                                      'LRLRL', self.cop_pars, [100e3, 100e3],
                                      [self.res_width, self.res_height, self.res_width],
                                      [np.pi / 2, np.pi / 2])
        self.design_dict['resonator_readline'] = self.res_readline

    def draw(self, parts=None):
        self.set_parts()
        if parts is None:
            parts = self.design_set
        self.region_ph.insert(self.chip_box)
        if 'readline' in parts:
            self.readline.place(self.region_ph)
        if 'transmon' in parts:
            self.transmon.place(self.region_ph)
        if 'resonator' in parts:
            self.resonator.place(self.region_ph)
        if 'filter' in parts:
            self.filter.place(self.region_ph)
        if 'coupler_capacitor' in parts:
            for cap in self.coupler_capacitor:
                cap.place(self.region_ph)
        if 'line_capacitor' in parts:
            for cap in self.line_capacitor:
                cap.place(self.region_ph)

        self.last_draw = parts

    def draw_for_res_sim(self):
        self.set_parts()
        self.region_ph.insert(self.chip_box)
        self.transmon.place(self.region_ph)
        self.resonator.place(self.region_ph)
        for cap in self.coupler_capacitor:
            cap.place(self.region_ph)
        self.res_readline.place(self.region_ph)

        self.last_draw = {'transmon', 'resonator', 'coupler_capacitor', 'resonator_readline'}


    def save_logdata(self, filename=PROJECT_DIR / 'purcell_filter.xml', comment=''):

        root = Element('Design')
        root.set('version', 'gamma')
        root.set('class', str(ChipDesign))
        generalComment = Comment(f'Purcell filter design log.')
        root.append(generalComment)
        head = SubElement(root, 'head')
        date = SubElement(head, 'creationDate')
        date.text = f'{ctime()}'
        seParts = SubElement(head, 'parts')
        strlist = ''
        for prt in self.last_draw:
            strlist += prt + ' '
        seParts.text = strlist
        seComment = SubElement(head, 'comment')
        seComment.text = comment

        body = SubElement(root, 'body')

        for prt in self.last_draw:
            detail = self.design_dict[prt]
            se = SubElement(body, prt)
            se.set('class', str(type(detail)))
            sePar = SubElement(se, 'params')
            if type(detail) == list:
                i = 0
                for ln in detail:
                    seLn = SubElement(se, f'line_{i}')
                    seLn.set('class', str(type(ln)))
                    sePar = SubElement(seLn, 'params')
                    for key, value in ln.get_geometry_params_dict().items():
                        sePar.set(key.replace(', um', ''), str(value))
                    i += 1
            else:
                for key, value in detail.get_geometry_params_dict().items():
                    sePar.set(key.replace(', um', ''), str(value))

        parsed = tostring(root, 'utf-8')
        reparsed = minidom.parseString(parsed)
        with open(filename, 'w') as f:
            f.write(reparsed.toprettyxml(indent='   '))


def simulate_Cfl():
    resolution_dx = 5e2
    resolution_dy = 5e2

    ### DRAWING SECTION START ###
    design = ResonatorWithFilter("testScript")
    design.draw({'filter', 'line_capacitor'})

    bot_cpw = design.line_capacitor[-2]
    top_cpw = design.line_capacitor[-1]
    center = Point(design.line_cap_mid.x + 1e3, design.line_cap_mid.y)
    width = top_cpw.end.y - bot_cpw.start.y

    dv = Point(design.line_capacitor_len + 1e3, width / 2)

    crop_box = pya.Box().from_dbox(pya.Box(
        center + dv,
        center + (-1) * dv
    ))
    design.crop(crop_box)
    dr = DPoint(0, 0) - crop_box.p1

    # finding the furthest edge of cropped resonator`s central line polygon
    # sonnet port will be attached to this edge
    reg1 = design.filter.metal_region & Region(crop_box)
    reg1.merge()
    max_distance = 0
    port_pt = None
    for poly in reg1.each():
        for edge in poly.each_edge():
            edge_center = (edge.p1 + edge.p2) / 2
            dp = edge_center - center
            d = max(abs(dp.x), abs(dp.y))
            if d > max_distance:
                max_distance = d
                port_pt = edge_center
    design.sonnet_ports.append(port_pt)

    reg1 = top_cpw.metal_region & Region(crop_box)
    reg1.merge()
    max_distance = 0
    port_pt = None
    for poly in reg1.each():
        for edge in poly.each_edge():
            edge_center = (edge.p1 + edge.p2) / 2
            dp = edge_center - center
            d = max(abs(dp.x), abs(dp.y))
            if d > max_distance:
                max_distance = d
                port_pt = edge_center
    design.sonnet_ports.append(port_pt)

    design.transform_region(design.region_ph, DTrans(dr.x, dr.y),
                            trans_ports=True)

    design.show()
    design.lv.zoom_fit()
    ### DRAWING SECTION END ###

    for prt in design.sonnet_ports:
        print(prt)

    simulate_C12(crop_box, design, filename="ps_Cfl_results.csv", resolution_dx=resolution_dx, resolution_dy=resolution_dy)


def simulate_Cfr():
    resolution_dx = 5e2
    resolution_dy = 5e2

    ### DRAWING SECTION START ###
    design = ResonatorWithFilter("testScript")
    design.draw({'coupler_capacitor'})

    left_cpw = design.coupler_capacitor[-2]
    right_cpw = design.coupler_capacitor[-1]
    center = Point(design.coup_cap_mid.x, design.coup_cap_mid.y)
    width = right_cpw.end.y - left_cpw.start.y

    dv = Point(width / 2, width / 2)

    crop_box = pya.Box().from_dbox(pya.Box(
        center + dv,
        center + (-1) * dv
    ))
    design.crop(crop_box)
    dr = DPoint(0, 0) - crop_box.p1

    # finding the furthest edge of cropped resonator`s central line polygon
    # sonnet port will be attached to this edge
    reg1 = left_cpw.metal_region & Region(crop_box)
    reg1.merge()
    max_distance = 0
    port_pt = None
    for poly in reg1.each():
        for edge in poly.each_edge():
            edge_center = (edge.p1 + edge.p2) / 2
            dp = edge_center - center
            d = max(abs(dp.x), abs(dp.y))
            if d > max_distance:
                max_distance = d
                port_pt = edge_center
    design.sonnet_ports.append(port_pt)

    reg1 = right_cpw.metal_region & Region(crop_box)
    reg1.merge()
    max_distance = 0
    port_pt = None
    for poly in reg1.each():
        for edge in poly.each_edge():
            edge_center = (edge.p1 + edge.p2) / 2
            dp = edge_center - center
            d = max(abs(dp.x), abs(dp.y))
            if d > max_distance:
                max_distance = d
                port_pt = edge_center
    design.sonnet_ports.append(port_pt)

    design.transform_region(design.region_ph, DTrans(dr.x, dr.y),
                            trans_ports=True)

    for prt in design.sonnet_ports:
        print(prt)

    design.show()
    design.lv.zoom_fit()
    ### DRAWING SECTION END ###

    simulate_C12(crop_box, design, filename="ps_Cfr_results.csv", resolution_dx=resolution_dx, resolution_dy=resolution_dy)


def simulate_Cqr():
    resolution_dx = 5e2
    resolution_dy = 5e2

    ### DRAWING SECTION START ###
    design = ResonatorWithFilter("testScript")

    design.draw({'transmon', 'resonator'})

    worm = design.resonator
    xmonCross = design.transmon
    worm_start = list(worm.primitives.values())[0].start

    # draw open end at the resonators start
    p1 = worm_start - DVector(design.cop_pars.b / 2, 0)
    rec = Rectangle(p1, design.cop_pars.b, design.cop_pars.b / 2,
                    inverse=True)
    rec.place(design.region_ph)

    if worm_start.x < xmonCross.center.x:
        dr = (worm_start - xmonCross.cpw_r.end)
    else:
        dr = (worm_start - xmonCross.cpw_l.end)
    dr.x = abs(dr.x)
    dr.y = abs(dr.y)

    box_side_x = 4 * xmonCross.sideX_length
    box_side_y = 4 * xmonCross.sideY_length
    dv = DVector(box_side_x / 2, box_side_y / 2)

    crop_box = pya.Box().from_dbox(pya.Box(
        xmonCross.center + dv,
        xmonCross.center + (-1) * dv
    ))
    design.crop(crop_box)
    dr = DPoint(0, 0) - crop_box.p1

    # finding the furthest edge of cropped resonator`s central line polygon
    # sonnet port will be attached to this edge
    reg1 = worm.metal_region & Region(crop_box)
    reg1.merge()
    max_distance = 0
    port_pt = None
    for poly in reg1.each():
        for edge in poly.each_edge():
            edge_center = (edge.p1 + edge.p2) / 2
            dp = edge_center - xmonCross.cpw_b.end
            d = max(abs(dp.x), abs(dp.y))
            if d > max_distance:
                max_distance = d
                port_pt = edge_center
    design.sonnet_ports.append(port_pt)
    design.sonnet_ports.append(xmonCross.cpw_l.end)

    design.transform_region(design.region_ph, DTrans(dr.x, dr.y),
                            trans_ports=True)

    design.show()
    design.lv.zoom_fit()
    ### DRAWING SECTION END ###

    for prt in design.sonnet_ports:
        print(prt)

    simulate_C12(crop_box, design, filename="ps_Cqr_results.csv", resolution_dx=resolution_dx, resolution_dy=resolution_dy)


def simulate_Cfork():
    resolution_dx = 2e3
    resolution_dy = 2e3

    ### DRAWING SECTION START ###
    design = ResonatorWithFilter("testScript")

    design.draw({'transmon', 'resonator'})

    worm = design.resonator
    xmonCross = design.transmon
    worm_start = list(worm.primitives.values())[0].start

    # draw open end at the resonators start
    p1 = worm_start - DVector(design.cop_pars.b / 2, 0)
    rec = Rectangle(p1, design.cop_pars.b, design.cop_pars.b / 2,
                    inverse=True)
    rec.place(design.region_ph)

    if worm_start.x < xmonCross.center.x:
        dr = (worm_start - xmonCross.cpw_r.end)
    else:
        dr = (worm_start - xmonCross.cpw_l.end)
    dr.x = abs(dr.x)
    dr.y = abs(dr.y)

    box_side_x = 4 * xmonCross.sideX_length
    box_side_y = 4 * xmonCross.sideY_length
    dv = DVector(box_side_x / 2, box_side_y / 2)

    crop_box = pya.Box().from_dbox(pya.Box(
        xmonCross.center + dv,
        xmonCross.center + (-1) * dv
    ))
    design.crop(crop_box)
    dr = DPoint(0, 0) - crop_box.p1

    # finding the furthest edge of cropped resonator`s central line polygon
    # sonnet port will be attached to this edge
    reg1 = worm.metal_region & Region(crop_box)
    reg1.merge()
    max_distance = 0
    port_pt = None
    for poly in reg1.each():
        for edge in poly.each_edge():
            edge_center = (edge.p1 + edge.p2) / 2
            dp = edge_center - xmonCross.cpw_b.end
            d = max(abs(dp.x), abs(dp.y))
            if d > max_distance:
                max_distance = d
                port_pt = edge_center
    design.sonnet_ports.append(port_pt)
    design.sonnet_ports.append(xmonCross.cpw_l.end)

    design.transform_region(design.region_ph, DTrans(dr.x, dr.y),
                            trans_ports=True)

    design.show()
    design.lv.zoom_fit()
    ### DRAWING SECTION END ###

    for prt in design.sonnet_ports:
        print(prt)

    simulate_C12(crop_box, design, filename="ps_Cqr_results.csv")


def simulate_C12(crop_box, design, filename='c12.csv', resolution_dx=2e3, resolution_dy=2e3):
    ### SIMULATION SECTION START ###
    ml_terminal = SonnetLab()
    # print("starting connection...")
    from sonnetSim.cMD import CMD

    ml_terminal._send(CMD.SAY_HELLO)
    ml_terminal.clear()
    simBox = SimulationBox(
        crop_box.width(),
        crop_box.height(),
        crop_box.width() / resolution_dx,
        crop_box.height() / resolution_dy
    )

    ml_terminal.set_boxProps(simBox)
    # print("sending cell and layer")
    from sonnetSim.pORT_TYPES import PORT_TYPES

    ports = [
        SonnetPort(design.sonnet_ports[0], PORT_TYPES.AUTOGROUNDED),
        SonnetPort(design.sonnet_ports[1], PORT_TYPES.AUTOGROUNDED)
    ]
    # for sp in ports:
    #     print(sp.point)
    ml_terminal.set_ports(ports)

    ml_terminal.send_polygons(design.cell, design.layer_ph)
    ml_terminal.set_linspace_sweep(7, 7, 1)
    print("simulating...")
    result_path = ml_terminal.start_simulation(wait=True)
    ml_terminal.release()

    ### SIMULATION SECTION END ###

    ### CALCULATE C_QR CAPACITANCE SECTION START ###
    C12 = None
    with open(result_path.decode("ascii"), "r") as csv_file:
        data_rows = list(csv.reader(csv_file))
        ports_imps_row = data_rows[6]
        R = float(ports_imps_row[0].split(' ')[1])
        data_row = data_rows[8]
        freq0 = float(data_row[0])

        s = [[0, 0], [0, 0]]  # s-matrix
        # print(data_row)
        for i in range(0, 2):
            for j in range(0, 2):
                s[i][j] = complex(float(data_row[1 + 2 * (i * 2 + j)]),
                                  float(data_row[
                                            1 + 2 * (i * 2 + j) + 1]))
        for i in range(0, 2):
            for j in range(0, 2):
                s[i][j] = complex(
                    float(data_row[1 + 2 * (i * 2 + j)]),
                    float(data_row[1 + 2 * (i * 2 + j) + 1]))

        import math
        delta = (1 + s[0][0]) * (1 + s[1][1]) - s[0][1] * s[1][0]
        y11 = 1 / R * ((1 - s[0][0]) * (1 + s[1][1]) + s[0][1] * s[1][0]) / delta
        y22 = 1 / R * ((1 - s[1][1]) * (1 + s[0][0]) + s[0][1] * s[1][0]) / delta
        C1 = -1e15 / (2 * math.pi * freq0 * 1e9 * (1 / y11).imag)
        C2 = -1e15 / (2 * math.pi * freq0 * 1e9 * (1 / y22).imag)
        # formula taken from https://en.wikipedia.org/wiki/Admittance_parameters#Two_port
        y21 = -2 * s[1][0] / delta * 1 / R
        C12 = 1e15 / (2 * math.pi * freq0 * 1e9 * (1 / y21).imag)
        C1 -= C12
        C2 -= C12

        ### CALCULATE C_QR CAPACITANCE SECTION START ###

        ### SAVING REUSLTS SECTION START ###
        design.layout.write(str(PROJECT_DIR / "purcell_filter" / "final_design" / "capacitance" / (filename[:-4] + '.gds')))
        output_filepath = PROJECT_DIR / "purcell_filter" / "final_design" / "capacitance" / filename
        if os.path.exists(str(output_filepath)):
            # append data to file
            with open(str(output_filepath), "a", newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(
                    [C12, C1, C2]
                )
        else:
            # create file, add header, append data
            with open(str(output_filepath), "w", newline='') as csv_file:
                writer = csv.writer(csv_file)
                # create header of the file
                writer.writerow(
                    ["C12, fF", "C1, fF", "C2, fF"])
                writer.writerow(
                    [C12, C1, C2]
                )

        design.save_logdata(PROJECT_DIR / "purcell_filter" / "final_design" / "capacitance" / (filename[:-4] + '.xml'),
                            comment=f'Capacitance simulation results saved at {filename}.')

        ### SAVING RESULTS SECTION END ###


def simulate_T_junction_S_pars(filename='t_junc_S12.csv'):
    resolution_dx = 2e3
    resolution_dy = 2e3

    ### DRAWING SECTION START ###
    design = ResonatorWithFilter("testScript")

    design.draw({'readline', 'line_capacitor'})

    worm = design.readline
    t_worm = design.line_capacitor[-1]

    box_side_x = 200e3
    box_side_y = 200e3
    dv = DPoint(box_side_x / 2, box_side_y / 2)
    center = DPoint(design.line_capacitor[-1].end.x, design.line_capacitor[-1].end.y)

    crop_box = pya.Box().from_dbox(pya.Box(
        center + dv,
        center + (-1) * dv
    ))
    design.crop(crop_box)
    dr = DPoint(0, 0) - crop_box.p1

    # finding the furthest edge of cropped resonator`s central line polygon
    # sonnet port will be attached to this edge
    reg1 = worm.metal_region & Region(crop_box)
    reg1.merge()
    max_distance = box_side_x/2
    port_pt = None
    for poly in reg1.each():
        for edge in poly.each_edge():
            edge_center = (edge.p1 + edge.p2) / 2
            dp = edge_center - center
            d = max(abs(dp.x), abs(dp.y))
            if d >= max_distance:
                max_distance = d
                design.sonnet_ports.append(edge_center)

    reg1 = t_worm.metal_region & Region(crop_box)
    reg1.merge()
    max_distance = 0
    port_pt = None
    for poly in reg1.each():
        for edge in poly.each_edge():
            edge_center = (edge.p1 + edge.p2) / 2
            dp = edge_center - center
            d = max(abs(dp.x), abs(dp.y))
            if d > max_distance:
                max_distance = d
                port_pt = edge_center
    design.sonnet_ports.append(port_pt)

    design.transform_region(design.region_ph, DTrans(dr.x, dr.y),
                            trans_ports=True)

    design.show()
    design.lv.zoom_fit()
    ### DRAWING SECTION END ###

    simulate_S_pars(design, crop_box, filename)


def simulate_only_filter_S_pars(filename='filter_S12.csv'):
    ### DRAWING SECTION START ###
    design = ResonatorWithFilter("testScript")

    design.draw({'readline', 'line_capacitor', 'filter'})

    worm = design.readline
    center = design.filter.start + DPoint(-design.res_width / 2, design.res_height - design.res_line_gap*2 + 1e3)

    box_side_x = 3 * design.res_width
    box_side_y = design.res_height * 2 + design.res_line_gap*4
    dv = DPoint(box_side_x / 2, box_side_y / 2)

    crop_box = pya.Box().from_dbox(pya.Box(
        center + dv,
        center + (-1) * dv
    ))
    design.crop(crop_box)
    dr = DPoint(0, 0) - crop_box.p1

    # finding the furthest edge of cropped resonator`s central line polygon
    # sonnet port will be attached to this edge
    reg1 = worm.metal_region & Region(crop_box)
    reg1.merge()
    max_distance = center.x
    port_pt = None
    for poly in reg1.each():
        for edge in poly.each_edge():
            edge_center = (edge.p1 + edge.p2) / 2
            dp = edge_center - design.line_cap_mid
            d = max(abs(dp.x), abs(dp.y))
            if edge_center.x > max_distance:
                port_pt = edge_center
                max_distance = edge_center.x
    design.sonnet_ports.append(port_pt)
    
    reg1 = worm.metal_region & Region(crop_box)
    reg1.merge()
    max_distance = center.x
    port_pt = None
    for poly in reg1.each():
        for edge in poly.each_edge():
            edge_center = (edge.p1 + edge.p2) / 2
            dp = edge_center - design.line_cap_mid
            d = max(abs(dp.x), abs(dp.y))
            if edge_center.x < max_distance:
                port_pt = edge_center
                max_distance = edge_center.x
    design.sonnet_ports.append(port_pt)

    design.transform_region(design.region_ph, DTrans(dr.x, dr.y),
                            trans_ports=True)

    design.show()
    design.lv.zoom_fit()
    ### DRAWING SECTION END ###

    for prt in design.sonnet_ports:
        print(prt)
    
    simulate_S_pars(design, crop_box, filename, 6.75, 7.5)


def simulate_filter_w_Cfr_S_pars(filename='filter_w_Cfr_S12.csv', min_freq=7.0, max_freq=7.5):
    ### DRAWING SECTION START ###
    design = ResonatorWithFilter("testScript")

    design.draw({'readline', 'line_capacitor', 'filter', 'coupler_capacitor'})

    worm = design.readline
    center = design.filter.start + DPoint(-design.res_width * design.capacitor_ratio,
                                          design.res_height - design.filter_resonator_gap/2 - design.res_line_gap)

    box_side_x = 3 * design.res_width
    box_side_y = design.res_height * 2 + design.res_line_gap * 4 + design.filter_resonator_gap
    dv = DPoint(box_side_x / 2, box_side_y / 2)

    crop_box = pya.Box().from_dbox(pya.Box(
        center + dv,
        center + (-1) * dv
    ))
    design.crop(crop_box)
    dr = DPoint(0, 0) - crop_box.p1

    # finding the furthest edge of cropped resonator`s central line polygon
    # sonnet port will be attached to this edge
    reg1 = worm.metal_region & Region(crop_box)
    reg1.merge()
    max_distance = center.x
    port_pt = None
    for poly in reg1.each():
        for edge in poly.each_edge():
            edge_center = (edge.p1 + edge.p2) / 2
            dp = edge_center - design.line_cap_mid
            d = max(abs(dp.x), abs(dp.y))
            if edge_center.x > max_distance:
                port_pt = edge_center
                max_distance = edge_center.x
    design.sonnet_ports.append(port_pt)

    reg1 = worm.metal_region & Region(crop_box)
    reg1.merge()
    max_distance = center.x
    port_pt = None
    for poly in reg1.each():
        for edge in poly.each_edge():
            edge_center = (edge.p1 + edge.p2) / 2
            dp = edge_center - design.line_cap_mid
            d = max(abs(dp.x), abs(dp.y))
            if edge_center.x < max_distance:
                port_pt = edge_center
                max_distance = edge_center.x
    design.sonnet_ports.append(port_pt)

    design.transform_region(design.region_ph, DTrans(dr.x, dr.y),
                            trans_ports=True)

    design.show()
    design.lv.zoom_fit()
    ### DRAWING SECTION END ###

    for prt in design.sonnet_ports:
        print(prt)

    simulate_S_pars(design, crop_box, filename, min_freq=min_freq, max_freq=max_freq)


def simulate_resonator_w_Cfr_S_pars(filename='resonator_w_Cfr_S12.csv', min_freq=7.0, max_freq=7.5):
    ### DRAWING SECTION START ###
    design = ResonatorWithFilter("testScript")

    design.draw_for_res_sim()

    worm = design.res_readline
    center = design.resonator.start + DPoint(0, -design.res_height/2)

    box_side_x = 2 * design.res_width + design.transmon.sideX_length*2
    box_side_y = design.res_height * 2 + design.res_line_gap * 8 + design.filter_resonator_gap
    dv = DPoint(box_side_x / 2, box_side_y / 2)

    crop_box = pya.Box().from_dbox(pya.Box(
        center + dv,
        center + (-1) * dv
    ))
    design.crop(crop_box)
    dr = DPoint(0, 0) - crop_box.p1

    # finding the furthest edge of cropped resonator`s central line polygon
    # sonnet port will be attached to this edge
    reg1 = worm.metal_region & Region(crop_box)
    reg1.merge()
    max_distance = center.x
    port_pt = None
    for poly in reg1.each():
        for edge in poly.each_edge():
            edge_center = (edge.p1 + edge.p2) / 2
            if edge_center.x < max_distance:
                port_pt = edge_center
                max_distance = edge_center.x

    for poly in reg1.each():
        for edge in poly.each_edge():
            edge_center = (edge.p1 + edge.p2) / 2
            if edge_center.x == max_distance:
                design.sonnet_ports.append(edge_center)

    design.transform_region(design.region_ph, DTrans(dr.x, dr.y),
                            trans_ports=True)

    design.show()
    design.lv.zoom_fit()
    ### DRAWING SECTION END ###

    for prt in design.sonnet_ports:
        print(prt)

    simulate_S_pars(design, crop_box, filename, min_freq=min_freq, max_freq=max_freq)


def simulate_filter_w_res_S_pars(filename='filter_w_res_S12.csv', min_freq=7.0, max_freq=7.5):
    ### DRAWING SECTION START ###
    design = ResonatorWithFilter("testScript")

    design.draw(parts={'readline', 'line_capacitor', 'filter', 'resonator', 'transmon'})

    worm = design.readline
    worm_start = worm.start

    box_side_x = design.readline_len
    box_side_y = 2 * (design.coup_cap_mid.y - design.transmon.center.y + design.transmon.sideY_length)
    dv = Point(box_side_x / 2, box_side_y / 2)

    crop_box = pya.Box().from_dbox(pya.Box(
        design.coup_cap_mid + dv,
        design.coup_cap_mid + (-1) * dv
    ))
    design.crop(crop_box)
    dr = DPoint(0, 0) - crop_box.p1

    # finding the furthest edge of cropped resonator`s central line polygon
    # sonnet port will be attached to this edge
    reg1 = worm.metal_region & Region(crop_box)
    reg1.merge()
    max_distance = design.readline.start.y - design.coup_cap_mid.y + design.cop_pars.width
    port_pt = None
    for poly in reg1.each():
        for edge in poly.each_edge():
            edge_center = (edge.p1 + edge.p2) / 2
            dp = edge_center - DPoint(design.coup_cap_mid.x, design.coup_cap_mid.y)
            d = max(abs(dp.x), abs(dp.y))
            if d > max_distance:
                design.sonnet_ports.append(edge_center)

    design.transform_region(design.region_ph, DTrans(dr.x, dr.y),
                            trans_ports=True)

    design.show()
    design.lv.zoom_fit()
    ### DRAWING SECTION END ###

    for prt in design.sonnet_ports:
        print(prt)

    simulate_S_pars(design, crop_box, filename, min_freq=min_freq, max_freq=max_freq)


def simulate_full_S_pars(filename='ps_S12.csv', min_freq=6.8, max_freq=7.8):
    ### DRAWING SECTION START ###
    design = ResonatorWithFilter("testScript")

    design.draw()

    worm = design.readline
    worm_start = worm.start

    box_side_x = design.res_width*4
    box_side_y = 2*(design.coup_cap_mid.y - design.transmon.center.y + design.transmon.sideY_length)
    dv = DPoint(box_side_x / 2, box_side_y / 2)

    crop_box = pya.Box().from_dbox(pya.Box(
        design.coup_cap_mid + dv,
        design.coup_cap_mid + (-1) * dv
    ))
    design.crop(crop_box)
    dr = DPoint(0, 0) - crop_box.p1

    # finding the furthest edge of cropped resonator`s central line polygon
    # sonnet port will be attached to this edge
    reg1 = worm.metal_region & Region(crop_box)
    reg1.merge()
    max_distance = box_side_x/2
    port_pt = None
    for poly in reg1.each():
        for edge in poly.each_edge():
            edge_center = (edge.p1 + edge.p2) / 2
            dp = edge_center - DPoint(design.coup_cap_mid.x, design.coup_cap_mid.y)
            d = abs(dp.x)
            if d >= max_distance:
                design.sonnet_ports.append(edge_center)

    design.transform_region(design.region_ph, DTrans(dr.x, dr.y),
                            trans_ports=True)

    design.show()
    design.lv.zoom_fit()
    ### DRAWING SECTION END ###

    for prt in design.sonnet_ports:
        print(prt)

    simulate_S_pars(design, crop_box, filename, min_freq=min_freq, max_freq=max_freq)

def simulate_filter_w_Cfr_S_pars_parametr_len(res_w, filename='filter_w_Cfr_S12', min_freq=7.0, max_freq=7.5):
    ### DRAWING SECTION START ###
    design = ResonatorWithFilter("testScript")
    design.filt_subheight = res_w

    design.draw({'readline', 'line_capacitor', 'filter', 'coupler_capacitor'})

    worm = design.readline
    center = design.filter.start + DPoint(-design.res_width * design.capacitor_ratio,
                                          design.res_height - design.filter_resonator_gap/2 - design.res_line_gap)

    box_side_x = 3 * design.res_width
    box_side_y = design.res_height * 2 + design.res_line_gap * 4 + design.filter_resonator_gap
    dv = DPoint(box_side_x / 2, box_side_y / 2)

    crop_box = pya.Box().from_dbox(pya.Box(
        center + dv,
        center + (-1) * dv
    ))
    design.crop(crop_box)
    dr = DPoint(0, 0) - crop_box.p1

    # finding the furthest edge of cropped resonator`s central line polygon
    # sonnet port will be attached to this edge
    reg1 = worm.metal_region & Region(crop_box)
    reg1.merge()
    max_distance = center.x
    port_pt = None
    for poly in reg1.each():
        for edge in poly.each_edge():
            edge_center = (edge.p1 + edge.p2) / 2
            dp = edge_center - design.line_cap_mid
            d = max(abs(dp.x), abs(dp.y))
            if edge_center.x > max_distance:
                port_pt = edge_center
                max_distance = edge_center.x
    design.sonnet_ports.append(port_pt)

    reg1 = worm.metal_region & Region(crop_box)
    reg1.merge()
    max_distance = center.x
    port_pt = None
    for poly in reg1.each():
        for edge in poly.each_edge():
            edge_center = (edge.p1 + edge.p2) / 2
            dp = edge_center - design.line_cap_mid
            d = max(abs(dp.x), abs(dp.y))
            if edge_center.x < max_distance:
                port_pt = edge_center
                max_distance = edge_center.x
    design.sonnet_ports.append(port_pt)

    design.transform_region(design.region_ph, DTrans(dr.x, dr.y),
                            trans_ports=True)

    design.show()
    design.lv.zoom_fit()
    ### DRAWING SECTION END ###

    for prt in design.sonnet_ports:
        print(prt)

    simulate_S_pars(design, crop_box, filename + f'_{res_w}.csv', min_freq=min_freq, max_freq=max_freq)

def simulate_resonator_w_Cfr_S_pars_parametr_len(res_w, filename='resonator_w_Cfr_S12', min_freq=7.0, max_freq=7.5):
    ### DRAWING SECTION START ###
    design = ResonatorWithFilter("testScript")
    design.res_subheight = res_w

    design.draw_for_res_sim()

    worm = design.res_readline
    center = design.resonator.start + DPoint(0, -design.res_height/2)

    box_side_x = 2 * design.res_width + design.transmon.sideX_length*4
    box_side_y = design.res_height * 2 + design.res_line_gap * 8 + design.filter_resonator_gap
    dv = DPoint(box_side_x / 2, box_side_y / 2)

    crop_box = pya.Box().from_dbox(pya.Box(
        center + dv,
        center + (-1) * dv
    ))
    design.crop(crop_box)
    dr = DPoint(0, 0) - crop_box.p1

    # finding the furthest edge of cropped resonator`s central line polygon
    # sonnet port will be attached to this edge
    reg1 = worm.metal_region & Region(crop_box)
    reg1.merge()
    max_distance = center.x
    port_pt = None
    for poly in reg1.each():
        for edge in poly.each_edge():
            edge_center = (edge.p1 + edge.p2) / 2
            if edge_center.x < max_distance:
                port_pt = edge_center
                max_distance = edge_center.x

    for poly in reg1.each():
        for edge in poly.each_edge():
            edge_center = (edge.p1 + edge.p2) / 2
            if edge_center.x == max_distance:
                design.sonnet_ports.append(edge_center)

    design.transform_region(design.region_ph, DTrans(dr.x, dr.y),
                            trans_ports=True)

    design.show()
    design.lv.zoom_fit()
    ### DRAWING SECTION END ###

    for prt in design.sonnet_ports:
        print(prt)

    simulate_S_pars(design, crop_box, filename + f'_{res_w}.csv', min_freq=min_freq, max_freq=max_freq)

def simulate_full_S_pars_variational(filename='ps_S12'):
    ### DATA SECTION ###
    Lf = np.array([204, 168, 134, 102])*1e3
    Lr = np.array([230, 194, 160, 128])*1e3
    vv = np.linspace(7.3, 7.9, 4)
    for i in range(len(vv)):
        ### DRAWING SECTION START ###
        design = ResonatorWithFilter("testScript")
        design.res_subheight = Lr[i]
        design.filt_subheight = Lf[i]

        design.draw()

        worm = design.readline
        worm_start = worm.start

        box_side_x = design.res_width*4
        box_side_y = 2*(design.coup_cap_mid.y - design.transmon.center.y + design.transmon.sideY_length)
        dv = DPoint(box_side_x / 2, box_side_y / 2)

        crop_box = pya.Box().from_dbox(pya.Box(
            design.coup_cap_mid + dv,
            design.coup_cap_mid + (-1) * dv
        ))
        design.crop(crop_box)
        dr = DPoint(0, 0) - crop_box.p1

        # finding the furthest edge of cropped resonator`s central line polygon
        # sonnet port will be attached to this edge
        reg1 = worm.metal_region & Region(crop_box)
        reg1.merge()
        max_distance = box_side_x/2
        port_pt = None
        for poly in reg1.each():
            for edge in poly.each_edge():
                edge_center = (edge.p1 + edge.p2) / 2
                dp = edge_center - DPoint(design.coup_cap_mid.x, design.coup_cap_mid.y)
                d = abs(dp.x)
                if d >= max_distance:
                    design.sonnet_ports.append(edge_center)

        design.transform_region(design.region_ph, DTrans(dr.x, dr.y),
                                trans_ports=True)

        design.show()
        design.lv.zoom_fit()
        ### DRAWING SECTION END ###

        for prt in design.sonnet_ports:
            print(prt)

        simulate_S_pars(design, crop_box, filename + f'_{vv[i]:.03f}.csv', min_freq=vv[i]-0.1, max_freq=vv[i]+0.1)


def simulate_S_pars(design, crop_box, filename, min_freq=6.5, max_freq=7.5, resolution_dx=2e3, resolution_dy=2e3):
    ### SIMULATION SECTION START ###
    ml_terminal = SonnetLab()
    from sonnetSim.cMD import CMD

    ml_terminal._send(CMD.SAY_HELLO)
    ml_terminal.clear()
    simBox = SimulationBox(
        crop_box.width(), crop_box.height(),
        crop_box.width() / resolution_dx,
        crop_box.height() / resolution_dy
    )

    ml_terminal.set_boxProps(simBox)
    from sonnetSim.pORT_TYPES import PORT_TYPES

    ports = [
        SonnetPort(prt, PORT_TYPES.BOX_WALL) for prt in design.sonnet_ports
    ]
    ml_terminal.set_ports(ports)
    ml_terminal.send_polygons(design.cell, design.layer_ph)
    ml_terminal.set_ABS_sweep(min_freq, max_freq)
    # print(f"simulating...{resonator_idx}")
    result_path = ml_terminal.start_simulation(wait=True)
    ml_terminal.release()

    all_params = design.get_geometry_parameters()

    # creating directory with simulation results
    results_dirpath = PROJECT_DIR / "purcell_filter" / "final_design" / "resonators_S12"

    shutil.copy(
        result_path.decode("ascii"),
        str(results_dirpath / filename)
    )

    design.layout.write(str(results_dirpath / (filename[:-4] + '.gds')))
    design.save_logdata(results_dirpath / (filename[:-4] + '.xml'),
                        comment=f'S-parameters simulation results saved at {filename}.')

    ### RESULT SAVING SECTION END ###



### MAIN FUNCTION ###
if __name__ == "__main__":
    my_design = ResonatorWithFilter("testScript")
    my_design.draw()
    my_design.show()

    length = my_design.resonator.length(exception='fork')
    light_speed = 299792458 / refractive_index  # m/s
    freq = light_speed / (4 * length)  # GHz
    print(f'l_r = {length} um')
    length = my_design.filter.length(exception='fork')
    light_speed = 299792458 / refractive_index  # m/s
    freq = light_speed / (4 * length)  # GHz
    print(f'f_f = {freq} GHz')
    print(f'l_f = {length} um')

    # simulate_filter_w_res_S_pars()

    # simulate_filter_w_Cfr_S_pars()

    # simulate_only_filter_S_pars()

    # simulate_T_junction_S_pars()

    # simulate_full_S_pars()

    # simulate_resonator_w_Cfr_S_pars()

    # simulate_Cfl()
    # simulate_Cfr()
    # simulate_Cqr()

    # for i in range(100, 260, 20):
    #     simulate_filter_w_Cfr_S_pars_parametr_len(i*1e3, min_freq=7.1, max_freq=8.)
    #     simulate_resonator_w_Cfr_S_pars_parametr_len(i*1e3, min_freq=7.1, max_freq=8.)
