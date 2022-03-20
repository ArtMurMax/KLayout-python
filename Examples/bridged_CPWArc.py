from typing import List
from importlib import reload
from collections import namedtuple, Counter
import copy
from typing import Union, Optional

import pya
from pya import Point, DPoint, Vector, DVector, DPath, \
    DSimplePolygon, SimplePolygon, DPolygon, Polygon, Region, Box, DBox
from pya import Trans, DTrans, CplxTrans, DCplxTrans, ICplxTrans

import classLib

reload(classLib)
from classLib.baseClasses import ComplexBase
from classLib.chipDesign import ChipDesign
from classLib.shapes import Circle, Rectangle
from classLib.coplanars import CPW, CPWParameters, CPW2CPW, \
    CPWArc, CPWRLPath, CPW2CPWArc, DPathCPW, Bridge1

from classLib.chipTemplates import CHIP_10x10_12pads

import classLib.josJ
reload(classLib.josJ)
from classLib.josJ import AsymSquidDCFlux, AsymSquidDCFluxParams
from classLib.shapes import DPathCL

# exclude
import numpy as np

from classLib import ElementBase

SQUID_PARAMETERS = AsymSquidDCFluxParams(
    pad_r=5e3, pads_distance=30e3,
    contact_pad_width=10e3, contact_pad_ext_r=200,
    sq_dy=15e3, sq_area=200e6,
    j1_dx=95, j2_dx=348,
    inter_leads_width=500, b_ext=2e3, j1_dy=94, n=20,
    bridge=180, j2_dy=250,
    flux_line_dx=30e3, flux_line_dy=10e3, flux_line_outer_width=1e3,
    flux_line_inner_width=370, flux_line_IO_transition_L=100
)


class MyDesign(ChipDesign):
    def draw(self):
        info_bridges1 = pya.LayerInfo(3, 0)  # bridge photo layer 1
        self.region_bridges1 = Region()
        self.layer_bridges1 = self.layout.layer(info_bridges1)

        info_bridges2 = pya.LayerInfo(4, 0)  # bridge photo layer 2
        self.region_bridges2 = Region()
        self.layer_bridges2 = self.layout.layer(info_bridges2)

        self.lv.add_missing_layers()

        self.draw_chip()
        self.draw_arc()
        self.bridgify_arc()

    def draw_chip(self):
        self.chip = CHIP_10x10_12pads
        self.chip_box: pya.DBox = self.chip.box

        self.region_bridges2.insert(self.chip_box)
        self.region_ph.insert(self.chip_box)

    def draw_arc(self):
        self.Z0 = CPWParameters(200e3, 100e3)
        cpw_start = (self.chip_box.p1 + self.chip_box.p2)/2
        self.arc = CPWArc(
            z0=self.Z0, start=cpw_start,
            R=1.2e6, delta_alpha=5/4*np.pi,
            trans_in=DTrans.R0
        )
        self.arc.place(self.region_ph)

    def bridgify_arc(self):
        print(isinstance(self.arc, CPWArc))
        print(type(self.arc))
        Bridge1.bridgify_CPW(
            self.arc,
            bridges_step=100e3,
            dest=self.region_bridges1,
            dest2=self.region_bridges2,
        )


    def _transfer_regs2cell(self):
        self.cell.shapes(self.layer_ph).insert(self.region_ph)
        self.cell.shapes(self.layer_el).insert(self.region_el)
        self.cell.shapes(self.layer_bridges1).insert(self.region_bridges1)
        self.cell.shapes(self.layer_bridges2).insert(self.region_bridges2)
        self.lv.zoom_fit()


### MAIN FUNCTION ###
if __name__ == "__main__":
    my_design = MyDesign("testScript")
    my_design.draw()
    my_design.show()
