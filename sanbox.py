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
    CPWArc, CPWRLPath, CPW2CPWArc, DPathCPW

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
    sq_len=15e3, sq_area=200e6,
    j1_dx=95, j2_dx=348,
    inter_leads_width=500, b_ext=2e3, j1_dy=94, n=20,
    bridge=180, j2_dy=250,
    flux_line_dx=30e3, flux_line_dy=10e3, flux_line_outer_width=1e3,
    flux_line_inner_width=370, flux_line_IO_transition_L=100
)


class MyDesign(ChipDesign):
    def draw(self):
        self.draw_DPathCPW_test()

    def draw_DPathCPW_test(self):
        origin=DPoint(0, 0)
        dx, dy = 5e3, 5e3

        from itertools import product
        from random import randrange, seed
        seed(52)
        for i_y, i_x in product(range(2), range(5)):
            origin_loc = DPoint(i_x * dx, i_y * dy)
            rec = Rectangle(origin_loc, dx, dy)
            rec.place(self.region_ph)
        for i_y, i_x in product(range(2), range(5)):
            origin_loc = DPoint(i_x * dx, i_y * dy)
            pts = [
                origin_loc + DPoint(randrange(dx), randrange(dy)) for _ in
                range(3)
            ]
            print(i_y, i_x)
            print("origin loc: ", origin_loc)
            print("points: ", [(p.x, p.y) for p in pts])
            print()
            cpw_path = DPathCPW(
                points=pts,
                cpw_parameters=[
                    CPWParameters(width=200, gap=100),
                    CPWParameters(smoothing=True),
                    CPWParameters(width=300, gap=150)
                ],
                turn_radiuses=150,
                trans_in=None
            )
            if i_x == i_y == 0:
                print(cpw_path._turn_angles)
                print(cpw_path._segment_lengths)
            cpw_path.place(self.region_ph)

    def draw_cpw2cpw_arc_test(self):
        Rectangle(DPoint(-2e3, -2e3), 4e3, 4e3).place(self.region_ph)
        cpw2cpwarc = CPW2CPWArc(
            origin=DPoint(0, 0), r=1e3,
            start_angle=-np.pi/2, end_angle=-np.pi,
            cpw1_params=CPWParameters(width=100, gap=100),
            cpw2_params=CPWParameters(width=200, gap=200),
            trans_in=None  # Trans.R90
        )
        cpw2cpwarc.place(self.region_ph)

        CPWRLPath(
            origin=DPoint(0, 0), shape="LRL",
            cpw_parameters=[
                CPWParameters(width=SQUID_PARAMETERS.flux_line_inner_width, gap=0),
                CPWParameters(smoothing=True),
                CPWParameters(width=SQUID_PARAMETERS.inter_leads_width, gap=0)
            ],
            turn_radiuses=SQUID_PARAMETERS.inter_leads_width,
            segment_lengths=[
                2e3,
                3e3
            ],
            turn_angles=[np.pi / 2],
            trans_in=None
        ).place(self.region_el)

    def draw_asym_squid_test(self):
        origin = DPoint(0, 0)
        sq = AsymSquidDCFlux(origin, SQUID_PARAMETERS)
        sq.place(self.region_el)
        self.region_el.merge()


### MAIN FUNCTION ###
if __name__ == "__main__":
    my_design = MyDesign("testScript")
    my_design.draw()
    my_design.show()
