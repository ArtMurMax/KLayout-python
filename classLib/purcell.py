import pya
from pya import Point, DPoint, Vector, DVector, DSimplePolygon, SimplePolygon, DPolygon, Polygon, Region, Box, DBox
from pya import Trans, DTrans, CplxTrans, DCplxTrans, ICplxTrans

import classLib
from classLib.chipDesign import ChipDesign
from classLib.shapes import TmonT, Rectangle
from classLib.coplanars import CPWRLPath, CPWParameters, CPW
from classLib.baseClasses import ElementBase, ComplexBase
from classLib.chipTemplates import CHIP_16p5x16p5_20pads
from classLib.resonators import EMResonatorTL3QbitWormRLTailXmonFork, EMResonatorTL3QbitWormRLTail
from collections import OrderedDict

from classLib.LogManager import LogManager


class CapacitorParams(ElementBase):
    plate_pars: CPWParameters
    cap_len: float
    cap_gap: float
    metal_gap: float

    def __init__(self, plate_pars, cap_len, cap_gap, metal_gap=False):
        self.plate_pars = plate_pars
        self.cap_len = cap_len
        self.cap_gap = cap_gap
        self.metal_gap = metal_gap

        self._geometry_parameters = {"plate_width, um": self.plate_pars.width,
                                     "plate_gnd_gap, um": self.plate_pars.gap,
                                     "cap_len, um": self.cap_len,
                                     "cap_gap, um": self.cap_gap,
                                     "metal_gap": self.metal_gap}


class EMResonatorCapCoupLineXmonFork(EMResonatorTL3QbitWormRLTailXmonFork):
    def __init__(self, start, Z0, cap_line_len: float,
                 reader_line_cap: CapacitorParams,
                 L_coupling, L0, L1, r, N,
                 tail_shape, tail_turn_radiuses,
                 tail_segment_lengths, tail_turn_angles,
                 fork_x_span, fork_y_span, fork_metal_width, fork_gnd_gap,
                 tail_trans_in=None,
                 coupling_ratio=0.5,
                 trans_in=None):

        self.start = start
        self.cop_pars = Z0
        self.cap_line_len = cap_line_len
        self.capacitor_ratio = coupling_ratio
        self.coup_capacitor_len = reader_line_cap.cap_len
        self.coup_cap_gap = reader_line_cap.cap_gap
        self.metal_gap = reader_line_cap.metal_gap

        self.cap_plate_par = reader_line_cap.plate_pars

        super().__init__(
            Z0, start + DPoint(L_coupling*coupling_ratio, cap_line_len + Z0.width/2),
            L_coupling, L0, L1, r, N,
            tail_shape, tail_turn_radiuses,
            tail_segment_lengths, tail_turn_angles,
            fork_x_span, fork_y_span, fork_metal_width, fork_gnd_gap,
            tail_trans_in, trans_in
        )

    def init_primitives(self):
        super().init_primitives()

        ### Setting filter-resonator capacitor ###
        coup_cap_start = DPoint(self.L_coupling*self.capacitor_ratio, self.cap_line_len + self.Z0.width/2)
        coup_cap_end = DPoint(self.L_coupling*self.capacitor_ratio, self.cap_line_len + self.Z0.width/2) - DPoint(0, self.cap_line_len)
        self.coup_cap_mid = (coup_cap_start + coup_cap_end) / 2

        # erase gaps at the capacitors corner
        self.primitives['corner_gap_1'] = CPW(
                    self.coup_cap_gap - 2 * self.cop_pars.gap, self.cap_plate_par.width + self.cap_plate_par.gap,
                    start=self.coup_cap_mid - DPoint(self.coup_capacitor_len / 2, 0),
                    end=self.coup_cap_mid - DPoint(self.coup_capacitor_len / 2 + self.cop_pars.gap, 0)
                )
        self.primitives['corner_gap_2'] = CPW(
                self.coup_cap_gap - 2 * self.cop_pars.gap, self.cap_plate_par.width + self.cap_plate_par.gap,
                start=self.coup_cap_mid - DPoint(-self.coup_capacitor_len / 2, 0),
                end=self.coup_cap_mid - DPoint(-self.coup_capacitor_len / 2 - self.cop_pars.gap, 0)
            )

        self.primitives['plate_1'] = CPW(
                start=self.coup_cap_mid - DPoint(self.coup_capacitor_len / 2,
                                                 self.coup_cap_gap / 2 + self.cap_plate_par.width / 2),
                end=self.coup_cap_mid - DPoint(-self.coup_capacitor_len / 2,
                                               self.coup_cap_gap / 2 + self.cap_plate_par.width / 2),
                cpw_params=self.cap_plate_par
            )
        self.primitives['plate_2'] = CPW(
                start=self.coup_cap_mid + DPoint(self.coup_capacitor_len / 2,
                                                 self.coup_cap_gap / 2 + self.cap_plate_par.width / 2),
                end=self.coup_cap_mid + DPoint(-self.coup_capacitor_len / 2,
                                               self.coup_cap_gap / 2 + self.cap_plate_par.width / 2),
                cpw_params=self.cap_plate_par
            )
        # add lines to resonators
        self.primitives['input_cap_line'] = CPW(start=coup_cap_start,
            end=self.coup_cap_mid + DPoint(0, self.coup_cap_gap / 2 + self.cap_plate_par.width),
            cpw_params=self.cop_pars)
        self.primitives['cap_reader_line'] = CPW(start=self.coup_cap_mid - DPoint(0, self.coup_cap_gap / 2 + self.cap_plate_par.width), end=coup_cap_end,
            cpw_params=self.cop_pars)


class LinearCapacitor(ComplexBase):

    def __init__(self, start, Z0, cap_line_len: float,
                 filter_line_cap: CapacitorParams,
                 trans_in=None):

        self.start = start
        self.cop_pars = Z0
        self.cap_line_len = cap_line_len

        self.coup_capacitor_len = filter_line_cap.cap_len
        self.coup_cap_gap = filter_line_cap.cap_gap

        self.cap_plate_par = filter_line_cap.plate_pars
        self.metal_gap = filter_line_cap.metal_gap

        super().__init__(start, trans_in=trans_in)

    def init_primitives(self):

        ### Setting capacitor ###
        coup_cap_start = DPoint(0,0)
        coup_cap_end = DPoint(0, -self.cap_line_len)
        self.coup_cap_mid = (coup_cap_start + coup_cap_end) / 2

        # erase gaps at the capacitors corner
        if self.metal_gap:
            self.primitives['corner_gap_1'] = CPW(
                self.coup_cap_gap - 2 * self.cop_pars.gap, self.cap_plate_par.width + self.cap_plate_par.gap,
                start=self.coup_cap_mid - DPoint(self.coup_capacitor_len / 2, 0),
                end=self.coup_cap_mid - DPoint(self.coup_capacitor_len / 2 + self.cop_pars.gap, 0)
            )
            self.primitives['corner_gap_2'] = CPW(
                self.coup_cap_gap - 2 * self.cop_pars.gap, self.cap_plate_par.width + self.cap_plate_par.gap,
                start=self.coup_cap_mid - DPoint(-self.coup_capacitor_len / 2, 0),
                end=self.coup_cap_mid - DPoint(-self.coup_capacitor_len / 2 - self.cop_pars.gap, 0)
            )
        else:
            self.primitives['corner_gap_1'] = CPW(
                    0, self.cap_plate_par.width + 3 / 2 * self.cap_plate_par.gap,
                    start=self.coup_cap_mid - DPoint(self.coup_capacitor_len / 2, 0),
                    end=self.coup_cap_mid - DPoint(self.coup_capacitor_len / 2 + self.cop_pars.gap, 0)
                )
            self.primitives['corner_gap_2'] = CPW(
                    0, self.cap_plate_par.width + 3 / 2 * self.cap_plate_par.gap,
                    start=self.coup_cap_mid - DPoint(-self.coup_capacitor_len / 2, 0),
                    end=self.coup_cap_mid - DPoint(-self.coup_capacitor_len / 2 - self.cop_pars.gap, 0)
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
        self.primitives['plate_1'] = CPW(
                start=self.coup_cap_mid - DPoint(self.coup_capacitor_len / 2,
                                                 self.coup_cap_gap / 2 + self.cap_plate_par.width / 2),
                end=self.coup_cap_mid - DPoint(-self.coup_capacitor_len / 2,
                                               self.coup_cap_gap / 2 + self.cap_plate_par.width / 2),
                cpw_params=self.cap_plate_par
            )
        self.primitives['plate_2'] = CPW(
                start=self.coup_cap_mid + DPoint(self.coup_capacitor_len / 2,
                                                 self.coup_cap_gap / 2 + self.cap_plate_par.width / 2),
                end=self.coup_cap_mid + DPoint(-self.coup_capacitor_len / 2,
                                               self.coup_cap_gap / 2 + self.cap_plate_par.width / 2),
                cpw_params=self.cap_plate_par
            )
        # add lines to resonators
        self.primitives['input_cap_line_coil'] = CPW(start=coup_cap_start,
            end=self.coup_cap_mid + DPoint(0, self.coup_cap_gap / 2 + self.cap_plate_par.width),
            cpw_params=self.cop_pars)
        self.primitives['cap_reader_line_coil'] = CPW(start=self.coup_cap_mid - DPoint(0, self.coup_cap_gap / 2 + self.cap_plate_par.width), end=coup_cap_end,
            cpw_params=self.cop_pars)

        self.connections = [coup_cap_start, coup_cap_end]
        self.angle_connections = [0, 180]