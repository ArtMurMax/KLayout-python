from collections import namedtuple
from typing import Union, List
from math import pi

import numpy as np

import pya
from pya import DPoint, DVector, DSimplePolygon, SimplePolygon
from pya import Trans, DTrans, DVector, DPath

from classLib.baseClasses import ElementBase, ComplexBase
from classLib.shapes import Circle, Kolbaska, DPathCL
from classLib.coplanars import CPW, CPWParameters, CPWRLPath, CPW2CPW


# print("josJ reloaded")
# nice solution for parameters, man. Had really appreciated this one and put to use already (SH).


class AsymSquidParams:
    def __init__(
            self,
            pads_d=20e3,
            squid_dx=10.5e3,
            squid_dy=5e3,
            TC_dx=8e3,
            TC_dy=10e3,
            BC_dx: Union[List[float], float] = 5e3,
            BC_dy=7e3,
            beta=0.5,
            TCW_dx=1e3,
            TCW_dy=10e3,
            BCW_dx: Union[List[float], float] = 1e3,
            BCW_dy=1.5e3,
            SQT_dy=500,
            SQLTT_dx=None,
            SQRTT_dx=None,
            SQB_dy=0,
            SQLBT_dx=None,
            SQRBT_dx=None,
            alpha=0.7,
            shadow_gap=200,
            TCB_dx=2e3,
            TCB_dy=8e3,
            BCB_dx=5e3,
            BCB_dy=20e3,
            band_el_tol=1e3,
            band_ph_tol=1.5e3,
            JJC=1e3,
            SQLTJJ_dx=114.551,  # j1_dx
            SQLBJJ_dy=114.551,  # j1_dy
            SQLBJJ_dx=2e3,
            SQRTJJ_dx=398.086,  # j2_dx
            SQRBJJ_dy=250,  # j2_dy
            SQRBJJ_dx=None,
            bot_wire_x=0,
            SQRBT_dy=None
    ):
        """
        For all undocumented pararmeters see corresponding PDF schematics.

        Parameters
        ----------
        pads_d : float
        squid_dx : float
        squid_dy : float
        TC_dx : float
        BC_dx : Union[List[float], float]
        beta : float
            0 < beta < 1
        TCW_dx : float
        BCW_dx : Union[List[float],float]
        SQT_dy : float
        SQLTT_dx : Optional[float]
        SQRTT_dx : Optional[float]
        SQB_dy : Optional[float]
        SQLBT_dx : Optional[float]
        SQRBT_dx : Optional[float]
        alpha : float
            0 < alpha < 1
        shadow_gap : float
        TCB_dx : float
        TCB_dy : float
        BCB_dx : float
        BCB_dy : float
        band_el_tol : float
        band_ph_tol : float
        bot_wire_x : Union[float, List[float]]
        """
        self.pads_d = pads_d
        self.squid_dx = squid_dx
        self.squid_dy = squid_dy

        try:
            iter(bot_wire_x)
        except TypeError:
            self.bot_wire_x = [bot_wire_x]
        else:
            self.bot_wire_x = bot_wire_x
        # squid loop top side width
        self.JJC = JJC
        self.SQT_dy = SQT_dy

        # squid loop left side width if not passed equals to top side width
        self.SQLTT_dx = SQLTT_dx if SQLTT_dx is not None else SQT_dy
        left_side_half = (squid_dy - shadow_gap) / 2
        self.SQLTT_dy = alpha * left_side_half
        self.SQRTT_dy = self.SQLTT_dy
        # squid loop right top side width if not passed, equals to top
        # side width
        self.SQRTT_dx = SQRTT_dx if SQRTT_dx is not None else SQT_dy
        # squid loop bottom left side width if not passed equals to top
        # side width
        self.SQLBT_dx = SQLBT_dx if SQLBT_dx is not None else SQT_dy
        self.SQRBT_dx = SQRBT_dx if SQRBT_dx is not None else SQT_dy
        self.SQB_dx = squid_dx + self.SQLBT_dx + self.SQRBT_dx
        # squid loop bottom side width if not passed equals to top side
        # width
        self.SQB_dy = SQB_dy if SQB_dy is not None else SQT_dy
        self.SQLBT_dy = left_side_half
        self.SQRBT_dy = SQRBT_dy if SQRBT_dy is not None else self.SQLBT_dy

        self.shadow_gap = shadow_gap

        self.TC_dx = TC_dx

        # check if iterable and set iterable
        try:
            iter(BC_dx)
        except TypeError:
            self.BC_dx = [BC_dx] * len(self.bot_wire_x)
        else:
            self.BC_dx = BC_dx

        pads_top_left = (pads_d / 2 - self.SQT_dy - squid_dy / 2)
        pads_bottom_left = (pads_d / 2 - self.SQB_dy - squid_dy / 2)
        self.beta = beta
        self.TC_dy = TC_dy
        self.BC_dy = BC_dy
        self.TCW_dx = TCW_dx
        self.TCW_dy = TCW_dy
        self.BCW_dy = BCW_dy
        try:
            iter(BCW_dx)
        except TypeError:
            self.BCW_dx = [BCW_dx] * len(self.bot_wire_x)
        else:
            self.BCW_dx = BCW_dx

        self.SQT_dx = self.squid_dx + self.SQLBT_dx / 2 - self.JJC + \
                      self.SQLTT_dx / 2 + self.SQRBT_dx / 2 - self.JJC + \
                      self.SQRTT_dx / 2

        self.TCB_dx = TCB_dx
        self.TCB_dy = TCB_dy

        self.BCB_dx = BCB_dx
        self.BCB_dy = BCB_dy

        self.band_el_tol = band_el_tol
        self.band_ph_tol = band_ph_tol

        self.SQLTJJ_dx = SQLTJJ_dx
        self.SQRTJJ_dx = SQRTJJ_dx if SQRTJJ_dx is not None else SQLTJJ_dx
        self.SQLTJJ_dy = (1 - alpha) * left_side_half
        self.SQRTJJ_dy = self.SQLTJJ_dy
        self.SQLBJJ_dy = SQLBJJ_dy
        self.SQLBJJ_dx = SQLBJJ_dx

        ''' right side of the squid parameters '''
        # RHS params, if not passed, equal to LHS params
        self.SQRBJJ_dy = SQRBJJ_dy if SQRBJJ_dy is not None else SQLBJJ_dy
        self.SQRBJJ_dx = SQRBJJ_dx if SQRBJJ_dx is not None else SQLBJJ_dx


class AsymSquid(ComplexBase):
    def __init__(self, origin: DPoint, params: AsymSquidParams,
                 trans_in=None):
        """

        Parameters
        ----------
        origin : DPoint
            where to put object's local coordinate system origin (after
            trans_in is performed in local reference frame)
        params : AsymSquidParams
            see `AsymSquid2Params` class description for parameters
            details
        trans_in : DCplxTrans
            transformation to perform in object's reference frame
        """
        self.center = origin
        self.squid_params = params
        super().__init__(origin=origin, trans_in=trans_in)

    def init_primitives(self):
        # introducing shorthands for long-named variables
        origin = DPoint(0, 0)
        pars = self.squid_params

        # (TC) Top contact polygon
        tc_p1 = DPoint(0, pars.squid_dy / 2 + pars.SQT_dy + pars.TCW_dy +
                       pars.TC_dy)
        tc_p2 = tc_p1 + DVector(0, -pars.TC_dy)
        self.TC = CPW(start=tc_p1, end=tc_p2, width=pars.TC_dx, gap=0)
        self.primitives["TC"] = self.TC

        # (TCW) Top contact wire
        tcw_p1 = self.TC.end
        tcw_p2 = tcw_p1 + DVector(0, -pars.TCW_dy)
        self.TCW = CPW(start=tcw_p1, end=tcw_p2, width=pars.TCW_dx, gap=0)
        self.primitives["TCW"] = self.TCW

        # (SQT) squid loop top
        sqt_p1 = origin + DVector(-pars.SQT_dx / 2,
                                  pars.squid_dy / 2 + pars.SQT_dy / 2)
        sqt_p2 = sqt_p1 + DVector(pars.SQT_dx, 0)
        self.SQT = CPW(start=sqt_p1, end=sqt_p2, width=pars.SQT_dy, gap=0)
        self.primitives["SQT"] = self.SQT

        # (SQLTT) squid loop left top thick
        sqltt_p1 = self.SQT.start + DVector(
            pars.SQLTT_dx / 2, -pars.SQT_dy / 2
        )
        sqltt_p2 = sqltt_p1 + DVector(0, -pars.SQLTT_dy)
        self.SQLTT = CPW(
            start=sqltt_p1, end=sqltt_p2,
            width=pars.SQLTT_dx, gap=0
        )
        self.primitives["SQLTT"] = self.SQLTT

        # (SQLTJJ) squid loop left top JJ
        sqltjj_p1 = self.SQLTT.end
        sqltjj_p2 = sqltjj_p1 + DVector(0, -pars.SQLTJJ_dy)
        self.SQLTJJ = CPW(start=sqltjj_p1, end=sqltjj_p2,
                          width=pars.SQLTJJ_dx, gap=0)
        self.primitives["SQLTJJ"] = self.SQLTJJ

        # (SQB) squid bottom
        sqb_p1 = origin + DVector(-pars.squid_dx / 2 - pars.SQLBT_dx,
                                  -pars.squid_dy / 2 - pars.SQB_dy / 2)
        sqb_p2 = sqb_p1 + DVector(pars.squid_dx + pars.SQLBT_dx +
                                  pars.SQRBT_dx, 0)
        self.SQB = CPW(start=sqb_p1, end=sqb_p2, width=pars.SQB_dy, gap=0)
        self.primitives["SQB"] = self.SQB

        # (SQLBT) squid left bottom thick
        sqlbt_p1 = self.SQB.start + \
                   DVector(pars.SQLBT_dx / 2, pars.SQB_dy / 2) + \
                   DVector(0, pars.SQLBT_dy)
        sqlbt_p2 = sqlbt_p1 + DVector(0, -pars.SQLBT_dy)
        self.SQLBT = CPW(start=sqlbt_p1, end=sqlbt_p2, width=pars.SQLBT_dx,
                         gap=0)
        self.primitives["SQLBT"] = self.SQLBT

        # (SQLBJJ) squid left botton JJ
        if ((pars.SQLBT_dx / 2 + pars.SQLBJJ_dx + pars.SQLTJJ_dx) <
                pars.JJC):
            raise Warning("please, increase SQLBJJ_dy is too low")
        sqlbjj_p1 = self.SQLBT.start + DVector(pars.SQLBT_dx / 2,
                                               -pars.SQLBJJ_dy / 2)
        sqlbjj_p2 = sqlbjj_p1 + DVector(pars.SQLBJJ_dx, 0)
        self.SQLBJJ = CPW(start=sqlbjj_p1, end=sqlbjj_p2,
                          width=pars.SQLBJJ_dy,
                          gap=0)
        self.primitives["SQLBJJ"] = self.SQLBJJ

        # (SQRTT) squid loop right top thick
        sqrtt_p1 = self.SQT.end + DVector(
            -pars.SQRTT_dx / 2, -pars.SQT_dy / 2
        )
        sqrtt_p2 = sqrtt_p1 + DVector(0, -pars.SQRTT_dy)
        self.SQRTT = CPW(
            start=sqrtt_p1, end=sqrtt_p2,
            width=pars.SQRTT_dx, gap=0
        )
        self.primitives["SQRTT"] = self.SQRTT

        # (SQRTJJ) squid loop right top JJ
        sqrtjj_p1 = self.SQRTT.end
        sqrtjj_p2 = sqrtjj_p1 + DVector(0, -pars.SQRTJJ_dy)
        self.SQRTJJ = CPW(start=sqrtjj_p1, end=sqrtjj_p2,
                          width=pars.SQRTJJ_dx, gap=0)
        self.primitives["SQRTJJ"] = self.SQRTJJ

        # (SQRBT) squid right bottom thick
        sqrbt_p1 = self.SQB.end + \
                   DVector(-pars.SQRBT_dx / 2, pars.SQB_dy / 2) + \
                   DVector(0, pars.SQRBT_dy)
        sqrbt_p2 = sqrbt_p1 + DVector(0, -pars.SQRBT_dy)
        self.SQRBT = CPW(start=sqrbt_p1, end=sqrbt_p2,
                         width=pars.SQRBT_dx, gap=0)
        self.primitives["SQRBT"] = self.SQRBT

        # (SQRBJJ) squid right botton JJ
        if ((pars.SQRBT_dx / 2 + pars.SQRBJJ_dx + pars.SQRTJJ_dx) <
                pars.JJC):
            raise Warning("please, increase SQLBJJ_dy is too low")
        sqrbjj_p1 = self.SQRBT.start + DVector(-pars.SQRBT_dx / 2,
                                               -pars.SQRBJJ_dy / 2)
        sqrbjj_p2 = sqrbjj_p1 + DVector(-pars.SQRBJJ_dx, 0)
        self.SQRBJJ = CPW(start=sqrbjj_p1, end=sqrbjj_p2,
                          width=pars.SQRBJJ_dy,
                          gap=0)
        self.primitives["SQRBJJ"] = self.SQRBJJ

        ''' following code can enclude multiple bottom connections '''
        self.BC_list = []
        self.BCW_list = []
        for i, x in enumerate(pars.bot_wire_x):
            if x is None:
                self.BC_list.append(None)
                self.BCW_list.append(None)
                continue
            # (BC) bottom contact pad polygon
            bc_p1 = DPoint(x, -pars.squid_dy / 2 - pars.BCW_dy)
            bc_p2 = bc_p1 + DVector(0, -pars.BC_dy)

            BCi = CPW(start=bc_p1, end=bc_p2, width=pars.BC_dx[i], gap=0)
            self.BC_list.append(BCi)
            self.primitives["BC" + str(i)] = BCi

            # (BCW) Bottom contact wire
            bcw_p1 = self.BC_list[-1].start + DVector(0, pars.BCW_dy)
            bcw_p2 = bcw_p1 + DVector(0, -pars.BCW_dy)
            BCWi = CPW(start=bcw_p1, end=bcw_p2, width=pars.BCW_dx[i], gap=0)
            self.BCW_list.append(BCWi)
            self.primitives["BCW" + str(i)] = BCWi

            # (BCE) bottom contact extension
            if x < (-pars.SQB_dx / 2 + pars.BCW_dx[i] / 2):
                bce_p1 = self.BCW_list[i].start + \
                         DVector(-pars.BCW_dx[i] / 2, pars.SQB_dy / 2)
                bce_p2 = self.SQB.start
            elif x > (pars.SQB_dx / 2 - pars.BCW_dx[i] / 2):
                bce_p1 = self.BCW_list[i].start + \
                         DVector(pars.BCW_dx[i] / 2, pars.SQB_dy / 2)
                bce_p2 = self.SQB.end
            else:
                continue
            name = "BCE" + str(i)
            setattr(
                self,
                name,
                CPW(start=bce_p1, end=bce_p2, width=pars.SQB_dy, gap=0)
            )
            self.primitives[name] = getattr(self, name)

    def _refresh_named_connections(self):
        self.center = self.origin


AsymSquidDCFluxParams = namedtuple(
    "AsymSquidParams",
    [
        "pad_r",
        "pads_distance",
        "contact_pad_width",
        "contact_pad_ext_r",
        "sq_dy",
        "sq_area",
        "j1_dy",
        "j2_dy",
        "inter_leads_width",
        "b_ext",
        "j1_dx",
        "n",
        "bridge",
        "j2_dx",
        # last 5
        "flux_line_dx",
        "flux_line_dy",
        "flux_line_outer_width",
        "flux_line_inner_width",
        "flux_line_contact_width"
    ],
    defaults=[
        5e3, 30e3, 10e3, 200, 15e3, 200e6,
        96, 348, 500, 1e3, 94, 20, 180, 250,
        30e3, 10e3, 1e3, 370, 5e3
    ]
)


class AsymSquidDCFlux(ComplexBase):
    def __init__(self, origin, params, side=0, trans_in=None):
        """
        Class to draw width symmetrical squid with
        outer positioning of the junctions.

        The notation 'length' is the dimension along the line
         which connects the contact pads,
        'width' is for the perpendicular direction.

        Parameters
        ----------
        origin : DPoint
            the geometrical center between centers of contact pads
        params : Union[tuple, AsymSquidParams]
        side : int
            only creates single JJ.
            `side == -1` - only left junction created
            `side == 1` - only right junction created
            `side == 0` - both junctions created (default)
        trans_in : Union[DCplxTrans, ICplxTrans]
            initial transformation in object's reference frame

        Notes
        ------
        pad_r: float
            A radius of the circle pad.
            Default - 5e3
        pads_distance:
            The distance between centers of contact pads.
            Default - 30e3
        contact_pad_width: float
            The width of contact pads made as rounded DPath
            Default - 10e3
        contact_pad_ext_r: float
            Radius of extension of contact pads along the y-axis
            Default - 200
        sq_dy: float
            The length of the squid, along contacts pad-to-pad direction.
            Default - 15e3
        sq_area: float
            The total area of the squid.
            (does not count the reduction of area due to
             shadow angle evaporation).
            Default - 200e6
        j1_dx: float
            The width of the upper thin lead on left
            side (straight) and also width width of
            the left junction
            Default - 96
        j2_dx: float
            The width of the upper small lead on right
            side (straight) and also width width of
            the junction
            Default - 348
        inter_leads_width: float
            The width of the thick lower/upper vertical leads from contact
            pads.
            Default - 500
        b_ext: float
            Length of small horizontal leads
            Default - 1e3
        j1_dy: float
            The dy of the left jj and the width of left lower
            small horizontal leads.
            Default - 94
        n: int
            The number of angle in regular polygon
            which serves as width large contact pad
            Default - 20
        bridge: float
            The value of the gap between vertical small leads on the upper
            side and horizontal small leads on the bottom side.
            Associated with an undercut's suspended bridge width formed
            during after e-beam lithography solvent developing.
            Default - 180
        j2_dy : float
            optional
            if present, `j1_dy` is interpreted as
            y-dimensions of left small bottom horizontal lead,
            and `j2_dy` is interpreted as
            y-dimensions of the right small bottom horizontal lead
            Default - 250
        """
        # To draw only width half of width squid use 'side'
        # side = -1 is left, 1 is right, 0 is both (default)

        # See description of AsymSquidParams tuple and the comment above
        self.params: AsymSquidDCFluxParams = params
        if (self.params.inter_leads_width < self.params.j1_dx) or \
                (self.params.inter_leads_width < self.params.j2_dx):
            raise ValueError("AsymSquid constructor:\n"
                             "intermediate lead width assumed "
                             "be bigger than width of each jj")
        if self.params.j2_dy is None:
            # workaround because namedtuple is immutable
            # there is `recordtype` that is mutable, but
            # in is not included in default KLayout build.
            self.params = AsymSquidDCFluxParams(*(self.params[:-1] + (
                self.params.j1_dy,)))
        self.side = side

        ''' Attributes corresponding to primitives '''
        self.pad_top: Circle = None
        self.ph_el_conn_pad: DPathCL = None
        self.bot_dc_flux_line_right: CPWRLPath = None
        self.bot_dc_flux_line_left: CPWRLPath = None

        super().__init__(origin, trans_in)

    def init_primitives(self):

        origin = DPoint(0, 0)
        if self.side == 0:
            # left
            self.init_half(origin, side=-1)
            # right
            self.init_half(origin, side=1)
        else:
            self.init_half(origin, side=self.side)

        ''' draw top contact pad '''
        origin = DPoint(0, 0)
        pars = self.params
        top_pad_center = origin + DVector(0, pars.pads_distance / 2)
        self.pad_top = Circle(
            top_pad_center, pars.pad_r,
            n_pts=pars.n, offset_angle=np.pi / 2
        )
        self.primitives["pad_top"] = self.pad_top

        self.ph_el_conn_pad = DPathCL(
            pts=[
                top_pad_center,
                origin + DPoint(0, pars.sq_dy / 2)
            ],
            width=pars.contact_pad_width
        )
        self.primitives["top_ph_el_conn_pad"] = self.ph_el_conn_pad

        ''' draw bottom DC flux line '''
        # print(self.bot_inter_lead_dx)
        self.bot_dc_flux_line_right = CPWRLPath(
            origin=origin + DPoint(
                pars.flux_line_dx / 2,
                -(pars.sq_dy / 2 + pars.flux_line_dy) -
                pars.flux_line_outer_width / 2
            ),
            shape="LRL",
            cpw_parameters=
            [
                CPWParameters(width=pars.flux_line_contact_width, gap=0),
                CPWParameters(smoothing=True),
                CPWParameters(width=pars.flux_line_outer_width, gap=0)
            ],
            turn_radiuses=max(pars.flux_line_outer_width,
                              pars.flux_line_contact_width),
            segment_lengths=[
                pars.flux_line_dy + pars.flux_line_outer_width,
                pars.flux_line_dx / 2 - self.bot_inter_lead_dx
            ],
            turn_angles=[np.pi / 2],
            trans_in=Trans.R90
        )
        self.primitives["bot_dc_flux_line_right"] = \
            self.bot_dc_flux_line_right

        self.bot_dc_flux_line_left = CPWRLPath(
            origin=origin + DPoint(
                -pars.flux_line_dx / 2,
                -(pars.sq_dy / 2 + pars.flux_line_dy) -
                pars.flux_line_outer_width / 2
            ),
            shape="LRL",
            cpw_parameters=
            [
                CPWParameters(width=pars.flux_line_contact_width, gap=0),
                CPWParameters(smoothing=True),
                CPWParameters(width=pars.flux_line_outer_width, gap=0)
            ],
            turn_radiuses=max(pars.flux_line_outer_width,
                              pars.flux_line_contact_width),
            segment_lengths=[
                pars.flux_line_dy + pars.flux_line_outer_width,
                pars.flux_line_dx / 2 - self.bot_inter_lead_dx
            ],
            turn_angles=[-np.pi / 2],
            trans_in=Trans.R90
        )
        self.primitives["bot_dc_flux_line_left"] = \
            self.bot_dc_flux_line_left

    def init_half(self, origin, side=-1):
        # side = -1 is width left half, 1 is width right half
        pars = self.params
        j_dy = pars.j1_dy if side < 0 else pars.j2_dy
        j_dx = pars.j1_dx if side < 0 else pars.j2_dx
        suff = "_left" if side < 0 else "_right"

        # euristic value is chosen to ensure that JJ lead do not suffer
        # overexposure due to proximity to top intermediate lead
        top_jj_lead_dy = 5 * pars.inter_leads_width
        ''' 
        Logic of parameters formulas by solving following equations
        by := bot_inter_lead_dy - j_dy/2
        ty := top_inter_lead_dy + top_jj_lead_dy + j_dy/2 + pars.bridge
        bx := bot_inter_lead_dx
        tx := top_inter_lead_dx
        phi := pars.intermediate_width/2 + 2/3*pars.b_ext
        L := pars.sq_dy
        A = pars.sq_area/2

        system:
        tx * ty + bx * tx = A
        ty + by = L
        bx - tx = phi - JJ's are located at 2/3 b_ext from bottom 
        intermediate leads
        by - ty = 0 - euristic, for completion

        If you will substitute values from definition above and look at 
        the design of width SQUID's layout you will get the idea about this 

        gives

        bx = A/L + phi/2
        tx = A/L - phi/2
        ty = L/2
        by = L/2

        and it follows that
        bot_inter_lead_dy = pars.sq_dy/2 + j_dy/2
        top_inter_lead_dy = pars.sq_dy/2 - (top_jj_lead_dy + j_dy/2 + 
        pars.bridge)
        bot_inter_lead_dx = pars.sq_area/pars.sq_dy/2 + 
        (pars.inter_leads_width/2 + 2/3*pars.b_ext)/2
        top_inter_lead_dx = pars.sq_area/pars.sq_dy/2 - 
        (pars.inter_leads_width/2 + 2/3*pars.b_ext)/2
        '''
        bot_inter_lead_dy = pars.sq_dy / 2 + j_dy / 2
        top_inter_lead_dy = pars.sq_dy / 2 - (top_jj_lead_dy + j_dy / 2 +
                                              pars.bridge)
        self.bot_inter_lead_dx = (
                pars.sq_area / pars.sq_dy / 2 +
                (pars.inter_leads_width / 2 + 2 / 3 * pars.b_ext) / 2
        )
        top_inter_lead_dx = (
                pars.sq_area / pars.sq_dy / 2 -
                (pars.inter_leads_width / 2 + 2 / 3 * pars.b_ext) / 2
        )

        ''' draw top intermediate lead'''
        # `pars.inter_leads_width/2` is made to ensure that SQUID countour
        # is connected at adjacent points lying on line x=0.
        top_inter_p1 = origin + DPoint(
            0,
            pars.sq_dy / 2
        )
        top_inter_p2 = top_inter_p1 + DPoint(
            side * top_inter_lead_dx,
            0
        )
        top_inter_p3 = top_inter_p2 + DPoint(
            0,
            -top_inter_lead_dy
        )

        self.primitives["top_inter_lead" + suff] = DPathCL(
            pts=[top_inter_p1, top_inter_p2, top_inter_p3],
            width=pars.inter_leads_width,
            bgn_ext=pars.inter_leads_width / 2,
            end_ext=pars.inter_leads_width / 4,
            round=True,
            bendings_r=pars.inter_leads_width
        )

        ''' draw top JJ lead '''
        top_jj_lead_p1 = top_inter_p3
        top_jj_lead_p2 = top_jj_lead_p1 + DPoint(
            0,
            -top_jj_lead_dy
        )
        self.primitives["top_jj_lead" + suff] = DPathCL(
            pts=[top_jj_lead_p1, top_jj_lead_p2],
            width=j_dx,
            bgn_ext=pars.inter_leads_width / 4,
            round=True
        )

        ''' draw buttom intermediate lead '''
        bottom_y = top_jj_lead_p2.y - pars.bridge - bot_inter_lead_dy
        bot_inter_lead_p1 = DPoint(0, bottom_y)
        bot_inter_lead_p2 = bot_inter_lead_p1 + DPoint(
            side * self.bot_inter_lead_dx,
            0
        )
        bot_inter_lead_p3 = bot_inter_lead_p2 + DPoint(
            0,
            bot_inter_lead_dy
        )
        self.primitives["bot_inter_lead" + suff] = CPWRLPath(
            origin=bot_inter_lead_p1, shape="LRL",
            cpw_parameters=[
                CPWParameters(width=pars.flux_line_inner_width, gap=0),
                CPWParameters(smoothing=True),
                CPWParameters(width=pars.inter_leads_width, gap=0)
            ],
            turn_radiuses=pars.inter_leads_width,
            segment_lengths=[
                bot_inter_lead_p1.distance(bot_inter_lead_p2),
                bot_inter_lead_p2.distance(bot_inter_lead_p3) +
                pars.inter_leads_width / 2
            ],
            turn_angles=[np.pi / 2],
            trans_in=Trans.M90 if side == -1 else None
        )

        ''' draw bottom JJ lead '''
        bot_jj_lead_p1 = bot_inter_lead_p3 + DPoint(
            -side * pars.inter_leads_width / 2,
            -j_dy / 2
        )
        bot_jj_lead_p2 = bot_jj_lead_p1 + DPoint(
            -side * pars.b_ext,
            0
        )

        self.primitives["bot_jj_lead" + suff] = DPathCL(
            pts=[bot_jj_lead_p1, bot_jj_lead_p2],
            width=j_dy,
            bgn_ext=pars.inter_leads_width / 4,
            round=True
        )


AsymSquidOneLegParams = namedtuple(
    "AsymSquidParams",
    [
        "pad_r",
        "pads_distance",
        "contact_pad_width",
        "contact_pad_ext_r",
        "sq_dy",
        "sq_area",
        "j1_dy",
        "j2_dy",
        "inter_leads_width",
        "b_ext",
        "j1_dx",
        "n",
        "bridge",
        "j2_dx",
        # last 5
        "flux_line_dx",
        "flux_line_dy",
        "flux_line_outer_width",
        "flux_line_inner_width",
        "flux_line_contact_width",
    ],
    defaults=[
        5e3, 30e3, 10e3, 200, 15e3, 200e6,
        96, 348, 500, 1e3, 94, 20, 180, 250,
        30e3, 10e3, 1e3, 370, 5e3
    ]
)


class AsymSquidOneLeg(ComplexBase):
    def __init__(self, origin, params, side=0, leg_side=0, trans_in=None):
        """
        Class to draw width symmetrical squid with
        outer positioning of the junctions.

        The notation 'length' or 'dy' is the dimension along the line
         which connects the contact pads,
        'width' or 'dx' is for the perpendicular direction.

        Parameters
        ----------
        origin : DPoint
            the geometrical center between centers of contact pads
        params : Union[tuple, AsymSquidOneLegParams]
        side : int
            only creates single JJ.
            `side == -1` - only left junction created
            `side == 1` - only right junction created
            `side == 0` - both junctions created (default)
        leg_side : int
            0 - draw both legs
            1 - right leg only
            -1 - left leg only
        trans_in : Union[DCplxTrans, ICplxTrans]
            initial transformation in object's reference frame

        Notes
        ------
        pad_r: float
            A radius of the circle pad.
            Default - 5e3
        pads_distance:
            The distance between centers of contact pads.
            Default - 30e3
        contact_pad_width: float
            The width of contact pads made as rounded DPath
            Default - 10e3
        contact_pad_ext_r: float
            Radius of extension of contact pads along the y-axis
            Default - 200
        sq_dy: float
            The length of the squid, along contacts pad-to-pad direction.
            Default - 15e3
        sq_area: float
            The total area of the squid.
            (does not count the reduction of area due to
             shadow angle evaporation).
            Default - 200e6
        j1_dx: float
            The width of the upper thin lead on left
            side (straight) and also width width of
            the left junction
            Default - 96
        j2_dx: float
            The width of the upper small lead on right
            side (straight) and also width width of
            the junction
            Default - 348
        inter_leads_width: float
            The width of the thick lower/upper vertical leads from contact
            pads.
            Default - 500
        b_ext: float
            Length of small horizontal leads
            Default - 1e3
        j1_dy: float
            The dy of the left jj and the width of left lower
            small horizontal leads.
            Default - 94
        n: int
            The number of curve points in regular polygon
            which serves as width large contact pad
            Default - 20
        bridge: float
            The value of the gap between vertical small leads on the upper
            side and horizontal small leads on the bottom side.
            Associated with an undercut's suspended bridge width formed
            during after e-beam lithography solvent developing.
            Default - 180
        j2_dy : float
            optional
            if present, `j1_dy` is interpreted as
            y-dimensions of left small bottom horizontal lead,
            and `j2_dy` is interpreted as
            y-dimensions of the right small bottom horizontal lead
            Default - 250
        """
        # To draw only width half of width squid use 'side'
        # side = -1 is left, 1 is right, 0 is both (default)

        # See description of AsymSquidParams tuple and the comment above
        self.params: AsymSquidDCFluxParams = params
        if (self.params.inter_leads_width < self.params.j1_dx) or \
                (self.params.inter_leads_width < self.params.j2_dx):
            raise ValueError("AsymSquid constructor:\n"
                             "intermediate lead width assumed to"
                             "be bigger than dx of each jj")
        if self.params.j2_dy is None:
            # workaround because namedtuple is immutable
            # there is `recordtype` that is mutable, but
            # in is not included in default KLayout build.
            self.params = AsymSquidDCFluxParams(*(self.params[:-1] + (
                self.params.j1_dy,)))
        self.side = side
        self.leg_side = leg_side

        ''' Attributes corresponding to primitives '''
        self.pad_top: Circle = None
        self.top_ph_el_conn_pad: DPathCL = None
        self.bot_dc_flux_line_right: CPWRLPath = None
        self.bot_dc_flux_line_left: CPWRLPath = None

        super().__init__(origin, trans_in)

    def init_primitives(self):

        origin = DPoint(0, 0)
        if self.side == 0:
            # left
            self.init_half(origin, side=-1)
            # right
            self.init_half(origin, side=1)
        else:
            self.init_half(origin, side=self.side)
        self.init_ph_el_conn_pads(leg_side=self.leg_side)

    def init_ph_el_conn_pads(self, leg_side=0):
        ''' draw top contact pad '''
        origin = DPoint(0, 0)
        pars = self.params
        top_pad_center = origin + DVector(0, pars.pads_distance / 2)
        self.pad_top = Circle(
            top_pad_center, pars.pad_r,
            n_pts=pars.n, offset_angle=np.pi / 2
        )
        self.primitives["pad_top"] = self.pad_top

        self.top_ph_el_conn_pad = DPathCL(
            pts=[
                top_pad_center,
                origin + DPoint(0, pars.sq_dy / 2)
            ],
            width=pars.contact_pad_width
        )
        self.primitives["top_ph_el_conn_pad"] = self.top_ph_el_conn_pad

        ''' draw bottom DC flux line '''
        if leg_side == 1 or leg_side == 0:
            # print(self.bot_inter_lead_dx)
            self.bot_dc_flux_line_right = CPWRLPath(
                origin=origin + DPoint(
                    pars.flux_line_dx / 2,
                    -(pars.sq_dy / 4 + pars.flux_line_dy) -
                    pars.flux_line_outer_width / 2
                ),
                shape="LRL",
                cpw_parameters=
                [
                    CPWParameters(width=pars.flux_line_contact_width,
                                  gap=0),
                    CPWParameters(smoothing=True),
                    CPWParameters(width=pars.flux_line_outer_width, gap=0)
                ],
                turn_radiuses=max(pars.flux_line_outer_width,
                                  pars.flux_line_contact_width),
                segment_lengths=[
                    pars.flux_line_dy + pars.flux_line_outer_width,
                    pars.flux_line_dx / 2 - self.bot_inter_lead_dx
                ],
                turn_angles=[np.pi / 2],
                trans_in=Trans.R90
            )
            self.primitives["bot_dc_flux_line_right"] = \
                self.bot_dc_flux_line_right
        if leg_side == 0 or leg_side == -1:
            self.bot_dc_flux_line_left = CPWRLPath(
                origin=origin + DPoint(
                    -pars.flux_line_dx / 2,
                    -(pars.sq_dy / 4 + pars.flux_line_dy) -
                    pars.flux_line_outer_width / 2
                ),
                shape="LRL",
                cpw_parameters=
                [
                    CPWParameters(width=pars.flux_line_contact_width,
                                  gap=0),
                    CPWParameters(smoothing=True),
                    CPWParameters(width=pars.flux_line_outer_width, gap=0)
                ],
                turn_radiuses=max(pars.flux_line_outer_width,
                                  pars.flux_line_contact_width),
                segment_lengths=[
                    pars.flux_line_dy + pars.flux_line_outer_width,
                    pars.flux_line_dx / 2 - self.bot_inter_lead_dx
                ],
                turn_angles=[-np.pi / 2],
                trans_in=Trans.R90
            )
            self.primitives["bot_dc_flux_line_left"] = \
                self.bot_dc_flux_line_left

    def init_half(self, origin, side=-1):
        # side = -1 is width left half, 1 is width right half
        pars = self.params
        j_dy = pars.j1_dy if side < 0 else pars.j2_dy
        j_dx = pars.j1_dx if side < 0 else pars.j2_dx
        suff = "_left" if side < 0 else "_right"

        # euristic value is chosen to ensure that JJ lead do not suffer
        # overexposure due to proximity to top intermediate lead
        top_jj_lead_dy = 5 * pars.inter_leads_width
        ''' 
        Logic of parameters formulas by solving following equations
        by := bot_inter_lead_dy - j_dy/2
        ty := top_inter_lead_dy + top_jj_lead_dy + j_dy/2 + pars.bridge
        bx := bot_inter_lead_dx
        tx := top_inter_lead_dx
        phi := pars.intermediate_width/2 + 2/3*pars.b_ext
        L := pars.sq_dy
        A = pars.sq_area/2

        system:
        tx * ty + bx * tx = A
        ty + by = L
        bx - tx = phi - JJ's are located at 2/3 b_ext from bottom 
        intermediate leads
        by - ty = 0 - euristic, for completion

        If you will substitute values from definition above and look at 
        the design of width SQUID's layout you will get the idea about this 

        gives

        bx = A/L + phi/2
        tx = A/L - phi/2
        ty = L/2
        by = L/2

        and it follows that
        bot_inter_lead_dy = pars.sq_dy/2 + j_dy/2
        top_inter_lead_dy = pars.sq_dy/2 - (top_jj_lead_dy + j_dy/2 + 
        pars.bridge)
        bot_inter_lead_dx = pars.sq_area/pars.sq_dy/2 + 
        (pars.inter_leads_width/2 + 2/3*pars.b_ext)/2
        top_inter_lead_dx = pars.sq_area/pars.sq_dy/2 - 
        (pars.inter_leads_width/2 + 2/3*pars.b_ext)/2
        '''
        bot_inter_lead_dy = pars.sq_dy / 2 + j_dy / 2
        top_inter_lead_dy = pars.sq_dy / 2 - (top_jj_lead_dy + j_dy / 2 +
                                              pars.bridge)
        self.bot_inter_lead_dx = (
                pars.sq_area / pars.sq_dy / 2 +
                (pars.inter_leads_width / 2 + 2 / 3 * pars.b_ext) / 2
        )
        top_inter_lead_dx = (
                pars.sq_area / pars.sq_dy / 2 -
                (pars.inter_leads_width / 2 + 2 / 3 * pars.b_ext) / 2
        )

        ''' draw top intermediate lead'''
        # `pars.inter_leads_width/2` is made to ensure that SQUID countour
        # is connected at adjacent points lying on line x=0.
        top_inter_p1 = origin + DPoint(
            0,
            pars.sq_dy / 2
        )
        top_inter_p2 = top_inter_p1 + DPoint(
            side * top_inter_lead_dx,
            0
        )
        top_inter_p3 = top_inter_p2 + DPoint(
            0,
            -top_inter_lead_dy
        )

        self.primitives["top_inter_lead" + suff] = DPathCL(
            pts=[top_inter_p1, top_inter_p2, top_inter_p3],
            width=pars.inter_leads_width,
            bgn_ext=pars.inter_leads_width / 2,
            end_ext=pars.inter_leads_width / 4,
            round=True,
            bendings_r=pars.inter_leads_width
        )

        ''' draw top JJ lead '''
        top_jj_lead_cpw2cpw_p1 = top_inter_p3
        top_jj_lead_cpw2cpw_p2 = top_jj_lead_cpw2cpw_p1 + DPoint(
            0,
            -top_jj_lead_dy / 2
        )
        top_jj_rigid_lead = CPW2CPW(
            Z0=CPWParameters(width=pars.inter_leads_width, gap=0),
            Z1=CPWParameters(width=j_dx, gap=0),
            start=top_jj_lead_cpw2cpw_p1,
            end=top_jj_lead_cpw2cpw_p2
        )
        self.primitives["top_jj_rigid_lead" + suff] = top_jj_rigid_lead

        top_jj_lead_p1 = top_jj_rigid_lead.end
        top_jj_lead_p2 = top_jj_lead_p1 + DPoint(
            0,
            -top_jj_lead_dy / 2
        )
        self.primitives["top_jj_lead" + suff] = DPathCL(
            pts=[top_jj_lead_p1, top_jj_lead_p2],
            width=j_dx,
            bgn_ext=pars.inter_leads_width / 4,
            round=True
        )

        ''' draw buttom intermediate lead '''
        bottom_y = top_jj_lead_p2.y - pars.bridge - bot_inter_lead_dy
        bot_inter_lead_p1 = DPoint(0, bottom_y)
        bot_inter_lead_p2 = bot_inter_lead_p1 + DPoint(
            side * self.bot_inter_lead_dx,
            0
        )
        bot_inter_lead_p3 = bot_inter_lead_p2 + DPoint(
            0,
            bot_inter_lead_dy
        )
        self.primitives["bot_inter_lead" + suff] = CPWRLPath(
            origin=bot_inter_lead_p1, shape="LRL",
            cpw_parameters=[
                CPWParameters(width=pars.flux_line_inner_width, gap=0),
                CPWParameters(smoothing=True),
                CPWParameters(width=pars.inter_leads_width, gap=0)
            ],
            turn_radiuses=pars.inter_leads_width,
            segment_lengths=[
                bot_inter_lead_p1.distance(bot_inter_lead_p2),
                bot_inter_lead_p2.distance(bot_inter_lead_p3) +
                pars.inter_leads_width / 2
            ],
            turn_angles=[np.pi / 2],
            trans_in=Trans.M90 if side == -1 else None
        )

        ''' draw bottom JJ lead '''
        bot_jj_lead_rigid_p1 = bot_inter_lead_p3 + DPoint(
            -side * pars.inter_leads_width / 2,
            -j_dy / 2
        )
        bot_jj_lead_rigid_p2 = bot_jj_lead_rigid_p1 + DPoint(
            -side * pars.b_ext / 3,
            0
        )
        bot_jj_lead_rigid = CPW2CPW(
            Z0=CPWParameters(width=pars.inter_leads_width, gap=0),
            Z1=CPWParameters(width=j_dy, gap=0),
            start=bot_jj_lead_rigid_p1,
            end=bot_jj_lead_rigid_p2
        )
        self.primitives["bot_jj_lead_rigid" + suff] = bot_jj_lead_rigid

        bot_jj_lead_p1 = bot_jj_lead_rigid_p1
        bot_jj_lead_p2 = bot_jj_lead_rigid_p1 + DPoint(
            -side * pars.b_ext,
            0
        )

        self.primitives["bot_jj_lead" + suff] = DPathCL(
            pts=[bot_jj_lead_p1, bot_jj_lead_p2],
            width=j_dy,
            bgn_ext=pars.inter_leads_width / 4,
            round=True
        )


class Squid(AsymSquid):
    '''
    Class to draw width symmetrical squid with outer positioning of the junctions.

    The notation 'length' is the dimension along the line which connects the contact pads,
    'width' is for the perpendicular direction.

    Notes
    -----------
        pad_side: float
            A length of the side of triangle pad.
        pad_r: float
            Radius of contact pads circle part.
        pads_distance: float
            The distance between triangle contact pads.
        contact_pad_width: float
            The width of curved rectangle leads which connect triangle contact pads and junctions.
        contact_pad_ext_r: float
            The angle outer_r of the pad extension
        sq_dy: float
            The length of the squid, along leads.
        sq_width: float
            The total area of the squid.
            (does not count the reduction of area due to shadow angle evaporation).
        j_width: float
            The width of the upper small leads (straight) and also width width of the junction
        inter_leads_width: float
            The width of the lower small bended leads before bending
        b_ext: float
            The extension of bended leads after bending
        j_length: float
            The length of the jj and the width of bended parts of the lower leads.
        n: int
            The number of angle in regular polygon which serves as width large contact pad
        bridge: float
            The value of the gap between two parts of junction in the design
        trans_in: Trans
            Initial transformation
    '''

    def __init__(self, origin, params, side=0, trans_in=None):
        # To draw only width half of width squid use 'side'
        # side = -1 is left, 1 is right, 0 is both (default)
        asymparams = AsymSquidParams(*params[:-3], *params[-2:])
        super().__init__(self, origin, asymparams, side, trans_in)


class LineNJJ(ElementBase):
    def __init__(self, origin, params, trans_in=None):
        self.params = params
        self.a = params[0]
        self.b = params[1]
        self.jos1_b = params[2]
        self.jos1_a = params[3]
        self.f1 = params[4]
        self.d1 = params[5]
        self.jos2_b = params[6]
        self.jos2_a = params[7]
        self.f2 = params[8]
        self.d2 = params[9]
        self.w = params[10]

        self.poly1 = self._make_polygon(self.b, self.w, self.d1, self.f1,
                                        self.d2)

        super().__init__(origin, trans_in)

    def _make_polygon(self, length, w, d, f, overlapping):
        polygon = DSimplePolygon
        p1 = DPoint(0, 0)
        p2 = p1 + DPoint(length, 0)
        p3 = p2 + DPoint(0, w)
        p4 = p3 - DPoint(overlapping, 0)
        p5 = p4 - DPoint(0, d)
        p6 = p5 - DPoint(f, 0)
        p7 = p6 + DPoint(0, d)
        p8 = p1 + DPoint(0, w)

        polygon = DSimplePolygon([p1, p2, p3, p4, p5, p6, p7, p8])
        return polygon

    def init_regions(self):
        self.metal_region.insert(SimplePolygon(self.poly1))
