from collections import namedtuple
from typing import Union
from math import pi

import numpy as np

import pya
from pya import DPoint, DSimplePolygon, SimplePolygon
from pya import Trans, DTrans, DVector, DPath

from classLib.baseClasses import ElementBase, ComplexBase
from classLib.shapes import Circle, Kolbaska, DPathCL
from classLib.coplanars import CPWParameters, CPWRLPath



# print("josJ reloaded")
# nice solution for parameters, man. Had really appreciated this one and put to use already (SH).


AsymSquidParams = namedtuple(
    "AsymSquidParams",
    [
        "pad_r",
        "pads_distance",
        "contact_pad_width",
        "contact_pad_ext_r",
        "sq_len",
        "sq_area",
        "j_width_1",
        "j_width_2",
        "inter_leads_width",
        "b_ext",
        "j_length_1",
        "n",
        "bridge",
        "j_length_2"
    ],
    defaults=[
        5e3, 30e3, 10e3, 200,15e3, 200e6,
        96, 348, 500, 1e3, 94, 20,
        180, 250
    ]
)


class AsymSquid(ComplexBase):
    def __init__(self, origin, params, side=0, trans_in=None):
        '''
        Class to draw width symmetrical squid with outer positioning of the junctions.

        The notation 'length' is the dimension along the line which connects the contact pads,
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
            initial transformation

        Notes
        ------
        Notes
        ------
        pad_r: float
            A radius of the circle pad.
            Default - 5e3
        pads_distance:
            The distance between centers of contact pads.
            Default - 30e3
        contact_pad_width: float
            The width of DPath leads which connect
            contact pads and junctions.
            Default - 10e3
        contact_pad_ext_r: float
            Curved extension of DPath leads along the leads direction
            Default - 200
        sq_len: float
            The length of the squid, along pad-to-pad direction.
            Default - 15e3
        sq_area: float
            The total area of the squid.
            (does not count the reduction of area due to
             shadow angle evaporation).
            Default - 200e6
        j_width_1: float
            The width of the upper small lead on left
            side (straight) and also width width of
            the left junction
            Default - 96
        j_width_2: float
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
        j_length_1: float
            The length of the jj and the width of left lower
            small horizontal leads.
            Default - 94
        n: int
            The number of angle in regular polygon
            which serves as width large contact pad
            Default - 20
        bridge: float
            The value of the gap between vertical small leads on the upper
            side and horizontal small leads on the bottom side
            Default - 180
        j_length_2 : float
            optional
            if present, `j_length_1` is interpreted as
            y-dimensions of left small bottom horizontal lead,
            and `j_length_2` is interpreted as
            y-dimensions of the right small bottom horizontal lead
            Default - 250
        '''
        # To draw only width half of width squid use 'side'
        # side = -1 is left, 1 is right, 0 is both (default)
        self.params: AsymSquidParams = params  # See description of AsymSquidParams tuple and the comment above
        if (self.params.inter_leads_width < self.params.j_width_1) or \
                (self.params.inter_leads_width < self.params.j_width_2):
            raise ValueError("AsymSquid constructor:\n"
                             "intermediate lead width assumed be bigger than width of each jj")
        if self.params.j_length_2 is None:
            # workaround because namedtuple is immutable
            # there is `recordtype` that is mutable, but
            # in is not included in default KLayout build.
            self.params = AsymSquidParams(*(self.params[:-1] + (self.params.j_length_1,)))
        self.side = side
        super().__init__(origin, trans_in)

    def init_primitives(self):
        origin = DPoint(0, 0)
        pars = self.params
        up_pad_center = origin + DVector(0, pars.pads_distance / 2)
        down_pad_center = origin + DVector(0, -pars.pads_distance / 2)
        self.pad_down = Circle(
            down_pad_center, pars.pad_r,
            n_pts=pars.n, offset_angle=pi / 2
        )
        self.primitives["pad_down"] = self.pad_down

        self.p_ext_down = Kolbaska(
            down_pad_center, origin + DPoint(0, -pars.sq_len / 2),
            pars.contact_pad_width, pars.contact_pad_ext_r
        )
        self.primitives["p_ext_down"] = self.p_ext_down

        self.pad_up = Circle(
            up_pad_center, pars.pad_r,
            n_pts=pars.n, offset_angle=-pi / 2
        )
        self.primitives["pad_up"] = self.pad_up

        self.p_ext_up = Kolbaska(
            up_pad_center, origin + DVector(0, pars.sq_len / 2),
            pars.contact_pad_width, pars.contact_pad_ext_r
        )
        self.primitives["p_ext_up"] = self.p_ext_up

        origin = DPoint(0, 0)
        if self.side == 0:
            self.init_half(origin, side=1)  # right
            self.init_half(origin, side=-1)  # left
        else:
            self.init_half(origin, side=self.side)

    def init_half(self, origin, side=-1):
        # side = -1 is left, 1 is right
        pars = self.params
        j_width = pars.j_width_1 if side < 0 else pars.j_width_2
        j_length_1 = pars.j_length_1 if side < 0 else pars.j_length_2
        suff = "_left" if side < 0 else "_right"
        up_st_gap = pars.sq_area / (2 * pars.sq_len)

        # exact correction in first row
        # additional extension to isolate jj's from intermediate bottom polygons
        # without correction horizontal faces of jj's will be adjacent
        # to thick intemediate polygons
        low_st_gap = up_st_gap + ((pars.j_length_1 + pars.j_length_2) / 2 + pars.inter_leads_width) + \
                     2 * pars.inter_leads_width

        ### upper and lower vertical intermediate and jj leads ###
        ## top leads ##
        upper_leads_extension = j_width / 4
        # upper intemediate lead
        up_st_start = self.primitives["p_ext_up"].connections[1] + \
                      DVector(side * up_st_gap / 2, 0)
        up_st_stop = origin + \
                     DVector(side * up_st_gap / 2, pars.bridge / 2 + upper_leads_extension)
        # top jj lead
        self.primitives["upp_st" + suff] = Kolbaska(
            up_st_start, up_st_stop, j_width, upper_leads_extension
        )
        # top intermediate lead
        upper_thin_part_len = 4 * pars.bridge + pars.inter_leads_width / 2
        self.primitives["upp_st_thick" + suff] = Kolbaska(
            up_st_start, up_st_stop + DPoint(0, upper_thin_part_len),
            pars.inter_leads_width, pars.inter_leads_width / 2
        )

        ## bottom leads ##
        low_st_start = self.primitives["p_ext_down"].connections[1] + \
                       DVector(side * low_st_gap / 2, 0)
        low_st_stop = origin + \
                      DVector(side * (low_st_gap / 2 + 2 * pars.inter_leads_width),
                              -pars.bridge / 2 - pars.inter_leads_width / 2)
        len_ly = (low_st_stop - low_st_start).y
        # bottom intermediate lead
        self.primitives["low_st" + suff] = CPWRLPath(
            low_st_start, 'LR', CPWParameters(pars.inter_leads_width, 0),
            pars.inter_leads_width / 2, [len_ly], [side * pi / 2], trans_in=DTrans.R90)
        # bottom jj lead (horizontal)
        low_st_end = self.primitives["low_st" + suff].connections[1]
        low_st_jj_start = low_st_end + DPoint(0, -j_length_1 / 2 + pars.inter_leads_width / 2)
        low_st_jj_stop = low_st_jj_start + DPoint(-side * pars.b_ext, 0)
        self.primitives["low_st_jj" + suff] = Kolbaska(low_st_jj_start, low_st_jj_stop, j_length_1,
                                                       j_length_1 / 2)


AsymSquidDCFluxParams = namedtuple(
    "AsymSquidParams",
    [
        "pad_r",
        "pads_distance",
        "contact_pad_width",
        "contact_pad_ext_r",
        "sq_len",
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
        sq_len: float
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
                origin + DPoint(0, pars.sq_len / 2)
            ],
            width=pars.contact_pad_width
        )
        self.primitives["ph_el_conn_pad"] = self.ph_el_conn_pad

        ''' draw bottom DC flux line '''
        # print(self.bot_inter_lead_dx)
        self.bot_dc_flux_line_right = CPWRLPath(
            origin=origin + DPoint(
                pars.flux_line_dx / 2,
                -(pars.sq_len / 2 + pars.flux_line_dy) -
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
                -(pars.sq_len / 2 + pars.flux_line_dy) -
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
        L := pars.sq_len
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
        bot_inter_lead_dy = pars.sq_len/2 + j_dy/2
        top_inter_lead_dy = pars.sq_len/2 - (top_jj_lead_dy + j_dy/2 + 
        pars.bridge)
        bot_inter_lead_dx = pars.sq_area/pars.sq_len/2 + 
        (pars.inter_leads_width/2 + 2/3*pars.b_ext)/2
        top_inter_lead_dx = pars.sq_area/pars.sq_len/2 - 
        (pars.inter_leads_width/2 + 2/3*pars.b_ext)/2
        '''
        bot_inter_lead_dy = pars.sq_len / 2 + j_dy / 2
        top_inter_lead_dy = pars.sq_len / 2 - (top_jj_lead_dy + j_dy / 2 +
                                               pars.bridge)
        self.bot_inter_lead_dx = (
                pars.sq_area / pars.sq_len / 2 +
                (pars.inter_leads_width / 2 + 2 / 3 * pars.b_ext) / 2
        )
        top_inter_lead_dx = (
                pars.sq_area / pars.sq_len / 2 -
                (pars.inter_leads_width / 2 + 2 / 3 * pars.b_ext) / 2
        )

        ''' draw top intermediate lead'''
        # `pars.inter_leads_width/2` is made to ensure that SQUID countour
        # is connected at adjacent points lying on line x=0.
        top_inter_p1 = origin + DPoint(
            0,
            pars.sq_len / 2
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
        "sq_len",
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
        sq_len: float
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
                origin + DPoint(0, pars.sq_len / 2)
            ],
            width=pars.contact_pad_width
        )
        self.primitives["ph_el_conn_pad"] = self.ph_el_conn_pad

        # ''' draw bottom DC flux line '''
        # # print(self.bot_inter_lead_dx)
        # self.bot_dc_flux_line_right = CPWRLPath(
        #     origin=origin + DPoint(
        #         pars.flux_line_dx / 2,
        #         -(pars.sq_len / 2 + pars.flux_line_dy) -
        #         pars.flux_line_outer_width / 2
        #     ),
        #     shape="LRL",
        #     cpw_parameters=
        #     [
        #         CPWParameters(width=pars.flux_line_contact_width, gap=0),
        #         CPWParameters(smoothing=True),
        #         CPWParameters(width=pars.flux_line_outer_width, gap=0)
        #     ],
        #     turn_radiuses=max(pars.flux_line_outer_width,
        #                       pars.flux_line_contact_width),
        #     segment_lengths=[
        #         pars.flux_line_dy + pars.flux_line_outer_width,
        #         pars.flux_line_dx / 2 - self.bot_inter_lead_dx
        #     ],
        #     turn_angles=[np.pi / 2],
        #     trans_in=Trans.R90
        # )
        # self.primitives["bot_dc_flux_line_right"] = \
        #     self.bot_dc_flux_line_right

        self.bot_dc_flux_line_left = CPWRLPath(
            origin=origin + DPoint(
                -pars.flux_line_dx / 2,
                -(pars.sq_len / 2 + pars.flux_line_dy) -
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
        L := pars.sq_len
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
        bot_inter_lead_dy = pars.sq_len/2 + j_dy/2
        top_inter_lead_dy = pars.sq_len/2 - (top_jj_lead_dy + j_dy/2 + 
        pars.bridge)
        bot_inter_lead_dx = pars.sq_area/pars.sq_len/2 + 
        (pars.inter_leads_width/2 + 2/3*pars.b_ext)/2
        top_inter_lead_dx = pars.sq_area/pars.sq_len/2 - 
        (pars.inter_leads_width/2 + 2/3*pars.b_ext)/2
        '''
        bot_inter_lead_dy = pars.sq_len / 2 + j_dy / 2
        top_inter_lead_dy = pars.sq_len / 2 - (top_jj_lead_dy + j_dy / 2 +
                                               pars.bridge)
        self.bot_inter_lead_dx = (
                pars.sq_area / pars.sq_len / 2 +
                (pars.inter_leads_width / 2 + 2 / 3 * pars.b_ext) / 2
        )
        top_inter_lead_dx = (
                pars.sq_area / pars.sq_len / 2 -
                (pars.inter_leads_width / 2 + 2 / 3 * pars.b_ext) / 2
        )

        ''' draw top intermediate lead'''
        # `pars.inter_leads_width/2` is made to ensure that SQUID countour
        # is connected at adjacent points lying on line x=0.
        top_inter_p1 = origin + DPoint(
            0,
            pars.sq_len / 2
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
        sq_len: float
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

        self.poly1 = self._make_polygon(self.b, self.w, self.d1, self.f1, self.d2)

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
        self.metal_region.insert(SimplePolygon().from_dpoly(self.poly1))
