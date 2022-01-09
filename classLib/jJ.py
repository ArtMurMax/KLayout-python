import pya
from math import sqrt, cos, sin, atan2, pi, copysign
from pya import Point,DPoint,DSimplePolygon,SimplePolygon, DPolygon, Polygon,  Region
from pya import Trans, DTrans, CplxTrans, DCplxTrans, ICplxTrans
from numpy import around
from classLib.baseClasses import *
from classLib.coplanars import *
from classLib._PROG_SETTINGS import *

class SQUIDManhattan(ComplexBase):

    def __init__(self, origin, w_JJ, h_JJ, asymmetry, bridge_width, structure_length,
                 squid_width_top=3e3, squid_width_bottom=2e3, pad_width=8e3, small_lead_width=.65e3,
                 top_tiny_lead_length = 1e3, bottom_tiny_lead_length = 1e3,
                 tiny_lead_outshoot=1200, trans_in=None):

        self._w_JJ = w_JJ
        self._h_JJ = h_JJ
        self._asymmetry = asymmetry
        self._bridge_width = bridge_width
        self._structure_length = structure_length
        self._squid_width_top = squid_width_top
        self._squid_width_bottom = squid_width_bottom
        self._pad_width = pad_width
        self._small_lead_width = small_lead_width
        self._top_tiny_lead_length = top_tiny_lead_length
        self._bottom_tiny_lead_length = bottom_tiny_lead_length
        self._tiny_lead_outshoot = tiny_lead_outshoot
        self._correction = -30

        self._bottom_left_w_JJ = sqrt(1 - self._asymmetry) * self._h_JJ
        self._bottom_right_w_JJ = 1.5 * (1 + self._asymmetry) * self._h_JJ
        self._top_left_h_JJ = sqrt(1 - self._asymmetry) * self._w_JJ
        self._top_right_h_JJ = self._w_JJ / 1.5

        app = pya.Application.instance()
        mw = app.main_window()
        lv = mw.current_view()
        cv = lv.active_cellview()

        super().__init__(origin, trans_in)
        # creating cell for BMSTU

        params = around([self._bottom_left_w_JJ,
                         self._bottom_right_w_JJ,
                         self._top_left_h_JJ,
                         self._top_right_h_JJ]).astype(int)
        # names = ["xmon_l", "xmon_r", "ground_l", "ground_r"]
        self._cell_name = "SQUID"
        self._cell_name += "(%dx%d)" % (params[0], params[2]) + "(%dx%d); " % (params[1], params[3]) \
                           + "bridge: %d; " % self._bridge_width + "corr: -30; " + "d: %.1f" % self._asymmetry

        layout = cv.layout()
        layout.dbu = 0.001

        if not layout.has_cell(self._cell_name):
            layout.create_cell(self._cell_name)

            old_cell_name = cv.cell_name
            cv.cell_name = self._cell_name
            self._cell = cv.cell
            cv.cell_name = old_cell_name

            ebeam = layout.find_layer(LAYERS.ebeam)
            self._cell.shapes(ebeam).insert(self.metal_region.merge())

        else:
            old_cell_name = cv.cell_name
            cv.cell_name = self._cell_name
            self._cell = cv.cell
            cv.cell_name = old_cell_name


    def init_primitives(self):

        bottom_left_w_JJ = self._bottom_left_w_JJ + self._correction
        bottom_right_w_JJ = self._bottom_right_w_JJ + self._correction
        top_left_h_JJ = self._top_left_h_JJ + self._correction
        top_right_h_JJ = self._top_right_h_JJ + self._correction

        # Top
        cursor = DPoint(0, self._structure_length/2)
        self.primitives["top_pad"] =\
            CPW(self._pad_width, 0, cursor,
                cursor+DPoint(0, -self._structure_length/4))
        cursor = self.primitives["top_pad"].end

        cursor_l = cursor + DPoint(-self._squid_width_top/2-self._small_lead_width/2, 0)
        self.primitives["top_left_small_lead"] =\
            CPW(self._small_lead_width, 0, cursor_l,
                    cursor_l+DPoint(0, -self._structure_length/4+top_left_h_JJ/2))
        cursor_l = self.primitives["top_left_small_lead"].end

        cursor_l += DPoint(-self._small_lead_width/2, -top_left_h_JJ/2)
        self.primitives["top_left_tiny_lead"] =\
            CPW(top_left_h_JJ, 0, cursor_l,
                    cursor_l+DPoint(self._top_tiny_lead_length+self._tiny_lead_outshoot, 0))

        cursor_r = cursor + DPoint(self._squid_width_top/2+self._small_lead_width/2, 0)
        self.primitives["top_right_small_lead"] =\
            CPW(self._small_lead_width, 0, cursor_r,
                cursor_r+DPoint(0, -self._structure_length/4+top_right_h_JJ/2))
        cursor_r = self.primitives["top_right_small_lead"].end

        cursor_r += DPoint(self._small_lead_width/2, -top_right_h_JJ/2)
        self.primitives["top_right_tiny_lead"] =\
            CPW(top_right_h_JJ, 0, cursor_r,
                    cursor_r+DPoint(-self._top_tiny_lead_length-self._tiny_lead_outshoot, 0))


        # Bottom

        cursor = DPoint(0, -self._structure_length/2)
        self.primitives["bottom_pad"] =\
            CPW(self._pad_width, 0, cursor,
                cursor+DPoint(0, self._structure_length/4))
        cursor = self.primitives["bottom_pad"].end

        cursor_l = cursor +\
            DPoint(-self._squid_width_bottom/2 \
                    - self._small_lead_width/2 \
                    , 0)
        self.primitives["bottom_left_small_lead"] =\
            CPW(self._small_lead_width, 0, cursor_l,
                cursor_l+DPoint(0, self._structure_length/4\
                    -self._bottom_tiny_lead_length))
        cursor_l = self.primitives["bottom_left_small_lead"].end

        cursor_l += DPoint(0, 0)
        self.primitives["bottom_left_tiny_lead"] =\
            CPW(bottom_left_w_JJ, 0, cursor_l,
                cursor_l+DPoint(0, self._bottom_tiny_lead_length\
                    - top_left_h_JJ/2 - self._bridge_width))

        cursor_r = cursor +\
            DPoint(self._squid_width_bottom/2 \
                    + self._small_lead_width/2, 0)
        self.primitives["bottom_right_small_lead"] =\
            CPW(self._small_lead_width, 0, cursor_r,
                cursor_r+DPoint(0, self._structure_length/4\
                                   -self._bottom_tiny_lead_length))
        cursor_r = self.primitives["bottom_right_small_lead"].end

        cursor_r += DPoint(0, 0)
        self.primitives["bottom_right_tiny_lead"] =\
            CPW(bottom_right_w_JJ, 0, cursor_r,
                cursor_r+DPoint(0, self._bottom_tiny_lead_length\
                    - top_right_h_JJ/2 - self._bridge_width))


class Line_N_JJCross( ElementBase ):
    def __init__( self, origin, params, trans_in=None  ):
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

        self.poly1 = self._make_polygon( self.b, self.w, self.d1, self.f1,   self.d2 )

        super( self ).__init__( origin, trans_in )
    def _make_polygon( self, length, w, d, f, overlapping ):
        polygon = DSimplePolygon
        p1 = DPoint(0,0)
        p2 = p1 + DPoint( length,0 )
        p3 = p2 + DPoint( 0, w )
        p4 = p3 - DPoint( overlapping,0 )
        p5 = p4 - DPoint( 0, d )
        p6 = p5 - DPoint( f,0 )
        p7 = p6 + DPoint( 0, d )
        p8 = p1 + DPoint( 0,w )

        polygon = DSimplePolygon( [p1,p2,p3,p4,p5,p6,p7,p8] )
        return polygon

    def init_regions( self ):
        self.metal_region.insert( SimplePolygon(self.poly1) )
