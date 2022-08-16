from math import cos, sin, atan2, pi
import itertools

import pya
from pya import Point, DPoint, DVector, DSimplePolygon, SimplePolygon, DPolygon, Polygon, Region
from pya import Trans, DTrans, CplxTrans, DCplxTrans, ICplxTrans

from classLib.baseClasses import ComplexBase
from classLib.coplanars import CPW, CPWArc
from classLib.resonators import Coil_type_1
from classLib.shapes import XmonCross

from sonnetSim.sonnetLab import SonnetLab, SonnetPort, SimulationBox



"""
This Code implements a Serpentine PCell 

It will generate a path starting from 0,0 and produce a serpentine this
way:

    +->+  +->    ^ 
    ^  |  ^      |
    |  |  |      |
    |  |  |      | s
    |  |  |      |
    |  |  |      |
    |  V  |      |
 +->+  +->+      V

    <-> u

The parameters are:
- l: the layer to use
- w: the width of the path
- n: the number of vertical paths
- u: see drawing above
- s: see drawing above
- first_length
- last_length

NOTE: using negative angles makes the Serpentine turn the other way

"""


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


    class Serpentine(pya.PCellDeclarationHelper):
        """
        The PCell declaration for the Serpentine
        """

        def __init__(self):

            # Important: initialize the super class
            super(Serpentine, self).__init__()

            # declare the parameters
            self.param("l", self.TypeLayer, "Layer", default=pya.LayerInfo(1, 0))
            self.param("n", self.TypeInt, "Number of points per full turn", default=5)
            self.param("w", self.TypeDouble, "The width", default=1.0)
            self.param("u", self.TypeDouble, "One turn's pitch", default=2.0)
            self.param("s", self.TypeDouble, "The turn's length", default=20.0)
            self.param("n_points", self.TypeInt, "", default=1000)
            #self.param("r", self.TypeDouble, "Radius of curvature of rounded corner", default=20.0)
            self.param("first_length", self.TypeDouble, "The first length", default=0.0)
            self.param("last_length", self.TypeDouble, "The first length", default=0.0)
            self.param("start_point_x", self.TypeDouble, "start point x", default=0.0)
            self.param("start_point_y", self.TypeDouble, "start point y", default=0.0)


        def display_text_impl(self):
            # Provide a descriptive text for the cell
            return "Serpentine(L=%s,S=%.12g,U=%.12g" % (str(self.l), self.s, self.u)

        def produce_impl(self):

            # This is the main part of the implementation: create the layout

            # compute the Serpentine: generate a list of spine points for the path and then
            # create the path

            pts = []

            x = self.start_point_x
            y = self.start_point_y
            #x = 0
            #y = 0
            pts.append(pya.DPoint(x - 3, y))
            x += self.first_length + self.w/2
            pts.append(pya.DPoint(x, y))
            for i in range(0, self.n):
                pts.append(pya.DPoint(x, y))
                #y += self.u + self.w
                pts.append(pya.DPoint(x, y))
                if (i % 2) == 0:
                    for j in range(1, self.n_points + 1):
                        x += (self.u + self.w) * sin(pi / (2 * self.n_points)) * cos(pi * j / self.n_points)
                        y += (self.u + self.w) * sin(pi / (2 * self.n_points)) * sin(pi * j / self.n_points)
                        pts.append(pya.DPoint(x, y))
                    x -= self.s + self.w
                else:
                    for j in range(1, self.n_points + 1):
                        x -= (self.u + self.w) * sin(pi / (2 * self.n_points)) * cos(pi * j / self.n_points)
                        y += (self.u + self.w) * sin(pi / (2 * self.n_points)) * sin(pi * j / self.n_points)
                        pts.append(pya.DPoint(x, y))
                    x += self.s + self.w
                pts.append(pya.DPoint(x, y))

            # One last point to move to the end location
            #y += self.u + self.w
            pts.append(pya.DPoint(x, y))
            if (self.n % 2) == 0:
                #x -= self.last_length + self.w/2
                for j in range(1, self.n_points + 1):
                    x += (self.u + self.w) * sin(pi / (2 * self.n_points)) * cos(pi * j / self.n_points)
                    y += (self.u + self.w) * sin(pi / (2 * self.n_points)) * sin(pi * j / self.n_points)
                    pts.append(pya.DPoint(x, y))
                x -= self.last_length + 3 + self.w/2
            else:
                for j in range(1, self.n_points + 1):
                    x -= (self.u + self.w) * sin(pi / (2 * self.n_points)) * cos(pi * j / self.n_points)
                    y += (self.u + self.w) * sin(pi / (2 * self.n_points)) * sin(pi * j / self.n_points)
                    pts.append(pya.DPoint(x, y))
                x += self.last_length + 3 + self.w/2
            pts.append(pya.DPoint(x, y))

            # create the shape
            path = pya.DPath(pts, self.w)
            #param = {"layer": layer_photo, "radius": 20.0, "npoints": 16, "path": path}
            #basicLib = pya.Library.library_by_name("Basic")
            #rPath = basicLib.layout().pcell_declaration("ROUND_PATH")
            #pcell = layout.add_pcell_variant(basicLib, rPath.id(), param)
            #t = pya.Trans(0, 0)

            cell.shapes(layer_photo).insert(path)
            #cell.insert(pya.CellInstArray(pcell, t))
            #cell.insert(pya.CellInstArray(pcell, t))


    class SerpentineLib(pya.Library):
        """
        The library where we will put the PCell into
        """

        def __init__(self):
            # Set the description
            self.description = "Serpentine PCell Library"

            # Create the PCell declarations
            self.layout().register_pcell("Serpentine", Serpentine())

            # Register us with the name "SerpentineLib".
            self.register("SerpentineLib")


    # Instantiate and register the library
    SerpentineLib()


    layer_info_photo = pya.LayerInfo(10, 0)
    layer_info_el = pya.LayerInfo(1, 0)
    layer_photo = layout.layer(layer_info_photo)
    layer_el = layout.layer(layer_info_el)

    # setting layout view
    lv.select_cell(cell.cell_index(), 0)
    lv.add_missing_layers()

    #DRAWING SECTION START
    # create a slot for a resonator
    length = 710.40625e3
    width = 635e3
    # параметры для скругления углов
    inner_radius = 10
    outer_radius = 10
    n_points = 1000
    main_box = pya.Box(DPoint(0, 0), DPoint(length, width))
    #box1 = pya.Box(DPoint(47.625e3, 71.4375e3), DPoint(623.09375e3, 615.15625e3))
    box1 = pya.DPolygon([DPoint(47.625e3, 71.4375e3), DPoint(623.09375e3, 71.4375e3), DPoint(623.09375e3, 615.15625e3), DPoint(47.625e3, 615.15625e3)])
    box1 = box1.round_corners(inner_radius*1000, outer_radius*1000, n_points*1000)
    merged_boxes = pya.Region(main_box) - pya.Region(box1)
    cell.shapes(layer_photo).insert(merged_boxes)
    #cell.shapes(layer_photo).insert(box1)
    # coplanar line
    box2 = pya.Box(DPoint(748.40625e3, 0), DPoint(808.40625e3, width))
    box3 = pya.Box(DPoint(846.40625e3, 0), DPoint(889e3, width))
    cell.shapes(layer_photo).insert(box2)
    cell.shapes(layer_photo).insert(box3)
    # draw inductivity
    meander = Serpentine()
    meander.n = 18
    meander.w = 6.35
    meander.u=12.7
    meander.s=165
    meander.r = 10
    meander.start_point_x=307
    meander.start_point_y=181
    meander.first_length=100
    meander.last_length=80
    meander.n_points=100
    meander.produce_impl()
    #draw capacitance
    capacitance = pya.DPolygon([DPoint(284, 122), DPoint(98, 122), DPoint(98, 565), DPoint(571, 565),
                               DPoint(571, 122), DPoint(331, 122), DPoint(331, 152), DPoint(455, 152),
                               DPoint(455, 559), DPoint(177, 559), DPoint(177, 152), DPoint(284, 152)])
    capacitance = capacitance.round_corners(inner_radius, outer_radius, n_points)

    cell.shapes(layer_photo).insert(capacitance)
    #draw connections
    connection1 = pya.DPolygon([DPoint(324, 546), DPoint(324, 560), DPoint(331, 560), DPoint(331, 546)])
    connection2 = pya.DPolygon([DPoint(304, 71), DPoint(310, 71), DPoint(310, 179), DPoint(304, 179)])
    cell.shapes(layer_photo).insert(connection1)
    cell.shapes(layer_photo).insert(connection2)
    #DRAWING SECTION END
    lv.zoom_fit()