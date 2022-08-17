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
    # DRAWING SECTION START
    box = pya.Region(pya.Box(DPoint(0, 0), DPoint(10000.0e3, 5000.0e3)))
    x0 = 500e3
    y0 = 1115e3
    length = 710e3
    width = 635e3
    x_shift = (9000e3 - 100e3 - width) / 4
    pts = [] # массив точек для копланарной линии
    #y_start = 300e3
    y_start = 2500e3
    pts.append(DPoint(0, y_start))
    r = 400e3  # радиус кривизны копланарной линии
    x1 = 778e3 + x0 - r
    y1 = y_start
    n_points = 100 # число точек в скруглении копланара
    for j in range(1, n_points + 1):
        x1 += 2*r * sin(pi / (4 * n_points)) * cos(pi / 2 * j / n_points)
        y1 += 2*r * sin(pi / (4 * n_points)) * sin(pi / 2 * j / n_points) * (1 - 2 * (i % 2))
        pts.append(DPoint(x1, y1))
    print(x1)
    #pts.append(DPoint(x1, y1))

    for i in range(5):
        if i != 0:
            box -= pya.Region(pya.Box(DPoint(x0 + x_shift*i, y0), DPoint(x0 + length + x_shift*i, y0 + width)))
        if i != 4:
            box -= pya.Region(pya.Box(DPoint(x0 + x_shift*i, y0 + 1500e3 + width), DPoint(x0 + length + x_shift*i, y0 + width + 1500e3 + width)))
        if i != 4 and i != 0:
            box -= pya.Region(pya.Box(DPoint(x1 + 49e3 + 90, y0 + width + 1500e3/2 - width/2), DPoint(x1 + 49e3 + 90 + length, y0 + width + 1500e3/2 + width - width / 2)))
        if i == 0 or i == 4:
            y1 += (5000e3 - 2 * 300e3 - 2200e3 - 2 * r) * (1 - 2 * (i % 2))
        else:
            y1 += (5000e3 - 2*300e3 - 2*r) * (1 - 2 * (i % 2))
        for j in range(1, n_points + 1):
            x1 += 2*r * sin(pi/(4 * n_points)) * sin(pi/2 * j/n_points)
            y1 += 2*r * sin(pi / (4 * n_points)) * cos(pi / 2 * j / n_points) * (1 - 2 * (i % 2))
            pts.append(DPoint(x1, y1))
        pts.append(DPoint(x1, y1))
        if i != 4:
            x1 += x_shift - 2*r
            pts.append(DPoint(x1, y1))
            for j in range(1, n_points + 1):
                x1 += 2*r * sin(pi / (4 * n_points)) * cos(pi / 2 * j / n_points)
                y1 -= 2*r * sin(pi / (4 * n_points)) * sin(pi / 2 * j / n_points) * (1 - 2 * (i % 2))
                pts.append(DPoint(x1, y1))
    pts.append(DPoint(10000.0e3, y1))
    print(pts)
    path1 = pya.DPath(pts, 49e3)
    path2 = pya.DPath(pts, 30e3)
    box -= pya.Region(path1)
    box += pya.Region(path2)
    cell.shapes(layer_photo).insert(box)


    lv.zoom_fit()



