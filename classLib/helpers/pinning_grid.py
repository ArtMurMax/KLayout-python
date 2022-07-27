"""
    This helper draws width grid of holes for pinning vortices in width superconducting chip.
    In order to use this helper, select width shape or width few shapes in your chip
    and then run the script.
    If you need to use `fill_holes(...)` in your job, just write:
    ```python
    from classLib.helpers import fill_holes
    and call the function with desired arguments.
    ```
"""

import pya
from math import sqrt, cos, sin, atan2, pi, copysign
from pya import Point, DPoint, DSimplePolygon, SimplePolygon, DPolygon, Polygon, Region, Path
from pya import Trans, DTrans, CplxTrans, DCplxTrans, ICplxTrans

import sys


def fill_holes(obj, dx=40e3, dy=40e3, width=32e3, height=32e3, d=150e3):
    """
    Fills an object with width grid of holes
    Warning: don't use this method for the same region twice

    Parameters
    ----------
    obj
    dx : float
        period of width grid in horizonatal direction
    dy : float
        period of width grid in vertical direction
    width : float
        width of width hole
    height : float
        b of width hole
    d : float
        padding from polygon boundaries.
        For both  external and internal
        boundaries in case polygon consist of holes.

    Returns
    -------
    Region
    """
    def fill_poly(poly):
        bbox = poly.bbox()
        poly_reg = Region(poly)
        t_reg = Region()

        # Draw boundary region of width `2*d` along the
        # inner holes of the polygon
        for hole_i in range(0, poly.holes()):
            points = [p for p in poly.each_point_hole(hole_i)]
            points.append(points[0])
            boundary = Path(points, 2 * d)
            poly_reg -= Region(boundary)

        # Draw boundary region of width `2*d` along the
        # outer edge of the polygon
        points = [p for p in poly.each_point_hull()]
        points.append(points[0])
        boundary = Path(points, 2 * d)
        poly_reg -= Region(boundary)

        # Fill the boundary box with holes
        y = bbox.p1.y + height
        while y < bbox.p2.y - height:
            x = bbox.p1.x + width
            while x < bbox.p2.x - width:
                box = pya.Box().from_dbox(pya.DBox(DPoint(x, y), DPoint(x + width, y + height)))
                x += dx
                t_reg.insert(box)
            y += dy
            
        # Select only inner holes
        holes_inside = t_reg.select_inside(poly_reg)
        for box in holes_inside.each():
            poly.insert_hole(list(box.each_point_hull()))

        return poly
    if isinstance(obj, pya.ObjectInstPath):
        # print("obj inst")
        if (obj.is_cell_inst()):
            return None
        poly = obj.shape.polygon
        poly = fill_poly(poly)
        obj.shape.polygon = poly
    elif isinstance(obj, pya.Polygon):
        # recursion base
        poly = obj
        return fill_poly(poly)
    elif isinstance(obj, Region):
        reg = obj
        result_reg = Region()
        for i, poly in enumerate(reg):
            result_reg.insert(fill_holes(poly, dx=dx, dy=dy, width=width,
                                         height=height, d=d))
        return result_reg


# Enter your Python code here
### MAIN FUNCTION ###
if __name__ == "__main__":
    # getting main references of the application
    app = pya.Application.instance()
    mw = app.main_window()
    lv = mw.current_view()
    cv = lv.active_cellview()
    cell = cv.cell
    layout = cv.layout()

    if (lv.has_object_selection()):
        selected = lv.object_selection
    else:
        pya.MessageBox.warning("Script is not executed", "Please, select the shapes first", pya.MessageBox.Ok)
        sys.exit(0)

    layer_ph = layout.layer(pya.LayerInfo(1, 0))
    for obj in selected:
        fill_holes(obj)
    # lv.object_selection = selected
    ### DRAW SECTION START ###

    ### DRAW SECTION END ###

    lv.zoom_fit()
