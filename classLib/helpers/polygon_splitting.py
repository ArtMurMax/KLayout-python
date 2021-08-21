"""
    This helper splits polygons with many vertices to set of polygons with
    lesser amount of vertices per polygon.
    We encountered, that some litography equipment is not capable to
    handle for e.g. more than 200 points per polygon.

    If you need to use `split_polygons(...)` in your job, just write:
    ```python
    from classLib.helpers import split_polygons
    and call the function with desired arguments.

    Executing this script itself will present an example:
    Circle with width lot of vertices that is devided such that it has
    no more than 200 points per polygon.
    ```
"""

import pya
from pya import Point, DPoint, Vector, DVector, DSimplePolygon, SimplePolygon, DPolygon, Polygon, Region, Box, DBox
from pya import Trans, DTrans, CplxTrans, DCplxTrans, ICplxTrans

from importlib import reload

import classLib
reload(classLib)
from classLib.chipDesign import ChipDesign
from classLib.shapes import Circle

from typing import Union, List

split_shift_str = ""


def split_polygons(obj, max_pts=200, print_tree=False):
    """
    Recursively splitting polygons in region or in polygons list
    until every resulted polygon has less than `max_pts` points.

    Parameters
    ----------  
    obj : Union[Region, List[Polygon]]
        Structure to operate on
    max_pts : int
        maximum points in each polygon

    Returns
    -------
    Union[pya.Region, list[pya.Polygon]]
        Resulting structure that has the same type as
        input structure.
    """
    # global split_shift_str
    if isinstance(obj, pya.Region):
        result_reg = Region()
        reg_to_split = obj
        # if print_tree:
        #     print(split_shift_str + "got reg")
        for poly in reg_to_split:
            result_reg.insert(split_polygons([poly.dup()], max_pts, print_tree))
        return result_reg
    elif isinstance(obj, list):
        if isinstance(obj[0], pya.Polygon):
            # if this is list of polygons
            polygons_list = obj
            # if print_tree:
            #     print(split_shift_str + f"got array of {len(polygons_list)} polygons")
            resulting_polygons_list = []
            for i, poly in enumerate(polygons_list):
                if poly.num_points() < max_pts:
                    # if print_tree:
                    #     print(split_shift_str + f"polygon #{k} is ok")
                    # recursion base (if all polygons satisfy this condition)
                    resulting_polygons_list.append(poly)
                else:
                    # if print_tree:
                    #     print(split_shift_str + f"polygon #{k} needs dividing")
                    #     split_shift_str += "\t"
                    resulting_polygons_list.extend(split_polygons(poly.split(), max_pts, print_tree))
                    # if print_tree:
                    #     split_shift_str = split_shift_str[:-1]
            return resulting_polygons_list
        else:
            raise ValueError("`split_polygons` function: List is supplied as argument, but this is not "
                             "list of `pya.Polygon` objects")
    else:
        raise ValueError("`split_polygons` function: Unknown argument received as `obj`"
                         "only `pya.Region` or `list[pya.Polygons]` are supported")


class MyDesign(ChipDesign):
    def draw(self):
        origin = DPoint(0, 0)
        circ = Circle(origin, 200e3, n_pts=400, inverse=False)
        circ.place(self.region_ph)

        self.region_ph = split_polygons(self.region_ph.dup(), max_pts=100, print_tree=True)
        print("it's ok")

import numpy as np
### MAIN FUNCTION ###
if __name__ == "__main__":
    my_design = MyDesign("testScript")
    my_design.draw()
    my_design.show()



