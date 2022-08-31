from typing import Hashable, Union, Dict, Any
import pya
from math import sqrt, cos, sin, atan2, pi, copysign
from pya import Point, DPoint, DSimplePolygon, SimplePolygon, DPolygon, Polygon, Region
from pya import Trans, DTrans, CplxTrans, DCplxTrans, ICplxTrans

from classLib._PROG_SETTINGS import PROGRAM

from collections import OrderedDict
import itertools


class ElementBase():
    """
    @brief: base class for simple single-layer or multi-layer elements and objects that are consisting of
            several polygons.
            metal_region polygons will be added to the design
            empty_region polygons will be erased from the background with
            metal region polygons already added.

    """

    def __init__(self, origin, trans_in=None, inverse=False,
                 region_id="default"):
        ## MUST BE IMPLEMENTED ##
        self.connections = []  # DPoint list with possible connection points
        self.connection_edges = []  # indexes of edges that are intended to connect to other polygons
        # indexes in "self.connection_edges" where Sonnet ports
        # should be placed
        self.sonnet_port_connections = []
        self.angle_connections = []  # list with angle of connecting elements
        ## MUST BE IMLPEMENTED END ##

        self.connection_ptrs = []  # pointers to connected structures represented by their class instances

        self.origin = origin
        self.inverse = inverse
        # TODO: after Region.insert() and/or? metal_region + other_region
        #  there is width problem that initial region has dimensions
        #  Region().bbox() = ((-1,-1,1,1)) or something like that
        #  Hence, if you wish to take Region.bbox() of the resulting region
        #  You will get incorrect value due to initial region having
        #  something "blank" at the creation moment. This may be solved
        #  by either shrinking resulting region to shapes it contains,
        #  or by refusing of usage of empty `Region()`s
        self.metal_regions: Dict[Any, Region] = OrderedDict()
        self.empty_regions: Dict[Any, Region] = OrderedDict()
        self.metal_region = Region()
        self.empty_region = Region()
        self.region_id = region_id
        self.metal_regions[self.region_id] = self.metal_region
        self.empty_regions[self.region_id] = self.empty_region
        self.region_ids = [self.region_id]

        self.metal_region.merged_semantics = True
        self.empty_region.merged_semantics = True
        self.DCplxTrans_init = None
        self.ICplxTrans_init = None

        if (trans_in is not None):
            # TODO: update with Klayouts new rules.
            # if( isinstance( trans_in, ICplxTrans ) ): <==== FORBIDDEN
            if (isinstance(trans_in, DCplxTrans)):
                self.DCplxTrans_init = trans_in
                self.ICplxTrans_init = ICplxTrans().from_dtrans(trans_in)
            elif (isinstance(trans_in, CplxTrans)):
                self.DCplxTrans_init = DCplxTrans().from_itrans(trans_in)
                self.ICplxTrans_init = ICplxTrans().from_trans(trans_in)
            elif (isinstance(trans_in, DTrans)):
                self.DCplxTrans_init = DCplxTrans(trans_in, 1)
                self.ICplxTrans_init = ICplxTrans(Trans().from_dtrans(trans_in), 1)
            elif (isinstance(trans_in, Trans)):
                self.DCplxTrans_init = DCplxTrans(DTrans().from_itrans(trans_in), 1)
                self.ICplxTrans_init = ICplxTrans(trans_in, 1)
            elif (isinstance(trans_in, ICplxTrans)):
                # not tested 14.08.2021
                self.DCplxTrans_init = DCplxTrans(trans_in)
                self.ICplxTrans_init = trans_in
        self._geometry_parameters = OrderedDict()
        self._init_regions_trans()

    def get_geometry_params_dict(self, prefix="", postfix=""):
        """
        Function return geometry parameters in format:
        dict[prefix + key + postfix, value]

        Parameters
        ----------
        prefix : str
        postfix : str

        Returns
        -------
        dict
        """
        if hasattr(self, "_geometry_parameters"):
            tmp_dict = {}
            for key, item in self._geometry_parameters.items():
                tmp_dict[prefix + key + postfix] = item
            return tmp_dict
        else:
            print("Geometry parameters for ", self.__class__ , " does not implemented")
            return {}

    def init_regions(self):
        raise NotImplementedError

    # first it makes trans_init displacement
    # then the rest of the trans_init
    # then displacement of the current state to the origin
    # after all, origin should be updated
    def _init_regions_trans(self):
        self.init_regions()  # must be implemented in child classes

        dr_origin = DSimplePolygon([DPoint(0, 0)])
        if (self.DCplxTrans_init is not None):
            # constructor trans displacement
            dCplxTrans_temp = DCplxTrans(1, 0, False, self.DCplxTrans_init.disp)
            self.make_trans(dCplxTrans_temp)
            dr_origin.transform(dCplxTrans_temp)

            # rest of the constructor trans functions
            dCplxTrans_temp = self.DCplxTrans_init.dup()
            dCplxTrans_temp.disp = DPoint(0, 0)
            self.make_trans(dCplxTrans_temp)
            dr_origin.transform(dCplxTrans_temp)

        # translation local coordinates to the program coordinates by
        # tranlating geometry into `self.origin` - user-requested point
        # in program's coordinate system
        # Note: self.connections are already contain proper values
        self.make_trans(DCplxTrans(1, 0, False, self.origin))  # move to the origin
        self.origin += dr_origin.point(0)

    def make_trans(self, dCplxTrans):
        if (dCplxTrans is not None):
            regions = itertools.chain(self.metal_regions.values(), self.empty_regions.values())
            for reg in regions:
                iCplxTrans = ICplxTrans().from_dtrans(dCplxTrans)
                reg.transform(iCplxTrans)
            self._update_connections(dCplxTrans)
            self._update_alpha(dCplxTrans)

    def _update_connections(self, dCplxTrans):
        if (dCplxTrans is not None):
            '''the problem is, if k construct polygon with multiple points
            their order in poly_temp.each_point() doesn't coinside
            with the order of the list that was passed to the polygon
            constructor so, when k perform transformation and
            try to read new values through poly_temp.each_point()
            they values are rearranged solution is: k need to create
            polygon for each point personally,
            then the initial order presists'''
            for i, pt in enumerate(self.connections):
                poly_temp = DSimplePolygon([pt])
                poly_temp.transform(dCplxTrans)
                self.connections[i] = poly_temp.point(0)
        self._refresh_named_connections()

    def _refresh_named_connections(self):
        """
            if there is connections in `self.connections` that
        have specific class attribute e.g. `self.end` is assumed
        to coincide with `self.connections[1]`, then this function
        is used to update this correspondance after connections
        are changed due to call of `self.make_trans`.
            See classLib.Coplanars.CPW class for example.
        """
        # can be implemented in child class
        # see classLib.Coplanars.CPW for example
        pass

    def _update_alpha(self, dCplxTrans):
        if (dCplxTrans is not None):
            dCplxTrans_temp = dCplxTrans.dup()
            dCplxTrans_temp.disp = DPoint(0, 0)

            for i, alpha in enumerate(self.angle_connections):
                poly_temp = DSimplePolygon([DPoint(cos(alpha), sin(alpha))])
                poly_temp.transform(dCplxTrans_temp)
                pt = poly_temp.point(0)
                self.angle_connections[i] = atan2(pt.y, pt.x)
            self._refresh_named_angles()

    def _refresh_named_angles(self):
        """
            If there is angles in `self.angle_connections` that
        have specific class attribute e.g. `self.end_angle` is assumed
        to coincide with `self.angle_connections[1]`, then this function
        is used to update this correspondance after connections
        are changed due to call of `self.make_trans`.
            See classLib.Coplanars.CPW class for example.
        """
        pass

    def _update_origin(self, dCplxTrans):
        if (dCplxTrans is not None):
            poly_temp = DSimplePolygon([self.origin])
            poly_temp.transform(dCplxTrans)
            self.origin = poly_temp.point(0)

    def change_region_id(self, old_reg_id, new_reg_id):
        self.metal_regions[new_reg_id] = self.metal_regions.pop(old_reg_id)
        self.empty_regions[new_reg_id] = self.empty_regions.pop(old_reg_id)
        self.region_id = new_reg_id

    def place(self, dest, layer_i=-1, region_id="default", merge=False):
        if all([
                region_id not in self.metal_regions,
                region_id not in self.empty_regions
        ]):
            return

        if (layer_i != -1):
            r_cell = Region(dest.begin_shapes_rec(layer_i))
            temp_i = dest.layout().layer(pya.LayerInfo(PROGRAM.LAYER1_NUM, 0))
            if (region_id in self.metal_regions):
                metal_region = self.metal_regions[region_id]
                r_cell += metal_region
            if (region_id in self.empty_regions):
                empty_region = self.empty_regions[region_id]
                r_cell -= empty_region

            if (merge is True):
                r_cell.merge()

            dest.shapes(temp_i).insert(r_cell)
            dest.layout().clear_layer(layer_i)
            dest.layout().move_layer(temp_i, layer_i)
            dest.layout().delete_layer(temp_i)
        if (layer_i == -1):  # dest is interpreted as instance of Region() class
            if (region_id in self.metal_regions):
                metal_region = self.metal_regions[region_id]
                dest += metal_region
            if (region_id in self.empty_regions):
                empty_region = self.empty_regions[region_id]
                dest -= empty_region
            if (merge is True):
                dest.merge()


class ComplexBase(ElementBase):
    def __init__(self, origin, trans_in=None, region_id="default"):
        super().__init__(origin, trans_in, region_id=region_id)
        # ensures sequential order of drawing primitives
        self.primitives: Dict[Hashable, Union[ElementBase, ComplexBase]] = OrderedDict()
        self._init_primitives_trans()

    def _init_regions_trans(self):
        pass

    def make_trans(self, dCplxTrans_temp: DCplxTrans):
        for primitive in self.primitives.values():
            primitive.make_trans(dCplxTrans_temp)
        for region in itertools.chain(self.metal_regions.values(), self.empty_regions.values()):
            iCplxTrans = ICplxTrans().from_dtrans(dCplxTrans_temp)
            region.transform(iCplxTrans)
        self._update_connections(dCplxTrans_temp)
        self._update_alpha(dCplxTrans_temp)

    def _init_primitives_trans(self):
        self.init_primitives()  # must be implemented in every subclass
        dr_origin = DSimplePolygon([DPoint(0, 0)])
        if (self.DCplxTrans_init is not None):
            # constructor trans displacement
            dCplxTrans_temp = DCplxTrans(1, 0, False, self.DCplxTrans_init.disp)
            self.make_trans(dCplxTrans_temp)
            dr_origin.transform(dCplxTrans_temp)

            # rest of the constructor trans functions
            dCplxTrans_temp = self.DCplxTrans_init.dup()
            dCplxTrans_temp.disp = DPoint(0, 0)
            self.make_trans(dCplxTrans_temp)
            dr_origin.transform(dCplxTrans_temp)

        dCplxTrans_temp = DCplxTrans(1, 0, False, self.origin.x, self.origin.y)
        self.make_trans(dCplxTrans_temp)  # move to the origin
        self.origin += dr_origin.point(0)

        # Intermediate object representation is kept in its metal regions.
        # This is a tradeoff biased into memory size to simplify and speedup analysis of
        # compound objects.
        for element in self.primitives.values():
            for reg_id in self.region_ids:
                element.place(self.metal_regions[reg_id], region_id=reg_id)

    def place(self, dest, layer_i=-1, region_id="default"):
        if (layer_i != -1):
            # `dest` is interpreted as `pya.Cell` object
            r_cell = Region(dest.begin_shapes_rec(layer_i))
            for primitive in self.primitives.values():
                primitive.place(r_cell, region_id=region_id)

            temp_i = dest.layout().layer(pya.LayerInfo(PROGRAM.LAYER1_NUM, 0))
            dest.shapes(temp_i).insert(r_cell)
            dest.layout().clear_layer(layer_i)
            dest.layout().move_layer(temp_i, layer_i)
            dest.layout().delete_layer(temp_i)
        else:
            # `dest` is interpreted as `pya.Region` object
            for primitive in self.primitives.values():
                primitive.place(dest, region_id=region_id)

    def init_primitives(self):
        raise NotImplementedError

    def init_regions(self):
        pass

    def length(self, exception=None):
        """

        Parameters
        ----------
        exception : str
            primitives which names contains this string will be skipped

        Returns
        -------

        """
        length = 0
        for name, primitive in self.primitives.items():
            # getting only those who has length
            if hasattr(primitive, "length"):
                if (exception is not None) and (exception in name):
                    continue

                dl = 0
                if isinstance(primitive, ComplexBase):
                    dl = primitive.length(exception=exception)
                elif isinstance(primitive, ElementBase):
                    dl = primitive.length()
                else:
                    raise Exception("unknown primitive found while "
                                    "searching for length. "
                                    "Terminating.")

                length += dl
            else:
                continue

        return length
