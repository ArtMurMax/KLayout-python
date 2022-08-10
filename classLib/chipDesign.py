import pya
from pya import Region, DPoint, Cell, Vector, Trans, DSimplePolygon

from classLib._PROG_SETTINGS import PROGRAM

from collections import OrderedDict
import numpy as np
from numbers import Number
from typing import Union


class ChipDesign:
    def __init__(self, cell_name="testScript"):
        """
        Inherit this class for working on width chip design
        and override draw() method where other drawing
        methods should be called from
        call show() to draw everything
        str cell_name - chip_name of width cell, e.g. 'testScript'

        Parameters
        ----------
        cell_name : str
            chip_name of cell design will be written into, e.g. 'testScript'
        """
        # getting main references of the application
        self.app = pya.Application.instance()
        self.mw = self.app.main_window()
        self.lv = self.mw.current_view()
        self.cv = None
        self.cell = None

        # basic regions for sample
        self.region_ph = Region()
        self.region_el = Region()

        # this insures that lv and cv are valid objects
        if (self.lv == None):
            self.cv = self.mw.create_layout(1)
            self.lv = self.mw.current_view()
        else:
            self.cv = self.lv.active_cellview()

        # find or create the desired by programmer cell and layer
        self.layout = self.cv.layout()
        self.layout.dbu = 0.001
        if (self.layout.has_cell(cell_name)):
            self.cell = self.layout.cell(cell_name)
        else:
            self.cell = self.layout.create_cell(cell_name)

        # basic layers for sample
        info = pya.LayerInfo(1, 0)
        info2 = pya.LayerInfo(2, 0)
        self.layer_ph = self.layout.layer(info)  # photoresist layer
        self.layer_el = self.layout.layer(info2)  # e-beam lithography layer

        # clear this cell and layer
        self.cell.clear()

        # setting layout view  
        self.lv.select_cell(self.cell.cell_index(), 0)
        self.lv.add_missing_layers()

        self.version = "not implemented"

        # additinal variables for convinience
        self.origin = DPoint(0, 0)

        # design parameters that were passed to the last
        # self.draw(...) call are stored here as ordered dict
        self.design_pars = OrderedDict()
        self.sonnet_ports: list[DPoint] = []

    def get_version(self):
        return self.version

    # Call other methods drawing parts of the design from here
    def draw(self, design_params=None):
        """
        Purely virtual base-class method that is ought to be
        implemented in child classes.
        Responsible for calling functions that draw separate
        objects.

        Must be started with self.deisgn_pars = design_params
                
        Parameters
        ----------
        design_params : OrderedDict
            dictionary that contains design parameters and
            used by other drawing routines
        Returns
        -------
        None
        """
        raise NotImplementedError

    # Call this m
    def show(self, design_params=None):
        self._transfer_regs2cell()

    def _transfer_regs2cell(self):
        # this too methods assumes that all previous drawing
        # functions are placing their object on regions
        # in order to avoid extensive copying of the polygons
        # to/from cell.shapes during the logic operations on
        # polygons
        # can be modified in child classes if there are different
        # layers set
        self.cell.shapes(self.layer_ph).insert(self.region_ph)
        self.cell.shapes(self.layer_el).insert(self.region_el)
        self.lv.zoom_fit()

    # Erases everything outside the box
    def crop(self, box, region=None):
        if region is None:
            self.__crop_box_in_region(self.region_ph, box)
            self.__crop_box_in_region(self.region_el, box)
        else:
            self.__crop_box_in_region(region, box)

    # Erases everything outside the box in width layer
    def __crop_box_in_region(self, region, box):
        box_reg = Region(box)
        region &= box_reg

    def _reg_from_layer(self, layer):
        if layer == self.layer_el:
            return self.region_el
        elif layer == self.layer_ph:
            return self.region_ph
        else:
            return None

    def inverse_destination(self, dest: Union[Region, Cell], layer_i: int = -1):
        """
            Inverses empty regions and solid polygons
        on the given destination `dest` and `layer`.
            If layer is not specified, destination `dest`
        is interpreted as `pya.Region` instance.
            Otherwise, `dest` is interpreted as `pya.Cell` instance

        Parameters
        ----------
        dest : Union[Region, Cell]
            destination, interpreted either as Region or Cell instances depending
            on whether layer was provided
        layer_i : Optional[int]
            positive layer index.
        Returns
        -------
        None
        """
        tmp_reg = Region()
        tmp_reg.insert(self.chip_box)

        if layer_i == -1:
            dest_reg = dest
            dest_reg ^= tmp_reg
        else:
            r_cell = Region(dest.begin_shapes_rec(layer_i))
            r_cell ^= tmp_reg
            temp_layer_i = dest.layout().layer(pya.LayerInfo(PROGRAM.LAYER1_NUM, 0))

            # Moving layers.
            # Due to internal representation, region polygons are actually
            # point to polygons in width cell. So we can
            dest.shapes(temp_layer_i).insert(r_cell)
            dest.layout().clear_layer(layer_i)
            dest.layout().move_layer(temp_layer_i, layer_i)
            dest.layout().delete_layer(temp_layer_i)

    def transform_region(self, reg, trans, trans_ports=False):
        """
        Performs transofmation of the layer desired.

        Parameters
        ----------
        reg : rEGiOn
            layer index, >0
        trans : Union[DcplxTrans, DTrans]
            transformation to perform
        trans_ports : bool
            If `True` also performs transform of `self.sonnet_ports`
            as they are vectors.

        Returns
        -------
        None
        """
        reg.transform(trans)

        if trans_ports:
            self.sonnet_ports = list(
                DSimplePolygon(self.sonnet_ports).transform(trans).each_point()
            )

    # Save your design as GDS-II
    def save_as_gds2(self, filename):
        slo = pya.SaveLayoutOptions()
        slo.format = 'GDS2'
        slo.gds2_libname = 'LIB'
        slo.gds2_max_cellname_length = 32000
        slo.gds2_max_vertex_count = 8000
        slo.gds2_write_timestamps = True
        slo.select_all_layers()
        self.lv.save_as(self.cell.cell_index(), filename, slo)

    # get all geometry parameters as dictionary (todo exists)
    def get_geometry_parameters(self):
        # TODO: add docstring and case with attributes that are not
        #  geometry parameters themselves but
        #  contain geometry parameters. e.g. `self.chip
        pars_dict = self.__dict__
        geometry_dict = {}  # dictionary containing geometry parameters

        # choose only numbers or array(iterables) with numbers
        for key, val in pars_dict.items():
            if isinstance(val, (list, np.ndarray)):
                for i, it_entry in enumerate(val):
                    if isinstance(it_entry, Number):
                        it_entry_key = key + "_" + str(i)  # for readability of pandas.DataFrame
                        geometry_dict[it_entry_key] = it_entry
                    else:
                        break
            elif isinstance(val, Number):
                geometry_dict[key] = val
        geometry_dict["design_version"] = self.get_version()
        return geometry_dict
