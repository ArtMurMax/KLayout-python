import pya
from pya import Region, DPoint, Cell, Vector, Trans, DSimplePolygon

from classLib._PROG_SETTINGS import PROGRAM

from collections import OrderedDict

from typing import Union


class ChipDesign:
    def __init__(self, cell_name="testScript"):
        """
        Inherit this class for working on width chip design
        and override draw() method where other drawing
        methods should be called from
        call show() to draw everything
        str cell_name - name of width cell, e.g. 'testScript'

        Parameters
        ----------
        cell_name : str
            name of cell design will be written into, e.g. 'testScript'
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

        # additinal variables for convinience
        self.origin = DPoint(0, 0)

        # design parameters that were passed to the last
        # self.draw(...) call are stored here as ordered dict
        self.design_pars = OrderedDict()
        self.sonnet_ports: list[DPoint] = []

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
    def crop(self, box, layer=None):
        if layer is None:
            self.__erase_in_layer(self.layer_ph, box)
            self.__erase_in_layer(self.layer_el, box)
        else:
            self.__erase_in_layer(layer, box)

    # Erases everything outside the box in width layer
    def __erase_in_layer(self, layer, box):
        reg_l = self._reg_from_layer(layer)
        box_reg = Region(box)
        reg_l &= box_reg

        temp_i = self.cell.layout().layer(pya.LayerInfo(PROGRAM.LAYER1_NUM, 0))
        self.cell.shapes(temp_i).insert(reg_l)
        self.cell.layout().clear_layer(layer)
        self.cell.layout().move_layer(temp_i, layer)
        self.cell.layout().delete_layer(temp_i)

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

    def transform_layer(self, layer_i, trans, trans_ports=False):
        """
        Performs transofmation of the layer desired.

        Parameters
        ----------
        layer_i : int
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
        r_cell = Region(self.cell.begin_shapes_rec(layer_i))

        r_cell.transform(trans)

        temp_i = self.cell.layout().layer(pya.LayerInfo(PROGRAM.LAYER1_NUM, 0))
        self.cell.shapes(temp_i).insert(r_cell)
        self.cell.layout().clear_layer(layer_i)
        self.cell.layout().move_layer(temp_i, layer_i)
        self.cell.layout().delete_layer(temp_i)

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
