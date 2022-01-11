from importlib import reload

from classLib.helpers import polygon_splitting
reload(polygon_splitting)

from classLib.helpers import pinning_grid
reload(pinning_grid)

from classLib.helpers import region_manipulation
reload(region_manipulation)

fill_holes = pinning_grid.fill_holes
split_polygons = polygon_splitting.split_polygons
extended_region = region_manipulation.extended_region