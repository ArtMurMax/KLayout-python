import pya
from pya import Region

def extended_region(reg, extension=0):
    """
    extends region in outer direction by `extension` value.
    extension is orthogonal to region`s polygons edges.
    Parameters
    ----------
    reg : Region
        pya.Region instance
    extension : float
        extension in nm
    Returns
    -------
    Region
        extended version of region
    """
    tmp_reg = Region()
    ep = pya.EdgeProcessor()
    for poly in reg.each():
        tmp_reg.insert(
            ep.simple_merge_p2p(
                [
                    poly.sized(
                        extension,
                        extension,
                        2
                    )
                ],
                False,
                False,
                1
            )
        )
    return tmp_reg