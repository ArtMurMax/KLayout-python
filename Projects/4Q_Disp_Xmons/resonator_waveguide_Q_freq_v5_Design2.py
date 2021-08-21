# Enter your Python code here
from math import cos, sin, atan2, pi
import itertools

import pya
from pya import Point, DPoint, DVector, DSimplePolygon, SimplePolygon, DPolygon, Polygon, Region
from pya import Trans, DTrans, CplxTrans, DCplxTrans, ICplxTrans

from importlib import reload
import classLib

reload(classLib)

from classLib.coplanars import CPW, CPWParameters
from classLib.shapes import XmonCross
from classLib.baseClasses import ComplexBase

from sonnetSim.sonnetLab import SonnetLab, SonnetPort, SimulationBox
from classLib.resonators import EMResonatorTL3QbitWormRLTailXmonFork


class CHIP:
    dx = 0.6e6
    dy = 2.7e6
    nX = 600  # simulation box cells along X-axis
    nY = 1350  # simulation box cells along Y-axis

    # connections of the chip (not used in this file)
    gap = 150.e3
    width = 260.e3
    b = 2 * gap + width
    origin = DPoint(0, 0)
    box = pya.DBox(origin, origin + DPoint(dx, dy))
    L1 = 220
    # only 4 connections programmed by now
    connections = [box.p1 + DPoint(L1 + b / 2, 0), box.p1 + DPoint(dx - (L1 + b / 2), 0),
                   box.p2 - DPoint(L1 + b / 2, 0), box.p1 + DPoint(L1 + b / 2, dy)]

    @staticmethod
    def get_geometry_params_dict(prefix="", postfix=""):
        from collections import OrderedDict
        geometry_params = OrderedDict(
            [
                ("dx, um", CHIP.dx / 1e3),
                ("dy, um", CHIP.dy / 1e3),
                ("nX", CHIP.nX),
                ("nY", CHIP.nY)
            ]
        )
        modified_dict = OrderedDict()
        for key, val in geometry_params.items():
            modified_dict[prefix + key + postfix] = val
        return modified_dict


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
        cell = layout.cell("testScript")
    else:
        cell = layout.create_cell("testScript")

    layer_info_photo = pya.LayerInfo(10, 0)
    layer_info_el = pya.LayerInfo(1, 0)
    layer_photo = layout.layer(layer_info_photo)
    layer_el = layout.layer(layer_info_el)

    # setting layout view
    lv.select_cell(cell.cell_index(), 0)
    lv.add_missing_layers()

    ## DRAWING SECTION START ##
    origin = DPoint(0, 0)
    resonators_dx = 900e3
    # resonators parameters
    L_coupling_list = [1e3 * x for x in [310, 320, 320, 310, 300]]
    # corresponding to resonanse freq is linspaced in interval [6,9) GHz
    L0 = 1150e3
    L1_list = [1e3 * x for x in [57.94293048, 85.84821562, 138.21119481, 131.12841218,  79.78515414]]
    r = 60e3
    N_coils = [3] * len(L1_list)
    to_line_list = [56e3] * len(L1_list)

    estimated_res_freqs_init = [6.3, 6.3, 6.4, 6.5, 6.86]  # GHz
    freqs_span_corase = 1.0  # GHz
    corase_only = False
    freqs_span_fine = 0.005

    L2_list = [r] * len(L1_list)
    L3_list = [0e3] * len(L1_list)
    L4_list = [r] * len(L1_list)
    width_res = 20e3
    gap_res = 10e3
    Z_res = CPW(width_res, gap_res, origin, origin)

    # xmon cross parameters
    xmon_x_distance: float = 545e3  # from simulation of g_12
    xmon_dys_Cg_coupling = [14e3] * 5

    cross_len_x = 180e3
    cross_width_x = 60e3
    cross_gnd_gap_x = 20e3
    cross_len_y = 155e3
    cross_width_y = 60e3
    cross_gnd_gap_y = 20e3

    # fork at the end of resonator parameters
    fork_metal_width = 10e3
    fork_gnd_gap = 15e3
    xmon_fork_gnd_gap = 14e3
    xmon_fork_penetration_list = [-25e3] * len(L1_list)
    fork_x_span = cross_width_y + 2 * (xmon_fork_gnd_gap + fork_metal_width)
    fork_y_spans = [x * 1e3 for x in [8.73781, 78.3046, 26.2982, 84.8277, 35.3751]]
    # Xmon-fork parameters
    # -20e3 for Xmons in upper sweet-spot
    # -10e3 for Xmons in lower sweet-spot

    ### RESONATORS TAILS CALCULATIONS SECTION START ###
    # key to the calculations can be found in hand-written format here:
    # https://drive.google.com/file/d/1wFmv5YmHAMTqYyeGfiqz79a9kL1MtZHu/view?usp=sharing

    # x span between left long vertical line and
    # right-most center of central conductors
    resonators_widths = [2 * r + L_coupling for L_coupling in L_coupling_list]
    x1 = 2 * resonators_dx + resonators_widths[2] / 2 - 2 * xmon_x_distance
    x2 = x1 + xmon_x_distance - resonators_dx
    x3 = resonators_widths[2] / 2
    x4 = 3 * resonators_dx - (x1 + 3 * xmon_x_distance)
    x5 = 4 * resonators_dx - (x1 + 4 * xmon_x_distance)

    res_tail_shape = "LRLRL"
    tail_turn_radiuses = r
    L0_list = [L0 - xmon_dy_Cg_coupling for xmon_dy_Cg_coupling in xmon_dys_Cg_coupling]
    L2_list[0] += 6 * Z_res.b
    L2_list[1] += 0
    L2_list[3] += 3 * Z_res.b
    L2_list[4] += 6 * Z_res.b

    L3_list[0] = x1
    L3_list[1] = x2
    L3_list[2] = x3
    L3_list[3] = x4
    L3_list[4] = x5

    L4_list[1] += 6 * Z_res.b
    L4_list[2] += 6 * Z_res.b
    L4_list[3] += 3 * Z_res.b
    tail_segment_lengths_list = [[L2, L3, L4] for L2, L3, L4 in zip(L2_list, L3_list, L4_list)]
    tail_turn_angles_list = [
        [pi / 2, -pi / 2],
        [pi / 2, -pi / 2],
        [pi / 2, -pi / 2],
        [-pi / 2, pi / 2],
        [-pi / 2, pi / 2],
    ]
    tail_trans_in_list = [
        Trans.R270,
        Trans.R270,
        Trans.R270,
        Trans.R270,
        Trans.R270
    ]
    ### RESONATORS TAILS CALCULATIONS SECTION END ###
    # print([L0 - xmon_dy_Cg_coupling for xmon_dy_Cg_coupling in xmon_dys_Cg_coupling])
    print(L1_list)
    print(L2_list)
    print(L3_list)
    print(L4_list)
    # print("Lc = ", L_coupling_list, flush=True)
    import time

    time.sleep(0.5)  # waiting for print of parameters above
    from itertools import product
    dL1_list = [0]
    pars = list(
        product(
            zip(
              L1_list, estimated_res_freqs_init,
              to_line_list, L_coupling_list,
              xmon_fork_penetration_list, tail_segment_lengths_list,
              tail_turn_angles_list, tail_trans_in_list,
              xmon_dys_Cg_coupling, L0_list,
              list(range(0, 5)), fork_y_spans, N_coils
            ),
        dL1_list
        )
    )
    for params, dL1 in pars:
        # parameters exctraction
        L1 = params[0] + dL1
        estimated_freq = params[1]
        to_line = params[2]
        L_coupling = params[3]
        xmon_fork_penetration = params[4]
        tail_segment_lengths = params[5]
        tail_turn_angles = params[6]
        tail_trans_in = params[7]
        xmon_dy_Cg_coupling = params[8]
        L0 = params[9]
        resonator_idx = params[10]
        fork_y_span = params[11]
        N = params[12]
        print(resonator_idx)

        # frequency and span approximation cycle in 2 steps
        # starting with a corase frequency
        freqs_span = freqs_span_corase
        fine_resonance_success = False  # Whether or not we found local minima with `freqs_span = freqs_span_fine`
        while not fine_resonance_success:
            # box parameters calculation
            """
                I could make `test_region = Region()` and place my objects there
                and then call `test_region.bbox()` method to get appropriate dimensions
                but `bbox()` returns bad values due to the fact that a lot of empty regions
                were placed inside the geometry objects. For those empty regions `bbox()`
                will be something like `pya.Box(Point(-1,1), Point(1,-1))` or something like that.
                Due to multiple displacements transformations in nested classes
                that do not affect empty regions (so their bbox stays the same)
                actual region bbox is over exaggerated
            """
            print(
                L_coupling, L0, L1, r, N, res_tail_shape, tail_turn_radiuses,
                tail_segment_lengths, tail_turn_angles, tail_trans_in, fork_x_span, fork_y_span,
                fork_metal_width, fork_gnd_gap
            )
            worm = EMResonatorTL3QbitWormRLTailXmonFork(
                Z_res, origin, L_coupling, L0, L1, r, N,
                tail_shape=res_tail_shape, tail_turn_radiuses=r,
                tail_segment_lengths=tail_segment_lengths,
                tail_turn_angles=tail_turn_angles, tail_trans_in=tail_trans_in,
                fork_x_span=fork_x_span, fork_y_span=fork_y_span,
                fork_metal_width=fork_metal_width, fork_gnd_gap=fork_gnd_gap
            )
            xmon_center = (worm.fork_x_cpw.start + worm.fork_x_cpw.end) / 2 + \
                          DVector(0, -xmon_dy_Cg_coupling - worm.fork_metal_width / 2)
            xmon_center += DPoint(
                0,
                -(cross_len_y + cross_width_x / 2 + min(cross_gnd_gap_y, xmon_fork_gnd_gap))
            )
            xmonCross = XmonCross(
                xmon_center,
                sideX_length=cross_len_x, sideX_width=cross_width_x, sideX_gnd_gap=cross_gnd_gap_x,
                sideY_length=cross_len_y, sideY_width=cross_width_y, sideY_gnd_gap=cross_gnd_gap_y
            )
            tmp_reg = Region()
            worm.place(tmp_reg)
            xmonCross.place(tmp_reg)
            bbox = tmp_reg.bbox()
            import math

            # main drive line coplanar
            width = 20e3
            gap = 10e3
            z0_cpw_params = CPWParameters(width, gap)
            CHIP.dx = 1.2 * bbox.width() + 8 * Z_res.b
            CHIP.nX = int(CHIP.dx / 2.5e3)

            y = CHIP.dy - 150e3 - (width + 2 * gap) / 2
            p1 = DPoint(0, y)
            p2 = DPoint(CHIP.dx, y)
            Z0 = CPW(start=p1, end=p2, cpw_params=z0_cpw_params)

            # clear this cell and layer
            cell.clear()
            
            import math

            if math.copysign(1, tail_turn_angles[0]) > 0:
                worm_x = 5 * Z_res.b + Z_res.b / 2
            else:
                worm_x = CHIP.dx - (5 * Z_res.b + L_coupling + r + Z_res.b / 2)

            worm_y = Z0.start.y - to_line

            worm = EMResonatorTL3QbitWormRLTailXmonFork(
                Z_res, DPoint(worm_x, worm_y), L_coupling, L0 - xmon_dy_Cg_coupling, L1, r, N,
                tail_shape=res_tail_shape, tail_turn_radiuses=r,
                tail_segment_lengths=tail_segment_lengths,
                tail_turn_angles=tail_turn_angles, tail_trans_in=tail_trans_in,
                fork_x_span=fork_x_span, fork_y_span=fork_y_span,
                fork_metal_width=fork_metal_width, fork_gnd_gap=fork_gnd_gap
            )
            xmon_center = (worm.fork_x_cpw.start + worm.fork_x_cpw.end) / 2 + \
                               DVector(0, -xmon_dy_Cg_coupling - worm.fork_metal_width / 2)
            xmon_center += DPoint(
                0,
                -(cross_len_y + cross_width_x / 2 + min(cross_gnd_gap_y, xmon_fork_gnd_gap))
            )
            xmonCross = XmonCross(
                xmon_center,
                sideX_length=cross_len_x, sideX_width=cross_width_x, sideX_gnd_gap=cross_gnd_gap_x,
                sideY_length=cross_len_y, sideY_width=cross_width_y, sideY_gnd_gap=cross_gnd_gap_y
            )

            # optimal to place everything in one region for polygon logic
            # operations and then transfer it to the cell
            photo_reg = Region()

            # calculate simulation box dimensions
            chip_box = pya.Box(DPoint(0, 0), DPoint(CHIP.dx, CHIP.dy))

            # placing all objects in proper order and translating them to origin
            photo_reg.insert(chip_box)

            # translating all objects so that chip.p1 at coordinate origin
            xmonCross.place(photo_reg)
            worm.place(photo_reg)

            xmonCross_corrected = XmonCross(
                xmon_center,
                sideX_length=cross_len_x, sideX_width=cross_width_x, sideX_gnd_gap=cross_gnd_gap_x,
                sideY_length=cross_len_y, sideY_width=cross_width_y,
                sideY_gnd_gap=min(cross_gnd_gap_y, xmon_fork_gnd_gap)
            )
            xmonCross_corrected.place(photo_reg)

            Z0.place(photo_reg)

            cell.shapes(layer_photo).insert(photo_reg)

            # # delete Xmon cross
            # shapes = cell.shapes(layer_photo)
            # for shape in shapes.each_overlapping(pya.Box(xmon_center, xmon_center)):
            #     # if shape is polygon and contains xmon_center inside
            #     # then delete this polygon
            #     if shape.is_polygon and shape.polygon.inside(xmon_center):
            #         shape.delete()


            # fine_resonance_success = True

            ## DRAWING SECTION END ##
            lv.zoom_fit()
            # fine_resonance_success = True  # break cycle if drawing only
            ### MATLAB COMMANDER SECTION START ###
            ml_terminal = SonnetLab()
            # print("starting connection...")
            from sonnetSim.cMD import CMD

            # waiting for 20 seconds or less for confirmation of matlab side ready
            # this demanded for large designs. Reason is to be determined.
            import time

            timeout = 20
            t0 = time.time()
            while True:
                try:
                    ml_terminal._send(CMD.SAY_HELLO)
                except RuntimeError as e:
                    if (time.time() - t0) > timeout:
                        time.sleep(1)
                        continue
                    else:
                        raise e
                else:
                    break

            ml_terminal.clear()
            simBox = SimulationBox(CHIP.dx, CHIP.dy, CHIP.nX, CHIP.nY)
            ml_terminal.set_boxProps(simBox)
            # print("sending cell and layer")
            from sonnetSim.pORT_TYPES import PORT_TYPES

            ports = [SonnetPort(point, PORT_TYPES.BOX_WALL) for point in [Z0.start, Z0.end]]
            ml_terminal.set_ports(ports)

            ml_terminal.send_polygons(cell, layer_photo)
            ml_terminal.set_ABS_sweep(estimated_freq - freqs_span / 2, estimated_freq + freqs_span / 2)
            print("simulating...")
            result_path = ml_terminal.start_simulation(wait=True)
            ml_terminal.release()
            ### MATLAB COMMANDER SECTION END ###

            ### RESONANCE FINDING SECTION START ###
            """
            intended to be working ONLY IF:
            s12 is monotonically increasing or decreasing over the chosen frequency band.
            That generally holds true for circuits with single resonator.
            """
            with open(result_path.decode('ascii'), "r", newline='') as file:
                # exctracting s-parameters in csv format
                # though we do not have csv module
                rows = [row.split(',') for row in list(file.readlines())[8:]]
                freqs = [float(row[0]) for row in rows]  # rows in GHz
                df = freqs[1] - freqs[0]  # frequency error
                s12_list = [float(row[3]) + 1j * float(row[4]) for row in rows]
                s12_abs_list = [abs(s12) for s12 in s12_list]
                min_freq_idx, min_s21_abs = min(enumerate(s12_abs_list), key=lambda x: x[1])
                min_freq = freqs[min_freq_idx]

            # processing the results
            if min_freq_idx == 0:
                derivative = (s12_list[1] - s12_list[0]) / df
                second_derivative = (s12_list[2] - 2 * s12_list[1] + s12_list[0]) / df ** 2
                print('resonance located the left of the current interval')
                # try adjacent interval to the left
                estimated_freq -= freqs_span
                continue
            elif min_freq_idx == (len(freqs) - 1):
                derivative = (s12_list[-1] - s12_list[-2]) / df
                second_derivative = (s12_list[-1] - 2 * s12_list[-2] + s12_list[-3]) / df ** 2
                print('resonance located the right of the current interval')
                # try adjacent interval to the right
                estimated_freq += freqs_span
                continue
            else:
                # local minimum is found
                print(f"fr = {min_freq:3.5} GHz,  fr_err = {df:.5}")
                estimated_freq = min_freq
                if freqs_span == freqs_span_corase:
                    if corase_only:
                        # terminate simulation after corase simulation
                        fine_resonance_success = True
                    else:
                        # go to fine approximation step
                        freqs_span = freqs_span_fine
                        continue
                elif freqs_span == freqs_span_fine:
                    # fine approximation ended, go to saving the result
                    fine_resonance_success = True  # breaking frequency locating cycle condition is True

            # unreachable code:
            # TODO: add approximation of the resonance if minimum is nonlocal during corase approximation
            # fr_approx = (2*derivative/second_derivative) + min_freq
            # B = -4*derivative**3/second_derivative**2
            # A = min_freq - 2*derivative**2/second_derivative
            # print(f"fr = {min_freq:3.3} GHz,  fr_err = not implemented(")
            ### RESONANCE FINDING SECTION END  ###

        ### RESULT SAVING SECTION START ###
        import shutil
        import os
        import csv

        # geometry parameters gathering
        worm_params = worm.get_geometry_params_dict(prefix="worm_")
        xmonCross_params = xmonCross.get_geometry_params_dict(prefix="xmonCross_")
        Z0_params = Z0.get_geometry_params_dict(prefix="S21Line_")
        CHIP_params = CHIP.get_geometry_params_dict(prefix="chip_")
        tail_turn_pars = {"tail_turn_angles": tail_turn_angles}

        from collections import OrderedDict

        all_params = OrderedDict(
            itertools.chain(
                worm_params.items(),
                xmonCross_params.items(),
                Z0_params.items(),
                CHIP_params.items(),
                tail_turn_pars.items(),
                {
                    "to_line, um": to_line / 1e3,
                    "filename": None,
                    "resonator_idx": resonator_idx
                }.items()
            )
        )

        project_dir = os.path.dirname(__file__)

        # creating directory with simulation results
        results_dirname = "resonator_waveguide_Q_freqs_v4_results"
        results_dirpath = os.path.join(project_dir, results_dirname)

        output_metaFile_path = os.path.join(
            results_dirpath,
            "resonator_waveguide_Q_freq_meta.csv"
        )
        try:
            # creating directory
            os.mkdir(results_dirpath)
        except FileExistsError:
            # directory already exists
            with open(output_metaFile_path, "r+", newline='') as csv_file:
                reader = csv.reader(csv_file)
                existing_entries_n = len(list(reader))
                all_params["filename"] = "result_" + str(existing_entries_n) + ".csv"

                writer = csv.writer(csv_file)
                # append new values row to file
                writer.writerow(list(all_params.values()))
        else:
            '''
                Directory did not exist and has been created sucessfully.
                So we create fresh meta-file.
                Meta-file contain simulation parameters and corresponding
                S-params filename that is located in this directory
            '''
            with open(output_metaFile_path, "w+", newline='') as csv_file:
                writer = csv.writer(csv_file)
                # create header of the file
                writer.writerow(list(all_params.keys()))
                # add first parameters row
                reader = csv.reader(csv_file)
                existing_entries_n = len(list(reader))
                all_params["filename"] = "result_1.csv"
                writer.writerow(list(all_params.values()))
        finally:
            # copy result from sonnet folder and rename it accordingly
            shutil.copy(
                result_path.decode("ascii"),
                os.path.join(results_dirpath, all_params["filename"])
            )
        ### RESULT SAVING SECTION END ###