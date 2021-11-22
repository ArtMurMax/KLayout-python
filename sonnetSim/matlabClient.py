import socket
import struct
import numpy as np

from .cMD import CMD
from .flags import FLAG, RESPONSE


class MatlabClient():
    MATLAB_PORT = 30000
    TIMEOUT = 10

    # internal state enumeration class
    class STATE:
        INITIALIZING = 0
        READY = 1
        BUSY = 2
        ERROR = 3
        BUSY_SIMULATING = 4
        SIMULATION_FINISHED = 5

    def __init__(self, host="localhost", port=MATLAB_PORT):
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.timeout = MatlabClient.TIMEOUT
        self.sock.settimeout(self.timeout)
        self.address = (host, port)
        self.state = self.STATE.INITIALIZING

        try:
            self.sock.connect(self.address)
            self.state = self.STATE.READY
        except ConnectionRefusedError as e:
            print("connection refused: ", e)
            self.state = self.STATE.ERROR

    def _send(self, byte_arr, confirmation_value=RESPONSE.OK):
        confirm_byte = None
        self.sock.sendall(byte_arr)

        # waiting for 2 confirmation bytes received or timeout expired
        try:
            while (True):
                confirm_byte = self.sock.recv(2, socket.MSG_PEEK)
                if (len(confirm_byte) == 2):
                    confirm_byte = self.sock.recv(2)
                    confirm_val = struct.unpack("!H", confirm_byte)[0]
                    if (confirm_val == confirmation_value):
                        return True
                    else:
                        self.state = self.STATE.ERROR
                        return False
        except Exception as e:
            print("exception on reception of confirm byte, following exception:")
            print(e)
            raise e

    def _close(self):
        self._send(CMD.CLOSE_CONNECTION)
        self.sock.close()

    def _send_float64(self, val):
        raw_data = struct.pack(">d", np.float64(val))
        self._send(raw_data)

    def _send_array_float64(self, array):
        raw_data = struct.pack(">{0}d".format(len(array)), *array)
        self._send_uint32(len(array))
        self._send(raw_data)

    def _send_uint32(self, val):
        raw_data = struct.pack("!I", np.uint32(val))
        self._send(raw_data)

    def _send_array_uint16(self, array):
        raw_data = struct.pack("!{0}H".format(len(array)), *array)
        self._send_uint32(len(array))
        self._send(raw_data)

    def _send_array_uint32(self, array):
        raw_data = struct.pack("!{0}I".format(len(array)), *array)
        self._send_uint32(len(array))
        self._send(raw_data)

    def read_line(self):
        self.sock.settimeout(None)  # entering nonblocking mode
        while (True):
            data = self.sock.recv(1024, socket.MSG_PEEK)
            idx = data.find(b'\n')
            if (idx != - 1):
                self.sock.settimeout(MatlabClient.TIMEOUT)  # leaving nonblocking mode
                data = self.sock.recv(idx + 1)[:-1]
                break
            else:
                continue

        return data

    def _send_polygon(self, array_x, array_y, port_edges_numbers_list=None, port_edges_types=None):
        self._send(CMD.POLYGON)
        # print(port_edges_numbers_list, port_edges_types)
        if (port_edges_numbers_list is None) or (len(port_edges_numbers_list) == 0):
            self._send(FLAG.FALSE)
        else:
            self._send(FLAG.TRUE)
            self._send_array_uint32(port_edges_numbers_list)
            self._send_array_uint16(port_edges_types)

        self._send_array_float64(array_x)
        self._send_array_float64(array_y)

    def _set_boxProps(self, dim_X_um, dim_Y_um, cells_X_num, cells_Y_num):
        self._send(CMD.BOX_PROPS)
        self._send_float64(dim_X_um)
        self._send_float64(dim_Y_um)
        self._send_uint32(cells_X_num)
        self._send_uint32(cells_Y_num)

    def _set_ABS_sweep(self, start_f, stop_f):
        self._send(CMD.SET_ABS)
        self._send_float64(start_f)
        self._send_float64(stop_f)

    def _set_linspace_sweep(self, start_f, stop_f, points_n):
        self._send(CMD.SET_LINSPACE_SWEEP)
        self._send_float64(start_f)
        self._send_float64(stop_f)
        self._send_uint32(points_n)

    def _send_simulate(self):
        self._send(CMD.SIMULATE, confirmation_value=RESPONSE.START_SIMULATION)
        self.state = self.STATE.BUSY_SIMULATING

    def _get_simulation_status(self):
        if (self.state == self.STATE.BUSY_SIMULATING):
            self.sock.settimeout(None)  # entering nonblocking mode
            try:
                response = self.sock.recv(2, socket.MSG_PEEK)
            except Exception as e:
                print(e)

            if (len(response) < 2):
                # leaving nonblocking mode on return
                self.sock.settimeout(MatlabClient.TIMEOUT)
                return self.state  # equals to self.STATE.BUSY_SIMULATING

            self.sock.settimeout(MatlabClient.TIMEOUT)  # leaving nonblocking mode

            response = self.sock.recv(2)  # transferring data from socket input QUEUE
            response = struct.unpack("!H", response)[0]
            if (response == RESPONSE.SIMULATION_FINISHED):
                self.state = self.STATE.SIMULATION_FINISHED
            else:
                self.state = self.STATE.ERROR

            return self.state

    def _visualize_sever(self):
        self._send(CMD.VISUALIZE)

    def _clear(self):
        self._send(CMD.CLEAR_POLYGONS)
