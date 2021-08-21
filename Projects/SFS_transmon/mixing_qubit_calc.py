#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import qutip as qp
from scipy import constants as const
from scipy import optimize as opt

#%%
N = 10
ch_op = qp.charge(N)
E_j = 17
ng = 0
def H_C(E_c,n_g,N):
    return E_c*(ch_op - n_g)**2 / 2
def H_J(E_j,N):
    return -E_j/2*qp.tunneling(2*N+1, 1)

#%%
def calc_params(filename):
    eps = (11.45 + 1)/2
    # shunted capacitance
    caps = pd.read_table(filename, skiprows=5, nrows=4, index_col=0,
                        usecols=range(1,6), float_precision="round_trip") * -1000 * eps
    C12 = caps["PIN1"]["PIN2"]
    Ccpw = caps["PIN1"]["CPW"] * caps["PIN2"]["CPW"] / (caps["PIN1"]["CPW"] + caps["PIN2"]["CPW"])
    Cgnd = caps["PIN1"]["GND"] * caps["PIN2"]["GND"] / (caps["PIN1"]["GND"] + caps["PIN2"]["GND"])
    Csh = C12 + Ccpw + Cgnd
    # charging energy
    E_c = 4 * const.e**2 / (Csh / 1e15) / const.h / 1e9
    # kappa
    C1 = C12 + caps["PIN1"]["GND"] + caps["PIN1"]["CPW"]
    C2 = C12 + caps["PIN2"]["GND"] + caps["PIN2"]["CPW"]
    kappa_denom = C1 * C2 - C12**2
    kappa1_num = C12 * caps["PIN1"]["CPW"] + C1 * caps["PIN2"]["CPW"]
    kappa2_num = C12 * caps["PIN2"]["CPW"] + C2 * caps["PIN1"]["CPW"]
    kappa = (kappa1_num + kappa2_num) / kappa_denom
    # eigenvalues and eigenvectors
    H = H_J(E_j,N) + H_C(E_c,ng,N)
    evls, evcs = H.eigenstates(sparse=True,eigvals=0)
    # Matrix element
    n01 = evcs[0].dag() * ch_op * evcs[1]
    n01_abs2 = (n01.conj() * n01).norm()
    # Qubit frequency
    nu01 = evls[1] - evls[0]
    # Anharmoniticity
    nu12 = evls[2] - evls[1]
    A = (nu01 - nu12) * 1000
    # Gamma 1
    R = 50
    gamma1 = 2 * R * const.h * nu01 * 4 * const.e**2 * n01_abs2 / const.hbar**2 * kappa**2
    T = 1 / gamma1
    gamma1 = gamma1 * 1000 / 2 / const.pi
    return Csh, E_c, kappa, n01_abs2, nu01, A, gamma1, T

#%%
def print_one(filename):
    Csh, E_c, kappa, n01_abs2, nu01, A, gamma1, T = calc_params(filename)
    data = {"Csh, fF": Csh,
            "E_c, GHz": E_c,
            "kappa": kappa,
            "E01, GHz": nu01,
            "Anharm., MHz": A,
            "Г1/(2π), MHz": gamma1,
            "T1, ns": T
            }
    return pd.DataFrame(data=data, index=[0])

#%%
fun = lambda x, a, b, c: a / x**b + c
popt, pconv = opt.curve_fit(fun, to_line_params, G1_arr)
xdata = np.linspace(25, 80)
ydata = fun(xdata, popt[0], popt[1], popt[2])
plt.figure(1)
plt.plot(to_line_params, G1_arr)
plt.plot(xdata, ydata)
plt.xlabel("to line, um")
plt.ylabel("Г1/(2π), MHz")
plt.grid(True)
plt.show()
print(popt)
y = 3.191396059815843
x = (popt[0] / (y - popt[2]))**(1/popt[1])
print(x)

#%%
plt.figure(1)
plt.plot(to_line_params, nu01_arr)
plt.xlabel("to line, um")
plt.ylabel("E01, GHz")
plt.grid(True)
plt.show()

#%%
# sizesy = [600, 1000, 1500, 2000, 3000]
sizesy = ["1000_600", "2000_600", "2000_1000", "2000_1500", "2000_2000", "2000_3000", "2000_3000_equal", "2000_3000_equal_tiny", "2000_2000_real"]
eps = (11.45 + 1)/2
C1c = []
C2c = []
C1g = []
C2g = []
C12 = []
for size in sizesy:
    filename = '~/Documents/Simulations/Simulations_new/MixingQubit_%s.txt' % size
    caps = pd.read_table(filename, skiprows=5, nrows=4, index_col=0,
                    usecols=range(1,6), float_precision="round_trip") * -1000 * eps
    C1c.append(caps["PIN1"]["CPW"])
    C2c.append(caps["PIN2"]["CPW"])
    C1g.append(caps["PIN1"]["GND"])
    C2g.append(caps["PIN2"]["GND"])
    C12.append(caps["PIN1"]["PIN2"])
data = {"size": sizesy,
        "C1_CPW": C1c,
        "C2_CPW": C2c,
        "C1_GND": C1g,
        "C2_GND": C2g,
        "C12": C12,
        }
pd.DataFrame(data=data)

#%%
Csh_arr = []
E_c_arr = []
kappa_arr = []
matr_arr = []
nu01_arr = []
A_arr = []
G1_arr = []
T_arr = []
for size in sizesy:
    filename = '~/Documents/Simulations/Simulations_new/MixingQubit_%s.txt' % size
    Csh, E_c, kappa, n01_abs2, nu01, A, gamma1, T = calc_params(filename)
    Csh_arr.append(Csh)
    E_c_arr.append(E_c)
    kappa_arr.append(kappa)
    matr_arr.append(n01_abs2)
    A_arr.append(A)
    nu01_arr.append(nu01)
    G1_arr.append(gamma1)
    T_arr.append(T)

data = {"size": sizesy,
        "Csh, fF": Csh_arr,
        "E_c, GHz": E_c_arr,
        "kappa": kappa_arr,
        "E01, GHz": nu01_arr,
        "Anharm., MHz": A_arr,
        "Г1/(2π), MHz": G1_arr,
        "T1, ns": T_arr
        }
pd.DataFrame(data=data)

#%%
# fun = lambda x, width, b, c: width / x**b + c
# popt, pconv = opt.curve_fit(fun, to_line_params, G1_arr)
# xdata = np.linspace(25, 80)
# ydata = fun(xdata, popt[0], popt[1], popt[2])
plt.figure(1)
plt.plot(sizesy, G1_arr)
# plt.plot(xdata, ydata)
plt.xlabel("chip_y, um")
plt.ylabel("Г1/(2π), MHz")
plt.grid(True)
plt.show()
# print(popt)
# y = 3.191396059815843
# x = (popt[0] / (y - popt[2]))**(1/popt[1])
# print(x)

#%%
plt.figure(1)
plt.plot(sizesy, nu01_arr)
plt.xlabel("chip_y, um")
plt.ylabel("E01, GHz")
plt.grid(True)
plt.show()


#%%
toline = [25, 30, 344, 35, 351, 40, 45]
eps = (11.45 + 1)/2
C1c = []
C2c = []
C1g = []
C2g = []
C12 = []
for size in toline:
    filename = '~/Documents/Simulations/Simulations_new/MixingQubit_2000_2000_toline_%d.txt' % size
    caps = pd.read_table(filename, skiprows=5, nrows=4, index_col=0,
                    usecols=range(1,6), float_precision="round_trip") * -1000 * eps
    C1c.append(caps["PIN1"]["CPW"])
    C2c.append(caps["PIN2"]["CPW"])
    C1g.append(caps["PIN1"]["GND"])
    C2g.append(caps["PIN2"]["GND"])
    C12.append(caps["PIN1"]["PIN2"])
data = {"toline": toline,
        "C1_CPW": C1c,
        "C2_CPW": C2c,
        "C1_GND": C1g,
        "C2_GND": C2g,
        "C12": C12,
        }
pd.DataFrame(data=data)

#%%
Csh_arr = []
E_c_arr = []
kappa_arr = []
matr_arr = []
nu01_arr = []
A_arr = []
G1_arr = []
T_arr = []
for size in toline:
    filename = '~/Documents/Simulations/Simulations_new/MixingQubit_2000_2000_toline_%d.txt' % size
    Csh, E_c, kappa, n01_abs2, nu01, A, gamma1, T = calc_params(filename)
    Csh_arr.append(Csh)
    E_c_arr.append(E_c)
    kappa_arr.append(kappa)
    matr_arr.append(n01_abs2)
    A_arr.append(A)
    nu01_arr.append(nu01)
    G1_arr.append(gamma1)
    T_arr.append(T)

data = {"to line, um": toline,
        "Csh, fF": Csh_arr,
        "E_c, GHz": E_c_arr,
        "kappa": kappa_arr,
        "E01, GHz": nu01_arr,
        "Anharm., MHz": A_arr,
        "Г1/(2π), MHz": G1_arr,
        "T1, ns": T_arr
        }
pd.DataFrame(data=data)

#%%
fun = lambda x, a, b, c: a / x**b + c
popt, pconv = opt.curve_fit(fun, toline, G1_arr)
xdata = np.linspace(25, 80)
ydata = fun(xdata, popt[0], popt[1], popt[2])
plt.figure(1)
plt.plot(toline, G1_arr)
plt.plot(xdata, ydata)
plt.xlabel("to line, um")
plt.ylabel("Г1/(2π), MHz")
plt.grid(True)
plt.show()
print(popt)
y = 3.2831954039093043
x = (popt[0] / (y - popt[2]))**(1/popt[1])
print(x)

#%%
