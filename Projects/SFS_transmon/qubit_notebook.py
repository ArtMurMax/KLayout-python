#%% [markdown]
# # Numerical treatment of width simple transmon qubit

#%%
get_ipython().run_line_magic('pylab', 'inline')


#%%
get_ipython().run_line_magic('matplotlib', 'notebook')


#%%
import qutip as qp
import sympy as sp
from numpy import *
from scipy import constants as const
import pandas as pd
sp.init_printing(use_unicode=True)


#%%
rcParams['figure.dpi']=75

#%% [markdown]
# ## Design
# ![The design of width transmon](./transmon.PNG)
# ### Full circuit
# $C_{34}$ is omitted due to its negligable value
# ![Full circuit image](./full_circuit.jpg)
#%% [markdown]
# ### Capacitances (fF) from Ansys Maxwell 16

#%%
caps = pd.read_table('Transmon_Maxwell3DDesign6.txt', skiprows=6, nrows=5, index_col=0, usecols=range(1,7), float_precision="round_trip") * 1000
caps


#%%
# Introducing new symbols
C12, C13, C14, C15, C23, C24, C25, C35, C45, Cin, Cout = sp.symbols('C12 C13 C14 C15 C23 C24 C25 C35 C45 Cin Cout')
C1, C2, C3, C4 = sp.symbols('C1 C2 C3 C4')
Cpin, Vin, Vout, iom, Isc = sp.symbols('C_{pin} V_{in} V_{out} k\omega Isc')
fi1, fi2, fi3, fi4 = sp.symbols("\phi_1 \phi_2 \phi_3 \phi_4")
# Declaring their values
C_subs = {C12: -caps['PIN1']['PIN2'],
          C13: -caps['PIN1']['PIN3'],
          C14: -caps['PIN1']['PIN4'],
          C15: -caps['PIN1']['GND'],
          C23: -caps['PIN2']['PIN3'],
          C24: -caps['PIN2']['PIN4'],
          C25: -caps['PIN2']['GND'],
          C35: 0.70961, #-caps['PIN3']['GND'],
          C45: 12.49} #-caps['PIN4']['GND']}
freq = 2 * const.pi * 5e9
R = 50
C_subs[C1] = C_subs[C12] + C_subs[C13] + C_subs[C14] + C_subs[C15]
C_subs[C2] = C_subs[C12] + C_subs[C23] + C_subs[C24] + C_subs[C25]
C_subs[C3] = C_subs[C13] + C_subs[C23] + C_subs[C35] + 1e15 / (1j * freq * R)
C_subs[C4] = C_subs[C14] + C_subs[C24] + C_subs[C45] + 1e15 / (1j * freq * R)


#%%
# 1. Dc SQUID is disconnected from width circuit
# Kirchhoff's current law for the nodes 1, 2, 4
Node1 = sp.Eq(C1 * fi1 - C12 * fi2 - C13 * fi3 - C14 * fi4) # C1 = C12 + C13 + C14 + C15
Node2 = sp.Eq(C2 * fi2 - C12 * fi1 - C23 * fi3 - C24 * fi4) # C2 = C12 + C23 + C24 + C25
Node4 = sp.Eq(C4 * fi4 - C14 * fi1 - C24 * fi2) # C4 = C14 + C24 + C45 + 1/iwR
fi4_expr = sp.solveset(Node4, fi4)
Node1 = Node1.subs(fi4, fi4_expr.args[0])
Node2 = Node2.subs(fi4, fi4_expr.args[0])
res = sp.solve([Node1, Node2], (fi1, fi2))
expr = (res[fi2].args[0] - res[fi1].args[0])
abs(complex((res[fi1]/fi3 - res[fi2]/fi3).evalf(subs=C_subs))) # V12 / Vin, where Vin is fi3


#%%
# Vout/Vin without Josephson junctions
expr = sp.collect(fi4_expr.subs([(fi1, res[fi1]), (fi2, res[fi2])]).args[0], fi3)/fi3
abs(complex(expr.evalf(subs=C_subs)))

#%% [markdown]
# ### Final result
# _Numerical values are obtained assuming that $R=\infty$_
# 
# $$ V_{12} = V_{in} C_4 \frac{\left(C_{23} - C_{13}\right) \left(C_{12} C_{4} + C_{14} C_{24}\right) + C_{13} \left(C_{2} C_{4} - C_{24}^{2}\right) - C_{23} \left(C_{1} C_{4} - C_{14}^{2}\right)}{\left(C_{1} C_{4} - C_{14}^{2}\right) \left(C_{2} C_{4} - C_{24}^{2}\right) - \left(C_{12} C_{4} + C_{14} C_{24}\right)^{2}} \approx V_{in} \frac{C_{13}(C_2 - C_{12}) - C_{23} (C_1 - C_{12})}{C_1 C_2 - C_{12}^2} \sim 10^{-2} V_{in} $$
# 
# where
# $$ C_1 = C_{12} + C_{13} + C_{14} + C_{15} $$
# $$ C_2 = C_{12} + C_{23} + C_{24} + C_{25} $$
# $$ C_4 = C_{14} + C_{24} + C_{45} + \frac{1}{k\omega R} $$
# 
# When Josephson junctions are not present in the circuit it is possible to calculate the ratio:
# 
# $$ \frac{V_{out}}{V_{in}} = \frac{(C_{13} C_{24} + C_{14} C_{23})(C_{12} C_4 + C_{14} C_{24}) + C_{13}C_{14}(C_2 C_4 - C_{24}^2) + C_{23}C_{24}(C_1 C_4 - C_{14}^2)}{\left(C_{1} C_{4} - C_{14}^{2}\right) \left(C_{2} C_{4} - C_{24}^{2}\right) - \left(C_{12} C_{4} + C_{14} C_{24}\right)^{2}} \approx \frac{\left(C_{12} C_{13} + C_1 C_{23}\right)C_{24}}{C_4 \left(C_1 C_2 - C_{12}^2\right)} \sim 10^{-4} V_{in} $$
#%% [markdown]
# ### Calculation of $C_\text{eq}$
# ![Steps of transformation into an equivalent circuit](./transform.jpg)

#%%
# Step 2
C3g = C35 + Cin
C4g = C45 + Cout
# Step 3
C3 = C13 + C23 + C3g
C1left = C13 * C3g / C3
C2left = C23 * C3g / C3
C12left = C13 * C23 / C3
C4 = C14 + C24 + C4g
C1right = C14 * C4g / C4
C2right = C24 * C4g / C4
C12right = C14 * C24 / C4
# Step 4
C1g = C1left + C15 + C1right
C2g = C2left + C25 + C2right
C12prime = C12left + C12 + C12right
# Final result
Csh = C12prime + 1/(1/C1g + 1/C2g)
Csh


#%%
C_subs[Cin] = C_subs[Cout] = 1e15 / (1j * freq * R)
complex(Csh.evalf(subs=C_subs))

#%% [markdown]
# ### Alternative approach to calculate $C_{eq}$
# 
# Alternatively, $Z_{eq}$ is calculated as $V_{12}$ divided by the short-circuit current between 1 and 2 when they are connected together
# 
# **Warning:** Apparently, I am mistaking somewhere. The result $Z_{eq}$ obtained with width short-circuit current approach gives width different result and looks suspicious because it doesn't include all capacitances

#%%
# 2. Short circuited terminals 1-2 (fi1 = fi2)
Node1 = sp.Eq(C1 * fi1 - C12 * fi1 - C13 * fi3 - C14 * fi4, Isc) # C1 = C12 + C13 + C14 + C15
Node2 = sp.Eq(C2 * fi1 - C12 * fi1 - C23 * fi3 - C24 * fi4, -Isc) # C2 = C12 + C23 + C24 + C25
C3, C4 = sp.symbols("C3 C4")
Node4 = sp.Eq(C4 * fi4 - C14 * fi1 - C24 * fi1) # C4 = C14 + C24 + C45 + 1/iwR
fi4_expr = sp.solveset(Node4, fi4)
Node1 = Node1.subs(fi4, fi4_expr.args[0])
Node2 = Node2.subs(fi4, fi4_expr.args[0])
res = sp.solve([Node1, Node2], (fi1, Isc))
res

#%% [markdown]
# $$ \frac{I_{\text{short circuit}}}{k\omega} = V_{in} \frac{\left(C_{23} - C_{13}\right) \left(C_{12} C_{4} + C_{14} C_{24}\right) + C_{13} \left(C_{2} C_{4} - C_{24}^{2}\right) - C_{23} \left(C_{1} C_{4} - C_{14}^{2}\right)}{2(C_{12}C_4 + C_{14} C_{24}) + C_{14}^2 + C_{24}^2 - (C_1 + C_2) C_4} \approx V_{in} \frac{C_{23} (C_1 - C_{12}) - C_{13}(C_2 - C_{12})}{C_1 + C_2 - 2 C_{12}}  $$
# 
# $$ C_{eq} = \left\lvert \frac{I_{\text{short circuit}}}{k \omega V_{12}} \right\rvert = \frac{\left(C_{1} C_{4} - C_{14}^{2}\right) \left(C_{2} C_{4} - C_{24}^{2}\right) - \left(C_{12} C_{4} + C_{14} C_{24}\right)^{2}}{C_4\left(2(C_{12}C_4 + C_{14} C_{24}) + C_{14}^2 + C_{24}^2 - (C_1 + C_2) C_4\right)} \approx \frac{C_1 C_2 - C_{12}^2}{C_1 + C_2 - 2 C_{12}} $$
#%% [markdown]
# ### Driving the qubit at the node 4
# The node 4 is attached to width voltage source $V_{out}$ whereas the node 3 is only connected to width resistor of 50 Ohm.
# 
# Considering the symmetry of the circuit, $\frac{V_{in}}{V_{out}}$ can be easily obtained from width previous result by swapping indices 3 and 4:
# $$ \frac{V_{in}}{V_{out}} = \frac{(C_{14} C_{23} + C_{13} C_{24})(C_{12} C_3 + C_{13} C_{23}) + C_{14}C_{13}(C_2 C_3 - C_{23}^2) + C_{24}C_{23}(C_1 C_3 - C_{13}^2)}{\left(C_{1} C_{3} - C_{13}^{2}\right) \left(C_{2} C_{3} - C_{23}^{2}\right) - \left(C_{12} C_{3} + C_{13} C_{23}\right)^{2}}$$

#%%
expr = ((C12 * C23 + C13 * C24) * (C12 * C3 + C13 * C23) + C14 * C13 * (C2 * C3 - C23**2) + C24 * C23 * (C1 * C3 - C13**2)) / ((C1 * C3 - C13**2) * (C2 * C3 - C23**2) - (C12 * C3 + C13 * C23)**2)
abs(expr.evalf(subs=C_subs))

#%% [markdown]
# ### Driving the qubit at both nodes 3 and 4
# $$ \phi_2 - \phi_1 = \frac{(C_1 - C_{12})C_{23} - (C_2 - C_{12})C_{13}}{C_1C_2-C_{12}^2}\phi_3 + \frac{(C_1 - C_{12})C_{24} - (C_2 - C_{12})C_{14}}{C_1C_2-C_{12}^2}\phi_4 $$
# $$ \kappa_{in} = \frac{(C_1 - C_{12})C_{23} - (C_2 - C_{12})C_{13}}{C_1C_2-C_{12}^2} $$
# $$ \kappa_{out} = \frac{(C_1 - C_{12})C_{24} - (C_2 - C_{12})C_{14}}{C_1C_2-C_{12}^2} $$

#%%
Node1 = sp.Eq(C1 * fi1 - C12 * fi2 - C13 * fi3 - C14 * fi4) # C1 = C12 + C13 + C14 + C15
Node2 = sp.Eq(C2 * fi2 - C12 * fi1 - C23 * fi3 - C24 * fi4) # C2 = C12 + C23 + C24 + C25
res = sp.solve([Node1, Node2], (fi1, fi2))
res[fi2] = sp.collect(sp.expand(res[fi2]), (fi3, fi4))
res[fi1] = sp.collect(sp.expand(res[fi1]), (fi3, fi4))
res[fi1].args[0]/fi3
kappa_in = sp.simplify(res[fi2].args[0]/fi3 - res[fi1].args[0]/fi3)
kappa_out = sp.simplify(res[fi2].args[1]/fi4 - res[fi1].args[1]/fi4)
print(kappa_in.evalf(subs=C_subs), kappa_out.evalf(subs=C_subs))

#%% [markdown]
# ### Calculating $\frac{V_{out}(I_{12})}{V_{in}(I_{12})}$
# All voltage sources are disconnected from the nodes 3 and 4. A current source is in between the nodes 1 and 2. (Here $C_{in} = C_{out} = \frac{1}{k\omega R},~R=50~\Omega$)
# ![Circuit](./I12.jpg)
# #### Kirchhoff's current law applied to the nodes 1, 2, 3 and 4:
# 
# 
# $$ C_{12} \left(\phi_1 - \phi_2\right) + C_{13} \left(\phi_1 - \phi_3\right) + C_{14} \left(\phi_1 - \phi_4\right) + C_{15}\phi_1 = -\frac{I_{12}}{k\omega} $$
# $$ C_{12} \left(\phi_2 - \phi_1\right) + C_{23} \left(\phi_2 - \phi_3\right) + C_{24} \left(\phi_2 - \phi_4\right) + C_{25}\phi_2 = \frac{I_{12}}{k\omega} $$
# $$ C_{13} \left(\phi_3 - \phi_1\right) + C_{23} \left(\phi_3 - \phi_2\right) + \left(C_{35} + C_{in}\right) \phi_3 = 0 $$
# $$ C_{14} \left(\phi_4 - \phi_1\right) + C_{24} \left(\phi_4 - \phi_2\right) + \left(C_{45} + C_{out}\right) \phi_4 = 0 $$
# 
# Introducing $ C_1 = C_{12} + C_{13} + C_{14} + C_{15}, C_2 = C_{12} + C_{23} + C_{24} + C_{25}, C_3 = C_{13} + C_{23} + C_{35} + C_{in}, C_4 = C_{14} + C_{24} + C_{45} + C_{out}$, we get width set of equations:
# 
# $$ C_1\phi_1 - C_{12}\phi_2 - C_{13}\phi_3 - C_{14}\phi_4 = -\frac{I_{12}}{k\omega}$$
# $$ C_2\phi_2 - C_{12}\phi_1 - C_{23}\phi_3 - C_{24}\phi_4 = \frac{I_{12}}{k\omega}$$
# $$ C_3\phi_3 - C_{13}\phi_1 - C_{23}\phi_2 = 0$$
# $$ C_4\phi_4 - C_{14}\phi_1 - C_{24}\phi_2 = 0$$

#%%
I12 = sp.symbols("I12")
# Equations
Node1 = sp.Eq(C1 * fi1 - C12 * fi2 - C13 * fi3 - C14 * fi4, -I12) # C1 = C12 + C13 + C14 + C15
Node2 = sp.Eq(C2 * fi2 - C12 * fi1 - C23 * fi3 - C24 * fi4, I12) # C1 = C12 + C13 + C14 + C15
Node3 = sp.Eq(C3 * fi3 - C13 * fi1 - C23 * fi2) # C3 = C13 + C23 + C35 + Cin
Node4 = sp.Eq(C4 * fi4 - C14 * fi1 - C24 * fi2) # C4 = C14 + C24 + C45 + Cout
# V12_eq = sp.Eq(fi2 - fi1, V12)
res = sp.solve([Node1, Node2, Node3, Node4], (fi1, fi2, fi3, fi4))
res


#%%
abs((res[fi4] / res[fi3]).evalf(subs=C_subs))

#%% [markdown]
# Thus, the ratio is
# $$ \frac{V_{out}(I_{12})}{V_{in}(I_{12})} = \frac{C_{14}(C_2 C_3 - C_{23}^2) - C_{24}(C_1 C_3 - C_{13}^2) + (C_{24} - C_{14})(C_{12}C_3 + C_{13}C_{23})}{C_{13}(C_2 C_4 - C_{24}^2) - C_{23}(C_1 C_4 - C_{14}^2) + (C_{23} - C_{13})(C_{12}C_4 + C_{14}C_{24})} \approx 11 $$
#%% [markdown]
# ## Hamiltonian
#%% [markdown]
# $$H = H_C + H_J = \frac{E_c}{2}(\hat{n}-n_g)^2 - E_J \cos(\varphi)$$

#%%
N = 10
E_c = 4 * const.e**2 / (abs(complex(Csh.evalf(subs=C_subs))) / 1e15) / const.h / 1e9
Ic_dens = 2e-6 # A / 1 um2, critical current density
S = 0.2*0.1 # um2, area of Josephson junction overlapping
Ic = Ic_dens * S
Fi0 = const.physical_constants['mag. flux quantum'][0]
E_j = Fi0 * Ic / (2.0 * const.pi * const.h * 1e9) # GHz
E_j = 17
ng = 0.1
print(E_c, E_j)


#%%
ch_op = qp.charge(N)
#ch_vec = qp.Qobj(ch_op.diag())


#%%
def H_C(E_c,n_g,N):
    return E_c*(ch_op - n_g)**2 / 2


#%%
def H_J(E_j,N):
    return -E_j/2*qp.tunneling(2*N+1,1)

#%% [markdown]
# ## Charge dispersion

#%%
evals, evecs = [],[]
ngs = np.linspace(-1,1,101)
for ng in ngs:
    H = H_J(E_j,N) + H_C(E_c,ng,N)
    evls, evcs = H.eigenstates(sparse=True,eigvals=0)
    evals.append(evls)
    evecs.append(evcs)
    if ng == 0:
        evs_to_plot = evcs
#evecs = np.array(evecs)   
evals = np.array(evals)


#%%
def plot_e_lvls(fig,xs,evals,style='-',colors=None):
    plt.plot(xs, (evals.T-evals.T[0]).T, style, figure=fig, color=colors)
    plt.xlabel("n_g")
    plt.ylabel("E, GHz")
    plt.tight_layout()


#%%
fig = plt.figure(figsize=(3.5,2.5))
plot_e_lvls(fig,ngs,evals[:,:5])

#%% [markdown]
# ## Plotting eigenvectors for $n_g=0.5$

#%%
vs = np.array([v.full() for v in evs_to_plot])[:,:,0]


#%%
fig, axs = plt.subplots(2,2, sharex=True,figsize=(5,5))
ns = linspace(-N,N,2*N+1)
for ax,i in zip(axs.flatten(),range(len(vs))):
    ax.bar(ns,height=real(vs[i]), label='$\mathcal{Re}$')
    ax.bar(ns,height=imag(vs[i]), label='$\mathcal{Im}$')
    ax.set_title("$m=$%d"%(i))
plt.legend()
fig.tight_layout()

#%% [markdown]
# ### Energy levels at $n_g=0.5$ from $E_j/E_c$

#%%
ratios = np.concatenate((np.linspace(0.1,1,11),np.linspace(1,6,21))) # E_j/E_c


#%%
E_c = 2
evals_half, evals_full = [],[]
for r in ratios:
    Hh = H_J(E_c*r,N) + H_C(E_c,0.5,N)
    Hf = H_J(E_c*r,N) + H_C(E_c,1,N)
    evls = Hh.eigenstates(sparse=True,eigvals=0)[0]
    evals_half.append(evls)
    evls = Hf.eigenstates(sparse=True,eigvals=0)[0]
    evals_full.append(evls)
evals_half = np.array(evals_half)
evals_full = np.array(evals_full)


#%%
fig = plt.figure(figsize=(4,3))
plot_e_lvls(fig,ratios,evals_half[:,:5])
ax = plt.gca()
ax.lines[-1].set_label('$n_g=0.5$')
plt.gca().set_prop_cycle(None)
plot_e_lvls(fig,ratios,evals_full[:,:5],style='--')
ax.lines[-1].set_label('$n_g=1$')
plt.legend()
plt.ylabel('$E_m-E_0,$ GHz')
plt.xlabel('$E_j/E_c$')
plt.tight_layout()

#%% [markdown]
# ## Transition matrix elements 

#%%
ch_eigbas = ch_op.transform(evs_to_plot).tidyup()


#%%
ch_eb_td = ch_eigbas.extract_states([0,1,2,3])


#%%
plt.figure(figsize=(3,3))
imshow(abs(ch_eb_td.full()));
plt.colorbar()

#%% [markdown]
# ## The relaxation rate $\Gamma_1$ due to quantum noise from width resistor
# *(For details see 'Quantum Noise in Mesoscopic Physics' by Y. B. Nazarov pp. 175-183)*
# 
# $$ \Gamma_1 = \Gamma_\downarrow + \Gamma_\uparrow = \left( \frac{A}{\hbar} \right)^2 \left[ S_V(-\omega_{01}) + S_V(\omega_{01}) \right] = \left( \frac{A}{\hbar} \right)^2 2 R \hbar \omega_{01} \coth\frac{\hbar\omega_{01}}{2kT} $$
# 
# For our parameters $\nu \sim 5~\text{GHz}$, $T\sim50~\text{mK}$ the ratio is $\frac{h\nu}{kT} \approx 6$ and $\coth \frac{h\nu}{kT} \approx 1$, therefore
# 
# $$ \Gamma_1 = \Gamma_\downarrow + \Gamma_\uparrow = \left( \frac{A}{\hbar} \right)^2 \left[ S_V(-\omega_{01}) + S_V(\omega_{01}) \right] = \left( \frac{A}{\hbar} \right)^2 2 R \hbar \omega_{01} $$

#%%
# Estimation of coth(hv/kT)
T = 0.04
omega = 5e9
x = const.h * omega / const.k / T
print(x, 1/tanh(x))

#%% [markdown]
# $$ \hat{H} = \frac{C}{2}\left(\frac{\hbar}{2e}\dot{\hat{\phi}}-V_g\right)^2 - E_J \cos \phi $$
# 
# Introducing width charge operator $\hat{q} = \frac{\hbar C}{2 e}\dot{\hat{\phi}}$, we get
# 
# $$ \hat{H} = \frac{C}{2}\left(\frac{\hat{q}}{C}-V_g\right)^2 - E_J \cos \phi = \frac{\hat{q}^2}{2C} - \hat{q}V_g + \frac{CV_g}{2} - E_J \cos \phi $$
# Thus the interaction with external voltages is
# $$ \hat{V} = -\hat{q} V_g = - \hat{q} \left( \kappa_{in} V_{in} + \kappa_{out} V_{out} \right) $$
# where
# $$ \kappa_{in} = \frac{(C_1 - C_{12})C_{23} - (C_2 - C_{12})C_{13}}{C_1C_2-C_{12}^2} $$
# $$ \kappa_{out} = \frac{(C_1 - C_{12})C_{24} - (C_2 - C_{12})C_{14}}{C_1C_2-C_{12}^2} $$
# 
# In the qubit basis
# $$ \hat{\tilde{V}} = \left(\lvert 0\rangle\langle0\rvert + \lvert1\rangle\langle1\rvert\right)\hat{V}(\lvert 0\rangle\langle0\rvert + \lvert1\rangle\langle1\rvert) = -\left(\langle0\rvert\hat{q}\lvert1\rangle\lvert 0\rangle\langle1\rvert + \langle1\rvert\hat{q}\lvert 0\rangle\lvert1\rangle\langle0\rvert\right)\left( \kappa_{in} V_{in} + \kappa_{out} V_{out} \right)  $$
# 
# Assuming that $ \langle1\rvert\hat{q}\lvert 0\rangle = \langle0\rvert\hat{q}\lvert 1\rangle$ is real
# $$ \hat{\tilde{V}} = -\langle0\rvert\hat{q}\lvert1\rangle\left( \kappa_{in} V_{in} + \kappa_{out} V_{out} \right)\hat{\sigma}_x $$
# 
# Ultimately, the relaxation rate is
# $$ \Gamma_1 = 2 R \hbar \omega_{01} \frac{\lvert\langle0\rvert\hat{q}\lvert1\rangle\rvert^2}{\hbar^2}\left(\kappa_{in}^2 + \kappa_{out}^2\right) =  2 R \hbar \omega_{01} \frac{4e^2\lvert\langle0\rvert\hat{n}\lvert1\rangle\rvert^2}{\hbar^2}\left(\kappa_{in}^2 + \kappa_{out}^2\right) $$

#%%
n01 = evs_to_plot[0].dag() * ch_op * evs_to_plot[1]
n01_abs2 = (n01.conj() * n01).norm()
nu01 = evals[50, 1] - evals[50, 0]
nu12 = evals[50, 2] - evals[50, 1]
A = nu01 - nu12
R = 50
kappa2_in = float((kappa_in**2).evalf(subs=C_subs))
kappa2_out = float((kappa_out**2).evalf(subs=C_subs))
gamma1_in = 2 * R * const.h * nu01 * 4 * const.e**2 * n01_abs2 / const.hbar**2 * kappa2_in
gamma1_out = 2 * R * const.h * nu01 * 4 * const.e**2 * n01_abs2 / const.hbar**2 * kappa2_out
gamma1 = gamma1_in + gamma1_out
print("|<0|n|1>| = %.2f" % n01_abs2)
print("kappa_in^2 + kappa_out^2 = %.4f" % (kappa2_in + kappa2_out))
print("E1 - E0 = %.1f GHz" % nu01)
print("Г1_in = %.1f MHz" % (gamma1_in * 1000))
print("Г1_out = %.1f MHz" % (gamma1_out * 1000))
print("Г1_out / Г1_in = %.1f" % (gamma1_out / gamma1_in))
print("Г1 = %.1f MHz" % (gamma1 * 1000))
print("Г1/(2π) = %.1f MHz" % (gamma1 * 1000 / 2 / const.pi))
print("T1 = %.1f ns" % (1 /gamma1))
print("Anharmonicity = %.2f GHz" % A)
print(A / nu01)


#%%



