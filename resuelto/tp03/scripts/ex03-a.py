from scipy.integrate import solve_ivp
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns 
import pylab
import pandas as pd
"""
Benchmark (hyperfine):
Benchmark #1: python ex03-a.py
  Time (mean ± σ):      3.980 s ±  0.451 s    [User: 3.124 s, System: 0.797 s]
  Range (min … max):    3.562 s …  4.719 s    10 runs
  
Hardware (neofetch):
Kernel: 5.9.16-1-MANJARO
CPU: AMD Ryzen 3 3250U with Radeon Graphics (4) @ 2.600GHz
GPU: AMD ATI 04:00.0 Picasso
Memory: 5434MiB / 13971MiB
"""
# plt.ion()
#%%
def plot_household(fig, ax, filename):
    ax.legend()
    ax.set_ylabel("$x(t)$")
    ax.set_xlabel("$t$")
    fig.tight_layout()
    fig.savefig(filename)
#%%
def aPhi(t, b, d, phi_0):
    r = 2 * b - d
    s = 2 * b
    c = s - r / phi_0
    return r / (s - c * np.exp( -r * t))

def F_1(t, b, d, phi_0):
    return - (d - 2 * b * (1 - 2 * aPhi(t, b, d, phi_0) ) )

def F_2(t, b, d, phi_0):
    phi_value = aPhi(t, b, d, phi_0) 
    return phi_value * (d + 2 * b * (1 - phi_value))

def fokker_planck(t, varianza, b , d, phi_0):
    args = t, b, d, phi_0
    return 2 * F_1(*args) * varianza + F_2(*args)
#%%       
# throw a p weighted coin
def random_p_binary_choice(p):
    return np.random.choice([0,1], size=1, p=(1-p, p))[0]

def random_gillespie(N, a_0, vacio, b):
    return (np.random.uniform(size=1)[0] * a_0 < vacio * N * b)

def simular(N, a_0, vacio, b):
    return N+1 if random_gillespie(N, a_0, vacio, b) else N-1

def tiempo(t, a_0):
    w = np.random.uniform(size=1)[0]
    tau = -np.log(w) / a_0
    return t + tau

def tuple_generar(t, b, d, phi_0, omega, N_steps):
    counter = N_steps
    
    # auxiliar variables
    b = 2 * b / omega
    N = int(phi_0 * omega + 0.5)
    vacio = omega - N
    a_0 = N * (vacio * b + d)
    yield (t, N)  # first iteration
    
    while N>0 and counter > 0 and N<1e4 or a_0==0:
        
        vacio = omega - N
        a_0 = N * (vacio * b + d)
        
        t = tiempo(t, a_0)
        N = simular(N, a_0, vacio, b)
        
        counter -= 1
        yield (t, N)
        
N_simulaciones = 500

aux = {'b':0.1,
       'd':0.05,
       'phi_0':0.2,
       'omega':100,
       'N_steps':700}
#####################################################
def simulation_histogram(data, n_simulacion, names, columns):
    size = len(data)
    iterator = zip([n_simulacion] * size, [*range(size)])
    tuple_index = [*iterator]
    index = pd.MultiIndex.from_tuples(tuple_index, names=names)
    aux_df = pd.DataFrame(data, columns=columns, index=index)
    return aux_df
#####################################################

names = ["n_simulacion", "indice"]
columns = ["tiempo", "N"]

simulaciones_df = pd.DataFrame()
for n_simulacion in range(N_simulaciones):
    iterador = tuple_generar(0,**aux)
    
    data = list(iterador)
      
    aux_df = simulation_histogram(data, n_simulacion, names, columns)
    simulaciones_df = simulaciones_df.append(aux_df)

#########################################
fig, ax = plt.subplots()
    
sns.histplot(
    simulaciones_df, 
    x='tiempo',
    y='N',
    stat='density',
    discrete=(False, True), 
    cbar=True, 
    ax=ax
)

fig.tight_layout()
ax.set_xlabel('$t$')
ax.set_ylabel('$x(t)$')
fig.savefig('../figuras/ex03-a-RENOMBRADO.pdf')

######################################### plot theorical thing
b, d, phi_0, omega, N_steps = aux.values()
t_max = simulaciones_df.tiempo.max() * 0.85
t = np.linspace(0, t_max, N_steps)

simulation = solve_ivp(
    fun=fokker_planck,
    t_span=[0, t_max],
    y0=[0],
    args = (b, d, phi_0),
    method="RK45",
    dense_output=True,
    )
varianza = simulation.sol(t)[0]
macro = omega * aPhi(t, b, d, phi_0)
fluct = np.sqrt(omega * varianza)
asint = np.full_like(fluct, fluct[-1])

aux_str = r"Ley macro $\Omega \phi (t)$"
ax.plot(t, macro,
        c="k", 
        alpha=0.7, 
        label=aux_str
)

aux_str = r"Fluctuaciones $\sqrt{\Omega \langle \xi^2 \rangle_t}$"
ax.plot(t, np.transpose([macro - fluct, macro + fluct]),
        c="k", 
        ls=":", 
        alpha=0.9, 
        label=aux_str
)

aux_str = r"Fluct. Asint. $\sqrt{\Omega \langle \xi^2 \rangle_{t \to \infty}}$"
ax.plot(t, np.transpose([macro - asint, macro + asint]),
        c="k", 
        ls="dashed", 
        alpha=0.7, 
        label=aux_str
)
handles, labels = ax.get_legend_handles_labels()
labels, ids = np.unique(labels, return_index=True)
handles = [handles[i] for i in ids]
ax.legend(handles[::-1], labels[::-1])
ax.set_xlabel(r"Tiempo $t$")
ax.set_ylabel("Población")
ax.set_title(rf"Modelo de supervivencia, $r>0$, {N_simulaciones} simulaciones")
fig.tight_layout()
fig.savefig('../figuras/ex03-a-RENOMBRADO.pdf')