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
def bh_f(n, r, K):
    """Funcion de Bevertoh-Holt"""
    return r * n / (1 + n * ( (r - 1) / K) )

class mapBH:
    def __init__(self,n0,r,K):
        self.current = n0
        self.r = r
        self.K = K
    
    def __iter__(self):
        return self

    def __next__(self):
        n = self.current
        self.current = bh_f(n, self.r, self.K)
        return n

def bh_map(n0, r, K, n_steps=20):
    """Mapeo de Beverton-Holt."""
    t = np.arange(n_steps)
    
    map = mapBH(n0, r, K)
    map = iter(map)
    n_t = [next(map) for _ in range(n_steps)]
    return t, n_t

def plot_map(ax, n0, r, K, n_steps, color=None):
    label = f'$K$ = {K:3.2f}' 
    t, n_t = bh_map(n0, r, K, n_steps)
    ax.scatter(t, n_t, label=label, color=color, s=20)
    return t, n_t
#%%
# throw a p weighted coin
def random_p_binary_choice(p):
    return np.random.choice([0,1], size=1, p=(1-p, p))[0]

def simular(N, b, d):
    return N+1 if random_p_binary_choice(b/(b+d)) else N-1

def tiempo(t, b, d):
    aLambda = (b + d)
    return t + np.random.exponential(scale=1/aLambda, size=1)[0]

def tuple_generar(t, N, b, d, N_steps):
    counter = N_steps
    while N>0 and counter > 0 and N<1e4:
        t = tiempo(t, b, d)
        N = simular(N, b, d)
        counter -= 1
        yield (t, N)
        
b_nacer = 0.04
d_morir = 0.01
N0 = 1
N_steps = 200
#####################################################
def simulation_histogram(data, n_simulacion, names, columns):
    tuple_index = list(zip(
                           [n_simulacion] * len(data), 
                           tuple(range(len(data)))
                           )
                       )
    index = pd.MultiIndex.from_tuples(tuple_index, names=names)
    aux_df = pd.DataFrame(data, columns=columns, index=index)
    return aux_df
#####################################################
names = ["n_simulacion", "indice"]
columns = ["tiempo", "N"]

simulaciones_df = pd.DataFrame()
for n_simulacion in range(N_simulaciones:=10):
    aux = {'b': b_nacer, 'd': d_morir, 'N_steps': N_steps}
    generador = tuple_generar(0, N0,**aux)
    
    data = list(generador)
    data.insert(0, (0, N0)) # insert initial condition
      
    aux_df = simulation_histogram(data, n_simulacion, names, columns)
    simulaciones_df = simulaciones_df.append(aux_df)
#####################################################
# colormap
num_iterations = N_simulaciones
cm_subsection = np.linspace(0, 1, num_iterations) 
colors = tuple( cm.gnuplot(x) for x in cm_subsection )    
#########################################
fig, ax = plt.subplots()
num_plots = 10
iterador_plot = range(0, num_iterations, num_iterations//num_plots)
for step in iterador_plot:    
    aux_df = simulaciones_df.loc[(step,)]
    aux_args = {
        'color': colors[step],
        # 'label': f'Media de $z$ = {z.mean():0.3f}',
        'mew': 0.2,
        'mec': 'k',
        # 'ax': ax,
        # 'legend': False,
        'markersize': 4.0,
        'marker':'o'}
    ax.plot(aux_df['tiempo'],
            aux_df['N'],
            **aux_args)
    # ax.plot(aux_df, **aux_args)

# aux_args = {'label': "Sin ruido",
#             'markersize': 4.0,
#             'mew': 0.2,
#             'mec': 'k',
#             'marker':'o'}
# ax.plot(map_nonoise, **aux_args)

ax.set_yscale('symlog', base=2)
ax.autoscale()  # auto-scale
ax.legend(ncol=2)

n_steps = int(ax.get_xlim()[1])

# for K in np.linspace(1, 2**10, num=10):
    # plot_map(ax, N0, 2, K, n_steps, color=None)
    
for r in np.linspace(1.1, 10, num=10):
    plot_map(ax, N0, r, 2**10, n_steps, color=None)

plot_household(fig, ax, '../figuras/ex03-c-mapeo.pdf')