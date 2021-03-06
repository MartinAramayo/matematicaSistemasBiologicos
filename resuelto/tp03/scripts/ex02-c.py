import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns 
import pylab
import pandas as pd
"""
Benchmark (hyperfine):
Benchmark #1: python ex02-a.py
  Time (mean ± σ):      3.980 s ±  0.451 s    [User: 3.124 s, System: 0.797 s]
  Range (min … max):    3.562 s …  4.719 s    10 runs
  
Hardware (neofetch):
Kernel: 5.9.16-1-MANJARO
CPU: AMD Ryzen 3 3250U with Radeon Graphics (4) @ 2.600GHz
GPU: AMD ATI 04:00.0 Picassoq
Memory: 5434MiB / 13971MiB
"""
#%%
def plot_household(fig, ax, filename):
    ax.legend()
    ax.set_ylabel("$x(t)$")
    ax.set_xlabel("$t$")
    fig.tight_layout()
    fig.savefig(filename)
#%%
def prod_mapeo(array, paso):
    return np.prod( tuple( array[i] for i in range(paso) ) )

def pot_mapeo(cte, x0, paso):
    return cte**(paso) * x0
#%%
# plt.ion()
x0 = 1
N_steps = 50
sigma = 0.2
a = 1.05

all_maps = {}
for t in range(N_simulaciones:=10):
    z = np.random.normal(loc=0, scale=sigma, size=N_steps)
    z += a
    map1d = [prod_mapeo(z, n) for n in range(1,N_steps)]
    map1d.insert(0,x0)
    all_maps.update({t: (map1d, z)})

map_nonoise = [pot_mapeo(a, x0, n) for n in range(1, N_steps)]
map_nonoise.insert(0,x0)   
####################################################################
# colormap
num_iterations = N_simulaciones
cm_subsection = np.linspace(0, 1, num_iterations) 
colors = tuple( cm.gnuplot(x) for x in cm_subsection )    
#########################################
fig, ax = plt.subplots()
num_plots = N_simulaciones
iterador_plot = range(0, num_iterations, num_iterations//num_plots)
for step in iterador_plot:    
    array, z = all_maps[step]
    start, end, binwidth = 0, max(array), 5
    aux_args = {'label': f'Media de $z$ = {z.mean():0.3f}',
                'color': colors[step],
                'mew': 0.2,
                'mec': 'k',
                'markersize': 4.0,
                'marker':'o'}
    ax.plot(array, **aux_args)

aux_args = {'label': "Sin ruido",
            'markersize': 4.0,
            'mew': 0.2,
            'mec': 'k',
            'marker':'o'}
ax.plot(map_nonoise, **aux_args)
plot_household(fig, ax, '../figuras/ex02-mapeo-02.pdf')