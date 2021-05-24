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
GPU: AMD ATI 04:00.0 Picasso
Memory: 5434MiB / 13971MiB
"""

# plt.ion()
#%%
## config
# Set figure size
SMALL_SIZE = int( 8 * 1.5)
MEDIUM_SIZE = int(10 * 1.5)
BIGGER_SIZE = int(12 * 1.5)

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

params = {
    "font.family": "serif",
    "text.usetex": True,
    "pgf.rcfonts": False,
}
pylab.rcParams.update(params)

#%%
def axes_no_corner(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

def plot_household(fig, ax, filename):
    ax.legend()
    axes_no_corner(ax)
    ax.set_ylabel("$x(t)$")
    ax.set_xlabel("$t$")
    fig.tight_layout()
    fig.savefig(filename)
#%%
def sum_mapeo(array, cte, paso):
    return sum(cte**( paso - (i+1)) * array[i] for i in range(paso))

def pot_mapeo(cte, x0, paso):
    return cte**(paso - 1) * x0

x0 = 1
N_steps = 50
sigma = 0.2
a = 1.05

all_maps = {}
for t in range(N_simulaciones:=10):
    z = np.random.normal(loc=0, scale=sigma, size=N_steps)
    map1d = [sum_mapeo(z, a, n) + pot_mapeo(a, x0, n) for n in range(1,N_steps)]
    map1d.insert(0,x0)
    all_maps.update({t: (map1d, z)})

map_nonoise = [pot_mapeo(a, x0, n) for n in range(1, N_steps)]
map_nonoise.insert(0,x0)

# # Compute solution if the map does not have a compact form
# x[0] = x0
# for n in range(1, N+1): # (1,N)
#     x[n] = a * x[n-1] 
    
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
plot_household(fig, ax, '../figuras/ex02-mapeo.pdf')