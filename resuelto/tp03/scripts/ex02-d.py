import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns 
import pylab
import pandas as pd
"""
Benchmark (hyperfine):
Benchmark #1: python ex02-b.py
  Time (mean ± σ):     17.988 s ±  0.861 s    [User: 14.772 s, System: 3.048 s]
  Range (min … max):   16.978 s … 19.299 s    10 runs
  
Hardware (neofetch):
Kernel: 5.9.16-1-MANJARO
CPU: AMD Ryzen 3 3250U with Radeon Graphics (4) @ 2.600GHz
GPU: AMD ATI 04:00.0 Picasso
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
## parametros
x0 = 1
N_steps = 50
sigma = 0.2
a = 1.05

all_maps = {}
for t in range(N_simulaciones:=4000):
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

def simulation_histogram(values, num_steps, columns):
    aux_array = np.asarray(list(zip(list(range(num_steps)), values)))
    aux_df = pd.DataFrame(aux_array, columns=columns)
    return aux_df

histo_df = pd.DataFrame()
for step in range(N_simulaciones):
    values, _ = all_maps[step]
    columns = ['t', 'x']
    aux_df = simulation_histogram(values, N_steps, columns)
    histo_df = pd.concat( (histo_df, aux_df) )

H, xedges, yedges = np.histogram2d(histo_df.t.values, 
                                   histo_df.x.values,
                                   bins=(50,200),
                                   density=True)
H = H.T

im = ax.imshow(
    H, interpolation='nearest', origin='lower',
    extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
    cmap='gnuplot'
)

cbar = fig.colorbar(im) 
cbar.set_label('Densidad')

def forceAspect(ax,aspect):
    im = ax.get_images()
    extent =  im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)

forceAspect(ax,aspect=1.0)

ax.set_xlabel('$t$')
ax.set_ylabel('$x(t)$')
fig.tight_layout()
fig.savefig('../figuras/ex02-histograma-02.pdf')
