import pylab
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

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

def axes_no_corner(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

## Start

def f1D(n, r):
    """Funcion de Bevertoh-Holt"""
    return r * n

class map1D:
    def __init__(self, n0, r):
        self.current = n0
        self.r = r

    def __iter__(self):
        return self

    def __next__(self):
        n = self.current
        # agregar un numero dado por poisson
        self.r = np.random.poisson(1.7, 1)[0]
        self.current = f1D(n, self.r)
        return n

def map_data(map_object, n0, r, n_steps=20):
    """Mapeo de Beverton-Holt."""
    t = np.arange(n_steps)

    map = map_object(n0, r)
    map = iter(map)
    n_t = [next(map) for _ in range(n_steps)]
    return t, n_t

def plot_map(ax, n0, r, n_steps, color=None):
    # label = f'$r$ = {r:3.2f}' 
    t, n_t = map_data(map1D, n0, r, n_steps)
    ax.plot(t, n_t, 
            # label=label, 
            color=color)
    return t, n_t

# interactive plots
plt.ion()

fig, ax = plt.subplots()

## set the colormap things
n_steps = 80
cm_subsection = np.linspace(0, 1, n_steps) 

colors = [ cm.turbo(x) for x in cm_subsection ]

# for every parameter
for i in range(n_steps): # in reverse
    t, n_t = plot_map(ax, n0=1, r=1, n_steps=40, color=colors[i])

# style
ax.set_ylabel("Población $n$")
ax.set_xlabel("Iteración")
ax.set_yscale('log')
axes_no_corner(ax)

ax.set_xlim(right=30)

fig.savefig('ex4.pdf',
            format='pdf',
            # bbox_extra_artists=(lgd,),
            bbox_inches='tight')

# plt.close('all')