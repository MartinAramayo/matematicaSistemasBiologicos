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
    label = f'$r$ = {r:3.2f}' 
    t, n_t = bh_map(n0, r, K, n_steps)
    ax.scatter(t, n_t, label=label, color=color, s=20)
    return t, n_t

# interactive plots
plt.ion()

fig, ax = plt.subplots()

## the array of parameters to numerically run
concat = (np.linspace(0.1, 0.9, num=6), np.asarray([1]))
array_r = np.concatenate(concat)

concat = (array_r, np.linspace(1.15, 3, num=6))
array_r = np.concatenate(concat)
array_r_size = array_r.size

## set the colormap things
number_of_lines = array_r_size
cm_subsection = np.linspace(0, 1, number_of_lines) 

colors = [ cm.gnuplot(x) for x in cm_subsection ]

## parameters
K = 0.5
N0 = 0.1
n_steps = 100

# for every parameter
for i, r in np.ndenumerate(np.flipud(array_r)): # in reverse
    indx = i[0]
    t, n_t = plot_map(ax, n0=N0, r=r, K=K, n_steps=n_steps, color=colors[indx])
    lgd = ax.legend(ncol=1, bbox_to_anchor=(1.3, 0.5), loc='center right')

# arrow that indicates increasing r
ax.arrow(40, -0.01, 0, 0.515, 
         head_width=1.75, 
         head_length=0.01, 
         fc='k', ec='k')
ax.text(36.5, 0.3, "$r$ creciente", rotation='vertical')

# annotate the r=1 case
ax.text(50, 0.105, "$r$ = 1")

# fix points
ax.text(70, 0.505, "$n^* = K$")
ax.axhline(0.5, color='k', linestyle='--')

ax.text(70, -0.025, "$n^* = 0$")
ax.axhline(0, color='k', linestyle='--')

# style
ax.set_ylabel("Población $n$")
ax.set_xlabel("Iteración")
axes_no_corner(ax)

fig.savefig('ex1.pdf',
            format='pdf',
            bbox_extra_artists=(lgd,),
            bbox_inches='tight')

# plt.close('all')