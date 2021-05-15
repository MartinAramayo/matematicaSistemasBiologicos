import pylab
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from jitcdde import jitcdde, y, t
from scipy.integrate import odeint
from math import pi
# pip3 install jitcdde 

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
def logistical(y, t, r, K):
    dydt = r * y * (1 - y / K) 
    return dydt

def sterile_logistical(N, t, a, b, n, K):
    dNdt = N * ( a * N / (N+n) - b - K * (N + n) )
    return dNdt

# interactive plots
# plt.ion()

## simulacion
# parametros de la ecuacion logistica
a = 0.4
b = 0.1
K = 1
t = np.linspace(0, 500, num=500)
aux_kargs = {
    'func': sterile_logistical,
    'y0': (N0:=K - 0.5),
    't': t,
    'args': (a, b, 0, K)
}

sol_all = odeint(**aux_kargs).flatten()

fig, ax = plt.subplots()
ax.plot(t,sol_all)
pseudo_capacidad = sol_all[-1]

ax.axhline(pseudo_capacidad/1)
ax.text(100, pseudo_capacidad/1 + 0.01, f'y={pseudo_capacidad/1:.3E}')

ax.axhline(pseudo_capacidad/4)
ax.text(-25, pseudo_capacidad/4 - 0.02, f'y={pseudo_capacidad/4:.3E}')

## the array of parameters to numerically run
# array_n = np.linspace(0, N0/9, num=20)
array_n = np.geomspace(0.06, N0/9, 20, endpoint=True)

## set the colormap things
number_of_lines = array_n.size
cm_subsection = np.linspace(0, 1, number_of_lines) 

colors = tuple( cm.gnuplot(x) for x in cm_subsection )

for j, n in np.ndenumerate(array_n):
            
    # primer iteracion
    aux_kargs.update({
        'args': (a, b, n, K)
    })

    sol_all = odeint(**aux_kargs).flatten()

    # my_label = f'$n$ = {n:3.2f}'
    my_label = f'$n$ = {n:.2E}'
    ax.plot(t, sol_all, label=my_label, color=colors[j[0]])


# style
lgd = ax.legend(ncol=1, 
                bbox_to_anchor=(1., 0.5), 
                loc='center left')
ax.set_ylabel("Población $N$")
ax.set_xlabel("tiempo")
# ax.set_title(f'$a$ = {a:3.2f}, $b$ = {b:4.3f}, $n$ = {n:4.3f}, $K$ = {K:4.3f}')
ax.set_title(f'$a$ = {a:3.2f}, $b$ = {b:4.3f}, $K$ = {K:4.3f}')
# ax.set_title(f'$a$ = {a:3.2f}, $b$ = {b:4.3f}, $n$ = {n:.3E}, $K$ = {K:4.3f}')
axes_no_corner(ax)

fig.savefig('../figuras/ex5' + '.pdf',
            format='pdf',
            bbox_extra_artists=(lgd,),
            bbox_inches='tight')

# plt.close('all')