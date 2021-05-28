import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib import cm
from numpy import linspace

def g_rep(x,a,b,c,h):
    y = (a / (b + c * (x**h)) )
    return y

def model_goodwin(y, t, am, ae, ap, bm, be, bp, a, b, c, h):
    m,e,p = y
    dmdt = am * g_rep(p,a,b,c,h) - bm * m
    dedt = ae * m - be * e
    dpdt = ap * e - bp * p
    return dmdt, dedt, dpdt

#######################################################
fig = plt.figure(figsize=(6,6))
gs = fig.add_gridspec(3, 1, hspace=0, wspace=0)
ax = gs.subplots(sharex='col')

t = np.linspace(0,500,10000)

# parametros
y0 = 0.1,0,0
am = ae = ap = 1
bm = be = bp = 0.8
a = b = c = 1
param_args = (am, ae, ap, bm, be, bp, a, b, c)

h_array = np.linspace(10, 10, 1)

## colormap
start = 0.0
stop = 1.0
number_of_lines = len(h_array)
cm_subsection = linspace(start, stop, number_of_lines + 1) 

colors1 = [ cm.summer(x) for x in cm_subsection ]
colors2 = [ cm.autumn(x) for x in cm_subsection ]
colors3 = [ cm.winter(x) for x in cm_subsection ]

for i, h in np.ndenumerate(h_array):
    i = i[0]
    sol = odeint(model_goodwin, y0, t, args=(*param_args, h))
    m, e, p = sol.T 
    ax[0].plot(t,m,label=f'$h=${h:3.1f}', color=colors1[i])
    ax[1].plot(t,e,label=f'$h=${h:3.1f}', color=colors2[i])
    ax[2].plot(t,p,label=f'$h=${h:3.1f}', color=colors3[i])

ax[0].set_ylabel('$m$')
ax[1].set_ylabel('$e$')
ax[2].set_ylabel('$p$')

ax[2].set_xlabel('$t$')

ax[0].tick_params(bottom=False)
ax[1].tick_params(bottom=False)

for i in range(len(ax)):
    ax[i].legend(ncol=4, fontsize=9)

fig.tight_layout()
fig.savefig('../figuras/ex01-concentracion-h-osc-kill.pdf')