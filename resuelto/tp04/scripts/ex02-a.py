import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib import cm
from numpy import linspace

def g_rep(x, a, b, c, h):
    return a / (b + c * (x**h)) 

def g_rep_der(x, a, b, c, h):
    return ( -a * h * c * (x**(h-1)) ) / ( b + c * (x**(h)) )**2

def model_goodwin(y, t, am, ae, ap, bm, be, bp, a, b, c, h):
    m, e, p = y
    g = g_rep(p, a, b, c, h)
    dmdt = am * g - bm * m
    dedt = ae * m - be * e
    dpdt = ap * e - bp * p
    return dmdt, dedt, dpdt

def model_switchgenetico(y, am, ap, bm, bp, a, b, c, h):
    p1,p2 = y
    gp1, gp2 = g_rep(p1, a, b, c, h), g_rep(p2, a, b, c, h)
    dp1dt = ap * (am/bm) * gp2 - bp * p1
    dp2dt = ap * (am/bm) * gp1 - bp * p2
    return [dp1dt, dp2dt]

def nulclinas_switchgenetico(y, am, ap, bm, bp, a, b, c, h):
    p1, p2 = y
    gp1, gp2 = g_rep(p1,a,b,c,h), g_rep(p2,a,b,c,h)
    yp1 = ( ap * am / (bp * bm)) * gp2
    yp2 = ( ap * am / (bp * bm)) * gp1
    return yp1, yp2

aDir = '../figuras/'
######################################## nuclinas
fig, ax = plt.subplots()
b_list = np.linspace(1, 15, 6)

## colormap
start, stop = 0.0, 1
number_of_lines = len(b_list)
color_slice = linspace(start, stop, number_of_lines+1) 

colors = [ cm.summer(x) for x in color_slice ]

for indx, b_val in np.ndenumerate(b_list):

    indx = indx[0]
    aux_args = {'start': 0, 'stop': 5, 'num': 1000}
        
    p1, p2 = np.linspace(**aux_args), np.linspace(**aux_args)
    
    am = ap = 1
    bm = bp = 1
    a_val, c_val, h_val = 5, 2, 3
    parameters = (am, ap, bm, bp, a_val, b_val, c_val, h_val)
    
    z_tuple = p1, p2
    yp1, yp2 = nulclinas_switchgenetico(z_tuple, *parameters)
    
    ax.plot(yp1, p2, color=colors[indx], label=f'$b=${b_val:4.2f}')
    ax.plot(p1, yp2, color=colors[indx])

new_xlim = [0, ax.get_xlim()[1] * 0.9]
ax.plot(new_xlim, new_xlim, '--k')

ax.set_xlabel('$p_{1}$')
ax.set_ylabel('$p_{2}$')

new_lim = 0, 2
new_lim = 0.25, 1.75
ax.set_xlim(new_lim)
ax.set_ylim(new_lim)

ax.legend()
fig.tight_layout()
fig.savefig(aDir + f'ex02-cosa1-{2}.pdf')