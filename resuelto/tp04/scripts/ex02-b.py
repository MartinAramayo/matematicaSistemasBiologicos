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
######################################## nuclinas y campo vectorial
b_list = np.asarray([1.9, 2.6, 2.8, 3, 5, 6, 7, 8])

subplts = [plt.subplots() for _ in range(len(b_list))]

## colormap
start, stop = 0.0, 1
number_of_lines = len(b_list)
color_slice = linspace(start, stop, number_of_lines+1) 

colors = [ cm.summer(x) for x in color_slice ]

for indx, b_val in np.ndenumerate(b_list):
    
    indx = indx[0]
    fig, ax = subplts[indx]

    aux_args = {'start': 0, 'stop': 5, 'num': 1000}
        
    p1, p2 = np.linspace(**aux_args), np.linspace(**aux_args)
    
    am = ap = 1
    bm = bp = 1
    a_val, c_val, h_val = 5, 2, 3
    parameters = (am, ap, bm, bp, a_val, b_val, c_val, h_val)
    
    z_tuple = p1, p2
    yp1, yp2 = nulclinas_switchgenetico(z_tuple, *parameters)
    
    ax.plot(yp1, p2, color=colors[-1], label=f'$b=${b_val:4.2f}')
    ax.plot(p1, yp2, color=colors[-1])
    
    aux_args = {'start': 0, 'stop': 2.8, 'num': 25}
    y1, y2 = np.linspace(**aux_args), np.linspace(**aux_args)
    Y1, Y2 = np.meshgrid(y1, y2)
    u, v = np.zeros(Y1.shape), np.zeros(Y2.shape)
    NI, NJ = Y1.shape

    for indx_i in range(NI):
        for indx_j in range(NJ):
            x = Y1[indx_i, indx_j]
            y = Y2[indx_i, indx_j]
            yprime = model_switchgenetico([x, y], *parameters)
            u[indx_i, indx_j] = yprime[0]
            v[indx_i, indx_j] = yprime[1]
      
    # #grafico el retrato de fase   
    norm = np.sqrt(u*u + v*v)
    
    # modulo en color
    s = ax.pcolor(Y1, Y2, norm, cmap='summer', shading='nearest', rasterized=True)
    cbar = fig.colorbar(s, ax=ax)
    cbar.set_label('Densidad')
    
    # direccion 
    Q = ax.quiver(Y1, Y2, u/norm, v/norm, 
                color=colors[-1], 
                scale=30, width=0.005)

    ax.set_xlabel('$p_{1}$')
    ax.set_ylabel('$p_{2}$')

    new_lim = 0, 2.8
    ax.set_xlim(new_lim)
    ax.set_ylim(new_lim)

    ax.legend()
    fig.tight_layout()

    fig.savefig(aDir + f'ex02-cosa2-{indx}.pdf')
