import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from matplotlib import cm
# plt.ion()
def discriminant(beta, tauI, tauR):
    a = ( beta * tauI - 1) / (tauI + tauR)
    b = ( a + (1 / tauR))**2
    c = ( beta * tauI - 1) / (tauI * tauR)
    disc = b - 4 * c
    return disc

# ecuaciones del SIRS 
def deriv(y, t, beta, tauI,tauR):
    S, I, R = y
    dSdt = -beta * S * I + R / tauR
    dIdt =  beta * S * I - I / tauI
    dRdt =  I / tauI - R / tauR
    return dSdt, dIdt, dRdt

#funcion que calcula el punto de equilibrio pasando los parámetros
def eq_point(beta, tauI, tauR):
    s_eq = 1/(beta * tauI)
    i_eq = (beta * tauI - 1)/(beta * (tauI + tauR))
    return s_eq, i_eq

def roots(tauI, tauR):
    a = tauR
    b = 2 - 4 * ( ( 1 + (tauR / tauI) )**2 )
    c = (1 / tauR) + (4 / tauI) * ( (1+(tauR / tauI))**2)
    return (1 / (2*a))*(-b - np.sqrt(b**2 - 4*a*c))

def func(Y, t, beta, tauI, tauR):
    s, i = Y
    return [-beta*s*i + (1/tauR)*(1-s-i), beta*s*i - (1/tauI)*(i)]

########################################################
args = beta, tauI, tauR = 0.15, 14, 120
print(f'Los parametros (beta, tauI, tauR) son: {args}')
discriminante = discriminant(*args)
print(f'El valor del discriminante es: {round(discriminante, 3)}')

## odeint args
beta, gamma = beta, 1./tauI
args_odeint = (beta, 1/gamma, tauR)
########################################################
fig, ax = plt.subplots()
aux_args = {'start': 0, 'stop': 1, 'num': 15}
y1, y2 = np.linspace(**aux_args), np.linspace(**aux_args)

t = 0
Y1, Y2 = np.meshgrid(y1, y2)
u, v = np.zeros(Y1.shape), np.zeros(Y2.shape)
NI, NJ = Y1.shape

for i in range(NI):
    for j in range(NJ):
        x = Y1[i, j]
        y = Y2[i, j]
        yprime = func([x, y], t, *args)
        u[i,j] = yprime[0]
        v[i,j] = yprime[1]
     
#grafico el retrato de fase
Q = ax.quiver(Y1, Y2, u, v, color='g')

## colormap
start, stop = 0.0, 1
number_of_lines = 10
color_slice = np.linspace(start, stop, number_of_lines+1) 

colors = [ cm.summer(x) for x in color_slice ]

#grafico algunas trayectorias del sistema en particular
for i in range(10):
    N = 10000
    I0, R0 = (i*900+1)/N, 0
    
    # the rest
    S0 = 1 - I0 - R0

    y0 = S0, I0, R0
    
    t = np.linspace(0, 2000, N)

    ret = odeint(deriv, y0, t, args=args_odeint)
    S, I, R = ret.T
    ax.plot(S,I, label=f'$I_0$={I0:.2E}', color=colors[i])

#grafico el punto de equilibrio
s_eq, i_eq = eq_point(*args)
seq, ieq = round(s_eq,2), round(i_eq,2)
print(f'El punto de equilibrio es: ({seq}, {ieq})')
ax.plot(s_eq, i_eq, 'ob')

#detalles del gráfico
ax.set_xlabel('$s$ (suceptibles)', fontsize=16)
ax.set_ylabel('$i$ (infectados)', fontsize=16)
ax.set_xlim([-0.05, 1.05])
ax.set_ylim([-0.05, 1.05])
ax.legend(framealpha=1)

args_str = (r'$\beta$ = '  + f'{args[0]:.2E},  ' 
          + r'$\tau_R$ = ' + f'{args[1]:.2E},  ' 
          + r'$\tau_I$ = ' + f'{args[2]:.2E}')
y0_str = (r'$s_0=1-r_0-i_0$, '
        #  + r'$i_0$ = ' + f'{I0:.2E}, '
         + r'$r_0$ = ' + f'{R0}')
ax.set_title(y0_str + '\n' + args_str)

fig.tight_layout()
fig.savefig('../figuras/ex01-a-vector.pdf')
########################################################

N = 10000
I0, R0 = 1/N, 0

# the rest
S0 = 1 - I0 - R0
y0 = S0, I0, R0

t = np.linspace(0, 1000, 1000)

ret = odeint(deriv, y0, t, args=args_odeint)
S, I, R = ret.T

# Plot the data on three separate curves for S(t), I(t) and R(t)
# fig = plt.figure(figsize=(12,8))
fig2, ax2 = plt.subplots()
aux_args = {'alpha':0.5, 'lw':3}
ax2.plot(t, S, 'b', label='Suceptibles', **aux_args)
ax2.plot(t, I, 'r', label='Infectados', **aux_args)
ax2.plot(t, R, 'g', label='Recuperados', **aux_args)

#grafico los puntos de equilibrio 
s_eq,i_eq = eq_point(beta,tauI,tauR)
r_eq = 1 - s_eq - i_eq
ax2.plot(t, np.zeros(shape=t.shape)+s_eq, '--b', alpha=0.5)
ax2.plot(t, np.zeros(shape=t.shape)+r_eq, '--g', alpha=0.5)
ax2.plot(t, np.zeros(shape=t.shape)+i_eq, '--r', alpha=0.5)

ax2.set_xlabel('Tiempo',fontsize=18)
ax2.set_ylabel('$s,i,r$',fontsize=18)
ax2.legend(fontsize=14)

args_str = (r'$\beta$ = '  + f'{args[0]:.2E},  ' 
          + r'$\tau_R$ = ' + f'{args[1]:.2E},  ' 
          + r'$\tau_I$ = ' + f'{args[2]:.2E}')
y0_str = (r'$s_0=1-r_0-i_0$, '
         + r'$i_0$ = ' + f'{I0:.2E}, '
         + r'$r_0$ = ' + f'{R0}')
ax2.set_title(y0_str + '\n' + args_str)

fig2.tight_layout()
fig2.savefig('../figuras/ex01-a-sir.pdf')