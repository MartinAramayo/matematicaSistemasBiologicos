import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as pat

## Start
def birth_death_all_logistic(n, N, a, b, K):
    dNdt = N * ( a * N / (N+n) - b - K * (N + n) )
    return dNdt

def nuclina(n, N, a, b, K):
    return a * N - b * (N + n) - K * (N + n) **2

def birth_death_sterile_logistic(n, N, a, b, K):
    dndt = -b * n - K * (n + N) * n
    return dndt

def gradient(X, Y, args):
    U = birth_death_sterile_logistic(X, Y, *args)
    V = birth_death_all_logistic(X, Y, *args)    
    return U, V

plt.rcParams["axes.grid"] = False

fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
fig3, ax3 = plt.subplots()

a, b, K = 20, 1, 2
args = (a, b, K)

#######################################################
#######################################################
#######################################################
eq = (a-b)/K

# parameters 
x_i, x_f = xlim = [-0.005, eq/2]
y_i, y_f = ylim = [-0.01, eq*1.05]

ax2.set_xlim(xlim)
ax2.set_ylim(ylim)

Y, X = np.mgrid[y_i:y_f:100j,x_i:x_f:100j]
U, V = gradient(X, Y, args)
speed = np.sqrt(U*U + V*V)

# Varying color along a streamline
s = ax1.pcolor(X, Y, speed, shading='nearest', rasterized=True)
fig1.colorbar(s, ax=ax1)

# nuclina
ax1.contour(X, Y, nuclina(X, Y, *args), [0], 
            linewidths=1, 
            colors='white')

# punto equilibrio
ax1.axhline([0], color='white')
ax1.axvline([0], color='white')
ax1.scatter([0,0], [0, eq], color='yellow', s=25)

# Campo vectorial
Y, X = np.mgrid[y_i:y_f:15j, x_i:x_f:15j]
U, V = gradient(X, Y, args)
speed = np.sqrt(U*U + V*V)


ax1.quiver(X, Y, U/speed, V/speed, 
           angles="xy", 
           color='yellow')

# etiquetas de ejes y titulo
ax1.set_xlabel('$n$')
ax1.set_ylabel('$N$')
ax1.set_title(f'$a$ = {a:3.2f}, $b$ = {b:4.3f}, $K$ = {K:4.3f}')

fig1.tight_layout()
fig1.savefig('../figuras/campo.pdf')

#######################################################
#######################################################
#######################################################
eq = 0.00001

# parameters 
w_i = eq * 0.9
w_f = eq * 1.05
x_i, x_f = xlim = [0, 0.2]
y_i, y_f = ylim = [eq, 0.1]

ax2.set_xlim(xlim)
ax2.set_ylim(ylim)

Y, X = np.mgrid[y_i:y_f:100j,x_i:x_f:100j]
U, V = gradient(X, Y, args)
speed = np.sqrt(U*U + V*V)

# Varying color along a streamline
s = ax2.pcolor(X, Y, speed, shading='nearest', rasterized=True)
fig2.colorbar(s, ax=ax2)

# nuclina
ax2.contour(X, Y, V, [0], linewidths=1, colors='white')

# punto equilibrio
ax2.scatter([0], [eq], color='yellow')

# Campo vectorial
Y, X = np.mgrid[y_i:y_f:15j, x_i:x_f:15j]
U, V = gradient(X, Y, args)
speed = np.sqrt(U*U + V*V)

ax2.quiver(X, Y, U/speed, V/speed, 
           angles="xy", 
           color='yellow')

# etiquetas de ejes y titulo
ax2.set_xlabel('$n$')
ax2.set_ylabel('$N$')
ax2.set_title(f'$a$ = {a:3.2f}, $b$ = {b:4.3f}, $K$ = {K:4.3f}')

fig2.tight_layout()
fig2.savefig('../figuras/campo-P1.pdf')

#######################################################
#######################################################
#######################################################
eq = (a-b)/K

# parameters 
w_i = eq * 0.9
w_f = eq * 1.05
x_i, x_f = xlim = [0, 0.05]
y_i, y_f = ylim = [w_i, w_f]

ax3.set_xlim(xlim)
ax3.set_ylim(ylim)

Y, X = np.mgrid[y_i:y_f:100j,x_i:x_f:100j]
U, V = gradient(X, Y, args)
speed = np.sqrt(U*U + V*V)

# Varying color along a streamline
s = ax3.pcolor(X, Y, speed, shading='nearest', rasterized=True)
fig3.colorbar(s, ax=ax3)

# nuclina
ax3.contour(X, Y, V, [0], linewidths=1, colors='white')

# punto equilibrio
ax3.scatter([0], [eq], color='yellow')

# Campo vectorial
Y, X = np.mgrid[y_i:y_f:15j, x_i:x_f:15j]
U, V = gradient(X, Y, args)
speed = np.sqrt(U*U + V*V)

ax3.quiver(X, Y, U/speed, V/speed, 
           angles="xy", 
           color='yellow')

# etiquetas de ejes y titulo
ax3.set_xlabel('$n$')
ax3.set_ylabel('$N$')
ax3.set_title(f'$a$ = {a:3.2f}, $b$ = {b:4.3f}, $K$ = {K:4.3f}')

fig3.tight_layout()
fig3.savefig('../figuras/campo-P2.pdf')