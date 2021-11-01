import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import pandas as pd
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
import datetime
from numpy import cosh

def sech_sir(x, a, b, c):
    return a*( cosh( b * x + c )**(-2) )

# plt.ion()
aDir = '../figuras/'

#%% # abrir y pre procesar
df = pd.read_csv('2021-tp05-covid19.csv', index_col='Date')
rename_dict = {'i':'infectados',
               'd':'fallecidos'}
df = df.rename(columns=rename_dict)
df.index = pd.to_datetime(df.index)
df.index = df.index.rename('Fecha')

#%% ## feriados
file1 = 'datasetFeriadosArgentina/feriadosArgentina2020.csv'
file2 = 'datasetFeriadosArgentina/feriadosArgentina2021.csv'
feriados_20_21 = (pd.read_csv(file1), pd.read_csv(file2))
feriados = pd.concat(feriados_20_21).reset_index()
month, day, year = feriados['month'], feriados['day'], feriados['year']
month, day, year = month.astype('str'), day.astype('str'), year.astype('str') 
feriados = month + ' ' + day + ' ' + year
feriados = pd.to_datetime(feriados, format='%B %d %Y')

## para plotear
rename_plot = {'infectados':'Infectados',
               'fallecidos':'Fallecidos',
               'tests': 'Testeados',
               'difIt': 'Infectados-Fallecidos',}
style = {'linewidth':0.5, 
         'ms':1.8, 
         'marker':'o', 
         'subplots':True}
plt_rename = {'columns':rename_plot}

#%% # diferencia entre infectados y testeados
axs = df.rename(**plt_rename).plot(**style)
figs = axs[0].get_figure()
figs.tight_layout()
figs.savefig(aDir + 'ex02-resumen.pdf')

#%% ## sin weekends
df_weekday_2 = df[(df.index.weekday != 5) # sabado 
                & (df.index.weekday != 6) # domingo
                # & (df.index.weekday != 4) # viernes
                # & (df.index.weekday != 3) # jueves
                # & (df.index.weekday != 2) # miercoles
                # & (df.index.weekday != 1) # martes
                # & (df.index.weekday != 0) # lunes
                ] 
df_weekday_2 = df_weekday_2[~df_weekday_2.index.isin(feriados)]

axs3 = df_weekday_2.rename(**plt_rename).plot(**style)
figs3 = axs3[0].get_figure()
figs3.tight_layout()
figs3.savefig(aDir + 'ex02-sin-Finde.pdf')

#%% # Primer pico
pico_1 = datetime.datetime.strptime('2020-12-15','%Y-%m-%d')
df_pico_11 = df[df.index < pico_1]
df_pico_11.index = (df_pico_11.index - df_pico_11.index[0]).days
axs4 = df_pico_11.rename(**plt_rename).plot(**style)
figs4 = axs4[0].get_figure()
figs4.tight_layout()
figs4.savefig(aDir + 'ex02-pico1.pdf')

aux_args = {'f':sech_sir, 
            'xdata':df_pico_11.index, 
            'ydata':df_pico_11['infectados'],
            'p0':(5000, 0.01, -1)}

popt, pcov = curve_fit(**aux_args)
a, b, c = popt

x = df_pico_11.index.values
y = sech_sir(x, a ,b ,c)

fig_qq, ax_qq = plt.subplots()
ax_qq.scatter(y, df_pico_11['infectados'], color='royalblue')
ylim = (min(y), max(y))
ax_qq.plot(ylim, ylim, '--k')
ax_qq.set_xlabel('Predicción')
ax_qq.set_ylabel('Valor real')
fig_qq.tight_layout()
fig_qq.savefig(aDir + 'ex02-qq.pdf')

fig_res, ax_res = plt.subplots()
ax_res.scatter(x, df_pico_11['infectados'] - y, color='royalblue')
ax_res.set_xlabel('Dias despues del 2020-03-05')
ax_res.set_ylabel('Residuos')
ax_res.axhline(y=0, color='k', ls='dashed')
fig_res.tight_layout()
fig_res.savefig(aDir + 'ex02-residuos.pdf')

fig_fit, ax_fit = plt.subplots()
ax_fit.scatter(x, df_pico_11['infectados'], color='royalblue')
ax_fit.plot(x, y, 'salmon')
ax_fit.set_xlabel('Dias despues del 2020-03-05')
ax_fit.set_ylabel('Infectados')
fig_fit.tight_layout()
fig_fit.savefig(aDir + 'ex02-fit.pdf')
# %%
pico_1 = datetime.datetime.strptime('2020-12-15','%Y-%m-%d')
df_pico_11 = df_weekday_2[df_weekday_2.index < pico_1]
df_pico_11.index = (df_pico_11.index - df_pico_11.index[0]).days
axs4 = df_pico_11.rename(**plt_rename).plot(**style)
figs4 = axs4[0].get_figure()
figs4.tight_layout()
figs4.savefig(aDir + 'ex02-sin-Finde-pico1.pdf')

aux_args = {'f':sech_sir, 
            'xdata':df_pico_11.index, 
            'ydata':df_pico_11['infectados'],
            'p0':(5000, 0.01, -1)}

popt, pcov = curve_fit(**aux_args)
a, b, c = popt

x = df_pico_11.index.values
y = sech_sir(x, a ,b ,c)

fig_qq, ax_qq = plt.subplots()
ax_qq.scatter(y, df_pico_11['infectados'], color='royalblue')
ylim = (min(y), max(y))
ax_qq.plot(ylim, ylim, '--k')
ax_qq.set_xlabel('Predicción')
ax_qq.set_ylabel('Valor real')
fig_qq.tight_layout()
fig_qq.savefig(aDir + 'ex02-qq-sin-Finde.pdf')

fig_res, ax_res = plt.subplots()
ax_res.scatter(x, df_pico_11['infectados'] - y, color='royalblue')
ax_res.set_xlabel('Dias despues del 2020-03-05')
ax_res.set_ylabel('Residuos')
ax_res.axhline(y=0, color='k', ls='dashed')
fig_res.tight_layout()
fig_res.savefig(aDir + 'ex02-residuos-sin-Finde.pdf')

fig_fit, ax_fit = plt.subplots()
ax_fit.scatter(x, df_pico_11['infectados'], color='royalblue')
ax_fit.plot(x, y, 'salmon')
ax_fit.set_xlabel('Dias despues del 2020-03-05')
ax_fit.set_ylabel('Infectados')
fig_fit.tight_layout()
fig_fit.savefig(aDir + 'ex02-fit-sin-Finde.pdf')
