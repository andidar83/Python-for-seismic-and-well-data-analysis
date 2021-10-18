# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 19:23:30 2020

@author: Andidar83
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import signal as sg
import scipy.interpolate 
from scipy.optimize import brentq as root
"""
data = pd.read_csv('WELLUAS.csv')
depth = data['DEPTH']
gr = data['GR']
rhob = data['RHOB']
cali = data['CALI']
induction = data['INDUCTION']
nphi = data['NPHI']
pwave = data['PWAVE']

frame = plt.figure(figsize=[10,10])

plot1 = frame.add_subplot(131)
plot1.set_xticks(np.arange(0,150,15))
plot1.plot(gr,depth)
plot1.title.set_text('Kedalaman vs GR')


plot2 = frame.add_subplot(132)
plot2.set_xticks(np.arange(1.35,3.0,0.3))
plot2.plot(rhob,depth)
plot2.title.set_text('Kedalaman vs RHOB')

plot3 = frame.add_subplot(133)
plot3.set_xticks(np.arange(7,23,3))
plot3.plot(cali,depth)
plot3.title.set_text('Kedalaman vs CALI')

frame = plt.figure(figsize=[10,8])

plot4 = frame.add_subplot(131)
plot4.set_xticks(np.arange(0.3,300,100))
plot4.plot(induction,depth)
plot4.title.set_text('Kedalaman vs INDUCTION')

plot5 = frame.add_subplot(132)
plot5.set_xticks(np.arange(0,0.6,0.1))
plot5.plot(nphi,depth)
plot5.title.set_text('Kedalaman vs NPHI')

plot6 = frame.add_subplot(133)
plot6.set_xticks(np.arange(20,140,20))
plot6.plot(pwave,depth)
plot6.title.set_text('Kedalaman vs P-WAVE')

plt.tight_layout()
"""
"""
# NO 2

data = pd.read_csv('CHK1.csv')
depth1 = data['DEPTH1']
TWT = data['TWT']

frame = plt.figure(figsize=[10,10])
plot7 = frame.add_subplot(111)
plot7.scatter(TWT,depth1,s=5,marker='o',color = 'b',linewidths=0.1)
plot7.invert_yaxis()
plot7.invert_xaxis()
plot7.set_xlabel('TWT')
plot7.set_ylabel('Depth (m)')
plot7.title.set_text('Check Shoot')
plot7.grid()
"""
"""
#NO 3
data = pd.read_csv('CHK1.csv')
TWT = data['TWT']
gr2 = data['GR2']
rhob2 = data['RHOB2']
cali2 = data['CALI2']
induction2 = data['INDUCTION2']
nphi2 = data['NPHI2']
pwave2 = data['PWAVE2']
velocity = 1/(pwave2*10**(-6)/0.305)


frame = plt.figure(figsize=[10,10])

plot1 = frame.add_subplot(131)
plot1.set_xticks(np.arange(0,150,15))
plot1.plot(gr2,TWT)
plot1.title.set_text('Time vs GR')


plot2 = frame.add_subplot(132)
plot2.set_xticks(np.arange(1.35,3.0,0.3))
plot2.plot(rhob2,TWT)
plot2.title.set_text('Time vs RHOB')

plot3 = frame.add_subplot(133)
plot3.set_xticks(np.arange(7,23,3))
plot3.plot(cali2,TWT)
plot3.title.set_text('Time vs CALI')

frame = plt.figure(figsize=[10,8])

plot4 = frame.add_subplot(131)
plot4.set_xticks(np.arange(0.3,300,100))
plot4.plot(induction2,TWT)
plot4.title.set_text('Time vs INDUCTION')

plot5 = frame.add_subplot(132)
plot5.set_xticks(np.arange(0,0.6,0.1))
plot5.plot(nphi2,TWT)
plot5.title.set_text('Time vs NPHI')

plot6 = frame.add_subplot(133)

plot6.plot(velocity,TWT)
plot6.title.set_text('Time vs P-WAVE')

plt.tight_layout()
"""

#NO 4
data = pd.read_csv('AI.csv')
pwave2 = data['PWAVE3']
rhob2 = data['RHOB3']
TWT = data['TWT3']

velocity = 1/(pwave2*10**(-6)/0.305)
AI =  rhob2*velocity
Xnew = np.linspace(min(TWT), max(TWT), num=1489, endpoint=True)
Xnew2 = np.arange(min(TWT), max(TWT)+1, 2)
f = scipy.interpolate.Rbf(TWT, AI, function='linear')
fig= plt.figure(figsize=(10,4))
plt.plot(TWT, AI, 'x', Xnew, f(Xnew),'-',color='b',markersize=0.1)
plt.plot(TWT, AI, 'x', Xnew2, f(Xnew2),'-', color='r',markersize=0.1)
plt.gca().invert_xaxis()
plt.legend(['','AI','','AI Resample 2m/s'], loc='best')
plt.xlabel('Time(s)')
plt.ylabel('AI')
plt.grid()
plt.show()

de = TWT[1:1489]
KRF=[]
for i in range(1488):
    a= AI[i+1]-AI[i]/AI[i+1]-AI[i]
    KRF.append(a)

frame = plt.figure(figsize=[10,10])
    
plot10 = frame.add_subplot(141)
plot10.plot(KRF,de)
plt.grid()
plot10.title.set_text('KOEF REFLEKSI')


def Ricker(f,t):
    wavelet = (1 - 2*np.pi*f*t**2)*np.exp(-np.pi*f*t**2)
    return wavelet

LstFq = []
LstT = []
strt = -0.5
LstT.append(strt)
LstFq.append(Ricker(30,strt))
for i in range(500):
    strt += 0.002
    LstFq.append(Ricker(30,strt))
    LstT.append(strt)    

convolusi = np.convolve(KRF,LstFq)

frame = plt.figure(figsize=[10,4])

plot11 = frame.add_subplot(111)
plot11.plot(convolusi,'black')
plot11.set_xticks(np.arange(0,2000,200))
plot11.title.set_text('Seismogram sintetik')
plt.grid()


"""
# NO 5

h1 = 0 #satuan km
v1 = 3 #km/s

xmin = 0.1
xmax = 3
dx = 0.05

x = np.arange(xmin,xmax,dx)
t1 = ((2*h1/v1)**2+(x/v1)**2)**0.5
#############################################################################
h2 = 0.4
v2 = 3


t2 = ((2*h2/v2)**2+(x/v2)**2)**0.5


##########################################################################33
h3 = 0.6
v3 = 2


t3 = ((2*h3/v3)**2+(x/v3)**2)**0.5
##############################################################################
h4 = 0.9
v4 = 3


t4 = ((2*h4/v4)**2+(x/v4)**2)**0.5
##############################################################################
h5 = 1.1
v5 = 3.5


t5 = ((2*h5/v5)**2+(x/v5)**2)**0.5
################################################################################
h6 = 1.5
v6 = 3.3


t6 = ((2*h6/v6)**2+(x/v6)**2)**0.5
###############################################################################33
h7 = 2.6
v7 = 4

t7 = ((2*h7/v7)**2+(x/v7)**2)**0.5
##################################################################################
h8 = 1.8
v8 = 3.4


t8 = ((2*h8/v8)**2+(x/v8)**2)**0.5
####################################################################################
h9 = 2
v9 = 3.6


t9 = ((2*h9/v9)**2+(x/v9)**2)**0.5

plt.plot(x,t1)
plt.plot(x,t2)
plt.plot(x,t3)
plt.plot(x,t4)
plt.plot(x,t5)
plt.plot(x,t6)
plt.plot(x,t7)
plt.plot(x,t8)
plt.plot(x,t9)
plt.gca().invert_yaxis()
plt.xlabel('Offset(km)')
plt.ylabel('Time(s)')
plt.title('Ray Tracing')
plt.show()
TotalTime= t1+t2+t3+t4+t5+t6+t7+t8+t9
print(TotalTime)
"""