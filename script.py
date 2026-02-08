#Particula en una dimension

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


q = -10
e0 = 1
mu0 = 1
c = 1/np.sqrt(e0*mu0)

res = 1000
min_ = 0
max_ = 10
space = np.linspace(min_,max_,res)
dx = (max_-min_)/res

phi = [space*0,space*0]
r = [5]

dt = 0.01
v = 0.0*c
std = 0.05
def FiniteDiffStep(t):
    nphi = np.zeros_like(phi[-1])
    delta = np.exp(-(space-r[-1])**2/(2*std**2))/np.sqrt(2*np.pi*std**2)
    for i in range(res):
        if i>0 and i<res-1:
            d2phid2x = (phi[-1][i+1]-2*phi[-1][i]+phi[-1][i-1])/dx**2
        elif i==0:
            d2phid2x = (phi[-1][i+1]-2*phi[-1][i]+phi[-1][-1])/dx**2
        elif i==res-1:
            d2phid2x = (phi[-1][0]-2*phi[-1][i]+phi[-1][i-1])/dx**2
        nphi[i] = 2*phi[-1][i]-phi[-2][i]+c**2*dt**2*(d2phid2x+q/e0*delta[i])
    if r[-1] > max_:
        r.append(0)
    elif r[-1]<min_:
        r.append(max_)
    else:
        r.append(r[-1]+v*dt)
    phi.append(nphi)

steps = 1000
for i in range(steps):
    FiniteDiffStep(i)


E = []
for i in range(steps):
    E.append(np.gradient(phi[i]))

print(E[0].shape,phi[0].shape)
fig = plt.figure()
ax = fig.add_subplot()

def animate(i):
    ax.clear()
    ax.plot(space,-E[i])
    ax.scatter(r[i],0)

ani = FuncAnimation(fig,animate,frames=steps)
plt.show()
