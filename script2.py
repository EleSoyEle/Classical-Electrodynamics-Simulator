#Particula en 2 dimensiones
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import SymLogNorm

q = -1
e0 = 1e-1
mu0 = 1e-1
c = 1/np.sqrt(e0*mu0)

res = 100
min_ = 0
max_ = 100

x0 = np.linspace(min_,max_,res)
x1 = np.linspace(min_,max_,res)

X,Y = np.meshgrid(x0,x1)

dx = (max_-min_)/res
dy = (max_-min_)/res

phi = [X*0,X*0]

r = [
    [[20,50],[60,50]],
    [[20,50],[60,50]],
    ]

dt = 0.05
v = np.array([[0.99,0],[0,0]])*c
std = 1


thickness = 5
sigma = np.zeros_like(X)
for i in range(thickness):
    val = ((thickness - i) / thickness) * 0.5
    sigma[:, i] = sigma[:, -i-1] = val
    sigma[i, :] = sigma[-i-1, :] = val



def FiniteDiffStep(t):
    nphi = np.zeros_like(phi[-1])
    delta1 = np.exp(-((X-r[-1][0][0])**2+(Y-r[-1][0][1])**2)/(2*std**2))/(2*np.pi*std**2)
    delta2 = np.exp(-((X-r[-1][1][0])**2+(Y-r[-1][1][1])**2)/(2*std**2))/(2*np.pi*std**2)
    
    print(delta1,delta2)
    d2phid2x = (np.roll(phi[-1],-1,axis=1)-2*phi[-1]+np.roll(phi[-1],1,axis=1))/dx**2
    d2phid2y = (np.roll(phi[-1],-1,axis=0)-2*phi[-1]+np.roll(phi[-1],1,axis=0))/dy**2
    
    parts_term = q/e0*(1*delta1+1*delta2)

    phi_amort = sigma*(phi[-1]-phi[-2])
    nphi = 2*phi[-1]-phi[-2]+c**2*dt**2*(d2phid2x+d2phid2y+parts_term)-phi_amort
    rn = []
    for p in range(len(r[0])):
        rp = []
        for i in range(2):
            if r[-1][p][i] > max_:
                rp.append(0)
            elif r[-1][p][i]<min_:
                rp.append(max_)
            else:
                rp.append(r[-1][p][i]+v[p][i]*dt)
        rn.append(rp)
    r[0] = r[1]
    r[1] = rn
    phi[0] = phi[1]
    phi[1] = nphi
steps = 1000
pre_calc = False

fig = plt.figure()
ax = fig.add_subplot()
s = 4
v_lim = 0
def animate(i):
    global v_lim
    print("Paso",i)
    ax.clear()
    FiniteDiffStep(0)
    En = np.gradient(phi[1],dy,dx)
    Enm = np.sqrt(En[0][::s,::s]**2+En[1][::s,::s]**2)
    max_v = 1.2*np.max(np.abs(phi[1]))
    v_lim =  v_lim if max_v<v_lim else max_v
    print(v_lim)
    Ex = -En[0][::s,::s]/Enm
    Ey = -En[1][::s,::s]/Enm
    
    ax.pcolormesh(X,Y,phi[1],cmap="seismic",vmin=-v_lim, vmax=v_lim)
    #ax.pcolormesh(X,Y,phi[1],cmap="seismic")
    ax.quiver(X[::s,::s],Y[::s,::s],Ex,Ey,alpha=0.2)
    for i in range(len(r[0])):
        ax.scatter(r[1][i][0],r[1][i][1])

ani = FuncAnimation(fig,animate,frames=steps)
#ani.save("ani1.mp4",fps=60,dpi=150)
plt.show()