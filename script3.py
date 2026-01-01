#Particula en 3 dimensiones

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize
from matplotlib import cm
q = -1
e0 = 1
mu0 = 1
c = 1/np.sqrt(e0*mu0)

res = 100
min_ = -30
max_ = 30

x0 = np.linspace(min_,max_,res)
x1 = np.linspace(min_,max_,res)
x2 = np.linspace(min_,max_,res)

X,Y,Z = np.meshgrid(x0,x1,x2)
xl,yl = np.meshgrid(x0,x1)

dx = (max_-min_)/res
dy = (max_-min_)/res
dz = (max_-min_)/res

phi = [X*0,X*0]

A = [
    np.array([X*0,X*0,X*0]),
    np.array([X*0,X*0,X*0])
]

r = [
    [[0,0,0]],
    [[0,0,0]],
    ]

dt = 0.05
q = [1,1,-1]
v = np.array([[0.5,0,0],[0,0,0],[0,0,0.1]])*c
std = 0.01


thickness = 5
sigma = np.zeros_like(X)
for i in range(thickness):
    val = ((thickness - i) / thickness) * 0.5
    sigma[:, i, :] = sigma[:, -i-1, :] = val
    sigma[i, :, :] = sigma[-i-1, :, :] = val
    sigma[:, :, i] = sigma[:, :, -i-1] = val


def MakeDelta(x,y,z,r0):
    return np.exp(-((x-r0[0])**2+(y-r0[1])**2+(z-r0[2])**2))/(2*np.pi*std**2)**(3/2)

def FiniteDiffStep(t):
    nphi = np.zeros_like(phi[-1])
    delta_t = np.zeros_like(X)
    current_term = np.zeros_like(A[-1])
    for i in range(len(r[0])):
        delta_i = q[i]*MakeDelta(X,Y,Z,r[-1][i])
        vi_vec = v[i][:,np.newaxis, np.newaxis, np.newaxis]
        Ji = vi_vec*delta_i

        delta_t += delta_i
        current_term += Ji

    d2phid2x = (np.roll(phi[-1],-1,axis=1)-2*phi[-1]+np.roll(phi[-1],1,axis=1))/dx**2
    d2phid2y = (np.roll(phi[-1],-1,axis=0)-2*phi[-1]+np.roll(phi[-1],1,axis=0))/dy**2
    d2phid2z = (np.roll(phi[-1],-1,axis=2)-2*phi[-1]+np.roll(phi[-1],1,axis=2))/dz**2

    d2Ad2x = (np.roll(A[-1], -1, axis=1) - 2*A[-1] + np.roll(A[-1], 1, axis=1))/dx**2
    d2Ad2y = (np.roll(A[-1], -1, axis=0) - 2*A[-1] + np.roll(A[-1], 1, axis=0))/dy**2
    d2Ad2z = (np.roll(A[-1], -1, axis=2) - 2*A[-1] + np.roll(A[-1], 1, axis=2))/dz**2
    
    parts_term = delta_t/e0
    current_term = mu0*current_term

    phi_amort = sigma*(phi[-1]-phi[-2])
    nphi = 2*phi[-1]-phi[-2]+c**2*dt**2*(d2phid2x+d2phid2y+d2phid2z+parts_term)-phi_amort

    A_amort = sigma*(A[-1]-A[-2])
    nA = 2*A[-1]-A[-2]+c**2*dt**2*(d2Ad2x+d2Ad2y+d2Ad2z+current_term)-A_amort
    rn = []
    for p in range(len(r[0])):
        rp = []
        for i in range(3):
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
    A[0] = A[1]
    A[1] = nA
steps = 5000
pre_calc = False


def Rot(A):
    Axy = np.gradient(A[0],axis=0)
    Axz = np.gradient(A[0],axis=2)

    Ayx = np.gradient(A[1],axis=1)
    Ayz = np.gradient(A[1],axis=2)

    Azx = np.gradient(A[2],axis=1)
    Azy = np.gradient(A[2],axis=0)

    Bx = Ayz-Azy
    By = Axz-Azx
    Bz = Ayx-Axy
    return Bx,By,Bz



fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(1,2,1,projection="3d")
bx = fig.add_subplot(1,2,2,projection="3d")
s = 10
v_lim = 0


cmap = cm.get_cmap("seismic")
norm = Normalize(-1,1)

def animate(i):
    global v_lim
    print("Paso",i)
    ax.clear()
    bx.clear()
    FiniteDiffStep(0)
    
    gradPhin = np.gradient(-phi[1],dy,dx,dz)
    Bx,By,Bz = Rot(A[1])
    En = gradPhin-(A[1]-A[0])/dt


    Bn_mag = np.sqrt(Bx[::s, ::s, ::s]**2 + By[::s, ::s, ::s]**2 + Bz[::s, ::s, ::s]**2)
    log_mag = np.log10(Bn_mag+1e-9)
    log_mag_min = log_mag.min()
    log_mag_max = log_mag.max()
    Bn_mag_norm = (log_mag - log_mag_min) / (log_mag_max - log_mag_min)


    Bcolores_rgba = np.zeros((Bx[::s,::s,::s].size, 4))
    Bcolores_rgba[:, 0] = 0.0  # Rojo
    Bcolores_rgba[:, 1] = 0.4  # Verde
    Bcolores_rgba[:, 2] = 1.0  #Azul
    Bcolores_rgba[:, 3] = Bn_mag_norm.flatten()**0.3
    
    En_mag = np.sqrt(En[0][::s, ::s, ::s]**2+En[1][::s, ::s, ::s]**2+En[2][::s, ::s, ::s]**2)
    log_mag = np.log10(En_mag+1e-9)
    log_mag_min = log_mag.min()
    log_mag_max = log_mag.max()
    En_elec_norm = (log_mag - log_mag_min) / (log_mag_max - log_mag_min)


    Ecolores_rgba = np.zeros((En[0][::s,::s,::s].size, 4))
    Ecolores_rgba[:, 0] = 0.0  # Rojo
    Ecolores_rgba[:, 1] = 0.4  # Verde
    Ecolores_rgba[:, 2] = 1.0  #Azul
    Ecolores_rgba[:, 3] = En_elec_norm.flatten()**0.3
    
    ax.axis("off")
    bx.axis("off")

    if i%2==0:
        ax.view_init(elev=5, azim=i//2, roll=0)
        bx.view_init(elev=5, azim=i//2, roll=0)
    
    ax.set_title("Campo electrico")
    
    ax.quiver(
        X[::s,::s,::s],Y[::s,::s,::s],Z[::s,::s,::s],
        En[0][::s,::s,::s],En[1][::s,::s,::s],En[2][::s,::s,::s],
        normalize=True,length=4,arrow_length_ratio=0.3,color=Ecolores_rgba)
    bx.set_title("Campo magnetico")
    bx.quiver(
        X[::s,::s,::s],Y[::s,::s,::s],Z[::s,::s,::s],
        Bx[::s,::s,::s],By[::s,::s,::s],Bz[::s,::s,::s],
        normalize=True,length=4,color=Bcolores_rgba)

    for i in range(len(r[0])):
        ax.scatter(r[1][i][0],r[1][i][1],r[1][i][2])
        bx.scatter(r[1][i][0],r[1][i][1],r[1][i][2])
        ax.quiver(r[1][i][0],r[1][i][1],r[1][i][2],v[i][0],v[i][1],v[i][2],length=3,color="black",normalize=True)
        bx.quiver(r[1][i][0],r[1][i][1],r[1][i][2],v[i][0],v[i][1],v[i][2],length=3,color="black",normalize=True)


ani = FuncAnimation(fig,animate,frames=steps)
ani.save("ani4.mp4",fps=60,dpi=150)
plt.show()