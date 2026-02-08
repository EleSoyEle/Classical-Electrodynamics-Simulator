#Particula en 3 dimensiones

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize
from matplotlib import cm
import torch
device = torch.device("cpu")

#Parametros fisicos
e0 = 1
mu0 = 1
c = 1/np.sqrt(e0*mu0)

#Parametros del simulador
res = 50
min_ = -10
max_ = 10

x0 = np.linspace(min_,max_,res)
x1 = np.linspace(min_,max_,res)
x2 = np.linspace(min_,max_,res)

X,Y,Z = np.meshgrid(x0,x1,x2)

X = torch.from_numpy(X).to(device)
Y = torch.from_numpy(Y).to(device)
Z = torch.from_numpy(Z).to(device)

xl,yl = np.meshgrid(x0,x1)

dx = (max_-min_)/res
dy = (max_-min_)/res
dz = (max_-min_)/res

phi = [X*0,X*0]
A = [
    torch.zeros((3, res, res, res), device=device),
    torch.zeros((3, res, res, res), device=device)
]

r = [
    [[0,0,0],[0.1,0.3,0.9]],
    [[0,0,0],[0.1,0.3,0.9]],
    ]

dt = 0.05

#Carga y velocidad de las particulas
q = [1,1,-1]
v = np.array([[0.5,0,0.1],[0,0,0],[0,0,0.1]])*c
std = 0.01


thickness = 10
sigma = torch.zeros_like(X)
for i in range(thickness):
    val = ((thickness - i) / thickness) * 0.5
    sigma[:, i, :] = sigma[:, -i-1, :] = val
    sigma[i, :, :] = sigma[-i-1, :, :] = val
    sigma[:, :, i] = sigma[:, :, -i-1] = val


def MakeDelta(x,y,z,r0):
    return torch.exp(-((x-r0[0])**2+(y-r0[1])**2+(z-r0[2])**2))/(2*np.pi*std**2)**(3/2)

def FiniteDiffStep(t):
    nphi = torch.zeros_like(phi[-1])
    delta_t = torch.zeros_like(X)
    current_term = torch.zeros_like(A[-1])
    for i in range(len(r[0])):
        delta_i = q[i]*MakeDelta(X,Y,Z,r[-1][i])
        vi_vec = torch.from_numpy(v[i][:,np.newaxis, np.newaxis, np.newaxis])
        Ji = vi_vec*delta_i

        delta_t += delta_i
        current_term += Ji

    d2phid2x = (torch.roll(phi[-1],-1,dims=1)-2*phi[-1]+torch.roll(phi[-1],1,dims=1))/dx**2
    d2phid2y = (torch.roll(phi[-1],-1,dims=0)-2*phi[-1]+torch.roll(phi[-1],1,dims=0))/dy**2
    d2phid2z = (torch.roll(phi[-1],-1,dims=2)-2*phi[-1]+torch.roll(phi[-1],1,dims=2))/dz**2

    d2Ad2x = (torch.roll(A[-1], -1, dims=1) - 2*A[-1] + torch.roll(A[-1], 1, dims=1))/dx**2
    d2Ad2y = (torch.roll(A[-1], -1, dims=0) - 2*A[-1] + torch.roll(A[-1], 1, dims=0))/dy**2
    d2Ad2z = (torch.roll(A[-1], -1, dims=2) - 2*A[-1] + torch.roll(A[-1], 1, dims=2))/dz**2
    
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
steps = 100
pre_calc = False


def Rot(A):
    Axy = np.gradient(A[0],axis=0)
    Axz = np.gradient(A[0],axis=2)

    Ayx = np.gradient(A[1],axis=1)
    Ayz = np.gradient(A[1],axis=2)

    Azx = np.gradient(A[2],axis=1)
    Azy = np.gradient(A[2],axis=0)

    Bx = torch.from_numpy(Ayz-Azy)
    By = torch.from_numpy(Axz-Azx)
    Bz = torch.from_numpy(Ayx-Axy)
    return Bx,By,Bz

def torch_gradient_3d(f, dx, dy, dz):

    dfdy = (torch.roll(f, -1, dims=0) - torch.roll(f, 1, dims=0)) / (2 * dy)
    dfdx = (torch.roll(f, -1, dims=1) - torch.roll(f, 1, dims=1)) / (2 * dx)
    dfdz = (torch.roll(f, -1, dims=2) - torch.roll(f, 1, dims=2)) / (2 * dz)
    return torch.stack([-dfdy, -dfdx, -dfdz])


fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(1,2,1,projection="3d")
bx = fig.add_subplot(1,2,2,projection="3d")
s = 10
v_lim = 0


cmap = cm.get_cmap("seismic")
norm = Normalize(-1,1)

#Vamos a cortar en gpu
def prepare_tensor_to_graph(T,stride):
    return T[::stride,::stride,::stride].cpu().numpy()

def animate(i):
    global v_lim
    print("Paso",i)
    ax.clear()
    bx.clear()
    FiniteDiffStep(0)
    
    gradPhin = torch_gradient_3d(phi[1],dx,dy,dz)
    Bx,By,Bz = Rot(A[1])

    En = gradPhin-(A[1]-A[0])/dt


    Bn_mag = np.sqrt(Bx[::s, ::s, ::s]**2 + By[::s, ::s, ::s]**2 + Bz[::s, ::s, ::s]**2)
    log_mag = np.log10(Bn_mag+1e-9)
    log_mag_min = log_mag.min()
    log_mag_max = log_mag.max()
    Bn_mag_norm = (log_mag - log_mag_min) / (log_mag_max - log_mag_min)


    Bcolores_rgba = torch.zeros((Bx[::s,::s,::s].numel(), 4)).cpu().numpy()
    Bcolores_rgba[:, 0] = 0.65  #Rojo
    Bcolores_rgba[:, 1] = 0.04 #Verde
    Bcolores_rgba[:, 2] = 0.21  #Azul
    Bcolores_rgba[:, 3] = Bn_mag_norm.flatten()**(0.2)
    
    En_mag = np.sqrt(En[0][::s, ::s, ::s]**2+En[1][::s, ::s, ::s]**2+En[2][::s, ::s, ::s]**2)
    log_mag = np.log10(En_mag+1e-9)
    log_mag_min = log_mag.min()
    log_mag_max = log_mag.max()
    En_elec_norm = (log_mag - log_mag_min) / (log_mag_max - log_mag_min)


    Ecolores_rgba = torch.zeros((En[0][::s,::s,::s].numel(), 4)).cpu().numpy()
    Ecolores_rgba[:, 0] = 0.0  #Rojo
    Ecolores_rgba[:, 1] = 0.4  #Verde
    Ecolores_rgba[:, 2] = 1.0  #Azul
    Ecolores_rgba[:, 3] = En_elec_norm.flatten()**0.3
    
    ax.axis("off")
    bx.axis("off")

    if i%2==0:
        ax.view_init(elev=10, azim=i//2, roll=0)
        bx.view_init(elev=10, azim=i//2, roll=0)
    
    ax.set_title("Campo electrico")
    
    Xp = prepare_tensor_to_graph(X,s)
    Yp = prepare_tensor_to_graph(Y,s)
    Zp = prepare_tensor_to_graph(Z,s)
    Enp = [prepare_tensor_to_graph(En[i],s) for i in range(3)]
    Bnp = [prepare_tensor_to_graph([Bx,By,Bz][i],s) for i in range(3)]

    ax.quiver(
        Xp,Yp,Zp,
        Enp[0],Enp[1],Enp[1],
        normalize=True,length=1,arrow_length_ratio=0.3,color=Ecolores_rgba)
    bx.set_title("Campo magnetico")
    bx.quiver(
        Xp,Yp,Zp,
        Bnp[0],Bnp[1],Bnp[2],
        normalize=True,length=1,color=Bcolores_rgba)

    for i in range(len(r[0])):
        ax.scatter(r[1][i][0],r[1][i][1],r[1][i][2])
        bx.scatter(r[1][i][0],r[1][i][1],r[1][i][2])
        ax.quiver(r[1][i][0],r[1][i][1],r[1][i][2],v[i][0],v[i][1],v[i][2],length=3,color="black",normalize=True)
        bx.quiver(r[1][i][0],r[1][i][1],r[1][i][2],v[i][0],v[i][1],v[i][2],length=3,color="black",normalize=True)


ani = FuncAnimation(fig,animate,frames=steps)
ani.save("ani4.mp4",fps=15,dpi=150)
plt.show()