#Particula en 3 dimensiones

#NOTA: Se resuelven las ecuaciones de Maxwell para los potenciales
# se simplifican aplicando la simetria gauge de Lorenz

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
res = 100  #Puntos por eje en la malla
min_ = -10 #Limite inferior
max_ = 10  #Limite superior

steps = 100 #Pasos de tiempo
s = 10 #Stride - Para no hacer pesada la graficación
dt = 0.05 #Salto de tiempo en cada paso


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

phi = [X*0,X*0] #Potencial escalar
A = [
    torch.zeros((3, res, res, res), device=device),
    torch.zeros((3, res, res, res), device=device)
] #Potencial vectorial

r = [
    [[0,0,0],[0.1,0.3,0.9]],
    [[0,0,0],[0.1,0.3,0.9]],
    ]

t = 0

#Carga y velocidad de las particulas
q = [1,1,-1]
v = np.array([[0.5,0,0.1],[0,0,0],[0,0,0.1]])*c
std = 0.01


thickness = 10 #Perdida de energia en la frontera de la caja
sigma = torch.zeros_like(X)
for i in range(thickness):
    val = ((thickness - i) / thickness) * 0.5
    sigma[:, i, :] = sigma[:, -i-1, :] = val
    sigma[i, :, :] = sigma[-i-1, :, :] = val
    sigma[:, :, i] = sigma[:, :, -i-1] = val


def MakeDelta(x,y,z,r0):
    return torch.exp(-((x-r0[0])**2+(y-r0[1])**2+(z-r0[2])**2))/(2*np.pi*std**2)**(3/2)

def FiniteDiffStep(t):
    nphi = torch.zeros_like(phi[-1]).to(device)
    delta_t = torch.zeros_like(X).to(device)
    current_term = torch.zeros_like(A[-1]).to(device)
    for i in range(len(r[0])):
        delta_i = q[i]*MakeDelta(X,Y,Z,r[-1][i])
        vi_vec = torch.from_numpy(v[i][:,np.newaxis, np.newaxis, np.newaxis]).to(device)
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
    
    #Lo que sigue de la función esta inacabado y poco optimizado
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

#Es para calcular el rotacional de forma optimizada en cuda
def torch_rot(A_tensor, dx, dy, dz):
    def d_dz(f): return (torch.roll(f, -1, dims=0) - torch.roll(f, 1, dims=0)) / (2 * dz)
    def d_dy(f): return (torch.roll(f, -1, dims=1) - torch.roll(f, 1, dims=1)) / (2 * dy)
    def d_dx(f): return (torch.roll(f, -1, dims=2) - torch.roll(f, 1, dims=2)) / (2 * dx)
    Bx = d_dy(A_tensor[2]) - d_dz(A_tensor[1])
    By = d_dz(A_tensor[0]) - d_dx(A_tensor[2])
    Bz = d_dx(A_tensor[1]) - d_dy(A_tensor[0])

    return Bx, By, Bz

#Analogamente con el gradiente
def torch_gradient_3d(f, dx, dy, dz):

    dfdy = (torch.roll(f, -1, dims=0) - torch.roll(f, 1, dims=0)) / (2 * dy)
    dfdx = (torch.roll(f, -1, dims=1) - torch.roll(f, 1, dims=1)) / (2 * dx)
    dfdz = (torch.roll(f, -1, dims=2) - torch.roll(f, 1, dims=2)) / (2 * dz)
    return torch.stack([-dfdy, -dfdx, -dfdz])


fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(1,2,1,projection="3d")
bx = fig.add_subplot(1,2,2,projection="3d")


cmap = cm.get_cmap("seismic")
norm = Normalize(-1,1)

#Vamos a cortar en gpu
def prepare_tensor_to_graph(T,stride):
    return T[::stride,::stride,::stride].cpu().numpy()

def animate(i):
    global t
    print("Paso",i)
    ax.clear()
    bx.clear()
    FiniteDiffStep(t+dt)
    t += dt
    
    gradPhin = torch_gradient_3d(phi[1],dx,dy,dz)
    Bx,By,Bz = torch_rot(A[1],dx,dy,dz)

    En = gradPhin-(A[1]-A[0])/dt

    Xp = prepare_tensor_to_graph(X,s)
    Yp = prepare_tensor_to_graph(Y,s)
    Zp = prepare_tensor_to_graph(Z,s)
    Enp = [prepare_tensor_to_graph(En[i],s) for i in range(3)]
    Bnp = [prepare_tensor_to_graph([Bx,By,Bz][i],s) for i in range(3)]

    #Nota sobre los exponentes en los mapas de colores:
    '''
    Se usan para que en la grafica de los campos vectoriales se muestren secciones que normalmente no se mostrarian
    Así evitamos mostrar todos los vectores a la vez y los mostramos por intensidad
    '''



    Bn_mag = np.sqrt(Bnp[0]**2 + Bnp[1]**2 + Bnp[2]**2)
    log_mag = np.log10(Bn_mag+1e-9)
    log_mag_min = log_mag.min()
    log_mag_max = log_mag.max()
    Bn_mag_norm = (log_mag - log_mag_min) / (log_mag_max - log_mag_min)

    Bcolores_rgba = np.zeros((Bnp[0].size,4))
    Bcolores_rgba[:, 0] = 0.65  #Rojo
    Bcolores_rgba[:, 1] = 0.04 #Verde
    Bcolores_rgba[:, 2] = 0.21  #Azul
    Bcolores_rgba[:, 3] = Bn_mag_norm.flatten()**(0.2) 
    


    En_mag = np.sqrt(Enp[0]**2+Enp[1]**2+Enp[2]**2)
    log_mag = np.log10(En_mag+1e-9)
    log_mag_min = log_mag.min()
    log_mag_max = log_mag.max()
    En_elec_norm = (log_mag - log_mag_min) / (log_mag_max - log_mag_min)


    Ecolores_rgba = np.zeros((Enp[0].size,4))
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
    
    ax.quiver(
        Xp,Yp,Zp,
        Enp[0],Enp[1],Enp[2],
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
#plt.show()
