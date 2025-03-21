import numpy as np
from scipy.sparse import coo_matrix,csc_matrix,csr_matrix,spdiags
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
class PoissonData1d:
    def __init__(self,L=0,R=1):
        self.L=L  #定义域左节点
        self.R=R  #定义域右节点
    def domain(self):
        return (self.L,self .R)
    def init_mesh(self,NS=10):   #平均生成节点
        x=np.linspace(self.L,self.R,num=NS+1)
        return x
    def solution(self,x):
        pi=np.pi
        return np.sin(4*pi*x)
    def source(self,x):
        pi=np.pi
        return 16.0*pi**2*np.sin(4*pi*x)
def FVM(data,NS):
    x=data.init_mesh(NS=NS)

    NN=len(x)
    h=x[1]-x[0]
    c1=-1/h
    c2=2/h
    #Au=b
    #构建A矩阵
    val=np.zeros((3,NN-2))
    val[0,:]=c1
    val[1,:]=c2
    val[2,:]=c1
    A=spdiags(val,[-1,0,1],NN-2,NN-2)
    pi=np.pi
    pi = np.pi
    F = -4 * pi * (np.cos(4 * pi * (x[1: NN - 1] + h / 2)) - np.cos(4 * pi * (x[1: NN- 1] - h / 2)))
    data.source(x[1: -1])

    u = np.zeros((NN,), dtype=np.float_)
    u[[0, -1]] = data.solution(x[[0, -1]])

    F[0] -= c1 * u[0]
    F[-1] -= c1 * u[-1]
    u[1: -1] = spsolve(A, F)
    return x, u

def error(x, u, uh):
    e = u - uh

    emax = np.max(np.abs(e))
    return emax
L = 0
R = 1
NS = 5

maxit = 5
data = PoissonData1d(L=L, R=R)
e = np.zeros((maxit, 1), dtype=np.float_)
fig = plt.figure()
axes = fig.gca()
for i in range(maxit):
    x, uh = FVM(data, NS)
    u = data.solution(x)
    e[i, 0] = error(x, u, uh)
    axes.plot(x, uh, linestyle= '-', marker = 'o', markersize = 4,label = " N=% d " % NS )
    axes.set_title(" The solution plot ")
    axes.set_xlabel(" x ")
    axes.set_ylabel(" u ")
    if i < maxit - 1:
        NS *= 2
axes.plot(x, u, label= ' exact ')
plt.legend(loc= 'upper right')
print(" error :\ n", e)
plt.show()
