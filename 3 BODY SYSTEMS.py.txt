import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

G=6.67e-11

def distance(i):

  if i == 1:
    return  ((r1[0] - r2[0])**2 + (r1[1] - r2[1])**2 + (r1[2] - r2[2] )**2)**0.5
    
  elif i == 2:
    return  ((r3[0] - r2[0])**2 + (r1[1] - r2[1])**2 + (r3[2] - r2[2] )**2)**0.5
    
  elif i == 3:
    return  ((r1[0] - r3[0])**2 + (r1[1] - r3[1])**2 + (r1[2] - r3[2] )**2)**0.5

  else:
    print("ERROR OCCURED")
    

def force_p1p2(r1,r2):  #FORCE ON PARTICLE 1 BY PARTICLE 2
    
    r = np.zeros(3)
    F = np.zeros(3)

    # print(type((r1[0] - r2[0])[0]))
    # print(r)

    r[0] = (r1[0] - r2[0])[0]    
    r[1] = (r1[1] - r2[1])[1]
    r[2] = (r1[2] - r2[2])[2]
    rmag = np.linalg.norm(r)
    Fmag = G*M1*M2/(rmag+1e-20)**2   #change np.linalg.norm(r) into rmag
    

    theta_x = math.acos(np.abs(r[0])/((rmag)+1e-20))
    theta_y = math.acos(np.abs(r[1])/((rmag)+1e-20))
    theta_z = math.acos(np.abs(r[2])/((rmag)+1e-20))

    F[0] = Fmag * np.cos(theta_x)
    F[1] = Fmag * np.cos(theta_y)
    F[2] = Fmag * np.cos(theta_z)


    if r[0] > 0:
        F[0] = -F[0]
    if r[1] > 0:
        F[1] = -F[1]
    if r[2] > 0:
        F[2] = -F[2]
        
    return F

def force_p2p3(r2,r3):   #FORCE ON PARTICLE 2 BY PARTICLE 3
    r = np.zeros(3)
    F = np.zeros(3)
    r[0] = int(r2[0] - r3[0])    
    r[1] = int(r2[1] - r3[1])
    r[2] = int(r2[2] - r3[2])
    rmag = np.linalg.norm(r)
    Fmag = G*M2*M3/(rmag+1e-20)**2   #change np.linalg.norm(r) into rmag
   

    theta_x = math.acos(np.abs(r[0])/((rmag)+1e-20))
    theta_y = math.acos(np.abs(r[1])/((rmag)+1e-20))
    theta_z = math.acos(np.abs(r[2])/((rmag)+1e-20))

    F[0] = Fmag * np.cos(theta_x)
    F[1] = Fmag * np.cos(theta_y)
    F[2] = Fmag * np.cos(theta_z)


    if r[0] > 0:
        F[0] = -F[0]
    if r[1] > 0:
        F[1] = -F[1]
    if r[2] > 0:
        F[2] = -F[2]
        
    return F
  
def force_p3p1(r3,r1): #FORCE ON PARTICLE 3 BY PARTICLE 1
    
    r = np.zeros(3)
    F = np.zeros(3)
    r[0] = (r3[0] - r1[0]) [0]   
    r[1] = (r3[1] - r1[1])[1]
    r[2] = (r3[2] - r1[2])[2]
  
    rmag = np.linalg.norm(r)
  
    Fmag = G*M1*M3/(np.linalg.norm(r)+1e-20)**2   #change np.linalg.norm(r) into rmag
    

    theta_x = math.acos(np.abs(r[0])/((rmag)+1e-20))
    theta_y = math.acos(np.abs(r[1])/((rmag)+1e-20))
    theta_z = math.acos(np.abs(r[2])/((rmag)+1e-20))

    F[0] = Fmag * np.cos(theta_x)
    F[1] = Fmag * np.cos(theta_y)
    F[2] = Fmag * np.cos(theta_z)


    if r[0] > 0:
        F[0] = -F[0]
    if r[1] > 0:
        F[1] = -F[1]
    if r[2] > 0:
        F[2] = -F[2]
        
    return F 
  
def force(particle):
    if particle == 1:
        f1 = force_p1p2(r1, r2) + force_p3p1(r3, r1)
        return f1
    elif particle == 2:
        f2 = force_p1p2(r1, r2) + force_p2p3(r2, r3)
        return f2
    elif particle == 3:
        f3 = force_p2p3(r2, r3) + force_p3p1(r3, r1)
        return f3
    else:
        print("ERROR: Invalid particle number.")


def dv_dt(t,r,v,particle,ro,vo):  #ACCELERATION
         
    F = force(particle)

    if particle == 1:
        y = F/M1
        
    elif particle == 2:
        y = F/M2

    elif particle == 3:
        y = F/M3
        
    else:
        print("ERROR OCCURED")
        
    return y

def dr_dt(t,r1,r2,r3,v,particle):
    return v

def KINETICE(particle,v):        # KINETIC ENERGY OF A PARTICLE
    v = np.linalg.norm(v)

    if particle == 1:
      return 0.5*M1*v**2
      
    if particle == 2:
      return 0.5*M2*v**2
      
    if particle == 3:
      return 0.5*M3*v**2
      
    else:
        print("ERROR OCCURED")

def P_E(particle,r1):                # POTENTIAL ENERGY OF A PARTICLE
    
    if particle == 1:
      print(-(G*M1*M2/(distance(1)+1e-20))+(G*M1*M3/(distance(3)+1e-20)))

    elif particle == 2:
      print(-(G*M1*M2/(distance(1)+1e-20))+(G*M2*M3/(distance(2)+1e-20)))

    elif particle == 3:
      print(-(G*M1*M3/(distance(3)+1e-20))+(G*M2*M3/(distance(2)+1e-20)))
    
    else:
        print("ERROR OCCURED")
    
def AM(r,v):
  
    r_m = np.linalg.norm(r)
    v_m = np.linalg.norm(v)
    r = r/r_m
    v = v/v_m
    rdotv = r[0]*v[0]+r[1]*v[1] # CROSS PRODUCT
    theta = math.acos(rdotv)    
    return r_m*v_m*np.sin(theta)

def AngMomentum(particle,r1,r2,r3,v1,v2,v3):     # ANGULAR MOMENTUM OF A PARTICLE
  
    if particle == 1:
      return M1*AM(r1,v1)
      
    if particle == 2:
      return M2*AM(r2,v2)
      
    if particle == 3:
      return M1*AM(r3,v3)
      
    else:
        print("ERROR OCCURED")

def RK4(t,r,v,h,planet,ro,vo):
    k11 = dr_dt(t,r,v,planet,ro,vo) 
    k21 = dv_dt(t,r,v,planet,ro,vo)
    
    k12 = dr_dt(t + 0.5*h,r + 0.5*h*k11,v + 0.5*h*k21,planet,ro,vo)
    k22 = dv_dt(t + 0.5*h,r + 0.5*h*k11,v + 0.5*h*k21,planet,ro,vo)
    
    k13 = dr_dt(t + 0.5*h,r + 0.5*h*k12,v + 0.5*h*k22,planet,ro,vo)
    k23 = dv_dt(t + 0.5*h,r + 0.5*h*k12,v + 0.5*h*k22,planet,ro,vo)
    
    k14 = dr_dt(t + h,r + h*k13,v + h*k23,planet,ro,vo)
    k24 = dv_dt(t + h,r + h*k13,v + h*k23,planet,ro,vo)
    
    y0 = r + h * (k11 + 2.*k12 + 2.*k13 + k14) / 6.
    y1 = v + h * (k21 + 2.*k22 + 2.*k23 + k24) / 6.
    
    z = np.zeros([2,3])
    z = [y0, y1]
    return z


M1 = int(input())                
M2 = int(input())                                    
M3 = int(input())                 
G = 6.673e-11             # Gravitational Constant

RR = 1                    # Normalizing distance in m 
MM = 1                    # Normalizing mass 1kg
TT = 1                    # Normalizing time 1 sec

FF = (G*MM**2)/RR**2          # Unit force
EE = FF*RR                    # Unit energy

GG = G

M1 = M1/MM                    # Normalized mass of particle 1
M2 = M2/MM                    # Normalized mass of particle 2 
M3 = M3/MM                # Normalized mass of particle 3

ti = 0                # initial time = 0

tf = 1800           # final time = 1800 sec

 



N =1*tf                   # 1points per sec
t = np.linspace(ti,tf,N)     # time array from ti to tf with N points 

h = t[2]-t[1]                # time step (uniform)


# Initialization

KE = np.zeros(N)            # Kinetic energy
PE = np.zeros(N)            # Potential energy
A_M = np.zeros(N)            # Angular momentum
AreaVal = np.zeros(N)

r1 = np.zeros([N,3])         # position vector of mass1
v1 = np.zeros([N,3])         # velocity vector of mass1
r2 = np.zeros([N,3])        # position vector of mass2
v2 = np.zeros([N,3])        # velocity vector of mass2
r3 = np.zeros([N,3])        # position vector of mass2
v3 = np.zeros([N,3])        # velocity vector of mass2



r_1 = input().split()   # initial position of mass1
r_2 = input().split()   # initial position of mass2
r_3 = input().split()   # initial position of mass3

v_1 = input().split()   # initial velocity of mass1
v_2 = input().split()   # initial velocity of mass2
v_3 = input().split()   # initial velocity of mass3

for i in range(3):
  r_1[i]=int(r_1[i])
  r_2[i]=int(r_2[i])
  r_3[i]=int(r_3[i])

for i in range(3):
  v_1[i]=int(v_1[i])
  v_2[i]=int(v_2[i])
  v_3[i]=int(v_3[i])
  
v_1_mag = (v1[0]**2+v1[1]**2+v1[2]**2)**0.5   # Magnitude of Earth's initial velocity 

v_2_mag = (v1[0]**2+v1[1]**2+v1[2]**2)**0.5         # Magnitude of Jupiter's initial velocity 
        

# Initializing the arrays with initial values.
t[0] = ti
r1[0,:] = r_1
v1[0,:] = v_1
r2[0,:] = r_2
v2[0,:] = v_2
r3[0,:] = r_3
v3[0,:] = v_3

"""
t1 = dr_dt(ti,ri,vi)
t2 = dv_dt(ti,ri,vi)..
print t1
print t2
"""
KE[0] = KINETICE(1,v1[0,:])
PE[0] = P_E(1,r1[0,:])
A_M[0] = AngMomentum(1,r1[0,:],r2[0,:],r3[0,:],v1[0,:],v2[0,:],v3[0,:])

for i in range(0,N-1):
    [r1[i+1,:],v1[i+1,:]]=RK4(t[i],r1[i,:],v1[i,:],h,1,r2[i,:],v2[i,:])
    [r2[i+1,:],v2[i+1,:]]=RK4(t[i],r2[i,:],v2[i,:],h,1,r1[i,:],v2[i,:])
        
    KE[i+1] = KINETICE(1,v1[i+1,:])
    PE[i+1] = P_E(1,r1[i+1,:])
    A_M[i+1] = AngMomentum(1,r1[i+1,:],r2[i+1,:],r3[i+1,:],v1[i+1,:],v2[i+1,:],v3[i+1,:])


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(r1[:,0], r1[:,1], r1[:,2], c='b')
ax.set_xlabel('x-position (m)')
ax.set_ylabel('y-position (m)')
ax.set_zlabel('z-position')

plt.show()


