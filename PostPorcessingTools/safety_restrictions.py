
#Simulation name:  
name_Simu='Tunis_Nice'

#Offset at boundary plots (in degrees):
offset=0.2

#Plot parametric rolling
plot_pr=1 #Yes=1 ; No=0

#Plot Surfriding
plot_sr=1 #Yes=1 ; No=0

# END OF USER INPUTS   #######################

import numpy as np
from func_postprocess import *
import matplotlib.pyplot as plt
import math as math
import os
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker 

arx = '../out/'+name_Simu+'.npz'
if os.path.exists(arx) == False:
    print('Simulation '+arx+' not exist')
    raise SystemExit
    
dat = np.load(arx)
LonMin=dat['arr_0'][0]
LonMax=dat['arr_0'][1]
LatMin=dat['arr_0'][2]
LatMax=dat['arr_0'][3]
v0=dat['arr_0'][4]
inc=dat['arr_0'][5] 
nodIni=int(dat['arr_0'][6])
nodEnd=int(dat['arr_0'][7])
t_ini=int(dat['arr_0'][8])
time_res=int(dat['arr_0'][9])
WEN_form=int(dat['arr_0'][10])
Lbp=dat['arr_0'][11]
DWT=dat['arr_0'][12]
hs=dat['arr_1']
fp=dat['arr_2']
dir=dat['arr_3']
L_Trip=dat['arr_4']
L_TripFix=dat['arr_5']
L_CostTrip=dat['arr_6']
L_ConsCostTrip=dat['arr_7']
Cost_Min=dat['arr_8']
ARX=dat['arr_9']

inc=inc/60.0    
#Re-build Mesh:
Nx=int(np.floor((LonMax-LonMin)/inc)+2)
Ny=int(np.floor((LatMax-LatMin)/inc)+2)
tira_lon=[]
for i in range(Nx):
    tira_lon.append(LonMin+i*inc)
tira_lat=[]
for j in range(Ny):  
    tira_lat.append(LatMin+j*inc)
nodes=np.zeros((Nx*Ny,2))
#print( ' Nx = {:6d} ---   Ny = {:4d}\n'.format(Nx,Ny))
#print('longituds    {:8.3f}    -----   {:8.3f} \n'.format(tira_lon[0],tira_lon[-1]))
#print('latituds     {:8.3f}    -----   {:8.3f} \n'.format(tira_lat[0],tira_lat[-1]))
for j in range(Ny):   
    for i in range(Nx):
        nodes[Nx*j +i,0]=tira_lon[i]
        nodes[Nx*j +i,1]=tira_lat[j]
inc=inc*60
print('Mesh Re-built')
vmax=np.nanmax(hs)  # maxix valor de hs posible en la simu (pel colorbar)

Tr=20
epsi=0.1
L_sr_bt=[]
L_pr=[]

#for i in range(4):
for i in range(len(L_Trip)-1):    
    Ni=L_Trip[i]
    Ne=L_Trip[i+1]
    loni,lati = nodes[Ni,0], nodes[Ni,1]
    lone,late = nodes[Ne,0], nodes[Ne,1]
 #   print(i,Ni,Ne)
    for k in range(2):
        if time_res==1:
            ti=np.rint(L_CostTrip[i+k])
        else:
           (a,b)=np.divmod(L_CostTrip[i+k],time_res) 
           if b>time_res/2:
               ti=a+1
           else:
               ti=a
        ti=int(ti)
        if k==0:
            hi=hs[Ni,ti]
            diri=dir[Ni,ti]
            fpi=fp[Ni,ti]
            rumb=rumIni(loni,lati,lone,late)
        else:
            hi=hs[Ne,ti]
            diri=dir[Ne,ti]
            fpi=fp[Ne,ti]
            rumb=rumEnd(loni,lati,lone,late)
        angEnc=ang_encounter(rumb,diri)
        v=1.8*np.sqrt(Lbp)/ np.cos(np.deg2rad(180-angEnc))
        if angEnc > 145 and angEnc<225 and v0>v:
            if k==0:
                print("Unstable sr-bt in  node ini  ", Ni)
                L_sr_bt.append(Ni)
            else:
                print("Unstable sr-bt in  node end  ", Ne)
                L_sr_bt.append(Ne)                
        Tw=fpi
        Te=3*Tw*Tw/(3*Tw+v0*np.cos(np.deg2rad(angEnc)))
#        print(Te,Tw,angEnc)       
        if (np.abs(Te-Tr)<epsi*Tr ) or (np.abs(2*Te-Tr)<epsi*Tr ):
            if k==0:
                print("Unstable parametric rolling in node ini ", Ni)
                L_pr.append(Ni)
            else:
                print("Unstable parametric rolling in node end", Ne)
                L_pr.append(Ne)      

LamC=ccrs.LambertConformal(central_longitude=(LonMin+LonMax)/2,central_latitude=(LatMin+LatMax)/2)
geo=ccrs.Geodetic()

res_ldc='10m'
extent=[LonMin-offset,LonMax+offset, LatMin-offset, LatMax+offset]
fig = plt.figure(figsize=(12,6))    
ax=plt.subplot(1,1,1, projection=LamC)
#ax.set_title('Safety restrictions: ' + name_Simu) #+ ' make_plot temps = {}'.format(i))
lon=nodes[L_Trip[0:-1],0]
lat=nodes[L_Trip[0:-1],1]
lonc=nodes[L_TripFix[0:-1],0]
latc=nodes[L_TripFix[0:-1],1]
#ax.plot(lonc,latc,'orange',transform=geo,label='Minimum distance route') 
ax.plot(lon,lat,'m',transform=geo,label='Optimized route') 
ax.set_extent(extent)
ax.plot([nodes[nodIni,0]],[nodes[nodIni,1]],'^b',transform=geo,label='Departure')
ax.plot([nodes[nodEnd,0]],[nodes[nodEnd,1]],'^r',transform=geo,label='Arrival') 
ax.gridlines()
ax.coastlines(resolution=res_ldc)
ax.add_feature(cfeature.LAND)
lprx=nodes[L_pr[:],0]
lpry=nodes[L_pr[:],1]
lsrx=nodes[L_sr_bt[:],0]
lsry=nodes[L_sr_bt[:],1]
if plot_pr==1:
    ax.plot(lprx,lpry,'oy',transform=geo,label='Param. rolling')
if plot_sr==1:
    ax.plot(lsrx,lsry,'og',transform=geo,label='Surfriding')
ax.legend(loc='best') 
ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

name_fig='../out/Unstable_motions'+name_Simu+'.png'
plt.savefig(name_fig,dpi=300) #, bbox_inches='tight')
plt.show()



