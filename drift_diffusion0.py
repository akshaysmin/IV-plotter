#simple drift diffusion model to solve three sets of equations
#(poissons eqn, current equations, continuity equations)
#referance : Optical and electrical study of organic solar cells with a 2D grating anode, Wei E.I. Sha, Wallace C.H. Choy, Yumao Wu, and Weng Cho Chew
#https://www.osapublishing.org/oe/fulltext.cfm?uri=oe-20-3-2572&id=226694#ref24

import matplotlib.pyplot as plt
#import matplotlib.animation as animation
import numpy as np
from mpplot import NBPlot
import time

#parameters

kb  = 1.38064852e-23 #m2 kg s-2 K-1, from google
T   = 300 #Kelvin
q   = 1.60217662e-19 #coulombs
eps0= 8.8541878128e-12	#F⋅m-1
eps = eps0*3.5 #3.5=dielectric constant of ptb7:pcbm blend


#generation rate as got from transfermatrix_copy.py
Gx=np.array([[6.87217437e+21, 6.88774763e+21, 6.90340702e+21, 6.91901898e+21,
  6.93445025e+21, 6.94956810e+21, 6.96424065e+21, 6.97833708e+21,
  6.99172792e+21, 7.00428533e+21, 7.01588330e+21, 7.02639798e+21,
  7.03570790e+21, 7.04369423e+21, 7.05024101e+21, 7.05523543e+21,
  7.05856805e+21, 7.06013304e+21, 7.05982841e+21, 7.05755622e+21,
  7.05322282e+21, 7.04673904e+21, 7.03802040e+21, 7.02698731e+21,
  7.01356523e+21, 6.99768489e+21, 6.97928240e+21, 6.95829946e+21,
  6.93468347e+21, 6.90838768e+21, 6.87937131e+21, 6.84759968e+21,
  6.81304428e+21, 6.77568290e+21, 6.73549967e+21, 6.69248515e+21,
  6.64663638e+21, 6.59795690e+21, 6.54645681e+21, 6.49215275e+21,
  6.43506792e+21, 6.37523209e+21, 6.31268149e+21, 6.24745889e+21,
  6.17961344e+21, 6.10920065e+21, 6.03628233e+21, 5.96092644e+21,
  5.88320704e+21, 5.80320411e+21, 5.72100347e+21, 5.63669660e+21,
  5.55038050e+21, 5.46215749e+21, 5.37213506e+21, 5.28042563e+21,
  5.18714640e+21, 5.09241906e+21, 4.99636963e+21, 4.89912815e+21,
  4.80082849e+21, 4.70160804e+21, 4.60160746e+21, 4.50097043e+21,
  4.39984328e+21, 4.29837480e+21, 4.19671584e+21, 4.09501908e+21,
  3.99343866e+21, 3.89212988e+21, 3.79124889e+21, 3.69095233e+21,
  3.59139702e+21, 3.49273963e+21, 3.39513631e+21, 3.29874238e+21,
  3.20371201e+21, 3.11019783e+21, 3.01835063e+21, 2.92831901e+21,
  2.84024906e+21, 2.75428400e+21, 2.67056391e+21, 2.58922533e+21,
  2.51040102e+21, 2.43421960e+21, 2.36080527e+21, 2.29027751e+21,
  2.22275078e+21, 2.15833426e+21, 2.09713157e+21, 2.03924051e+21,
  1.98475282e+21, 1.93375394e+21, 1.88632279e+21, 1.84253155e+21,
  1.80244546e+21, 1.76612266e+21, 1.73361399e+21, 1.70496286e+21,
  1.68020508e+21, 1.65936877e+21, 1.64247423e+21, 1.62953387e+21,
  1.62055209e+21, 1.61552529e+21, 1.61444176e+21, 1.61728171e+21,
  1.62401725e+21, 1.63461238e+21, 1.64902305e+21, 1.66719720e+21,
  1.68907481e+21, 1.71458801e+21, 1.74366116e+21, 1.77621101e+21,
  1.81214677e+21, 1.85137033e+21, 1.89377640e+21, 1.93925271e+21,
  1.98768019e+21, 2.03893325e+21, 2.09287996e+21, 2.14938233e+21,
  2.20829657e+21, 2.26947339e+21, 2.33275828e+21, 2.39799181e+21,
  2.46500999e+21, 2.53364456e+21, 2.60372337e+21, 2.67507072e+21,
  2.74750773e+21, 2.82085271e+21, 2.89492157e+21, 2.96952818e+21,
  3.04448476e+21, 3.11960232e+21, 3.19469106e+21, 3.26956076e+21,
  3.34402120e+21, 3.41788260e+21, 3.49095602e+21, 3.56305378e+21,
  3.63398989e+21, 3.70358044e+21, 3.77164404e+21, 3.83800221e+21,
  3.90247982e+21, 3.96490544e+21, 4.02511176e+21, 4.08293598e+21,
  4.13822017e+21, 4.19081167e+21, 4.24056339e+21, 4.28733423e+21,
  4.33098934e+21, 4.37140050e+21, 4.40844640e+21, 4.44201294e+21,
  4.47199351e+21, 4.49828925e+21, 4.52080929e+21, 4.53947100e+21,
  4.55420019e+21, 4.56493130e+21, 4.57160755e+21, 4.57418118e+21,
  4.57261346e+21, 4.56687494e+21, 4.55694542e+21, 4.54281411e+21,
  4.52447967e+21, 4.50195019e+21, 4.47524326e+21, 4.44438595e+21,
  4.40941472e+21, 4.37037546e+21, 4.32732332e+21, 4.28032270e+21,
  4.22944704e+21, 4.17477877e+21, 4.11640908e+21, 4.05443780e+21,
  3.98897312e+21, 3.92013146e+21, 3.84803717e+21, 3.77282227e+21,
  3.69462622e+21, 3.61359559e+21, 3.52988372e+21, 3.44365048e+21,
  3.35506184e+21, 3.26428954e+21, 3.17151073e+21, 3.07690759e+21,
  2.98066690e+21, 2.88297965e+21, 2.78404062e+21, 2.68404795e+21,
  2.58320270e+21, 2.48170841e+21, 2.37977063e+21, 2.27759647e+21,
  2.17539417e+21, 2.07337257e+21, 1.97174070e+21, 1.87070728e+21,
  1.77048027e+21, 1.67126637e+21, 1.57327059e+21, 1.47669576e+21,
  1.38174210e+21, 1.28860674e+21, 1.19748327e+21, 1.10856133e+21,
  1.02202614e+21, 9.38058129e+20, 8.56832469e+20, 7.78518714e+20]])#220 elements



grid_len = Gx.size
dx       = 1e-9 #m , from transfermatrix_copy.py 

#plots
gen_rate_fig = plt.figure()
gen_rate_ax1 = gen_rate_fig.add_subplot(1,1,1)
print(max((Gx/10**20)[0]))
x_axis = np.arange(0,Gx.size).reshape(1,Gx.size)
print(x_axis.shape,(Gx/10**20)[0].shape)
gen_rate_ax1.plot(x_axis,(Gx)[0].reshape(1,Gx.size))
gen_rate_ax1.set_title('Generation Rate')

#nx = np.zeros((1,grid_len),dtype=float)#float is actually default ;)
#px = nx.copy()

mun = mup = 10e-7 #m/Vs
Dn  = Dp  = mun*kb*T/q #m2/s
delta_t = .00001 

Q = 1 #exciton dissociation probability
R = 0 #recombination rate

print('parameters used:')
print('Boltzman constant: ',kb,'m2 kg s-2 K-1','\nTemperature: ',T,'Kelvin','\nelectric charge: ',q,'coulombs')
print('electron mobility: ',mun,'m/Vs')
print('hole mobility: ',mup,'m/Vs')
print('electron Diffusion coefficient: ',Dn,'m2/s')
print('hole Diffusion coefficient: ',Dp,'m2/s')

#basic equations



recomb_const  = Q*Gx - (1-Q)*R
desired_distance = 1
icount = 0
icount_max = 100
def step(nx, px, icount, not_converged):
    
	#step2: find Ex
	Ex = np.cumsum(px-nx).reshape(1,grid_len)*(-q/eps)*dx#poisson's equation inegration

	#step3: find grad_nx and grad_px
	grad_nx = np.append((nx[0,1:]-nx[0,:-1])/dx, 0).reshape(1,grid_len)
	grad_px = np.append((px[0,1:]-px[0,:-1])/dx, 0).reshape(1,grid_len)

	#step4: find delta_nx, delta_px
	Jn  = -mun*nx*Ex + Dn*grad_nx #q is not used as it will cancel out in next equation
	delta_nx = np.append(Jn[0,1:]-Jn[0,:-1], 0).reshape(1,grid_len) + recomb_const
	delta_nx*= delta_t

	Jp  = -mup*px*Ex + Dp*grad_px
	delta_px = np.append(Jp[0,1:]-Jp[0,:-1], 0).reshape(1,grid_len) + recomb_const
	delta_px*= delta_t

	#step5: find new nx, px
	nx_new = nx + delta_nx
	px_new = px + delta_px

	#step6: check convergance to stop
	distance_nx = np.linalg.norm(nx_new-nx)
	distance_px = np.linalg.norm(px_new-px)
	distance = distance_nx + distance_px
	print('distance: ',distance_nx,distance_px)
	print('iteration: ',icount)

	if distance <desired_distance or distance == float('+inf'):
		not_converged = False
	else:
		not_converged = True

	nx = nx_new
	px = px_new
	icount += 1

	return nx, px, icount, not_converged

def main():
    #step1: initialise nx and px
    grid_len = Gx.size
    nx = np.random.rand(1,grid_len)
    px = np.random.rand(1,grid_len)
    icount = 0
    print(nx.shape,px.shape)
    #in loop till nx, px converges
    not_converged = True
    pl = NBPlot()#for mutiprocessed plotting
    x_axis = list(range(grid_len))
    while not_converged:
        pl.plot(x_y = (x_axis,list(nx[0])))
        nx, px, icount, not_converged = step(nx, px, icount, not_converged)
        time.sleep(2)
    #after converging nx and px ; find Jn and Jp
    Ex = np.cumsum(px-nx).reshape(1,grid_len)*(-q/eps)*dx
    grad_nx = np.append((nx[0,1:]-nx[0,:-1])/dx, 0).reshape(1,grid_len)
    grad_px = np.append((px[0,1:]-px[0,:-1])/dx, 0).reshape(1,grid_len)
    Jn  = (-mun*nx*Ex + Dn*grad_nx)*q
    Jp  = (-mup*px*Ex + Dp*grad_px)*q

if __name__=='__main__':
    main()


