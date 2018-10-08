import numpy as np
import scipy as sp
from scipy.linalg import circulant
import itertools
from matplotlib import pyplot as plt
import os

class setup_2d_schrodinger(object):
	def __init__(self,N=50,V0=1e4,pad=0.05,figpath="./figures/"):
		self.N=N
		self.V0=V0
		self.pad=pad
		self.figpath=figpath
		self.x=np.linspace(-1.,1.,self.N)
		self.y=np.linspace(-1.,1.,self.N)
		self.h=np.mean(self.x[1:]-self.x[:-1])
		self.cidx=[t for t in itertools.product(np.arange(self.N),np.arange(self.N))]
		self.coord=[t for t in itertools.product(self.x,self.y)]
		self.potential_type=["SW","CW","PW","HW","MP"]
		self.get_d2_operator()

	def get_potential(self,wtype="HW"):
		if wtype=="SW":
			self.square_well()
			self.fig_prefix="2D_SW"
		elif wtype=="CW":
			self.circular_well()
			self.fig_prefix="2D_CW"
		elif wtype=="PW":
			self.parabolic_well()
			self.fig_prefix="2D_PW"
		elif wtype=="HW":
			self.hexagonal_well()
			self.fig_prefix="2D_HW"
		elif wtype=="MP":
			self.medulung_potential()
			self.fig_prefix="2D_MP"

	def get_eigen(self):
		Ov=np.diag(self.V)
		self.En,temp_psi=np.linalg.eigh(-self.O+Ov)
		self.psi=np.zeros((np.size(self.En),self.N,self.N),float)
		for i in range(np.size(self.En)):
			for j,c in enumerate(self.cidx):
				self.psi[i,c[0],c[1]]=temp_psi[j,i]

	def get_d2_operator(self,wtype="SQW"):
		self.O=np.zeros((self.N**2,self.N**2),float)
		for i,t1 in enumerate(self.coord):
			for j,t2 in enumerate(self.coord):
				if np.all(t1==t2):
					self.O[i,j]=-4./self.h/self.h
				if t1[0]!=t2[0] and t1[1]!=t2[1]:
					self.O[i,j]=0
				elif abs(abs(t1[0]-t2[0])/self.h-1.)<1.e-10:
					self.O[i,j]=1./self.h/self.h
				elif abs(abs(t1[1]-t2[1])/self.h-1.)<1.e-10:
					self.O[i,j]=1./self.h/self.h

	def square_well(self):
		self.V=np.zeros(self.N*self.N,float)
		for i,xy in enumerate(self.coord):
			if xy[0]<self.x[int(np.ceil(self.pad*self.N))] or xy[0]>=self.x[self.N-int(np.ceil(self.pad*self.N))]:
				self.V[i]=1.
			if xy[1]<self.y[int(np.ceil(self.pad*self.N))] or xy[1]>=self.y[self.N-int(np.ceil(self.pad*self.N))]:
				self.V[i]=1.
		self.V=self.V0*self.V/max(self.V)-self.V0
		
	def circular_well(self):
		self.V=np.zeros(self.N*self.N,float)
		rad=self.x[self.N-int(np.ceil(self.pad*self.N))]
		for i,xy in enumerate(self.coord):
			rxy=np.sqrt(xy[0]**2. + xy[1]**2.)
			if rxy<=rad:
				self.V[i]=0.
			else:
				self.V[i]=1.
		self.V=self.V0*self.V/max(self.V)-self.V0

	def parabolic_well(self):
		self.V=np.zeros(self.N*self.N,float)
		for i,xy in enumerate(self.coord):
			self.V[i]=xy[0]**2. + xy[1]**2.
		self.V=self.V0*self.V/max(self.V)-self.V0

	def hexagonal_well(self):
		self.V=np.zeros(self.N*self.N,float)
		def hexagon(pos):
			s=self.x[self.N-int(np.ceil(self.pad*self.N))]
			x, y = map(abs, pos)
			return y < 3**0.5 * min(s - x, s / 2)
		for i,xy in enumerate(self.coord):
			if not(hexagon(xy)):
				self.V[i]=1.
		self.V=self.V0*self.V/max(self.V)-self.V0
		
	def medulung_potential(self):
		self.V=np.zeros(self.N*self.N,float)
		for i,xy1 in enumerate(self.cidx):
			for j,xy2 in enumerate(self.cidx):
				if j!=i:
					x1=self.coord[i][0] ; y1=self.coord[i][1] ; s1=np.sum(xy1)
					x2=self.coord[j][0] ; y2=self.coord[j][1] ; s2=np.sum(xy2)
					d=np.sqrt((x1-x2)**2. + (y1-y2)**2.)
					self.V[i]=self.V[i] + ((-1.)**s1)*((-1.)**s2)/d

	def return_2D_V(self):
		temp_V=np.zeros((self.N,self.N),float)
		for i,c in enumerate(self.cidx):
			temp_V[c]=self.V[i]
		return temp_V

	def gen_movie(self,Num_En=30):
		cmd="mkdir " + self.figpath ; os.system(cmd)

		plt.ioff()
		for i in range(Num_En+1):
			plt.figure()
			if i==0:
				temp_V=self.return_2D_V()
				temp_V=temp_V/np.max(abs(temp_V))
				plt.imshow(temp_V,origin="lower") ; plt.colorbar()
				plt.title("$V(x,y)/V_0$")
				plt.savefig(self.figpath  + self.fig_prefix  + "_eigenstates" + str(i).zfill(2) + ".jpeg")
			else:
				plt.imshow(self.psi[i-1,:,:]/np.max(abs(self.psi[i-1,:,:])),origin="lower") ; plt.colorbar()
				plt.title("$E_{" + str(i-1) + "}=$" + str(round(self.En[i-1],3)))
				plt.savefig(self.figpath  + self.fig_prefix  + "_eigenstates" + str(i).zfill(2) + ".jpeg")
			plt.close()
		workdir=os.getcwd()
		os.chdir(self.figpath)
		cmd="convert -quality 99 -density 150 -delay 120 -loop 0 "
		cmd=cmd + self.fig_prefix + "_eigenstate*.jpeg "
		cmd=cmd + self.fig_prefix + "_eigenstates.gif"
		#print cmd
		os.system(cmd)
		os.system("rm *.jpeg")
		os.chdir(workdir)


