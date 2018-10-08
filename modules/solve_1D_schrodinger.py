import numpy as np
import scipy as sp
from scipy.linalg import circulant
from matplotlib import pyplot as plt
import os

class setup_1d_schrodinger(object):
	def __init__(self,N=500,pad=0.05,V0=1e4,figpath="./figures/",periodic_bc=False):
		self.N=N
		self.pad=pad
		self.V0=V0
		self.figpath=figpath
		self.x=np.linspace(-1.,1.,self.N)
		self.h=np.mean(self.x[1:]-self.x[:-1])
		self.potential_type=["SW","PW","VW"]
		self.get_d2_operator(periodic_bc)

	def get_potential(self,wtype="SW"):
		self.V=np.zeros(self.N,float)
		if wtype=="SW":
			self.V[self.x<self.x[int(np.ceil(self.pad*self.N))]]=1.
			self.V[self.x>=self.x[self.N-int(np.ceil(self.pad*self.N))]]=1.
			self.V=self.V0*self.V/max(self.V)-self.V0
			self.fig_prefix="1D_SW"
		elif wtype=="PW":
			self.V=self.x**2.
			self.V=self.V0*self.V/max(self.V)-self.V0
			self.fig_prefix="1D_PW"
		elif wtype=="VW":
			sl=30.
			self.V[self.x<self.x[self.N-int(np.ceil(self.pad*self.N))]]=-sl*self.x[self.x<self.x[self.N-int(np.ceil(self.pad*self.N))]]
			self.V[self.x>=self.x[self.N-int(np.ceil(self.pad*self.N))]]=max(self.V)
			self.V=self.V0*self.V/max(self.V)-self.V0
			self.fig_prefix="1D_VW"

	def get_eigen(self):
		Ov=np.diag(self.V)
		self.En,self.psi=np.linalg.eigh(-self.O+Ov)

	def get_d2_operator(self,periodic_bc="False"):
		if periodic_bc:
			sO=np.zeros(np.size(self.x),float)
			sO[0]=-2 ; sO[1]=1 ;  sO[self.N-1]=1
			self.O=circulant(sO)/self.h/self.h
		else:
			self.O=np.zeros((self.N,self.N),float)
			for i,tx1 in enumerate(self.x):
				for j,tx2 in enumerate(self.x):
					if tx1==tx2:
						self.O[i,j]=-2./self.h/self.h
					if abs(abs(tx1-tx2)/self.h-1.)<1.e-10:
						self.O[i,j]=1./self.h/self.h

	def gen_movie(self,Num_En=30):
		cmd="mkdir " + self.figpath ; os.system(cmd)
		
		plt.ioff()
		for i in range(Num_En):
			plt.figure()
			plt.plot(self.x,self.psi[:,i]/max(self.psi[:,i]),label="$\psi_{" + str(i) + "}(x)$" )
			plt.plot(self.x,self.V/max(abs(self.V)),"k--",label="$V(x)/V_0$",alpha=0.6)
			plt.title("$E_{" + str(i) + "}=$" + str(round(self.En[i],3)))
			plt.legend(loc=0)
			plt.ylim(-1.5,1.5)
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
