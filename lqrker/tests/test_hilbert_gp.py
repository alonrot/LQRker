import tensorflow as tf
import pdb
import math
import matplotlib.pyplot as plt


class ReducedRankBayesianLinearRegression:
	"""
	"""
	def __init__(self, dim, Nfeat, L, sigma_n):
		"""
		
		dim: Dimensionality of the input space
		Nfeat: Number of features
		L: Half Length of the hypercube. Each dimension has length [-L, L]
		"""
		self.dim = dim
		self.Nfeat = Nfeat
		self.L = tf.constant([L]*dim)
		self.jj = tf.reshape(tf.range(1,self.Nfeat+1,dtype=tf.float32),(-1,1,1)) # [Nfeat, 1, 1]
		self.eigvals = self._get_eigenvalues()
		self.sigma_n = sigma_n
		self.sigma2_n = sigma_n**2

		self.spectral_density = lambda w: self.spectral_density_matern(w)
		# self.spectral_density = lambda w: 1.0*self.spectral_density_SE(w)

	def update_dataset(self,X,Y):
		self.X = X

		if Y.ndim == 1:
			self.Y = tf.reshape(Y,(-1,1))
		else:
			assert Y.ndim == 2
			self.Y = Y

		self._update_features()

	def add2dataset(self,xnew,ynew):
		pass

	def spectral_density_matern(self,omega_vec):
		# TODO: Make sure that here we are required to enter a vetor, not a scalar (see appendix from Sarkaa)
		# TODO: Look at eq. (4.15) from Rasmussen. It actually depends on the dimensionality D, and it's not multidimensional!!! :(

		# Using now the N-dimensional formulation from Rasmussen

		p = 3
		nu = p + 0.5
		ls = 0.5
		lambda_val = tf.sqrt(2*nu)/ls

		# S_vec = ((2*tf.sqrt(math.pi)*tf.exp(tf.math.lgamma(nu+0.5))) / (tf.exp(tf.math.lgamma(nu)))) * lambda_val**(2*nu)/((lambda_val**2 + omega_vec**2)**(nu+0.5))
		const = ((2*tf.sqrt(math.pi))**self.dim)*tf.exp(tf.math.lgamma(nu+0.5*self.dim))*lambda_val**(2*nu) / tf.exp(tf.math.lgamma(nu))
		S_vec = const / ((lambda_val**2 + omega_vec**2)**(nu+self.dim*0.5)) # Using omega directly (Sarkka) as opposed to 4pi*s (rasmsusen)

		# print("S_vec:",S_vec)

		# from scipy.special import gamma
		# import numpy as np
		# S_vec = self.sigma2_n * ((2*np.sqrt(np.pi)*gamma(nu+0.5)) / (gamma(nu))) * lambda_val**(2*nu)/((lambda_val**2 + omega_vec**2)**(nu+0.5))
		# print("S_vec np:",S_vec)

		# pdb.set_trace()

		return S_vec


	def spectral_density_SE(self,omega_vec):

		ls = 0.1

		const = (tf.sqrt(2*math.pi)*ls)**self.dim
		S_vec = const * tf.exp( -2*math.pi**2 * ls**2 * omega_vec**2 )

		return S_vec

	def _update_features(self):
		"""

		Cache the expensive operation
		"""

		self.PhiX = self.get_features_mat(self.X)
		Lambda_inv_times_noise_var = tf.linalg.diag( self.sigma2_n * 1./self.spectral_density(tf.sqrt(self.eigvals)) )
		self.Lchol = tf.linalg.cholesky(tf.transpose(self.PhiX) @ self.PhiX + Lambda_inv_times_noise_var ) # Lower triangular

	def _get_eigenvalues(self):

		Lstack = tf.stack([self.L]*self.Nfeat) # [Nfeat, dim]
		jj = tf.reshape(tf.range(1,self.Nfeat+1,dtype=tf.float32),(-1,1)) # [Nfeat, 1]
		Ljj = (math.pi * jj / (2.*Lstack))**2 # [Nfeat, dim]

		return tf.reduce_sum(Ljj,axis=1) # [Nfeat,]

	def get_features_mat(self,x):
		
		xx = tf.stack([x]*self.Nfeat) # [Nfeat, Npoints, dim]
		# jj = tf.reshape(tf.range(self.Nfeat,dtype=tf.float32),(-1,1,1)) # [Nfeat, 1, 1]
		# pdb.set_trace()
		feat = 1/tf.sqrt(self.L) * tf.sin( math.pi*self.jj*(xx + self.L)/(2.*self.L) ) # [Nfeat, Npoints, dim]
		return tf.transpose(tf.reduce_prod(feat,axis=-1)) # [Npoints, Nfeat]


	def get_predictive_moments(self,xpred):
		
		# MATLAB code from Särkkä:
		# % Eigenfunctions
		# Phit   = eigenfun(NN,xt);
		# Phi    = eigenfun(NN,x); 
		# PhiPhi = Phi'*Phi;        % O(nm^2)
		# Phiy   = Phi'*y;
		# lambda = eigenval(NN)';

		#   % Solve GP with optimized hyperparameters and 
		# % return predictive mean and variance 
		# k = S(sqrt(lambda),lengthScale,magnSigma2);
		# L = chol(PhiPhi + diag(sigma2./k),'lower'); 
		# Eft = Phit*(L'\(L\Phiy));
		# Varft = sigma2*sum((Phit/L').^2,2); 

		# % Notice boundaries
		# Eft(abs(xt) > Lt) = 0; Varft(abs(xt) > Lt) = 0;

		Phi_pred = self.get_features_mat(xpred)
		
		# Get mean:
		PhiXY = tf.transpose(self.PhiX) @ self.Y
		mean_pred = Phi_pred @ tf.linalg.cholesky_solve(self.Lchol, PhiXY) # In the Sarkka paper Phi_pred is transposed, but it should be wrong...

		# Get covariance:
		cov_pred = self.sigma2_n * Phi_pred @ tf.linalg.cholesky_solve(self.Lchol, tf.transpose(Phi_pred))

		return tf.squeeze(mean_pred), cov_pred


def cost(X):
	return tf.reduce_sum( X**2 , axis = 1)

if __name__ == "__main__":
	
	dim = 1

	# Hypercube domain:
	L = 4.0
	Lred = 0.5*L # Training data should be in a reduced domain due to the Dirichlet boundary conditions
	Nfeat = 200 # Number of features

	# R-R-BLR:
	rrblr = ReducedRankBayesianLinearRegression(dim=dim,Nfeat=Nfeat,L=4.0,sigma_n=0.1)

	# Evaluate:
	Nevals = 10
	X = tf.random.uniform(shape=(Nevals,dim),minval=-Lred,maxval=Lred)
	Y = cost(X)
	
	rrblr.update_dataset(X,Y)

	# Prediction/test locations:
	Npred = 200
	if dim == 1:
		xpred = tf.reshape(tf.linspace(-L,L,Npred),(-1,1))
	else:
		xpred = tf.random.uniform(shape=(20,dim),minval=-L,maxval=L)

	# Compute predictive moments:
	mean_pred, cov_pred = rrblr.get_predictive_moments(xpred)
	std_pred = tf.sqrt(tf.linalg.diag_part(cov_pred))

	if dim == 1:
		hdl_fig, hdl_splots = plt.subplots(1,1,figsize=(14,10),sharex=True)
		hdl_fig.suptitle("Reduced-rank GP (Särkkä)")
		hdl_splots.plot(xpred,mean_pred)
		fpred_quan_plus = mean_pred + std_pred
		fpred_quan_minus = mean_pred - std_pred
		hdl_splots.fill(tf.concat([xpred, xpred[::-1]],axis=0),tf.concat([fpred_quan_minus,(fpred_quan_plus)[::-1]],axis=0),\
			alpha=.2, fc="blue", ec='None')
		hdl_splots.plot(X,Y,color="black",linestyle="None",markersize=5,marker="o")
		hdl_splots.set_xlim([xpred[0,0],xpred[-1,0]])
		plt.show(block=True)



