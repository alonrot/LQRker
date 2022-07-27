from abc import ABC, abstractmethod
import tensorflow as tf
import tensorflow_probability as tfp
import math
import pdb
import numpy as np

from lqrker.models.rrtp import ReducedRankStudentTProcessBase
from lqrker.spectral_densities.base import SpectralDensityBase
from lqrker.utils.parsing import get_logger
logger = get_logger(__name__)


class MultiObjectiveRRTPRegularFourierFeatures():

	def __init__(self, dim: int, cfg: dict, spectral_density: SpectralDensityBase, Xtrain, Ytrain):

		"""
		
		Initialize "dim" number of RRTPRegularFourierFeatures() models, one per output channel		
		"""

		self.dim_in = dim
		self.dim_out = dim

		# We use a single spectral density instance for all the models. For that instance, we compute the needed frequencies
		# and use them throughout all the models
		self.spectral_density = spectral_density 

		self.rrgpMO = [None]*self.dim_out
		for ii in range(self.dim_out):
			self.rrgpMO[ii] = RRTPRegularFourierFeatures(dim=self.dim_in,cfg=cfg.gpmodel,spectral_density=self.spectral_density)
			self.rrgpMO[ii].select_output_dimension(dim_out_ind=ii)

		self.update_model(X=Xtrain,Y=Ytrain)

		self.X = Xtrain
		self.Y = Ytrain


	def update_model(self,X,Y):
		for ii in range(self.dim_out):
			self.rrgpMO[ii].update_model(X,Y[:,ii:ii+1]) # Update model indexing the target outputs at the corresponding dimension

	def predict_at_locations(self,xpred,from_prior=False):
		"""

		xpred: [Npoints,self.dim]

		"""

		MO_mean_pred = [None]*self.dim_out
		MO_std_pred = [None]*self.dim_out

		# Compute predictive moments:
		for ii in range(self.dim_out):
			MO_mean_pred[ii], cov_pred = self.rrgpMO[ii].predict_at_locations(xpred)
			MO_std_pred[ii] = tf.sqrt(tf.linalg.diag_part(cov_pred))

		return MO_mean_pred, MO_std_pred


	def sample_path_from_predictive(self,xpred,Nsamples,from_prior=False):
		"""

		xpred: [Npoints,self.dim]
		samples_xpred: [Npoints,]*self.dim_out

		"""

		samples_xpred = [None]*self.dim_out
		for ii in range(self.dim_out):
			samples_xpred[ii] = self.rrgpMO[ii].sample_path_from_predictive(xpred,Nsamples,from_prior)

		return samples_xpred


	def get_sample_path_callable(self,Nsamples,from_prior=False):

		def nonlinfun_sampled_callable(x):

			out = []
			for ii in range(self.dim_out):
				fx_ii = self.rrgpMO[ii].get_sample_path_callable(Nsamples,from_prior)
				out_ii = fx_ii(x) # [Npoints,Nsamples]
				out += [out_ii]

			return tf.stack(out,axis=1) # [Npoints,self.dim_out,Nsamples]

		return nonlinfun_sampled_callable

	def sample_state_space_from_prior_recursively(self,x0,x1,traj_length,sort=False,plotting=False):
		"""

		Pass two initial latent values
		x0: [1,self.dim]
		x1: [1,self.dim]

		The GP won't be training during sampling, i.e., we won't call self.train_model()


		return:
		xsamples_X: [Npoints,self.dim_out,Nsamples]
		xsamples_Y: [Npoints,self.dim_out,Nsamples]

		"""

		# xmin = -6.
		# xmax = +3.
		# Ndiv = 201
		# xpred = tf.linspace(xmin,xmax,Ndiv)
		# xpred = tf.reshape(xpred,(-1,1))

		Nsamples = 1
		assert Nsamples == 1
		fx = self.get_sample_path_callable(Nsamples=Nsamples)

		# yplot_true_fun = self.spectral_density._nonlinear_system_fun(xpred)
		# yplot_sampled_fun = fx(xpred)

		assert traj_length > 2

		Xtraining = tf.identity(self.X) # Copy tensor
		Ytraining = tf.identity(self.Y) # Copy tensor
		Xtraining_and_new = tf.concat([Xtraining,x0],axis=0)
		Ytraining_and_new = tf.concat([Ytraining,x1],axis=0)
		self.update_model(X=Xtraining_and_new,Y=Ytraining_and_new)

		if plotting:
			hdl_fig, hdl_splots = plt.subplots(2,1,figsize=(12,8),sharex=True)
			hdl_fig.suptitle(r"Kink function simulation $x_{t+1} = f(x_t) + \varepsilon$"+", kernel: {0}".format("kink"),fontsize=fontsize_labels)

		xsamples = np.zeros((traj_length,self.dim_out,Nsamples),dtype=np.float32)
		xsamples[0,:,0] = x0
		xsamples[1,:,0] = x1
		resample_mvt0 = True
		for ii in range(1,traj_length-1):

			xsample_tp = tf.convert_to_tensor(value=xsamples[ii:ii+1,:,0],dtype=np.float32)

			if ii > 1: 
				resample_mvt0 = False

			# xsamples[ii+1,:] = self.sample_path_from_predictive(xpred=xsample_tp,Nsamples=Nsamples,resample_mvt0=resample_mvt0)
			xsamples[ii+1,...] = fx(xsample_tp)

			Xnew = tf.convert_to_tensor(value=xsamples[0:ii,:,0],dtype=np.float32)
			Ynew = tf.convert_to_tensor(value=xsamples[1:ii+1,:,0],dtype=np.float32)

			Xtraining_and_new = tf.concat([Xtraining,Xnew],axis=0)
			Ytraining_and_new = tf.concat([Ytraining,Ynew],axis=0)
			self.update_model(X=Xtraining_and_new,Y=Ytraining_and_new)

			if plotting:
				# Plot what's going on at each iteration here:
				MO_mean_pred, cov_pred = self.predict_at_locations(xpred)
				MO_std_pred = tf.sqrt(tf.linalg.diag_part(cov_pred))
				hdl_splots[0].cla()
				hdl_splots[0].plot(xpred,MO_mean_pred,linestyle="-",color="b",lw=3)
				hdl_splots[0].fill_between(xpred[:,0],MO_mean_pred - 2.*MO_std_pred,MO_mean_pred + 2.*MO_std_pred,color="cornflowerblue",alpha=0.5)
				hdl_splots[0].plot(Xnew[:,0],Ynew[:,0],marker=".",linestyle="--",color="gray",lw=0.5,markersize=5)
				# hdl_splots[0].set_xlim([xmin,xmax])
				hdl_splots[0].set_ylabel(r"$x_{t+1}$",fontsize=fontsize_labels)
				# hdl_splots[0].plot(xpred,yplot_true_fun,marker="None",linestyle="-",color="grey",lw=1)
				# hdl_splots[0].plot(xpred,yplot_sampled_fun,marker="None",linestyle="--",color="r",lw=0.5)

				plt.pause(0.1)
				# input()

		xsamples_X = xsamples[0:-1,:]
		xsamples_Y = xsamples[1::,:]

		if sort:
			assert x0.shape[1] == 1, "This only makes sense in 1D"
			ind_sort = tf.argsort(xsamples_X[:,0],axis=0)
			# pdb.set_trace()
			xsamples_X = xsamples_X[ind_sort,:]
			xsamples_Y = xsamples_Y[ind_sort,:]

		# Go back to the dataset at it was:
		self.update_model(X=Xtraining,Y=Ytraining)

		return xsamples_X, xsamples_Y


class RRTPRegularFourierFeatures(ReducedRankStudentTProcessBase):
	"""

	
	This model assumes a dim-dimensional input and a scalar output.


	
	As described in [1, Sec. 2.3.3], which is analogous to [2].

	[1] Hensman, J., Durrande, N. and Solin, A., 2017. Variational Fourier Features for Gaussian Processes. J. Mach. Learn. Res., 18(1), pp.5537-5588.
	[2] Solin, A. and Särkkä, S., 2020. Hilbert space methods for reduced-rank Gaussian process regression. Statistics and Computing, 30(2), pp.419-446.


	TODO:
	3) Add other hyperparameters as trainable variables to the optimization
	4) Refactor all this in different files
	5) How can we infer the dominant frquencies from data? Can we compute S(w|Data) ?
	"""

	def __init__(self, dim: int, cfg: dict, spectral_density: SpectralDensityBase, dim_out_int=0):

		super().__init__(dim,cfg,spectral_density,dim_out_int)

	def get_features_mat(self,X):
		"""

		X: [Npoints, dim]
		return: PhiX: [Npoints, Nfeat]
		"""

		# pdb.set_trace()
		WX = X @ tf.transpose(self.W_samples) # [Npoints, Nfeat]
		harmonics_vec = tf.math.cos(WX + tf.transpose(self.phi_samples_vec)) # [Npoints, Nfeat]
		harmonics_vec_scaled = harmonics_vec * tf.reshape(self.S_samples_vec,(1,-1)) # [Npoints, Nfeat]

		return harmonics_vec_scaled

	def get_cholesky_of_cov_of_prior_beta(self):
		return tf.linalg.diag(tf.math.sqrt(tf.reshape(self.S_samples_vec,(-1)) + self.get_noise_var()))
		
	def get_Sigma_weights_inv_times_noise_var(self):
		return self.get_noise_var() * tf.linalg.diag(1./tf.reshape(self.S_samples_vec,(-1)))

	def get_logdetSigma_weights(self):
		return tf.math.reduce_sum(tf.math.log(self.S_samples_vec))


class RRTPRandomFourierFeatures(ReducedRankStudentTProcessBase):

	def __init__(self, dim: int, cfg: dict, spectral_density: SpectralDensityBase, dim_out_int=0):

		super().__init__(dim,cfg,spectral_density,dim_out_int)

	def get_features_mat(self,X):
		"""

		X: [Npoints, dim]
		return: PhiX: [Npoints, Nfeat]
		"""

		u_samples = tfp.distributions.Uniform(low=0.0, high=2.*math.pi).sample(sample_shape=(1,self.Nfeat))
		WX = tf.transpose(self.W_samples @ tf.transpose(X)) # [Npoints, Nfeat]
		harmonics_vec = tf.math.cos(WX + u_samples) # [Npoints, Nfeat]

		return harmonics_vec

	def get_Sigma_weights_inv_times_noise_var(self):
		return self.get_noise_var() * (self.Nfeat/self.prior_var) * tf.eye(self.Nfeat)

	def get_cholesky_of_cov_of_prior_beta(self):
		return tf.eye(self.Nfeat)*tf.math.sqrt((self.prior_var/self.Nfeat + self.get_noise_var()))

	def get_logdetSigma_weights(self):
		return self.Nfeat*tf.math.log(self.prior_var)

class RRTPSarkkaFeatures(ReducedRankStudentTProcessBase):
	"""

	TODO: Incompatible with current implementation of SpectralDensityBase.
	Needs refactoring following RRTPRandomFourierFeatures
	"""

	def __init__(self, dim: int, cfg: dict):

		raise NotImplementedError("Needs refactoring following RRTPRandomFourierFeatures")

		super().__init__(dim,cfg)

		self.Nfeat = cfg.hyperpars.weights_features.Nfeat
		self.L = cfg.hyperpars.L
		self.jj = tf.reshape(tf.range(1,self.Nfeat+1,dtype=tf.float32),(-1,1,1)) # [Nfeat, 1, 1]

		# Spectral density to be used:
		# self.spectral_density = MaternSpectralDensity(cfg.spectral_density,dim)

	def _get_eigenvalues(self):
		"""

		Eigenvalues of the laplace operator
		"""

		Lstack = tf.stack([self.L]*self.Nfeat) # [Nfeat, dim]
		jj = tf.reshape(tf.range(1,self.Nfeat+1,dtype=tf.float32),(-1,1)) # [Nfeat, 1]
		# pdb.set_trace()
		Ljj = (math.pi * jj / (2.*Lstack))**2 # [Nfeat, dim]

		return tf.reduce_sum(Ljj,axis=1) # [Nfeat,]

	def get_features_mat(self,X):
		"""

		X: [Npoints, dim]
		return: PhiX: [Npoints, Nfeat]
		"""
		
		xx = tf.stack([X]*self.Nfeat) # [Nfeat, Npoints, dim]
		# jj = tf.reshape(tf.range(self.Nfeat,dtype=tf.float32),(-1,1,1)) # [Nfeat, 1, 1]
		# pdb.set_trace()
		feat = 1/tf.sqrt(self.L) * tf.sin( math.pi*self.jj*(xx + self.L)/(2.*self.L) ) # [Nfeat, Npoints, dim]
		return tf.transpose(tf.reduce_prod(feat,axis=-1)) # [Npoints, Nfeat]

	def get_Sigma_weights_inv_times_noise_var(self):
		omega_in = tf.sqrt(self._get_eigenvalues())
		S_vec = self.spectral_density.unnormalized_density(omega_in)
		ret = self.get_noise_var() * tf.linalg.diag(1./S_vec)

		# if tf.math.reduce_any(tf.math.is_nan(ret)):
		# 	pdb.set_trace()

		return ret

	def get_logdetSigma_weights(self):
		omega_in = tf.sqrt(self._get_eigenvalues())
		S_vec = self.spectral_density.unnormalized_density(omega_in)
		return tf.reduce_sum(tf.math.log(S_vec))

class RRTPQuadraticFeatures(ReducedRankStudentTProcessBase):
	"""

	TODO: Incompatible with current implementation of SpectralDensityBase.
	Needs refactoring following RRTPRandomFourierFeatures
	"""

	def __init__(self, dim: int, cfg: dict):

		raise NotImplementedError("Needs refactoring following RRTPRandomFourierFeatures")

		super().__init__(dim,cfg)

		# Elements of the diagonal matrix Lambda:
		# TODO: Test the line below
		self.log_diag_vals = self.add_weight(shape=(Nfeat,), initializer=tf.keras.initializers.Zeros(), trainable=True,name="log_diag_vars")

	def get_features_mat(self,X):
		"""
		X: [Npoints, in_dim]
		return: PhiX: [Npoints, Nfeat]
		"""

		Npoints = X.shape[0]
		SQRT2 = math.sqrt(2)

		if self.dim == 1:
			PhiX = tf.concat([ tf.ones((Npoints,1)) , SQRT2*X , X**2 ],axis=1)
		else:
			PhiX = tf.concat([ tf.ones((Npoints,1)) , SQRT2*X , SQRT2*tf.math.reduce_prod(X,axis=1,keepdims=True) , X**2 ],axis=1)

		assert cfg.weights_features.Nfeat == PhiX.shape[1], "Basically, for quadratic features the number of features is given a priori; the user cannot choose"

		return PhiX # [Npoints, Nfeat]

	def get_Sigma_weights_inv_times_noise_var(self):
		"""
		The Lambda matrix depends on the choice of features

		TODO: Test this function
		"""
		return self.get_noise_var()*tf.linalg.diag(tf.exp(-self.log_diag_vals))

	def get_logdetSigma_weights(self):
		# TODO: Test this function
		return tf.reduce_sum(self.log_diag_vals)

class RRTPLQRfeatures(ReducedRankStudentTProcessBase):
	"""

	TODO: Incompatible with current implementation of SpectralDensityBase.
	Needs refactoring following RRTPRandomFourierFeatures
	"""

	def __init__(self, dim: int, cfg: dict):

		raise NotImplementedError("Needs refactoring following RRTPRandomFourierFeatures")

		super().__init__(dim,cfg)

		# Get parameters:
		nu = cfg.hyperpars.nu
		Nsys = cfg.hyperpars.weights_features.Nfeat # Use as many systems as number of features

		# TODO: Test the line below
		self.log_diag_vals = self.add_weight(shape=(Nsys,), initializer=tf.keras.initializers.Zeros(), trainable=True,name="log_diag_vars")
		
		from lqrker.objectives.lqr_cost_chi2 import LQRCostChiSquared
		self.lqr_cost = LQRCostChiSquared(dim_in=dim,cfg=cfg,Nsys=Nsys)
		print("Make sure we're NOT using noise in the config file...")
		pdb.set_trace()

	def get_features_mat(self,X):
		"""
		X: [Npoints, in_dim]
		return: PhiX: [Npoints, Nfeat]
		"""

		cost_values_all = self.lqr_cost.evaluate(X,add_noise=False,verbo=True)

		return cost_values_all

	def get_Sigma_weights_inv_times_noise_var(self):
		# TODO: Test this function
		return self.get_noise_var()*tf.linalg.diag(tf.exp(-self.log_diag_vals))

	def get_logdetSigma_weights(self):
		# TODO: Test this function
		return tf.reduce_sum(self.log_diag_vals)

