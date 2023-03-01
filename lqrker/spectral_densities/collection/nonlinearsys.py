import tensorflow as tf
import pdb
import math
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_probability as tfp
from lqrker.utils.parsing import get_logger
from lqrker.spectral_densities.base import SpectralDensityBase
from lqrker.utils.common import CommonUtils
logger = get_logger(__name__)


class NonLinearSystemSpectralDensity(SpectralDensityBase):
# class NonLinearSystemSpectralDensity(tfp.distributions.distribution.AutoCompositeTensorDistribution,SpectralDensityBase):
	
	# @tf.function
	def __init__(self, cfg: dict, cfg_sampler: dict, dim: int, integration_method: str, Xtrain=None, Ytrain=None):
		super().__init__(cfg_sampler,dim)

		assert cfg.x_lim_max > cfg.x_lim_min
		assert cfg.Nsteps_integration > 3

		# Parameters:
		x_lim_min = cfg.x_lim_min
		x_lim_max = cfg.x_lim_max
		Nsteps_integration = cfg.Nsteps_integration

		assert integration_method in ["integrate_with_regular_grid","integrate_with_irregular_grid","integrate_with_bayesian_quadrature","integrate_with_data","integrate_with_regular_grid_randomized_parameters"]

		logger.info("Integration method: {0:s}".format(integration_method))

		if integration_method == "integrate_with_regular_grid_randomized_parameters":
			
			assert Xtrain is None and Ytrain is None, "Assert this for clarity"
			assert self.dim == 1 # For now, this is only for kink kernel

			Nsystem_randomized_parametrizations = 80
			Nsteps_integration_per_system = (Nsteps_integration-1) // Nsystem_randomized_parametrizations

			xgrid_data_per_system = CommonUtils.create_Ndim_grid(xmin=x_lim_min,xmax=x_lim_max,Ndiv=Nsteps_integration_per_system,dim=self.dim) # [Ndiv**dim_x,dim_x]

			xgrid_data = tf.concat([xgrid_data_per_system]*Nsystem_randomized_parametrizations,axis=0)
			Nsteps_integration = xgrid_data.shape[0]

			# Collect system evaluations for different parametrizations:
			fx_data_vec = np.zeros((Nsystem_randomized_parametrizations,Nsteps_integration_per_system,1))
			for jj in range(Nsystem_randomized_parametrizations):

				a0_min = -1.0; a0_max = +1.0;
				a0 = a0_min + (a0_max-a0_min)*np.random.rand(1)

				a1_min = -2.0; a1_max = +2.0;
				a1 = a1_min + (a1_max-a1_min)*np.random.rand(1)

				a2_min = -4.0; a2_max = -1.0;
				a2 = a2_min + (a2_max-a2_min)*np.random.rand(1)

				random_pars = dict(a0=a0,a1=a1,a2=a2)
				fx_data_vec[jj,...] = self._nonlinear_system_fun_static(xgrid_data_per_system,use_nominal_model=False,random_pars=random_pars) # [Nsteps,dim]

			self.fdata = np.reshape(fx_data_vec,(-1,1))
			
			# self.xdata = tf.reshape(xgrid_data,(-1,self.dim)) # [Nsteps,dim]
			self.xdata = xgrid_data # [Nsteps,dim]
			# self.dX = (self.xdata[1,-1] - self.xdata[0,-1])**self.dim # Equivalent to ((x_lim_max - x_lim_min) / Nsteps_integration)**self.dim
			dX_new = (x_lim_max-x_lim_min)**self.dim / self.xdata.shape[0] # Equivalent to ((x_lim_max - x_lim_min) / Nsteps_integration)**self.dim
			self.update_dX_voxels(dX_new=dX_new)
			logger.info("voxel value Dx_j = Dx for computing S() and varphi(): {0:f}".format(self.dX))

			# pdb.set_trace()

			# Fourier transform factor:
			# self.factor_Fourier = 1./(2.*math.pi)**(self.dim/2) # Unitary convention; would need to multiply the rpior mean by this factor as well
			self.factor_Fourier = 1./(2.*math.pi)**(self.dim) # Non-unitary convention: since we only care about S(w) in relation to the final f(x), we multiply the two terms directly here

		elif integration_method == "integrate_with_regular_grid":

			assert Xtrain is None and Ytrain is None, "Assert this for clarity"

			xgrid_data = CommonUtils.create_Ndim_grid(xmin=x_lim_min,xmax=x_lim_max,Ndiv=Nsteps_integration,dim=self.dim) # [Ndiv**dim_x,dim_x]
			# pdb.set_trace()
			self.xdata = tf.reshape(xgrid_data,(-1,self.dim)) # [Nsteps,dim]
			# self.dX = (self.xdata[1,-1] - self.xdata[0,-1])**self.dim # Equivalent to ((x_lim_max - x_lim_min) / Nsteps_integration)**self.dim
			dX_new = (x_lim_max-x_lim_min)**self.dim / self.xdata.shape[0] # Equivalent to ((x_lim_max - x_lim_min) / Nsteps_integration)**self.dim
			self.update_dX_voxels(dX_new=dX_new)
			logger.info("voxel value Dx_j = Dx for computing S() and varphi(): {0:f}".format(self.dX))

			self.fdata = self._nonlinear_system_fun(self.xdata) # [Nsteps,dim]
			if self.fdata.ndim == 1:
				self.fdata = tf.reshape(self.fdata,(-1,1))

			# Fourier transform factor:
			# self.factor_Fourier = 1./(2.*math.pi)**(self.dim/2) # Unitary convention; would need to multiply the rpior mean by this factor as well
			self.factor_Fourier = 1./(2.*math.pi)**(self.dim) # Non-unitary convention: since we only care about S(w) in relation to the final f(x), we multiply the two terms directly here

		elif integration_method == "integrate_with_irregular_grid":

			raise NotImplementedError("This case uses fixed voxels with an irregular grid, so it won't work unless the irrgular grid is really fine")

			Npoints = cfg.Nsteps_integration
			self.xdata = tf.random.uniform(shape=(Npoints**self.dim,self.dim),minval=x_lim_min,maxval=x_lim_max)
			self.fdata = self._nonlinear_system_fun(self.xdata) # [Nsteps,dim]
			if self.fdata.ndim == 1:
				self.fdata = tf.reshape(self.fdata,(-1,1))
			self.factor_Fourier = 1./(2.*math.pi)**(self.dim) # Non-unitary convention: since we only care about S(w) in relation to the final f(x), we multiply the two terms directly here
			# self.dX = np.prod(np.mean(abs(np.diff(self.xdata.numpy(),axis=0)),axis=0))
			# self.dX = 1./tf.reduce_sum(self.fdata)
			dX_new = (x_lim_max-x_lim_min)**self.dim / self.xdata.shape[0]
			self.update_dX_voxels(dX_new=dX_new)

			# pdb.set_trace()
			# 
		
		elif integration_method == "integrate_with_bayesian_quadrature":

			assert Xtrain is not None and Ytrain is not None
			raise NotImplementedError("This case needs work amid the new input arguments to the class, Xtrain, Ytrain")

			self.xdata = Xtrain

			# Data empirical mean and variance:
			xdata_mean = tf.reduce_mean(self.xdata,axis=0) # [1,dim]
			# pdb.set_trace()
			# assert self.dim == xdata_mean.shape[0]
			xdata_cov = (self.xdata.shape[0]/(self.xdata.shape[0]-1))*tfp.stats.covariance(x=self.xdata, y=self.xdata, sample_axis=0, event_axis=-1) # [dim,dim]
			# NOTE: The TF formula divides by N; we correct it multipliying by (N/(N-1))


			# Sample from normal-Wishart distribution:
			# posterior: we don't do posterior for now; just set the above quantities as if they were the posterior
			# https://en.wikipedia.org/wiki/Wishart_distribution
			# https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/WishartTriL
			# df = xdata_cov.shape[1]
			df = xdata_cov.shape[1] # as
			dist_wishart = tfp.distributions.WishartTriL(df=df,scale_tril=tf.linalg.cholesky(xdata_cov)) # df > (xdata_cov.shape[1]-1)

			Nsamples_tot = 5000
			Ngaussians = 1
			assert Nsamples_tot % Ngaussians == 0
			Sigma_sample = 1./30*dist_wishart.sample(sample_shape=(Ngaussians)) # [Ngaussians,dim,dim]

			Sigma_sample = tf.expand_dims(xdata_cov,axis=0)

			rand_vec_standard_normal = tf.random.normal(shape=(Nsamples_tot//Ngaussians,Ngaussians,xdata_mean.shape[0],1),mean=0.0,stddev=1.0) # [Nsamples_tot//Ngaussians,Ngaussians,dim,1]
			cov_samples = tf.expand_dims(tf.linalg.cholesky(Sigma_sample),axis=0) @ rand_vec_standard_normal # [1,Ngaussians,dim,dim] @ [Nsamples_tot//Ngaussians,Ngaussians,dim,1] = [Nsamples_tot//Ngaussians,Ngaussians,dim,1] || # Lower triangular A = L.L^T
			xsamples = xdata_mean + tf.squeeze(cov_samples) # [,dim] + [Nsamples_tot//Ngaussians,Ngaussians,dim] = [Nsamples_tot//Ngaussians,Ngaussians,dim]
			xsamples_tot = tf.reshape(xsamples,(Nsamples_tot,-1)) # [Nsamples_tot,dim] sorted as [ [Ngaussians, Ngaussians, ..., Ngaussians]   ,dim ] 

			print(xdata_mean)

			hdl_fig_data, hdl_splots_data = plt.subplots(1,1,figsize=(12,8),sharex=True)
			hdl_splots_data.plot(self.xdata[:,0],self.xdata[:,1],marker=".",linestyle="None",color="grey",alpha=0.3)
			hdl_splots_data.plot(xsamples[:,0],xsamples[:,1],marker=".",linestyle="None",color="blue",alpha=0.3)

			plt.show(block=True)

			pdb.set_trace()


			# # Use a voroni diagram to compute the integration steps:
			# Npoints = cfg.Nsteps_integration
			# self.xdata = tf.random.uniform(shape=(Npoints**self.dim,self.dim),minval=x_lim_min,maxval=x_lim_max)
			# self.fdata = self._nonlinear_system_fun(self.xdata) # [Nsteps,dim]
			# if self.fdata.ndim == 1:
			# 	self.fdata = tf.reshape(self.fdata,(-1,1))
			# self.factor_Fourier = 1./(2.*math.pi)**(self.dim) # Non-unitary convention: since we only care about S(w) in relation to the final f(x), we multiply the two terms directly here

		elif integration_method == "integrate_with_data":

			assert Xtrain is not None and Ytrain is not None

			assert Xtrain.shape[1] == self.dim
			assert Ytrain.shape[0] == Xtrain.shape[0]
			assert Ytrain.shape[1] >= 1

			# Fourier transform factor:
			# self.factor_Fourier = 1./(2.*math.pi)**(self.dim/2) # Unitary convention; would need to multiply the rpior mean by this factor as well
			self.factor_Fourier = 1./(2.*math.pi)**(self.dim) # Non-unitary convention: since we only care about S(w) in relation to the final f(x), we multiply the two terms directly here

			xmin = tf.reduce_min(Xtrain)
			xmax = tf.reduce_max(Xtrain)
			# dX_new = (xmax-xmin)**self.dim / Xtrain.shape[0] # Voxel value
			dX_new = 1./ Xtrain.shape[0] # Voxel value
			self.update_dX_voxels(dX_new=dX_new)
			
			# Naming convention:
			self.xdata = Xtrain
			self.fdata = Ytrain

	def update_dX_voxels(self,dX_new):

		assert dX_new is not None

		# logger.info("Updating voxel value Dx_t for computing S() and varphi()")
		if isinstance(dX_new,float):
			# logger.info("New Dx_t: {0:f}".format(dX_new))
			pass
		elif len(tf.squeeze(dX_new)) == 1:
			# logger.info("New Dx_t: {0:f}".format(dX_new[0]))
			pass
		else:
			# logger.info("New Dx_t: {0:s}".format(str(dX_new)))
			pass
		
		self.dX = dX_new

	def update_fdata(self,fdata):
		assert fdata.shape[1] == 1
		self.fdata = fdata

	# @tf.function
	def unnormalized_density(self,omega_in,log=False):
		"""
		
		in: omega_in [Nfeat,dim]
		return: 
			Sw_vec [Nfeat,dim]
			phiw_vec [Nfeat,dim]
		"""

		Sw_vec, phiw_vec = self._MVFourierTransform(omega_in) # [Nfeat,dim]

		if log == True:
			return tf.math.log(Sw_vec)

		return Sw_vec, phiw_vec

	# @tf.function
	def _MVFourierTransform(self,omega_vec):
		"""
		
		self.fdata: [Nxpoints,1]
		self.dX: float OR [Nxpoints,1]
		self.factor_Fourier: float
		self.xdata: [Nxpoints,dim_in]

		omega_vec: [Nomegas,dim_in]
		return:
			Sw: [Nomegas,1]
			phiw: [Nomegas,1]
		"""

		assert self.fdata.shape[1] == 1

		assert omega_vec.shape[1] == self.dim
		omega_times_X = omega_vec @ tf.transpose(self.xdata) # [Nomegas,Nxpoints]

		if isinstance(self.dX,float):
			# logger.info("Using ONLY ONE voxels dX for getting a(wj), b(wj)")
			part_real = self.dX*self.factor_Fourier*(tf.math.cos(omega_times_X) @ self.fdata)
			part_imag = self.dX*self.factor_Fourier*(tf.math.sin(omega_times_X) @ self.fdata)
		else:
			assert self.dX.shape[1] == 1
			# logger.info("Using MULTIPLE INDIVIDUAL voxels dX for getting a(wj), b(wj)")
			part_real = self.factor_Fourier*(tf.math.cos(omega_times_X) @ (self.fdata*self.dX))
			part_imag = self.factor_Fourier*(tf.math.sin(omega_times_X) @ (self.fdata*self.dX))

		# self.part_real_dbg = part_real
		# self.part_imag_dbg = part_imag

		# Modulus (spectral density):
		Sw = tf.math.sqrt(part_real**2 + part_imag**2) # [Nomegas,dim_in]

		# Argument:
		phiw = tf.math.atan2(y=-part_imag,x=part_real) # [Nomegas,dim_in]

		return Sw, phiw

	@abstractmethod
	def _nonlinear_system_fun(self,x):
		raise NotImplementedError


class KinkSpectralDensity(NonLinearSystemSpectralDensity):

	def __init__(self, cfg: dict, cfg_sampler: dict, dim: int, integration_method: str, Xtrain=None, Ytrain=None, use_nominal_model=True):
		assert dim == 1
		self.use_nominal_model = use_nominal_model
		super().__init__(cfg,cfg_sampler,dim,integration_method,Xtrain,Ytrain)

	def _nonlinear_system_fun(self,x):
		return self._nonlinear_system_fun_static(x,use_nominal_model=self.use_nominal_model)

	@staticmethod
	def _nonlinear_system_fun_static(x,use_nominal_model=True,random_pars=None):
		"""

		Kink function, as described in [1]

		f(x) = 0.8 + (x + 0.2)*(1. - 5./(1 + tf.math.exp(-2.*x)) )

		[1] Ialongo, A.D., Van Der Wilk, M., Hensman, J. and Rasmussen, C.E., 2019, May. 
		Overcoming mean-field approximations in recurrent Gaussian process models. In International Conference on Machine Learning (pp. 2931-2940). PMLR.
		"""

		# True parameters:
		a0 = 0.8
		a1 = 0.2
		a2 = -2.

		if not use_nominal_model:
			a0 = 3.0
			a1 = -1.0
			a2 = -5.0

		if random_pars is not None:
			# logger.info("Using random parametrization in Kink!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
			a0 = random_pars["a0"]
			a1 = random_pars["a1"]
			a2 = random_pars["a2"]

		# return 0.8 + (x + 0.2)*(1. - 5./(1 + tf.math.exp(-2.*x)) )
		return a0 + (x + a1)*(1. - 5./(1 + tf.math.exp(a2*x)) )



class ParaboloidSpectralDensity(NonLinearSystemSpectralDensity):

	def __init__(self, cfg: dict, cfg_sampler: dict, dim: int, integration_method: str, Xtrain=None, Ytrain=None):
		super().__init__(cfg,cfg_sampler,dim,integration_method,Xtrain,Ytrain)

	def _nonlinear_system_fun(self,x):
		"""
		x: [Npoints,self.dim]
		"""
		# return tf.math.reduce_sum(x**2,axis=1)
		# return tf.math.reduce_sum(x**2,axis=1,keepdims=True)
		return self._nonlinear_system_fun_static(x)

	@staticmethod
	def _nonlinear_system_fun_static(x):
		return tf.math.reduce_sum(x**2,axis=1,keepdims=True)

class NoNameSpectralDensity(NonLinearSystemSpectralDensity):

	def __init__(self, cfg: dict, cfg_sampler: dict, dim: int, integration_method: str):
		assert dim == 1
		super().__init__(cfg,cfg_sampler,dim)

	def _nonlinear_system_fun(self,x,u=0.):

		"""

		Toy example unction, as described in [1, Sec. 6]

		[1] Frigola, R., Lindsten, F., Schön, T.B. and Rasmussen, C.E., 2013. Bayesian inference and learning in Gaussian process state-space models with particle MCMC. Advances in neural information processing systems, 26.
		"""

		self.a = 0.5
		self.b = 25.
		self.c = 8.
		# self.q = 10.

		return self.a*x + self.b*x / (1. + x**2) + self.c*u


class KinkSharpSpectralDensity(NonLinearSystemSpectralDensity):

	def __init__(self, cfg: dict, cfg_sampler: dict, dim: int, integration_method: str, Xtrain=None, Ytrain=None):
		assert dim == 1
		super().__init__(cfg,cfg_sampler,dim,integration_method,Xtrain,Ytrain)

	def _nonlinear_system_fun(self,x):

		"""

		Toy example unction, as described in [1, Sec. 6]

		[1] Frigola, R., Chen, Y. and Rasmussen, C.E., 2014. Variational Gaussian process state-space models. Advances in neural information processing systems, 27.
		"""

		ind_smaller = x < 4
		out = x.numpy()
		out[ind_smaller] += 1.
		out[~ind_smaller] = -4.*out[~ind_smaller] + 21.

		return tf.convert_to_tensor(out,dtype=tf.float32)

		# if x < 4:
		# 	return x + 1
		# else:
		# 	return -4.*x + 21.

		# return self.a*x + self.b*x / (1. + x**2) + self.c*u


class VanDerPolSpectralDensity(NonLinearSystemSpectralDensity):

	def __init__(self, cfg: dict, cfg_sampler: dict, dim: int, integration_method: str, use_nominal_model=True, Xtrain=None, Ytrain=None):
		
		assert dim == 2
		self.use_nominal_model = use_nominal_model
		super().__init__(cfg,cfg_sampler,dim,integration_method,Xtrain,Ytrain)

	def _nonlinear_system_fun(self,x):
		"""
		x: [Npoints,self.dim]
		"""

		# return self._controlled_vanderpol_dynamics(x=x[:,0:1],y=x[:,1::],u1=0.,u2=0.,use_nominal_model=self.use_nominal_model)
		return self._controlled_vanderpol_dynamics(state_vec=x,control_vec="gather_data_policy",use_nominal_model=self.use_nominal_model)

	@staticmethod
	# def _controlled_vanderpol_dynamics(x, y, u1, u2, use_nominal_model=True):
	def _controlled_vanderpol_dynamics(state_vec, control_vec, use_nominal_model=True):
		"""
		state_vec: [Npoints,self.dim]
		control_vec: [Npoints,dim_u]
		"""

		# State:
		x = state_vec[:,0:1]
		y = state_vec[:,1:2]

		# Control:
		if control_vec == "gather_data_policy" and state_vec.shape[1] == 4:
			# Assume that we're concatenating the state and the control input
			u1 = state_vec[:,2:3]
			u2 = state_vec[:,3:4]
		elif control_vec == "gather_data_policy" and state_vec.shape[1] == 2: 
			u1 = 0.0
			u2 = 0.0
		elif control_vec == "gather_data_policy":
			raise NotImplementedError("Wrong state dimensionality")
		else:
			u1 = control_vec[0:1,:]
			u2 = control_vec[1:2,:]
		
		deltaT = 0.01 # NOTE: move this up to user choices

		# True parameters:
		a0 = 1.0
		a1 = 1.0
		a2 = 1.0
		a3 = 1.0

		if not use_nominal_model:
			a0 = 2.0
			a1 = 0.1
			a2 = 1.5
			a3 = 0.01

		x_dot =-a0*y + u1
		y_dot = a1*x - a2*y + a3*(x**2)*y + u2

		x_next = x_dot*deltaT + x
		y_next = y_dot*deltaT + y
		xy_next = tf.concat([x_next,y_next],axis=1)

		return xy_next

class DubinsCarSpectralDensity(NonLinearSystemSpectralDensity):

	# @tf.function
	def __init__(self, cfg: dict, cfg_sampler: dict, dim: int, integration_method: str, use_nominal_model=True, Xtrain=None, Ytrain=None):
		
		# assert dim == 2
		self.use_nominal_model = use_nominal_model
		super().__init__(cfg,cfg_sampler,dim,integration_method,Xtrain,Ytrain)

	# @tf.function
	def _nonlinear_system_fun(self,x):
		"""
		x: [Npoints,self.dim]
		"""

		# return self._controlled_dubinscar_dynamics(x=x[:,0:1],y=x[:,1:2],th=x[:,2:3],u1=0.,u2=0.,use_nominal_model=self.use_nominal_model)

		# By default, we assume that x concatenates state and control input, i.e., x = [xt,ut]
		return self._controlled_dubinscar_dynamics(state_vec=x,control_vec="gather_data_policy",use_nominal_model=self.use_nominal_model)

	@staticmethod
	# def _controlled_dubinscar_dynamics(x, y, th, u1, u2, use_nominal_model=True):
	# @tf.function
	def _controlled_dubinscar_dynamics(state_vec, control_vec, use_nominal_model=True):
		"""
		state_vec: [Npoints,self.dim]
		control_vec: [Npoints,dim_u]
		"""

		# State:
		x = state_vec[:,0:1]
		y = state_vec[:,1:2]
		th = state_vec[:,2:3]


		# Control:
		if isinstance(control_vec,str):
			if control_vec == "gather_data_policy" and state_vec.shape[1] == 5:
				print("@_controlled_dubinscar_dynamics: if control_vec == 'gather_data_policy' and state_vec.shape[1] == 5:")
				u1 = state_vec[:,3:4]
				u2 = state_vec[:,4:5]
			elif control_vec == "gather_data_policy" and state_vec.shape[1] == 3: # Infinitely growing spiral
				print("@_controlled_dubinscar_dynamics: elif control_vec == 'gather_data_policy' and state_vec.shape[1] == 3: # Infinitely growing spiral")
				u1 = 0.16 # Randomly chosen values, then fixed
				u2 = 0.11 # Randomly chosen values, then fixed
			elif control_vec == "gather_data_policy":
				print("@_controlled_dubinscar_dynamics: elif control_vec == 'gather_data_policy':")
				u1 = 0.0
				u2 = 0.0
		else:
			# print("@_controlled_dubinscar_dynamics: else")
			u1 = control_vec[:,0:1]
			u2 = control_vec[:,1:2]

		# assert u1 >= 0.0, "u1 = {} | Do something about this! The input velocity can't be negative. One solution is to flip the heading angle 180 degrees when u1 is negative".format(u1)
		# if u1 < 0.0:
		# 	print("u1 < 0.0")


		deltaT = 0.01 # NOTE: move this up to user choices

		# True parameters:
		vel_lin_min = 0.0
		vel_ang_min = 0.0
		vel_ang_max = +float("Inf")

		# Add dynamics imperfections:
		if not use_nominal_model:

			# which_alteration = "slacking"
			which_alteration = "disturbance"
			assert which_alteration in ["slacking","disturbance"]

			# Change model slacking the input
			if which_alteration == "slacking":
				vel_lin_min = 1.5
				vel_ang_min = 0.1
				vel_ang_max = 0.5
				u1_in = tf.math.sign(u1)*tf.clip_by_value(t=tf.math.abs(u1),clip_value_min=vel_lin_min,clip_value_max=float("Inf"))
				u2_in = tf.math.sign(u2)*tf.clip_by_value(t=tf.math.abs(u2),clip_value_min=vel_ang_min,clip_value_max=vel_ang_max)

			# Change the model adding a constant disturbance:
			if which_alteration == "disturbance": # IMPORTANT: This needs to agree with get_sequence_of_feedback_gains_finite_horizon_LQR() @ test_dubin_car.py
				u1_in = u1*2.0
				# u1_in = u1/2.0
				# u1_in = u1 - 0.5
				# u2_in = u2 - 1.0
				# u2_in = u2
				u2_in = u2*3.0
		else:
			u1_in = u1
			u2_in = u2


		# Integrate dynamics:
		x_next = deltaT*u1_in*tf.math.cos(th) + x
		y_next = deltaT*u1_in*tf.math.sin(th) + y
		th_next = deltaT*u2_in + th

		if tf.math.reduce_any(tf.math.is_inf(x_next)):
		# if np.any(np.isinf(x_next)):
			print("x_next is inf")
			pdb.set_trace()

		if tf.math.reduce_any(tf.math.is_inf(y_next)):
		# if np.any(np.isinf(y_next)):
			print("x_next is inf")
			pdb.set_trace()

		if tf.math.reduce_any(tf.math.is_inf(th_next)):
		# if np.any(np.isinf(th_next)):
			print("x_next is inf")
			pdb.set_trace()

		xyth_next = tf.concat([x_next,y_next,th_next],axis=1)

		if tf.math.reduce_all(xyth_next == 0.0):
		# if np.all(xyth_next == 0.0):
			print("xyth_next is all zeroes")
			pdb.set_trace()

		return xyth_next


	# def _log_prob(self,value):
	# 	"""
		# See how the uniform distribution is implemented in tensorflow
	# 	value: float or double Tensor
	# 	return: a Tensor of shape sample_shape(x) + self.batch_shape with values of type self.dtype
	# 	"""

	# 	out = self.unnormalized_density(omega_in=value,log=True)
	# 	pdb.set_trace()

	# 	return out

	# def _sample_n(self, n, seed=None):
		# See how the uniform distribution is implemented in tensorflow
	# 	samples = get_samples(n)

	# 	pdb.set_trace()

	# 	return samples
	# 	


	# # @tf.function
	# def _MVFourierTransform_bayesian_quadrature(self,omega_vec):
	# 	"""

	# 	omega_vec: [Npoints,dim]
	# 	return:
	# 		Sw: [Npoints,dim]
	# 		phiw: [Npoints,dim]
	# 	"""

	# 	assert omega_vec.shape[1] == self.dim
	# 	omega_times_X = omega_vec @ tf.transpose(self.xdata) # [Npoints,Nsteps]


	# 	if self.integrate_with_bayesian_quadrature:
	# 		intergation_steps


	# 	part_real = self.dX*self.factor_Fourier*(tf.math.cos(omega_times_X) @ self.fdata)
	# 	part_imag = self.dX*self.factor_Fourier*(tf.math.sin(omega_times_X) @ self.fdata)

	# 	# self.part_real_dbg = part_real
	# 	# self.part_imag_dbg = part_imag

	# 	# Modulus (spectral density):
	# 	Sw = tf.math.sqrt(part_real**2 + part_imag**2) # [Npoints,dim]

	# 	# Argument:
	# 	phiw = tf.math.atan2(y=-part_imag,x=part_real) # [Npoints,dim]

	# 	return Sw, phiw

	# def _monte_carlo_sum_approx_single_point(self,omega_in,Xdata,fXdata):
	# 	"""

	# 	omega_in: [1,dim]
	# 	Xdata: [Npoints,dim]
	# 	fXdata: [Npoints]

	# 	"""

	# 	M = Xdata.shape[0]
	# 	sum_all = tf.math.sum(fXdata**2)
	# 	for ii in range(M):
	# 		for jj in range(ii+1,M):
	# 			sum_all += 2.*fXdata[ii]*fXdata[jj]*tf.math.cos(omega_in @ (Xdata[ii:ii+1,:] - Xdata[jj:jj+1,:]))

	# 	sum_all = sum_all / M**2

	# 	return sum_all

