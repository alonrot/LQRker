import tensorflow as tf
import pdb
import math
from abc import ABC, abstractmethod
import numpy as np
import tensorflow_probability as tfp
from lqrker.utils.parsing import get_logger
from lqrker.spectral_densities.base import SpectralDensityBase
from lqrker.utils.common import CommonUtils
logger = get_logger(__name__)


class NonLinearSystemSpectralDensity(SpectralDensityBase):
# class NonLinearSystemSpectralDensity(tfp.distributions.distribution.AutoCompositeTensorDistribution,SpectralDensityBase):
	
	# @tf.function
	def __init__(self, cfg: dict, cfg_sampler: dict, dim: int):
		super().__init__(cfg_sampler,dim)

		assert cfg.x_lim_max > cfg.x_lim_min
		assert cfg.Nsteps_integration > 3

		# Parameters:
		x_lim_min = cfg.x_lim_min
		x_lim_max = cfg.x_lim_max
		Nsteps_integration = cfg.Nsteps_integration

		# pdb.set_trace()

		xgrid_data = CommonUtils.create_Ndim_grid(xmin=x_lim_min,xmax=x_lim_max,Ndiv=Nsteps_integration,dim=self.dim) # [Ndiv**dim_x,dim_x]
		self.xdata = tf.reshape(xgrid_data,(-1,self.dim)) # [Nsteps,dim]
		self.dX = (self.xdata[1,-1] - self.xdata[0,-1])**self.dim # Equivalent to ((x_lim_max - x_lim_min) / Nsteps_integration)**self.dim

		self.fdata = self._nonlinear_system_fun(self.xdata) # [Nsteps,dim]
		if self.fdata.ndim == 1:
			self.fdata = tf.reshape(self.fdata,(-1,1))

		# Fourier transform factor:
		# self.factor_Fourier = 1./(2.*math.pi)**(self.dim/2) # Unitary convention; would need to multiply the rpior mean by this factor as well
		self.factor_Fourier = 1./(2.*math.pi)**(self.dim) # Non-unitary convention: since we only care about S(w) in relation to the final f(x), we multiply the two terms directly here

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

		omega_vec: [Npoints,dim]
		return:
			Sw: [Npoints,dim]
			phiw: [Npoints,dim]
		"""

		assert omega_vec.shape[1] == self.dim
		omega_times_X = omega_vec @ tf.transpose(self.xdata) # [Npoints,Nsteps]

		part_real = self.dX*self.factor_Fourier*(tf.math.cos(omega_times_X) @ self.fdata)
		part_imag = self.dX*self.factor_Fourier*(tf.math.sin(omega_times_X) @ self.fdata)

		# Modulus (spectral density):
		Sw = tf.math.sqrt(part_real**2 + part_imag**2) # [Npoints,dim]

		# Argument:
		phiw = tf.math.atan2(y=-part_imag,x=part_real) # [Npoints,dim]

		return Sw, phiw

	@abstractmethod
	def _nonlinear_system_fun(self,x):
		raise NotImplementedError

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



class KinkSpectralDensity(NonLinearSystemSpectralDensity):

	def __init__(self, cfg: dict, cfg_sampler: dict, dim: int, use_nominal_model=True):
		assert dim == 1
		self.use_nominal_model = use_nominal_model
		super().__init__(cfg,cfg_sampler,dim)

	def _nonlinear_system_fun(self,x):
		return self.kink_dynamics(x,use_nominal_model=self.use_nominal_model)

	@staticmethod
	def kink_dynamics(x,use_nominal_model=True):
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
			logger.info("DBG")
			pdb.set_trace()
			a0 = -0.2
			a1 = 0.9
			a2 = -0.5

		# return 0.8 + (x + 0.2)*(1. - 5./(1 + tf.math.exp(-2.*x)) )
		return a0 + (x + a1)*(1. - 5./(1 + tf.math.exp(a2*x)) )



class ParaboloidSpectralDensity(NonLinearSystemSpectralDensity):

	def __init__(self, cfg: dict, cfg_sampler: dict, dim: int):
		super().__init__(cfg,cfg_sampler,dim)

	def _nonlinear_system_fun(self,x):
		"""
		x: [Npoints,self.dim]
		"""
		return tf.math.reduce_sum(x**2,axis=1)

class NoNameSpectralDensity(NonLinearSystemSpectralDensity):

	def __init__(self, cfg: dict, cfg_sampler: dict, dim: int):
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

	def __init__(self, cfg: dict, cfg_sampler: dict, dim: int):
		assert dim == 1
		super().__init__(cfg,cfg_sampler,dim)

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

	def __init__(self, cfg: dict, cfg_sampler: dict, dim: int, use_nominal_model=True):
		
		assert dim == 2
		self.use_nominal_model = use_nominal_model
		super().__init__(cfg,cfg_sampler,dim)

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
	def __init__(self, cfg: dict, cfg_sampler: dict, dim: int, use_nominal_model=True):
		
		# assert dim == 2
		self.use_nominal_model = use_nominal_model
		super().__init__(cfg,cfg_sampler,dim)

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
