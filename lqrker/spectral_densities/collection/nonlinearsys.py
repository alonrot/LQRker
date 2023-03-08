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
	"""

	Spectral density and phase through Fourier transform for channel f_d(x_in) of a dynamical system.

	The scalar function f_d \mathcal{X} \rightarrow \mathbb{R} induces a scala spectral density S(w) and phase varphi(w)

	"""


	def __init__(self, cfg: dict, cfg_sampler: dict, dim_in: int, integration_method: str, Xtrain: tf.Tensor, Ytrain: tf.Tensor):
		super().__init__(cfg_sampler,dim_in)

		assert Xtrain.shape[1] == self.dim_in
		assert Ytrain.shape[0] == Xtrain.shape[0]
		assert Ytrain.shape[1] == 1, "This spectral density encodes a single channel"

		# Fourier transform factor:
		# self.factor_Fourier = 1./(2.*math.pi)**(self.dim_in/2) # Unitary convention; would need to multiply the rpior mean by this factor as well
		self.factor_Fourier = 1./(2.*math.pi)**(self.dim_in) # Non-unitary convention: since we only care about S(w) in relation to the final f(x), we multiply the two terms directly here

		self.update_integration_data_fx_and_x(fx_data=Ytrain,x_data=Xtrain) # Xtrain: [Npoints,self.dim_in] | Ytrain: [Npoints,1] -> We pass only the Ytrain corresponding to a single channel
		self.dX_voxel = None
		
	def update_integration_dX_voxels(self,dX_voxel_new):
		"""
		dX_voxel_new: [Nxpoints,1]
		"""
		self.dX_voxel = dX_voxel_new

	def update_integration_data_fx_and_x(self,fx_data,x_data):
		"""
		fx_data: [Npoints,1]
		x_data: [Npoints,self.dim_in]
		"""
		assert x_data.shape[1] == self.dim_in
		assert fx_data.shape[0] == x_data.shape[0]
		assert fx_data.shape[1] == 1, "This spectral density encodes a single channel"
		self.fx_data = fx_data
		self.x_data = x_data

	def unnormalized_density(self,omega_in,log=False):
		"""
		
		in: omega_in [Nfeat,self.dim_in]
		return: 
			Sw_vec [Nfeat,1]
			phiw_vec [Nfeat,1]
		"""

		Sw_vec, phiw_vec = self._MVFourierTransform(omega_in) # [Nfeat,dim]

		if log == True:
			return tf.math.log(Sw_vec)

		return Sw_vec, phiw_vec

	def _MVFourierTransform(self,omega_vec):
		"""
		
		self.fx_data: [Nxpoints,1]
		self.dX_voxel: [Nxpoints,1]
		self.factor_Fourier: float
		self.x_data: [Nxpoints,self.dim_in]

		omega_vec: [Nomegas,self.dim_in]
		return:
			Sw: [Nomegas,1]
			phiw: [Nomegas,1]
		"""

		assert self.dX_voxel is not None, "You probably need to call self.update_integration_dX_voxels() first"
		assert self.fx_data is not None, "You probably need to call self.update_integration_data_fx_and_x() first"
		assert self.fx_data.shape[1] == 1
		assert self.dX_voxel.shape[1] == 1

		assert self.fx_data.shape[0] == self.x_data.shape[0]
		assert self.fx_data.shape[0] == self.dX_voxel.shape[0]

		assert omega_vec.shape[1] == self.dim_in
		omega_times_X = omega_vec @ tf.transpose(self.x_data) # [Nomegas,Nxpoints]

		part_real = self.factor_Fourier*(tf.math.cos(omega_times_X) @ (self.fx_data*self.dX_voxel))
		part_imag = self.factor_Fourier*(tf.math.sin(omega_times_X) @ (self.fx_data*self.dX_voxel))

		# Modulus (spectral density):
		Sw = tf.math.sqrt(part_real**2 + part_imag**2) # [Nomegas,dim_in]

		# Argument:
		phiw = tf.math.atan2(y=-part_imag,x=part_real) # [Nomegas,dim_in]

		return Sw, phiw

	@abstractmethod
	def _nonlinear_system_fun(self,x):
		raise NotImplementedError


class KinkSpectralDensity(NonLinearSystemSpectralDensity):

	def __init__(self, cfg: dict, cfg_sampler: dict, dim_in: int, integration_method: str, Xtrain: tf.Tensor, Ytrain: tf.Tensor):
	
		super().__init__(cfg,cfg_sampler,dim_in,integration_method,Xtrain,Ytrain)

	def _nonlinear_system_fun(self,x):
		return self._nonlinear_system_fun_static(x,model_pars=None)

	@staticmethod
	def _nonlinear_system_fun_static(x,model_pars=None):
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

		if model_pars is not None:
			a0 = model_pars["a0"]
			a1 = model_pars["a1"]
			a2 = model_pars["a2"]

		# return 0.8 + (x + 0.2)*(1. - 5./(1 + tf.math.exp(-2.*x)) )
		return a0 + (x + a1)*(1. - 5./(1 + tf.math.exp(a2*x)) )



class ParaboloidSpectralDensity(NonLinearSystemSpectralDensity):

	def __init__(self, cfg: dict, cfg_sampler: dict, dim_in: int, integration_method: str, Xtrain: tf.Tensor, Ytrain: tf.Tensor):
		super().__init__(cfg,cfg_sampler,dim_in,integration_method,Xtrain,Ytrain)

	def _nonlinear_system_fun(self,x):
		return self._nonlinear_system_fun_static(x,model_pars=None)

	@staticmethod
	def _nonlinear_system_fun_static(x,model_pars=None):

		# True parameters:
		a0 = 0.0
		a1 = 0.0
		a2 = 1.0

		if model_pars is not None:
			a0 = model_pars["a0"]
			a1 = model_pars["a1"]
			a2 = model_pars["a2"]

		return a0 + tf.math.reduce_sum(a2*(x-a1)**2,axis=1,keepdims=True)

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
		x: [Npoints,self.dim_in]
		"""

		# return self._controlled_vanderpol_dynamics(x=x[:,0:1],y=x[:,1::],u1=0.,u2=0.,use_nominal_model=self.use_nominal_model)
		return self._controlled_vanderpol_dynamics(state_vec=x,control_vec="gather_data_policy",use_nominal_model=self.use_nominal_model)

	@staticmethod
	# def _controlled_vanderpol_dynamics(x, y, u1, u2, use_nominal_model=True):
	def _controlled_vanderpol_dynamics(state_vec, control_vec, use_nominal_model=True):
		"""
		state_vec: [Npoints,self.dim_in]
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

	def __init__(self, cfg: dict, cfg_sampler: dict, dim: int, integration_method: str, use_nominal_model=True, Xtrain=None, Ytrain=None):
		
		# assert dim == 2
		self.use_nominal_model = use_nominal_model
		super().__init__(cfg,cfg_sampler,dim,integration_method,Xtrain,Ytrain)

	def _nonlinear_system_fun(self,x):
		"""
		x: [Npoints,self.dim_in]
		"""

		# return self._controlled_dubinscar_dynamics(x=x[:,0:1],y=x[:,1:2],th=x[:,2:3],u1=0.,u2=0.,use_nominal_model=self.use_nominal_model)

		# By default, we assume that x concatenates state and control input, i.e., x = [xt,ut]
		return self._controlled_dubinscar_dynamics(state_vec=x,control_vec="gather_data_policy",use_nominal_model=self.use_nominal_model)

	@staticmethod
	# def _controlled_dubinscar_dynamics(x, y, th, u1, u2, use_nominal_model=True):
	def _controlled_dubinscar_dynamics(state_vec, control_vec, use_nominal_model=True,control_vec_prev=None):
		"""
		state_vec: [Npoints,self.dim_in]
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
			# which_alteration = "lowpass"
			assert which_alteration in ["slacking","disturbance","lowpass"]

			# Change model slacking the input
			if which_alteration == "slacking":
				vel_lin_min = 1.5
				vel_ang_min = 0.1
				vel_ang_max = 0.5
				u1_in = tf.math.sign(u1)*tf.clip_by_value(t=tf.math.abs(u1),clip_value_min=vel_lin_min,clip_value_max=float("Inf"))
				u2_in = tf.math.sign(u2)*tf.clip_by_value(t=tf.math.abs(u2),clip_value_min=vel_ang_min,clip_value_max=vel_ang_max)

			# Change the model adding a constant disturbance:
			if which_alteration == "disturbance": # IMPORTANT: This needs to agree with the linearized model at get_sequence_of_feedback_gains_finite_horizon_LQR() @ test_dubin_car.py
				u1_in = u1*2.0
				# u1_in = u1/2.0
				# u1_in = u1 - 0.5
				# u2_in = u2 - 1.0
				# u2_in = u2
				u2_in = u2*3.0

			# Change the model adding a constant disturbance:
			if which_alteration == "lowpass": # IMPORTANT: This needs to agree with the linearized model at get_sequence_of_feedback_gains_finite_horizon_LQR() @ test_dubin_car.py
				if control_vec_prev is not None:
					u1p = control_vec_prev[:,0:1]
					u2p = control_vec_prev[:,1:2]

					u1_in = deltaT*u1 - u1p*(deltaT-1.) # ref_t - u_t*(deltaT-1)
					u2_in = deltaT*u2 - u2p*(deltaT-1.) # ref_t - u_t*(deltaT-1)
				else:
					u1_in = u1
					u2_in = u2
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

