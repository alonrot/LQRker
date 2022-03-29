import tensorflow as tf
import pdb
import math
from abc import ABC, abstractmethod
import numpy as np
import tensorflow_probability as tfp
from lqrker.utils.parsing import get_logger
logger = get_logger(__name__)
ZERO_NUM = 0.0


class SpectralDensity(ABC):
	"""
	Collection of Spectral densities, in correspondance with a particular type of
	dynamical system
	"""

	def __init__(self):
		self.normal_sampler_fn = lambda seed: tfp.distributions.Normal(loc=np.float32(1), scale=np.float32(5)).sample(seed=seed)

	@abstractmethod
	def unnormalized_density(self):
		raise NotImplementedError

	@abstractmethod
	def normalized_density(self):
		raise NotImplementedError

	@abstractmethod
	def log_normalized_density(self):
		raise NotImplementedError

	@abstractmethod
	def update_pars(self,args):
		raise NotImplementedError

	def get_samples(self):
		raise NotImplementedError

	def sampleESS(self,log_likelihood_fn,Nsamples):
		"""

		TODO:
		1) Sample within some boundaries, i.e., positive omegas
		2) Leverage trace_fn and others
		3) Understand parallel iterations
		4) Pass the starting state as input argument, as its dimensions are problem-dependent


		"""
		

		# logger.info("Initializing MCMC ESS kernel...")
		# kernel_mcmc = tfp.experimental.mcmc.EllipticalSliceSampler(	normal_sampler_fn=self.normal_sampler_fn,
		# 															log_likelihood_fn=log_likelihood_fn,
		# 															name=None)

		logger.info("Initializing MCMC NUTS kernel...")
		kernel_mcmc = tfp.experimental.mcmc.NoUTurnSampler(
			target_log_prob_fn=log_likelihood_fn, step_size=tf.constant([[1.,1.]]), max_tree_depth=10, unrolled_leapfrog_steps=1,
			num_trajectories_per_step=1, use_auto_batching=True, stackless=False,
			backend=None, seed=None, name=None
			)

		logger.info("Getting MCMC chains for "+str(Nsamples)+ " samples ...")
		samples_vec = tfp.mcmc.sample_chain(
			num_results=Nsamples,
			current_state=tf.constant([[1.,1.]]),
			kernel=kernel_mcmc,
			num_burnin_steps=5,
			trace_fn=None,
			parallel_iterations=5)  # For determinism.

		return samples_vec

class MaternSpectralDensity(SpectralDensity):

	def __init__(self,cfg: dict, dim: int):
		super().__init__()

		# Parameters:
		self.nu = cfg.nu*1.0
		assert self.nu > 2.0, "nu must be > 2"
		self.ls = cfg.ls*1.0
		self.prior_var = cfg.prior_var*1.0
		self.dim = dim

		# Constant parameters:
		self.lambda_val = tf.sqrt(2*self.nu)/self.ls
		self.const = ((2*tf.sqrt(math.pi))**self.dim)*tf.exp(tf.math.lgamma(self.nu+0.5*self.dim))*self.lambda_val**(2*self.nu) / tf.exp(tf.math.lgamma(self.nu))

	def unnormalized_density(self,omega_in,log=True):
		"""
		in: omega_in [Nfeat,dim]
		return: S_vec [Nfeat,]

		Using now the N-dimensional formulation from Rasmussen

		Outputs a vector 

		Matern kernel spectral density
		"""

		S_vec = self.const / ((self.lambda_val**2 + omega_in**2)**(self.nu+self.dim*0.5)) # Using omega directly (Sarkka) as opposed to 4pi*s (rasmsusen)

		# Multiply dimensions (i.e., assume that matern density factorizes w.r.t omega input dimensionality). Also, pump variance into it:
		S_vec = tf.reduce_prod(S_vec,axis=1)*self.prior_var

		if log == True:
			return tf.math.log(S_vec)

		return S_vec

	def get_samples(self,Nsamples,args):

		log_likelihood_fn = lambda omega_in: self.unnormalized_density(omega_in,log=True)
		
		return super().sampleESS(log_likelihood_fn,Nsamples)

	def update_pars(self,args):
		pass
		# raise NotImplementedError("Child Class MaternSpectralDensity")

	def normalized_density(self):
		raise NotImplementedError("Child Class MaternSpectralDensity")

	def log_normalized_density(self):
		raise NotImplementedError("Child Class MaternSpectralDensity")


class CartPoleSpectralDensity(SpectralDensity):

	def __init__(self,cfg: dict):
		super().__init__()

		# Parameters:
		self.fric = cfg.model_pars.friction_cart
		self.r = cfg.model_pars.length_pend
		self.m = cfg.model_pars.mass
		self.g = cfg.model_pars.gravity

		# Linearization point:
		self.phi0 = None
		self.u0 = None

		# Numerical zero for omega:
		self.omega_lim = 1e-5
		self.log_omega_lim = tf.math.log(1e-5)

		# Pre-sample for the complicated cases of S_3(w) and S_4(w):
		self.first_time = True
		# logger.info("Pre-sampling ")
		# Nsamples = 1000 # TODO: Do not hard-code this!
		# self.samples_vec_x3 = self.get_samples(Nsamples,2)
		# self.samples_vec_x4 = self.get_samples(Nsamples,3)

	def update_pars(self,args):
		self.update_linearization_point(args["phi0"],args["u0"])

	def update_linearization_point(self,phi0,u0):

		self.phi0 = phi0
		self.u0 = u0

		# Update matrix entries:
		self.a21 = (self.g * math.cos(self.phi0) + self.u0*math.sin(self.phi0)) / self.r
		self.a22 = -self.fric / (self.m*self.r**2)
		self.b2 = -math.cos(self.phi0)/self.r

	def unnormalized_density(self,omega,state_ind,log=True):

		if log == True:

			if state_ind == 0:
				return self._log_unnormalized_density_x1(omega)
			elif state_ind == 1:
				return self._log_unnormalized_density_x2(omega)
			elif state_ind == 2:
				return self._log_unnormalized_density_x3(omega)
			elif state_ind == 3:
				return self._log_unnormalized_density_x4(omega)
			else:
				raise ValueError("Invalid state")

		else:

			if state_ind == 0:
				return self._unnormalized_density_x1(omega)
			elif state_ind == 1:
				return self._unnormalized_density_x2(omega)
			elif state_ind == 2:
				return self._unnormalized_density_x3(omega)
			elif state_ind == 3:
				return self._unnormalized_density_x4(omega)
			else:
				raise ValueError("Invalid state")

	def _unnormalized_density_x1(self,omega):
		return tf.math.abs(self.b2) / tf.sqrt( (self.a21 + omega**2)**2 + (omega*self.a22)**2 )

	def _unnormalized_density_x2(self,omega):
		return tf.math.abs(omega)*self._unnormalized_density_x1(omega)

	def _unnormalized_density_x3(self,omega):

		if tf.math.abs(omega) < self.omega_lim:
			omega_val = self.omega_lim
			# logger.warning("[WARNING]: omega close to zero!")
		else:
			omega_val = tf.math.abs(omega)

		return 1./omega_val**2

	def _unnormalized_density_x4(self,omega):

		if tf.math.abs(omega) < self.omega_lim:
			omega_val = self.omega_lim
			# logger.warning("[WARNING]: omega close to zero!")
		else:
			omega_val = tf.math.abs(omega)

		return 1./omega_val

	def _log_unnormalized_density_x1(self,omega):
		return tf.math.log(tf.math.abs(self.b2)) - 0.5*tf.math.log( (self.a21 + omega**2)**2 + (omega*self.a22)**2 )

	def _log_unnormalized_density_x2(self,omega):
		return tf.math.log(tf.math.abs(omega)) + self._log_unnormalized_density_x1(omega)

	def _log_unnormalized_density_x3(self,omega):
		
		if tf.math.abs(omega) < self.omega_lim:
			log_val = self.log_omega_lim
			# logger.warning("[WARNING]: omega close to zero!")
		else:
			log_val = tf.math.log(tf.math.abs(omega))

		return -2.0*log_val

	def _log_unnormalized_density_x4(self,omega):

		if tf.math.abs(omega) < self.omega_lim:
			log_val = self.log_omega_lim
			# logger.warning("[WARNING]: omega close to zero!")
		else:
			log_val = tf.math.log(tf.math.abs(omega))
			
		return -log_val

	def get_samples(self,Nsamples,state_ind):

		# Pre-sampling and storing for x3, as it is too slow:
		if state_ind == 2 and not self.first_time:
			return self.samples_vec_x3
		elif state_ind == 2:
			self.first_time = False
			log_likelihood_fn = lambda omega_in: self.unnormalized_density(omega_in,state_ind,log=True)
			self.samples_vec_x3 = super().sampleESS(log_likelihood_fn,Nsamples)
			return self.samples_vec_x3

		# Pre-sampling and storing for x4, as it is too slow:
		if state_ind == 3 and not self.first_time:
			return self.samples_vec_x4
		elif state_ind == 3:
			self.first_time = False
			log_likelihood_fn = lambda omega_in: self.unnormalized_density(omega_in,state_ind,log=True)
			self.samples_vec_x4 = super().sampleESS(log_likelihood_fn,Nsamples)
			return self.samples_vec_x4

		log_likelihood_fn = lambda omega_in: self.unnormalized_density(omega_in,state_ind,log=True)
		return super().sampleESS(log_likelihood_fn,Nsamples)

	def get_cont_time_linear_system(self):
		"""
		TODO: Remove if not needed
		"""

		if self.a21 is None or self.a22 is None or self.b2 is None:
			raise ValueError("Supply linearization point first!")

		# Matrix A:
		A = tf.Tensor([	[0,   			  1, 0, 0],
						[self.a21, self.a22, 0, 0],
						[0,   			  0, 0, 1],
						[0,   			  0, 0, 0]])

		# Matrix B:
		B = tf.Tensor([[0,self.b2,0,1]])

		return A,B

	def normalized_density(self):
		raise NotImplementedError("Child Class CartPoleSpectralDensity")

	def log_normalized_density(self):
		raise NotImplementedError("Child Class CartPoleSpectralDensity")



class MultiDimensionalFourierTransformQuadratureFromData(SpectralDensity):

	def __init__(self,cfg: dict, X, Yproj):
		super().__init__()

		assert X.shape[0] == Yproj.shape[0] # Trajectories x_{i,t+1} = f_i(x_t)
		assert Yproj.shape[1] == 1

		self.X = X # [Npoints,dim]
		self.Yproj = Yproj # [Npoints,1]

	def unnormalized_density(self,omega_in,log=True):
		"""

		Approximate the quadrature of a multivariate Fourier transform using data

		omega_in: [Npoints_omega, dim]
		
		"""

		Sw_vec = self.quadrature_multivariate_Fourier_from_data(omega_in=omega_in,
																X=self.X,
																Yproj=self.Yproj,
																log=log)

		return Sw_vec

	@staticmethod
	def quadrature_multivariate_Fourier_from_data(omega_in,X,Yproj,log=True,get_parts=False):
		"""

		Approximate the quadrature of a multivariate Fourier transform using data

		omega_in: [Npoints_omega, dim]
		X: [Npoints, dim]
		Yproj: [Npoints, 1]
		
		"""


		omega_times_X = X @ tf.transpose(omega_in)  # [Npoints, Npoints_omega]
		real_part = tf.reduce_mean(Yproj * tf.math.cos(omega_times_X),axis=0)
		img_part = tf.reduce_mean(Yproj * tf.math.sin(omega_times_X),axis=0)

		if get_parts:
			return tf.reshape(real_part,(-1,1)), -tf.reshape(img_part,(-1,1))

		if log:
			# TODO: Check for numerical zero
			Sw_vec = 0.5*tf.math.log(real_part**2 + img_part**2)
		else:
			Sw_vec = tf.math.sqrt(real_part**2 + img_part**2)


		Sw_vec = tf.reshape(Sw_vec,(-1,1))			

		return Sw_vec


	def get_samples(self,Nsamples,state_ind):
		log_likelihood_fn = lambda omega_in: self.unnormalized_density(omega_in,log=True)
		return super().sampleESS(log_likelihood_fn,Nsamples)

	def normalized_density(self):
		raise NotImplementedError("Child Class MultiDimensionalFourierTransformQuadratureFromData")

	def log_normalized_density(self):
		raise NotImplementedError("Child Class MultiDimensionalFourierTransformQuadratureFromData")

	def update_pars(self,args):
		pass


class MultiDimensionalFourierTransformQuadratureCartPoleStructure(SpectralDensity):

	def __init__(self,cfg: dict, X, u_vec, ind):
		super().__init__()

		"""

		TODOs:
		1) Add numerical zeroes
		2) Add log functions
		3) Pass the policy, instead of the u signal. In principle, we may want to integrate over the whole domain, and for that we need the policy pi(x)
		5) Fix the TODOs on the parent class, sampling part. Also, test this on test_MultiDimensionalFourierTransformQuadratureCartPoleStructure.py
		6) Make the (symmetric) hypercubic domain input-dependent, i.e., L_i instead of a global L. Also, self.L is hard-coded right now...

		"""

		# assert X.shape[0] == Yproj.shape[0] # Trajectories x_{i,t+1} = f_i(x_t)
		# assert Yproj.shape[1] == 1

		# self.policy_fun = policy_fun
		self.u_vec = u_vec

		self.X = X # [Npoints,dim]
		# self.Yproj = Yproj # [Npoints,1]

		# Define hypercubic symmetric domain:
		self.L = 5.0

		self.which_state = ind

		# Parameters:
		self.deltaT = cfg.system_pars.deltaT
		self.a = cfg.model_pars.gravity / cfg.model_pars.length_pend
		self.b = -1./cfg.model_pars.length_pend
		self.c = -cfg.model_pars.friction_cart/(cfg.model_pars.mass*cfg.model_pars.length_pend**2)

	def definite_integral_constant(self,omega_in,ignore_const=True,abs_value=True):
		"""

		Univariate integral

		int_{-L}^{L} e^(-j w x) dx

		omega_in: [Npoints,dim]

		"""

		dim = omega_in.shape[1]
		const = 2.0**dim
		if ignore_const:
			const = 1.0

		Sw_vec_all_dim = tf.math.sin(self.L * omega_in) / omega_in
		if abs_value:
			Sw_vec_all_dim = tf.math.abs(Sw_vec_all_dim)

		Sw_vec = tf.reshape(tf.reduce_prod(Sw_vec_all_dim,axis=1),(-1,1))

		return const * Sw_vec

	def definite_integral_linear(self,omega_in,ignore_const=True,abs_value=True):
		"""

		Univariate integral

		int_{-L}^{L} e^(-j w x) x dx

		omega_in: [Npoints,1]

		"""
		assert omega_in.shape[1] == 1

		const = 2.0
		if ignore_const:
			const = 1.0

		Sw_vec = (tf.math.sin(self.L * omega_in) - self.L * omega_in) / (omega_in**2)
		if abs_value:
			Sw_vec = tf.math.abs(Sw_vec)

		return const * Sw_vec


	def definite_integral_sinx(self,omega_in,ignore_const=True,abs_value=True):
		"""

		Univariate integral

		int_{-L}^{L} e^(-j w x) sin(x) dx

		omega_in: [Npoints,1]

		"""

		assert omega_in.shape[1] == 1

		const = 2.0
		if ignore_const:
			const = 1.0


		sign = 1
		if not abs_value:
			sign = -1

		num_vec = tf.math.cos(self.L) * tf.math.sin(self.L * omega_in) - 2.0*omega_in*tf.math.sin(self.L) * tf.math.cos(self.L * omega_in)
		den_vec = omega_in**2 - 1.0

		Sw_vec = num_vec / den_vec

		if abs_value:
			Sw_vec = tf.math.abs(Sw_vec)

		return sign * const * Sw_vec


	def delete_one_dimension(self,omega_in,ind):
		"""

		Return a copy of the input vector, but delete the column indexed by ind

		omega_in: [Npoints,dim]
		"""

		dim = omega_in.shape[1]
		assert ind < dim

		omega_in_left = omega_in[:,0:ind]
		omega_in_right = omega_in[:,ind+1::]

		omega_in_reducted = tf.concat((omega_in_left,omega_in_right),axis=1)
		return omega_in_reducted

	def definite_multivariate_integral_singled_out_linear(self,omega_in,ind,abs_value=True):
		"""

		Multivariate integral

		int_{X} e^(-j w^T x) x_i  dx, where x in R^{n}, w in R^{n}, x_i in R

		The linear term x_i in the integrand corresponds to the element [x]_i.
		The integrand factorizes, and the contribution of e^(-j w_i^T x_i)x_i can be
		computed separately.

		omega_in: [Npoints,dim]
		ind: index that corresponds to x_i inside the vector x

		abs_value: By setting this to false, we return the complex number 0 -j*integral(...), which happens to have
		only an imaginary part.
		"""

		sign = +1
		if not abs_value:
			sign = -1

		omega_in_no_x_i = self.delete_one_dimension(omega_in,ind=ind) # Constant part, no x_i
		const_part_no_x_i = self.definite_integral_constant(omega_in_no_x_i,abs_value=abs_value)
		linear_part_with_x_i = self.definite_integral_linear(omega_in[:,ind:ind+1],abs_value=abs_value)

		Sw_vec = sign * const_part_no_x_i * linear_part_with_x_i

		# pdb.set_trace()

		return Sw_vec


	def _unnormalized_density_x1(self,omega_in):


		part_x2 = self.definite_multivariate_integral_singled_out_linear(omega_in,ind=1)
		part_x1 = self.definite_multivariate_integral_singled_out_linear(omega_in,ind=0)

		Sw_vec = self.deltaT * part_x2 + part_x1

		return Sw_vec


	def _unnormalized_density_x2(self,omega_in):


		# Part sin(x1):
		part_sinx1_img = self.definite_integral_sinx(omega_in[:,0:1],abs_value=False) * self.definite_integral_constant(self.delete_one_dimension(omega_in,ind=0),abs_value=False) # The transfer function is 0 -j*integral(...)

		# Part u*cos(x1):
		Y_u_cosx1 = self.u_vec*tf.math.cos(self.X[:,0:1])
		part_u_cosx1_real, part_u_cosx1_img = MultiDimensionalFourierTransformQuadratureFromData.quadrature_multivariate_Fourier_from_data(	omega_in=omega_in,
																												X=self.X,
																												Yproj=Y_u_cosx1,
																												log=False,
																												get_parts=True)

		# Part x2:
		part_x2_img = self.definite_multivariate_integral_singled_out_linear(omega_in,ind=1,abs_value=False) # The transfer function is 0 -j*integral(...)
		
		# Parts real and imaginary:
		part_total_real = part_u_cosx1_real
		part_total_img = self.deltaT*self.a*part_sinx1_img + self.deltaT*self.b*part_u_cosx1_img + (self.c*self.deltaT+1)*part_x2_img


		# Spectral density:
		Sw_vec = tf.math.sqrt(part_total_real**2 + part_total_img**2)

		return Sw_vec



	def _unnormalized_density_x3(self,omega_in):

		part_x4 = self.definite_multivariate_integral_singled_out_linear(omega_in,ind=3)
		part_x3 = self.definite_multivariate_integral_singled_out_linear(omega_in,ind=2)

		Sw_vec = self.deltaT * part_x4 + part_x3

		return Sw_vec


	def _unnormalized_density_x4(self,omega_in):

		part_x4_img = self.definite_multivariate_integral_singled_out_linear(omega_in,ind=3,abs_value=False) # The transfer function is 0 -j*integral(...)
		part_u_real, part_u_img = MultiDimensionalFourierTransformQuadratureFromData.quadrature_multivariate_Fourier_from_data(	omega_in=omega_in,
																												X=self.X,
																												Yproj=self.u_vec,
																												log=False,
																												get_parts=True)

		part_total_real = self.deltaT*part_u_real
		part_total_img = self.deltaT*part_u_img + part_x4_img
		Sw_vec = tf.math.sqrt(part_total_real**2 + part_total_img**2)

		return Sw_vec


	def get_samples(self,Nsamples,state_ind):
		log_likelihood_fn = lambda omega_in: self.unnormalized_density(omega_in,log=True)
		return super().sampleESS(log_likelihood_fn,Nsamples)

	def unnormalized_density(self,omega,log=True):

		if log == True:

			if self.which_state == 0:
				return self._log_unnormalized_density_x1(omega)
			elif self.which_state == 1:
				return self._log_unnormalized_density_x2(omega)
			elif self.which_state == 2:
				return self._log_unnormalized_density_x3(omega)
			elif self.which_state == 3:
				return self._log_unnormalized_density_x4(omega)
			else:
				raise ValueError("Invalid state")

		else:

			if self.which_state == 0:
				return self._unnormalized_density_x1(omega)
			elif self.which_state == 1:
				return self._unnormalized_density_x2(omega)
			elif self.which_state == 2:
				return self._unnormalized_density_x3(omega)
			elif self.which_state == 3:
				return self._unnormalized_density_x4(omega)
			else:
				raise ValueError("Invalid state")

	def normalized_density(self):
		raise NotImplementedError("Child Class MultiDimensionalFourierTransformQuadratureFromData")

	def log_normalized_density(self):
		raise NotImplementedError("Child Class MultiDimensionalFourierTransformQuadratureFromData")

	def update_pars(self,args):
		pass


		
