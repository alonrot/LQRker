import tensorflow as tf
import pdb
import math
from abc import ABC, abstractmethod
import tensorflow_probability as tfp
from lqrker.utils.parsing import get_logger
from ood.utils.common import CommonUtils
import numpy as np
logger = get_logger(__name__)

class SpectralDensityBase(ABC):
	"""
	Collection of Spectral densities, in correspondance with a particular type of
	dynamical system.

	Base class
	"""

	def __init__(self,cfg_samplerHMC,dim):

		self.num_burnin_steps = cfg_samplerHMC.num_burnin_steps
		self.Nsamples_per_state0 = cfg_samplerHMC.Nsamples_per_state0
		self.initial_states_sampling = eval(cfg_samplerHMC.initial_states_sampling)
		self.step_size_hmc = cfg_samplerHMC.step_size_hmc
		self.num_leapfrog_steps_hmc = cfg_samplerHMC.num_leapfrog_steps_hmc
		self.dim = dim
		assert self.Nsamples_per_state0 % 2 == 0, "Need an even number, for now"
		self.adaptive_hmc = None
		self.Sw_points, self.phiw_points, self.W_points = None, None, None

	@abstractmethod
	def unnormalized_density(self):
		raise NotImplementedError

	# @abstractmethod
	def normalized_density(self):
		raise NotImplementedError

	# @abstractmethod
	def log_normalized_density(self):
		raise NotImplementedError

	# @abstractmethod
	def update_pars(self,args):
		raise NotImplementedError

	def initialize_HMCsampler(self,log_likelihood_fn):
		"""

		Notes from [1]:

		 - Tuning HMC will usually require preliminary runswith trial values for step_size and num_leapfrog_steps
		 - The choice of stepsize is almost independent ofhow many leapfrog steps are done

		 - Too large a stepsize will result in a very low acceptance rate for states proposed by simulating trajectories. 
		 - Too small a stepsize will either waste computation time, by the same factor as the stepsize is toosmall, or (worse) 
		   will lead to slow exploration by a random walk, if the trajectory length step_size*num_leapfrog_steps is then too short.

		Notes from [2]:
		 - The output of target_log_prob_fn(*current_state) should sum log-probabilities across all event dimensions
		 - target_log_prob_fn needs to be proportional to log p(x)
		 - Slices along the rightmost dimensions may have different target distributions; 
		   for example, current_state[0, :] could have a different target distribution from current_state[1, :]
		   
		   [my own conclusion; might be wrong]: So, basically, the first index of current_state accounts for the output dimensionality. For example, if we have
		   S(w) = [S_1(w),...S_D(w)], where S(w) is a vector with D channels. So, current_state = [d,w], where d selects the dimensionality.


		[1] Neal, R.M., 2011. MCMC using Hamiltonian dynamics. Handbook of markov chain monte carlo, 2(11), p.2.
		[2] https://www.tensorflow.org/probability/api_docs/python/tfp/mcmc/HamiltonianMonteCarlo


		"""

		logger.info("Initializing tfp.mcmc.HamiltonianMonteCarlo()...")
		adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(	
			tfp.mcmc.HamiltonianMonteCarlo(
					target_log_prob_fn=log_likelihood_fn,
					step_size=self.step_size_hmc, # Step size of the leapfrog integration method. If too large, the trajectories can diverge. If we want a different step size for each dimension, a tensor can be entered as well.
					num_leapfrog_steps=self.num_leapfrog_steps_hmc, # How many steps forward the Hamiltonian dynamics are simulated
					state_gradients_are_stopped=False,
					step_size_update_fn=None,
					store_parameters_in_results=False,
					experimental_shard_axis_names=None,
					name=None),
				num_adaptation_steps=int(self.num_burnin_steps * 0.8))

		return adaptive_hmc

	def get_samples_HMC(self,log_likelihood_fn,Nsamples=None):
		"""

		self.Nsamples_per_state0: int
		self.initial_states_sampling: [Nstates0,dim]
		self.num_burnin_steps: int
		"""

		if self.adaptive_hmc is None:
			self.adaptive_hmc = self.initialize_HMCsampler(log_likelihood_fn)

		if Nsamples is not None:
			self.Nsamples_per_state0 = Nsamples

		Nsamples_per_restart_half = self.Nsamples_per_state0 // 2

		logger.info("Getting MCMC chains for {0} states, with {1} samples each; total: {2} samples".format(self.initial_states_sampling.shape[0],self.Nsamples_per_state0,self.initial_states_sampling.shape[0]*self.Nsamples_per_state0))
		samples, is_accepted = tfp.mcmc.sample_chain(
			num_results=Nsamples_per_restart_half,
			num_burnin_steps=self.num_burnin_steps,
			current_state=self.initial_states_sampling,
			kernel=self.adaptive_hmc,
			trace_fn=lambda _, pkr: pkr.inner_results.is_accepted)

		samples = tf.reshape(samples,(-1,self.dim))
		samples = tf.concat([samples,-samples],axis=0)

		return samples

	def get_normalization_constant_numerical(self,omega_vec):
		"""

		omega_vec: [Npoints,dim]
		return:
			const: scalar
		"""

		Sw, _ = self.unnormalized_density(omega_vec)
		Dw_prod = (omega_vec[1,-1] - omega_vec[0,-1])**self.dim # Equivalent to math.pi/L for self.spectral_density.get_Wpoints_discrete()
		const = tf.math.reduce_sum(Sw*Dw_prod,axis=0) # [self.dim,]

		return const

	def get_Wsamples_from_Sw(self,Nsamples=None):

		# Get samples:
		log_likelihood_fn = lambda omega_in: self.unnormalized_density(omega_in,log=True)
		W_samples_vec = self.get_samples_HMC(log_likelihood_fn,Nsamples)

		# Evaluate spectral density and argument:
		Sw_vec, phiw_vec = self.unnormalized_density(W_samples_vec)

		# Coarse normalization:
		# Sw_vec_nor = Sw_vec / tf.math.reduce_sum(Sw_vec)
		Sw_vec_nor = Sw_vec

		return Sw_vec_nor, phiw_vec, W_samples_vec


	def get_Wpoints_discrete(self,L,Ndiv,normalize_density_numerically=False,reshape_for_plotting=False):

		assert Ndiv % 2 != 0 and Ndiv > 2, "Ndiv must be an odd positive integer"

		j_indices = CommonUtils.create_Ndim_grid(xmin=-(Ndiv-1)//2,xmax=(Ndiv-1)//2,Ndiv=Ndiv,dim=self.dim) # [Ndiv**dim_x,dim_x]

		omegapred = tf.cast((math.pi/L) * j_indices,dtype=tf.float32)

		Sw_vec, phiw_vec = self.unnormalized_density(omegapred)

		if normalize_density_numerically:
			raise NotImplementedError("Check for dim > 1")
			normalization_constant_kernel = self.get_normalization_constant_numerical(omegapred)
			logger.info("normalization_constant_kernel: "+str(normalization_constant_kernel))
			Sw_vec = Sw_vec / normalization_constant_kernel

		if reshape_for_plotting and self.dim == 2:

			S_vec_plotting = [ np.reshape(Sw_vec[:,ii],(Ndiv,Ndiv)) for ii in range(self.dim) ]
			Sw_vec = np.stack(S_vec_plotting) # [dim, Ndiv, Ndiv]

			if np.any(phiw_vec != 0.0):
				phiw_vec_plotting = [ np.reshape(phiw_vec[:,ii],(Ndiv,Ndiv)) for ii in range(self.dim) ]
				phiw_vec = np.stack(phiw_vec_plotting) # [dim, Ndiv, Ndiv]

		return Sw_vec, phiw_vec, omegapred


	def get_Wpoints_on_regular_grid(self,omega_min=-5.,omega_max=+5.,Ndiv=51,normalize_density_numerically=False,reshape_for_plotting=False):
		"""

		return (with reshape_for_plotting=False):
			Sw_vec: [Nfeatures,self.dim]
			phiw_vec: [Nfeatures,self.dim]
			omegapred: [Nfeatures,self.dim]

			Nfeatures = Ndiv**self.dim

		"""

		omegapred = CommonUtils.create_Ndim_grid(xmin=omega_min,xmax=omega_max,Ndiv=Ndiv,dim=self.dim) # [Ndiv**dim_x,dim_x]
		Sw_vec, phiw_vec = self.unnormalized_density(omegapred)

		if normalize_density_numerically:
			raise NotImplementedError("Check for dim > 1")
			normalization_constant_kernel = self.get_normalization_constant_numerical(omegapred)
			logger.info("normalization_constant_kernel: "+str(normalization_constant_kernel))
			Sw_vec = Sw_vec / normalization_constant_kernel

		if reshape_for_plotting and self.dim == 2:

			S_vec_plotting = [ np.reshape(Sw_vec[:,ii],(Ndiv,Ndiv)) for ii in range(self.dim) ]
			Sw_vec = np.stack(S_vec_plotting) # [dim, Ndiv, Ndiv]

			if np.any(phiw_vec != 0.0):
				phiw_vec_plotting = [ np.reshape(phiw_vec[:,ii],(Ndiv,Ndiv)) for ii in range(self.dim) ]
				phiw_vec = np.stack(phiw_vec_plotting) # [dim, Ndiv, Ndiv]

		return Sw_vec, phiw_vec, omegapred

	def update_Wsamples(self,Nsamples=None):
		self.Sw_points, self.phiw_points, self.W_points = self.get_Wsamples_from_Sw(Nsamples)

	def update_Wpoints_regular(self,omega_min=-5.,omega_max=+5.,Ndiv=51,normalize_density_numerically=False,reshape_for_plotting=False):
		self.Sw_points, self.phiw_points, self.W_points = self.get_Wpoints_on_regular_grid(omega_min,omega_max,Ndiv,normalize_density_numerically,reshape_for_plotting)

	def update_Wpoints_discrete(self,L,Ndiv,normalize_density_numerically=False,reshape_for_plotting=False):
		self.Sw_points, self.phiw_points, self.W_points = self.get_Wpoints_discrete(L,Ndiv,normalize_density_numerically,reshape_for_plotting)

