import tensorflow as tf
import pdb
from abc import ABC, abstractmethod
import tensorflow_probability as tfp
from lqrker.utils.parsing import get_logger
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

	def get_samples(self,Nsamples=None):

		# Get samples:
		log_likelihood_fn = lambda omega_in: self.unnormalized_density(omega_in,log=True)
		W_samples_vec = self.get_samples_HMC(log_likelihood_fn,Nsamples)

		# Evaluate spectral density and argument:
		Sw_vec, phiw_vec = self.unnormalized_density(W_samples_vec)

		# Coarse normalization:
		# Sw_vec_nor = Sw_vec / tf.math.reduce_sum(Sw_vec)
		Sw_vec_nor = Sw_vec

		return W_samples_vec, Sw_vec_nor, phiw_vec

	def get_normalization_constant_numerical(self,omega_vec):
		"""

		omega_vec: [Npoints,dim]
		return:
			const: scalar
		"""

		Sw, _ = self.unnormalized_density(omega_vec)
		dw = omega_vec[1,0] - omega_vec[0,0]
		const = tfp.math.trapz(y=Sw,dx=dw)

		return const


