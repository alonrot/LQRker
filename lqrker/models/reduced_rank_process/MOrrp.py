import tensorflow as tf
import tensorflow_probability as tfp
import math
import pdb
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sys
from datetime import datetime
import pickle

from lqrker.models import RRPRegularFourierFeatures, RRPDiscreteCosineFeatures, RRPLinearFeatures, RRPDiscreteCosineFeaturesVariableIntegrationStep
from lqrker.spectral_densities.base import SpectralDensityBase
from lqrker.utils.parsing import get_logger
logger = get_logger(__name__)

markersize_x0 = 10
markersize_trajs = 0.4
fontsize_labels = 25
matplotlib.rc('xtick', labelsize=fontsize_labels)
matplotlib.rc('ytick', labelsize=fontsize_labels)
matplotlib.rc('text', usetex=True)
matplotlib.rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
plt.rc('legend',fontsize=fontsize_labels+2)


class MultiObjectiveReducedRankProcess(tf.keras.layers.Layer):

	# @tf.function
	def __init__(self, dim_in: int, cfg: dict, spectral_density: SpectralDensityBase, Xtrain, Ytrain, using_deltas=False, learn_correlation_noise=False, **kwargs):

		"""
		
		Initialize "dim" number of RRPRegularFourierFeatures() models, one per output channel		

		We use a single spectral density instance for all the models. For that instance, 
		we compute the needed frequencies and use them throughout all the models

		"""


		super().__init__(**kwargs)


		self.dim_in = dim_in
		self.dim_out = Ytrain.shape[1]
		self.using_deltas = using_deltas
		self.learn_correlation_noise = learn_correlation_noise

		assert cfg.gpmodel.which_features in ["RRPLinearFeatures", "RRPDiscreteCosineFeatures", "RRPRegularFourierFeatures", "RRPRandomFourierFeatures","RRPDiscreteCosineFeaturesVariableIntegrationStep"]

		self.rrgpMO = [None]*self.dim_out
		for ii in range(self.dim_out):

			if cfg.gpmodel.which_features == "RRPLinearFeatures":
				self.rrgpMO[ii] = RRPLinearFeatures(dim=self.dim_in,cfg=cfg.gpmodel,spectral_density=spectral_density,dim_out_ind=ii)
			elif cfg.gpmodel.which_features == "RRPDiscreteCosineFeatures":
				self.rrgpMO[ii] = RRPDiscreteCosineFeatures(dim=self.dim_in,cfg=cfg.gpmodel,spectral_density=spectral_density,dim_out_ind=ii)

				# self.rrgpMO[ii].update_model(Xtrain,Ytrain[:,ii:ii+1]) # Update model indexing the target outputs at the corresponding dimension
				# self.rrgpMO[ii].get_MLII_loss_gaussian_predictive(xpred=Xtrain)

			elif cfg.gpmodel.which_features == "RRPDiscreteCosineFeaturesVariableIntegrationStep":
				# self.rrgpMO[ii] = RRPDiscreteCosineFeaturesVariableIntegrationStep(dim=self.dim_in,cfg=cfg.gpmodel,spectral_density=spectral_density,dim_out_ind=ii)
				self.rrgpMO[ii] = RRPDiscreteCosineFeaturesVariableIntegrationStep(dim=self.dim_in,cfg=cfg.gpmodel,spectral_density=spectral_density[ii],dim_out_ind=ii)

			elif cfg.gpmodel.which_features == "RRPRegularFourierFeatures":
				self.rrgpMO[ii] = RRPRegularFourierFeatures(dim=self.dim_in,cfg=cfg.gpmodel,spectral_density=spectral_density,dim_out_ind=ii)
			elif cfg.gpmodel.which_features == "RRPRandomFourierFeatures":
				self.rrgpMO[ii] = RRPRandomFourierFeatures(dim=self.dim_in,cfg=cfg.gpmodel,spectral_density=spectral_density,dim_out_ind=ii)

			# self.rrgpMO[ii].select_output_dimension(dim_out_ind=ii)


		# Weights are initialized inside each self.rrgpMO[ii] and accesible from this upper layer via self.get_weights()

		# # Weights:
		# dbg_fac = 2.5
		# self.log_noise_std_vec = self.add_weight(shape=(self.dim_out), initializer=tf.keras.initializers.Constant(dbg_fac+tf.math.log(cfg.gpmodel.hyperpars.noise_std_process)), trainable=True,name="log_noise_std_vec")
		# self.update_weights_in_individual_models()



		# Correlate states by means of a noise correlation matrix:
		dbg_flag = False
		dim_chol_corr_noise_mat = (self.dim_out * (self.dim_out + 1)) // 2
		if self.learn_correlation_noise and dbg_flag:
			self.chol_corr_noise_mat_vectorized = 0.1*tf.ones(dim_chol_corr_noise_mat)
		elif self.learn_correlation_noise:
			initializer_corr_noise_mat = tf.keras.initializers.RandomUniform(minval=-cfg.gpmodel.hyperpars.chol_corr_noise_mat_lim_init, maxval=cfg.gpmodel.hyperpars.chol_corr_noise_mat_lim_init)
			regularizer_corr_noise_mat = tf.keras.regularizers.L1(l1=0.01) # https://www.tensorflow.org/api_docs/python/tf/keras/regularizers/L1
			self.chol_corr_noise_mat_vectorized = self.add_weight(	shape=(dim_chol_corr_noise_mat,), initializer=initializer_corr_noise_mat,\
																	regularizer=regularizer_corr_noise_mat, trainable=True, name="corr_noise_mat")

		self.update_model(X=Xtrain,Y=Ytrain)

	# @tf.function
	def update_model(self,X,Y):

		# self.update_weights_in_individual_models()
		# pdb.set_trace()

		for ii in range(self.dim_out):
			logger.info("Updating model for output dimension {0:d} / {1:d} ...".format(ii+1,self.dim_out))
			self.rrgpMO[ii].update_model(X,Y[:,ii:ii+1]) # Update model indexing the target outputs at the corresponding dimension

	# @tf.function
	def update_features(self):
		for ii in range(self.dim_out):
			self.rrgpMO[ii]._update_features()
			self.rrgpMO[ii].prior_beta_already_computed = False
			self.rrgpMO[ii].predictive_beta_already_computed = False

	@tf.function
	# def update_weights_in_individual_models(self):
	# 	"""
	# 	Update weights to all independent GPs
	# 	"""
	# 	for ii in range(self.dim_out):
	# 		self.rrgpMO[ii].log_noise_std.assign([self.log_noise_std_vec[ii]])
	# 		# self.rrgpMO[ii].log_noise_std[0].assign(self.log_noise_std_vec[ii]) # DBG/TODO: alternative

	# @tf.function
	def get_training_data(self):
		X = self.rrgpMO[0].X
		Y_ii = []
		for ii in range(self.dim_out):
			Y_ii += [self.rrgpMO[ii].Y]
		Y = tf.concat(Y_ii,axis=1)

		return tf.identity(X), tf.identity(Y) # Copy tensor

	# @tf.function
	def get_log_noise_std_vec(self):
		"""
		NOTE: This assumes that the noise matrix is diagonal
		"""
		log_noise_std_vec = tf.TensorArray(dtype=tf.float32,size=self.dim_out,dynamic_size=False)
		for ii in range(self.dim_out):
			log_noise_std_vec = log_noise_std_vec.write(ii,self.rrgpMO[ii].get_log_noise_std())
		return tf.squeeze(log_noise_std_vec.stack()) # [self.dim_out,]

	# @tf.function
	def get_noise_var_vec(self):
		"""
		NOTE: This assumes that the noise matrix is diagonal
		"""
		# noise_var_vec = [None]*self.dim_out
		noise_var_vec = tf.TensorArray(dtype=tf.float32,size=self.dim_out,dynamic_size=False)
		for ii in range(self.dim_out):
			noise_var_vec = noise_var_vec.write(ii,self.rrgpMO[ii].get_noise_var())
		return tf.squeeze(noise_var_vec.stack()) # [self.dim_out,]

	def get_chol_corr_noise_mat(self):		
		return tfp.math.fill_triangular(self.chol_corr_noise_mat_vectorized, upper=False)

	# @tf.function
	def predict_at_locations(self,xpred,from_prior=False):
		"""

		xpred: [Npoints,self.dim]

		return:
			MO_mean_pred: [Npoints,self.dim]
			MO_std_pred: [Npoints,self.dim]

		"""

		MO_mean_pred = [None]*self.dim_out
		MO_std_pred = [None]*self.dim_out

		# Compute predictive moments:
		for ii in range(self.dim_out):
			MO_mean_pred[ii], cov_pred = self.rrgpMO[ii].predict_at_locations(xpred,from_prior)

			MO_mean_pred[ii] = tf.reshape(MO_mean_pred[ii],(-1,1))
			MO_std_pred[ii] = tf.reshape(tf.sqrt(tf.linalg.diag_part(cov_pred)),(-1,1))

		MO_mean_pred = tf.concat(MO_mean_pred,axis=1)
		MO_std_pred = tf.concat(MO_std_pred,axis=1)

		return MO_mean_pred, MO_std_pred


	# @tf.function
	def sample_path_from_predictive(self,xpred,Nsamples,from_prior=False):
		"""

		xpred: [Npoints,self.dim]
		samples_xpred: [Npoints,]*self.dim_out

		"""

		samples_xpred = [None]*self.dim_out
		for ii in range(self.dim_out):
			samples_xpred[ii] = self.rrgpMO[ii].sample_path_from_predictive(xpred,Nsamples,from_prior)

		return samples_xpred


	# @tf.function
	def get_sample_path_callable(self,Nsamples,from_prior=False,sample_mv0=None):

		def nonlinfun_sampled_callable(x):
			"""
			Args:
				x: [Npoints,self.dim_in]
			
			Returns:
				function
			"""

			out = []
			for ii in range(self.dim_out):
				fx_ii = self.rrgpMO[ii].get_sample_path_callable(Nsamples,from_prior,sample_mv0)
				out_ii = fx_ii(x) # [Npoints,Nsamples]
				out += [out_ii]

			return tf.stack(out,axis=1) # [Npoints,self.dim_out,Nsamples]

		return nonlinfun_sampled_callable

	# @tf.function
	def get_predictive_beta_distribution(self):
		"""

		return:
			MO_mean_beta: [self.dim_out,self.dim_in]
			MO_cov_beta_chol: [self.dim_out,self.dim_in,self.dim_in]
		"""

		MO_mean_beta = [None]*self.dim_out
		MO_cov_beta_chol = [None]*self.dim_out

		# Compute predictive moments:
		for ii in range(self.dim_out):
			MO_mean_beta[ii], MO_cov_beta_chol[ii] = self.rrgpMO[ii].get_predictive_beta_distribution()
			MO_mean_beta[ii] = tf.reshape(MO_mean_beta[ii],(1,-1))

		MO_mean_beta = tf.concat(MO_mean_beta,axis=0)
		MO_cov_beta_chol = tf.stack(MO_cov_beta_chol,axis=0)

		return MO_mean_beta, MO_cov_beta_chol


	# @tf.function
	def train_model(self,verbosity=False):
		for ii in range(self.dim_out):
			self.rrgpMO[ii].train_model(verbosity)

	# @tf.function
	def _rollout_model_given_control_sequence_tf(self,x0,Nsamples,Nrollouts,u_traj,traj_length,sort=False,plotting=False,dbg=True,verbo=False,str_progress_bar="",from_prior=False,sample_fx_once=False):
		"""

		Pass two initial latent values
		x0: [1,dim_x]
		u_traj: [traj_length,dim_u]
		
		return:
		xsamples_X: [Npoints,self.dim_out,Nsamples]
		xsamples_Y: [Npoints,self.dim_out,Nsamples]

		Regardless of self.using_deltas, this function returns the actual state, not the deltas
		
		TODO: Refactor as
		Nsamples_per_particle <- Nsamples
		Nparticles <- Npoints
		"""

		# Parsing arguments:
		if dbg:
			assert u_traj is not None, "For now, we assume that we pass a control sequence"
			if u_traj is not None: # Concatenate the inputs to the model as (x_t,u_t)
				if verbo: logger.info(" * Open-loop model x_{t+1} = f(x_t,u_t)")
				if verbo: logger.info(" * Input to the model: x0, and u_traj")
				assert x0.shape[1] == self.dim_out
				assert x0.shape[0] == 1
				assert self.dim_in == self.dim_out + u_traj.shape[1]
				assert traj_length == -1, "Pass -1 to emphasize that traj_length is inferred from u_traj"
				traj_length = u_traj.shape[0]
			else: # The input to the model is the state, directly
				if verbo: logger.info(" * Closed-loop model x_{t+1} = f(x_t)")
				if verbo: logger.info(" * Input to the model: x0")
				assert x0.shape[1] == self.dim_out
				assert x0.shape[1] == self.dim_in
				assert traj_length > 2

			assert Nsamples > 0

		assert Nsamples == 1, "We need one fx sample per roll-out"


		# Initialize:
		state_trajectory_per_rollout = tf.TensorArray(dtype=tf.float32,size=traj_length,dynamic_size=False)
		state_trajectories_all = tf.TensorArray(dtype=tf.float32,size=Nrollouts,dynamic_size=False) # [Nrollouts,traj_length,self.dim_out]

		if sample_fx_once: # Pre-sample the noise vector that wil characterize the models, one per roll-out. Keep the same sample (i.e., model) accross all calls to this function _rollout_model_given_control_sequence_tf()
			Nfeat = self.rrgpMO[0].S_samples_vec.shape[0]
			sample_mv0 = tf.random.normal(shape=(Nrollouts,Nfeat,Nsamples),mean=0.0,stddev=1.0) # [Nrollouts,Nfeat,Nsamples=1]
		else:
			sample_mv0 = [None]*Nrollouts


		if self.learn_correlation_noise:
			sample_mv0_noise_corr_mat = tf.random.normal(shape=(Nrollouts,traj_length-1,self.dim_out),mean=0.0,stddev=1.0) # [Nrollouts,Nfeat,Nsamples=1]
		
		for ss in range(Nrollouts):

			self._print_progess_bar(ii=ss,Ntot=Nrollouts,name=str_progress_bar+"Rolling out model ")

			fx = self.get_sample_path_callable(Nsamples=Nsamples,from_prior=from_prior,sample_mv0=sample_mv0[ss]) # resample: We sample the possible models only once, and keep the sample for all the rollouts

			state_curr = x0 # [Npoints,self.dim_in], with Npoints=1
			state_trajectory_per_rollout = state_trajectory_per_rollout.write(0, tf.reshape(x0,(self.dim_out)))

			for ii in range(0,traj_length-1):
				xsample_curr = tf.concat([state_curr,u_traj[ii:ii+1,:]],axis=1) # [ [Npoints,self.dim_in] , [1,dim_u] ], with Npoints=1
				state_next = fx(xsample_curr) # xsample_next: [Npoints,self.dim_out,Nsamples], with Npoints=1, Nsamples=1

				if self.using_deltas:
					state_next += tf.expand_dims(state_curr,axis=2)

				if self.learn_correlation_noise:
					state_next += self.get_chol_corr_noise_mat() @ tf.reshape(sample_mv0_noise_corr_mat[ss,ii,:],(-1,1))

				state_trajectory_per_rollout = state_trajectory_per_rollout.write(ii+1, tf.reshape(state_next,(self.dim_out)))
				state_curr =  tf.reshape(state_next,(1,self.dim_out))

			state_trajectories_all = state_trajectories_all.write(ss, state_trajectory_per_rollout.stack())

		xsamples_X = state_trajectories_all.stack()[:,0:-1,:] # [Nrollouts,traj_length-1,self.dim_out]
		xsamples_Y = state_trajectories_all.stack()[:,1::,:] # [Nrollouts,traj_length-1,self.dim_out]

		return xsamples_X, xsamples_Y



	# @tf.function
	def _get_negative_log_evidence_and_predictive_trajectory_chunk(self,Xstate_real,u_traj_real,Nsamples,Nrollouts,str_progress_bar="",from_prior=False,scale_loss_entropy=1.0,scale_prior_regularizer=1.0,sample_fx_once=False):
		"""

		Xstate_real: [Nrollouts,traj_length,self.dim_out], Nrollouts=1

		return:
			loss_val_avg: scalar [,]
			x_traj_pred: [Nrollouts,traj_length-1,self.dim_out]
			y_traj_pred: [Nrollouts,traj_length-1,self.dim_out]

		"""

		log_noise_std_vec = self.get_log_noise_std_vec()
		noise_var_vec = self.get_noise_var_vec()

		# Tensorflow:
		x0_tf = tf.convert_to_tensor(value=Xstate_real[0,0:1,:],dtype=tf.float32) # [Npoints,self.dim_in], with Npoints=1
		u_applied_tf = tf.convert_to_tensor(value=u_traj_real,dtype=tf.float32) # [Npoints,self.dim_in], with Npoints=1
		# Regardless of self.using_deltas, the function below returns the actual state, not the deltas
		x_traj_pred, y_traj_pred = self._rollout_model_given_control_sequence_tf(x0=x0_tf,Nsamples=1,Nrollouts=Nrollouts,u_traj=u_applied_tf,traj_length=-1,
																				sort=False,plotting=False,str_progress_bar=str_progress_bar,from_prior=from_prior,
																				sample_fx_once=sample_fx_once)
		# x_traj_pred: [Nrollouts,traj_length-1,self.dim_out]
		# y_traj_pred: [Nrollouts,traj_length-1,self.dim_out]

		# Compute the error excluding the initial state because Xstate_real[:,0,:] = x_traj_pred[:,0,:], which yields zero error and diesn't contribute
		error = (y_traj_pred - Xstate_real[:,1::,:])**2 # [Nrollouts,traj_length-1,self.dim_out]
		error_weighted = error / tf.reshape(noise_var_vec,(1,1,self.dim_out))

		# Loss || Averaged log-likelihood:
		term_data_fit_vec = -0.5*tf.math.reduce_sum(error_weighted) # [,]
		term_model_complexity_vec = -tf.math.reduce_sum(log_noise_std_vec)*y_traj_pred.shape[0]*y_traj_pred.shape[1] # [,]
		loss_log_evidence = -(term_data_fit_vec + term_model_complexity_vec) # [,]
		loss_log_evidence_avg = loss_log_evidence / (y_traj_pred.shape[0]*y_traj_pred.shape[1])

		# Loss || Averaged entropy:
		# We need the predictive variance for each state
		# xpred: [Npoints,self.dim]
		# x_traj_pred: [Nrollouts,traj_length-1,self.dim_out]
		# Concatenate the predictions with their respective control input. We need them in order to compute the predictive variance:
		u_applied_tf_tiled = tf.tile(u_applied_tf[0:-1],[x_traj_pred.shape[0],1])
		x_traj_pred_tiled = tf.reshape(x_traj_pred,(-1,x_traj_pred.shape[2]))
		x_traj_pred_with_u_applied = tf.concat((x_traj_pred_tiled,u_applied_tf_tiled),axis=1)
		_, MO_std_pred = self.predict_at_locations(xpred=x_traj_pred_with_u_applied)
		entropy_term = tf.reduce_mean(tf.math.log(MO_std_pred)) # Why isn't it -log()? Because the minus sign apepars only in the definition of entropy. After developing the terms, it remains +log(2*pi*e*var); this makes sense: the larger var is, the higher the log and the higher the entropy
		loss_entropy = -entropy_term # We flip the sign becase in ELBO we are maximizing entropy_term

		# Prior || Averaged minus cross entropy (penalize distance between consecutive points)
		# x_traj_pred: [Nrollouts,traj_length-1,self.dim_out]
		# y_traj_pred: [Nrollouts,traj_length-1,self.dim_out]
		sigma_prior = 0.1
		prior_regularizer_term = -((y_traj_pred - x_traj_pred) / sigma_prior)**2
		loss_prior_regularizer_term = -tf.reduce_mean(prior_regularizer_term)

		# Total loss:
		loss_tot = loss_log_evidence_avg + scale_loss_entropy*loss_entropy + scale_prior_regularizer*loss_prior_regularizer_term

		return loss_tot, x_traj_pred, y_traj_pred



	def get_elbo_loss_for_predictions_in_full_trajectory_with_certain_horizon(self,Nsteps_tot,Nhorizon_rec,sample_fx_once=True):

		assert self.z_vec_real is not None, "Call update_dataset_predictive_loss()"

		loss_val = 0.0
		x_traj_pred_all_vec = np.zeros((Nsteps_tot,self.Nrollouts,Nhorizon_rec,self.dim_out)) # For plotting
		loss_val_per_step = np.zeros(Nsteps_tot) # For plotting
		for tt in range(Nsteps_tot):

			x_traj_real_applied = self.z_vec_real[tt:tt+Nhorizon_rec,:]
			x_traj_real_applied_tf = tf.reshape(x_traj_real_applied,(1,Nhorizon_rec,self.dim_out))
			u_applied_tf = self.u_traj_real[tt:tt+Nhorizon_rec,:]
			str_progress_bar = "Prediction with horizon = {0:d}; tt: {1:d} / {2:d} | ".format(Nhorizon_rec,tt+1,Nsteps_tot)
			loss_val_new, x_traj_pred, y_traj_pred = self._get_negative_log_evidence_and_predictive_trajectory_chunk(x_traj_real_applied_tf,u_applied_tf,Nsamples=1,
																												Nrollouts=self.Nrollouts,str_progress_bar=str_progress_bar,from_prior=False,
																												scale_loss_entropy=self.scale_loss_entropy,
																												scale_prior_regularizer=self.scale_prior_regularizer,
																												sample_fx_once=sample_fx_once)

			loss_val += loss_val_new

			# Save variables for plotting:
			# Add the last prediction y_traj_pred[:,-1::,:]; so x_traj_pred_all_vec[tt,...] contains [x0,predictions,last_prediction]
			x_traj_pred_all_vec[tt,...] = np.concatenate([x_traj_pred,y_traj_pred[:,-1::,:]],axis=1) # [Nsteps_tot,Nrollouts,Nhorizon_rec,self.dim_out]
			loss_val_per_step[tt] = loss_val_new


		loss_avg = loss_val / Nsteps_tot # For optimizer

		return loss_avg, x_traj_pred_all_vec, loss_val_per_step


	def update_dataset_predictive_loss(self,z_vec_real,u_traj_real,learning_rate,Nepochs,stop_loss_val,scale_loss_entropy,scale_prior_regularizer,Nrollouts):
		self.z_vec_real = z_vec_real
		self.u_traj_real = u_traj_real
		self.learning_rate = learning_rate
		self.Nepochs = Nepochs
		self.stop_loss_val = stop_loss_val
		assert scale_loss_entropy > 0.0
		self.scale_loss_entropy = scale_loss_entropy
		assert scale_prior_regularizer > 0.0
		self.scale_prior_regularizer = scale_prior_regularizer
		self.Nrollouts = Nrollouts

	def set_dbg_flag(self,flag=True):

		for ii in range(self.dim_out):
			self.rrgpMO[ii].dbg_flag = True

	# @tf.function
	def train_MOrrp_predictive(self,Nsteps_tot,Nhorizon_rec,sample_fx_once=True,verbosity=False,save_weights=True,path2save=None):
		"""

		"""

		str_banner = " << Training the model to predict better >> "


		# https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam
		# optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
		# optimizer_list = [tf.keras.optimizers.Adam(learning_rate=self.learning_rate)]*self.dim_out
		# trainable_weights_best_list = [None]*self.dim_out
		optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
		# Use a learning rate scheduler: https://arxiv.org/pdf/1608.03983.pdf
		# https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules/CosineDecay

		epoch = 0
		done = False
		loss_value_best = float("Inf")
		trainable_weights_best = self.get_weights()
		# for dd in range(self.dim_out):
		# 	trainable_weights_best_list[dd] = self.rrgpMO[dd].get_weights()
		plotting_dict = dict(plotting=False)
		print_every = 1
		while epoch < self.Nepochs and not done:

			if (epoch+1) % print_every == 0:
				logger.info("="*len(str_banner))
				logger.info(str_banner)
				logger.info(" << Epoch {0:d} / {1:d} >> ".format(epoch+1, self.Nepochs))
				logger.info("="*len(str_banner))

			# with tf.GradientTape(persistent=True) as tape:
			with tf.GradientTape() as tape:
				# loss_val_per_dim = self.get_negative_log_evidence_predictive_full_trajs_in_batch(self.z_vec_real,self.u_traj_real,self.Nhorizon,update_features=True)
				# print("Shouldn't this be update_features=epoch==0 ????????????")
				# loss_value,_ = self.get_negative_log_evidence_predictive_full_trajs_in_batch(update_features=epoch>0,plotting_dict=plotting_dict,Nrollouts=self.Nrollouts,from_prior=False)
				# loss_value = self.get_loss_debug()
				# loss_value = self.get_loss_debug_2(self.z_vec_real,self.u_traj_real,self.Nhorizon)
				loss_value, _, _ = self.get_elbo_loss_for_predictions_in_full_trajectory_with_certain_horizon(Nsteps_tot,Nhorizon_rec,sample_fx_once=sample_fx_once)

			# pdb.set_trace()
			# for dd in range(self.dim_out):
			# 	# grads = tape.gradient(loss_value_per_dim[dd], self.rrgpMO[dd].trainable_weights)
			# 	grads = tape.gradient(loss_value, self.rrgpMO[dd].trainable_weights)
			# 	optimizer_list[dd].apply_gradients(zip(grads, self.rrgpMO[dd].trainable_weights))

			# for dd in range(self.dim_out):
			# 	# grads = tape.gradient(loss_value_per_dim[dd], self.rrgpMO[dd].trainable_weights)
			# 	grads = tape.gradient(loss_value, self.trainable_weights)
			# 	print(grads)

			# pdb.set_trace()
			grads = tape.gradient(loss_value, self.trainable_weights)
			if tf.reduce_any([grads_el is None for grads_el in grads]):
				grads = tf.constant([[0.0],[0.0],[0.0]],dtype=tf.float32)
			optimizer.apply_gradients(zip(grads, self.trainable_weights))

			# loss_value = tf.math.reduce_sum(loss_val_per_dim)

			if (epoch+1) % print_every == 0:
				logger.info("    * Predictive loss (current): {0:.4f}".format(float(loss_value)))
				logger.info("    * Predictive loss (best): {0:.4f}".format(float(loss_value_best)))
				logger.info("    * Weights (current): {0:s}".format(self._weights2str(self.trainable_weights)))
				logger.info("    * Weights (best): {0:s}".format(self._weights2str(trainable_weights_best)))
				logger.info("    * Gradients (current): {0:s}".format(self._weights2str(grads)))

			if loss_value <= self.stop_loss_val:
				done = True
			
			if loss_value < loss_value_best:
				loss_value_best = loss_value
				trainable_weights_best = self.get_weights()
				# for dd in range(self.dim_out):
				# 	trainable_weights_best_list[dd] = self.rrgpMO[dd].get_weights()
			
			epoch += 1

		if done == True:
			logger.info(" * Training finished because loss_value = {0:f} (<= {1:f})".format(float(loss_value),float(self.stop_loss_val)))

		self.set_weights(weights=trainable_weights_best)
		# for dd in range(self.dim_out):
		# 	self.rrgpMO[dd].set_weights(weights=trainable_weights_best)

		# if verbosity:
		# 	self.rrgpMO[dd].print_weights_info()

		logger.info("Training finished!")

		# Save weights:
		weights_trained = self.get_weights()

		# Save data:
		if save_weights:

			name_file_data = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
			name_file = "{0:s}_weights_trained_MOrrp_model.pickle".format(name_file_data)
			path2save_full = "{0:s}/{1:s}".format(path2save,name_file)
			logger.info("Saving data at {0:s} ...".format(path2save_full))
			file = open(path2save_full, 'wb')
			pickle.dump(weights_trained,file)
			file.close()
			logger.info("Done saving data!")


	def export_tensors_needed_for_sampling_predictions_using_sampled_model_instances(self,path2save):

		Nomegas = self.rrgpMO[0].S_samples_vec.shape[0]

		S_samples_all_dim = np.zeros((self.dim_out,Nomegas,1))
		phi_samples_all_dim = np.zeros((self.dim_out,Nomegas,1))
		W_samples_all_dim = np.zeros((self.dim_out,Nomegas,self.dim_in))
		dw_samples_all_dim = np.zeros((self.dim_out,Nomegas,1))
		mean_beta_pred_all_dim = np.zeros((self.dim_out,Nomegas,1))
		cov_beta_pred_chol_all_dim = np.zeros((self.dim_out,Nomegas,Nomegas))
		for ii in range(self.dim_out):
			S_samples_all_dim[ii,...] = self.rrgpMO[ii].S_samples_vec # [Nomegas,1]
			phi_samples_all_dim[ii,...] = self.rrgpMO[ii].phi_samples_vec # [Nomegas,1]
			W_samples_all_dim[ii,...] = self.rrgpMO[ii].W_samples # [Nomegas,self.dim_in]
			dw_samples_all_dim[ii,...] = self.rrgpMO[ii].dw_vec # [Nomegas,1]
			mean_beta_pred_all_dim[ii,...], cov_beta_pred_chol_all_dim[ii,...] = self.rrgpMO[ii].predict_beta(from_prior=False)

		if self.learn_correlation_noise:
			chol_corr_noise_mat = self.get_chol_corr_noise_mat()
		else:
			chol_corr_noise_mat = None

		# def feat_mat(X,W_samples,phi_samples):
		# 	"""

		# 	X: [Nxpoints, dim_in]
		# 	W_samples: [Nomegas, dim_in]
		# 	phi_samples: [Nomegas,1]

		# 	return: PhiX: [Nxpoints, Nomegas]
		# 	"""
		# 	WX = X @ tf.transpose(W_samples) # [Nxpoints, Nomegas]
		# 	harmonics_vec = tf.math.cos(WX + tf.transpose(phi_samples)) # [Nxpoints, Nomegas]

		# 	return harmonics_vec

		# feat_mat_tf = tf.function(feat_mat)

		feat_mat_tf = None

		tensors4predictions = dict(	S_samples_all_dim=S_samples_all_dim,
									phi_samples_all_dim=phi_samples_all_dim,
									W_samples_all_dim=W_samples_all_dim,
									dw_samples_all_dim=dw_samples_all_dim,
									mean_beta_pred_all_dim=mean_beta_pred_all_dim,
									cov_beta_pred_chol_all_dim=cov_beta_pred_chol_all_dim,
									chol_corr_noise_mat=chol_corr_noise_mat,
									which_features="RRPDiscreteCosineFeaturesVariableIntegrationStep",
									feat_mat=feat_mat_tf)


		name_file_data = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
		name_file = "{0:s}_tensors.pickle".format(name_file_data)
		path2save_full = "{0:s}/{1:s}".format(path2save,name_file)
		logger.info("Saving data at {0:s} ...".format(path2save_full))
		file = open(path2save_full, 'wb')
		pickle.dump(tensors4predictions,file)
		file.close()
		logger.info("Done saving data!")

		return tensors4predictions


	def compute_fast_predictions_from_saved_tensors(self,path2data,x0,u_traj,Nsteps_tot,Nrollouts):
		"""
		
		x0: [1,self.dim_x]
		u_traj: [Nhor,self.dim_u]

		"""

		raise NotImplementedError("Probably better to just do it in c++")

		path2data = "/Users/alonrot/work/code_projects_WIP/ood_project/ood/experiments/dubins_car_receding/tensors4fast_prediction_with_MOrrp_model"
		file_name = "2023_03_16_19_55_36_tensors.pickle"
		path2data_full = "{0:s}/{1:s}".format(path2data,file_name)

		logger.info("Loading {0:s} ...".format(path2data_full))
		file = open(path2data_full, 'rb')
		tensors4predictions = pickle.load(file)
		file.close()

		S_samples_all_dim = tensors4predictions["S_samples_all_dim"]
		phi_samples_all_dim = tensors4predictions["phi_samples_all_dim"]
		W_samples_all_dim = tensors4predictions["W_samples_all_dim"]
		dw_samples_all_dim = tensors4predictions["dw_samples_all_dim"]
		mean_beta_pred_all_dim = tensors4predictions["mean_beta_pred_all_dim"]
		cov_beta_pred_chol_all_dim = tensors4predictions["cov_beta_pred_chol_all_dim"]
		chol_corr_noise_mat = tensors4predictions["chol_corr_noise_mat"]
		which_features = tensors4predictions["which_features"]
		feat_mat = tensors4predictions["feat_mat"]

		Nhorizon_rec = u_traj.shape[0] + 1

		dim_x = x0.shape[1]

		x_traj_pred_all_vec = np.zeros((Nsteps_tot,Nrollouts,Nhorizon_rec,dim_x)) # For plotting
		for tt in range(Nsteps_tot):

			for rr in range(Nrollouts):

				for hh in range(Nhorizon_rec):

					for dd in dim_x:
						# Get state
						pass

					# Roll out entire model for this specific model instance rr
					pass
					# feat_mat(X=x_traj_pred_all_vec[tt,rr,],W_samples,phi_samples)

				# Repeat for all roll outs
				pass
				# If we pre-sample the model instances, maybe this explicit loop can be avoided

			# Repeat for all time steps
			pass
			# Compute here the ELBO loss if desired

		return x_traj_pred_all_vec

	def _weights2str(self,trainable_weights):
		
		assert len(trainable_weights) > 0
		if tf.is_tensor(trainable_weights[0]):
			which_type = "tfvar"
		elif isinstance(trainable_weights[0],np.ndarray):
			which_type = "nparr"
		elif trainable_weights[0] is None:
			which_type = "none"
		else:
			raise ValueError("trainable_weights has an unspecificed type")

		str_weights = "[ "
		for ii in range(len(trainable_weights)-1):
			# if which_type == "tfvar": str_weights += str(trainable_weights[ii].numpy())
			# elif which_type == "nparr": str_weights += str(trainable_weights[ii])
			# elif which_type == "none": str_weights += str(None)
			try: str_weights += str(trainable_weights[ii].numpy());
			except: str_weights += str(trainable_weights[ii]);
			str_weights += " , "

		try: str_weights += str(trainable_weights[len(trainable_weights)-1].numpy());
		except: str_weights += str(trainable_weights[len(trainable_weights)-1]);
		# if which_type == "tfvar": str_weights += str(trainable_weights[len(trainable_weights)-1].numpy())
		# elif which_type == "nparr": str_weights += str(trainable_weights[len(trainable_weights)-1])
		# elif which_type == "none": str_weights += str(None)

		str_weights += " ]"
		return str_weights


	def _print_progess_bar(self,ii,Ntot,name):
		assert ii < Ntot
		bar_size = math.ceil(ii/(Ntot-1) * 20)
		sys.stdout.write('\r')
		# the exact output you're looking for:
		# sys.stdout.write("[%-20s] %d%%" % ('='*i, 5*i))
		# sys.stdout.write("[{0:20s}] {1:s} {2:d} / {3:d} ".format("="*bar_size, name, ii+1, Ntot))
		sys.stdout.write("{0:s} [{1:20s}] {2:d} / {3:d} ".format(name,"="*bar_size, ii+1, Ntot))
		# logger.info("[{0:50s}] "+name+" {1:d} / {2:d} ".format("="*bar_size, ii+1, Ntot))
		sys.stdout.flush()
		if ii == Ntot-1:
			print("")















	"""
	Deprecated
	"""

	# # @tf.function
	# def get_negative_log_evidence_predictive_full_trajs_in_batch(self,update_features,plotting_dict,Nrollouts,from_prior=False):
	# 	# """

	# 	# return:
	# 	# x_traj_pred_chunks: [Nchunks,Nrollouts,Nhorizon,self.dim_out]
	# 	# """

	# 	raise NotImplementedError("This function is deprecated; it computes predictions in discrete chunks; we need predictions at every time step. Replaced by get_elbo_loss_for_predictions_in_full_trajectory_with_certain_horizon()")

	# 	if plotting_dict["plotting"]:
	# 		block_plot = plotting_dict["block_plot"]
	# 		title_fig = plotting_dict["title_fig"]
	# 		z_vec = plotting_dict["z_vec"]
	# 		z_vec_changed_dyn = plotting_dict["z_vec_changed_dyn"]
	# 		ref_xt_vec = plotting_dict["ref_xt_vec"]
	# 		hdl_fig_pred, hdl_splots_pred = plt.subplots(1,1,figsize=(12,8),sharex=True)
	# 		hdl_fig_pred.suptitle(title_fig, fontsize=16)
	# 		hdl_splots_pred.set_xlabel(r"$x_1$",fontsize=fontsize_labels); hdl_splots_pred.set_ylabel(r"$x_2$",fontsize=fontsize_labels)

	# 	if update_features:
	# 		self.update_features()

	# 	loss_val = 0.0
	# 	Nsteps = self.z_vec_real.shape[0]
	# 	Nchunks = Nsteps//self.Nhorizon
	# 	verbo_loss_val_vec = np.zeros(Nchunks)
	# 	logger.info(" * Trajectory prediction: xpred_[t:t+H] given {x_t,u_t,u_[t+1],...,u_[t+H]}")
	# 	logger.info(" * Loss: E[log p(yreal_[t:t+H] | xpred_[t:t+H])] + H(p), with E[] wrt posterior p(x_[t:t+H] | Data) and H(p) its entropy")
	# 	logger.info("    * Trajectory divided in {0:d} chunks, with Nhorizon = {1:d}".format(Nchunks,self.Nhorizon))
	# 	logger.info("    * Control sequence length: {0:d} steps".format(Nsteps))
	# 	logger.info(" * Weights (current): {0:s}".format(self._weights2str(self.get_weights())))

	# 	x_traj_pred_chunks = np.zeros((Nchunks,Nrollouts,self.Nhorizon,self.dim_out))
	# 	for ii in range(Nchunks):

	# 		# logger.info("    * Prediction for chunk {0:d} / {1:d} || Steps: [{2:d}:{3:d}]".format(ii+1,Nchunks,ii*self.Nhorizon,(ii+1)*self.Nhorizon))
	# 		str_progress_bar = "Prediction for chunk {0:d} / {1:d} || Steps: [{2:d}:{3:d}] || ".format(ii+1,Nchunks,ii*self.Nhorizon,(ii+1)*self.Nhorizon)

	# 		# Extract chunk of real trajectory, to compare the predictions with:
	# 		x_traj_real = self.z_vec_real[ii*self.Nhorizon:(ii+1)*self.Nhorizon,:]
	# 		x_traj_real_applied_tf = tf.reshape(x_traj_real,(1,self.Nhorizon,self.dim_out))
	# 		u_applied_tf = self.u_traj_real[ii*self.Nhorizon:(ii+1)*self.Nhorizon,:]

	# 		# x_traj_real_applied = np.reshape(x_traj_real,(1,self.Nhorizon,self.dim_out))
	# 		# x_traj_real_applied_tf = tf.convert_to_tensor(value=x_traj_real_applied,dtype=tf.float32) # [Npoints,self.dim_in], with Npoints=1
	# 		# u_applied_tf = tf.convert_to_tensor(value=u_applied,dtype=tf.float32) # [Npoints,self.dim_in], with Npoints=1
	# 		loss_val, x_traj_pred, y_traj_pred = self._get_negative_log_evidence_and_predictive_trajectory_chunk(x_traj_real_applied_tf,u_applied_tf,Nsamples=1,
	# 																											Nrollouts=Nrollouts,str_progress_bar=str_progress_bar,from_prior=from_prior,
	# 																											scale_loss_entropy=self.scale_loss_entropy,
	# 																											scale_prior_regularizer=self.scale_prior_regularizer)

	# 		# Store some variables for plotting:
	# 		x_traj_pred_all = np.concatenate([x_traj_pred,y_traj_pred[:,-1::,:]],axis=1)
	# 		x_traj_pred_chunks[ii,...] = x_traj_pred_all

	# 		verbo_loss_val_vec[ii] = loss_val.numpy()

	# 		logger.info("    * Done! Cumulated average loss: {0:f}".format(float(np.mean(verbo_loss_val_vec[0:ii+1]))))

	# 		if plotting_dict["plotting"]:
	# 			for ss in range(y_traj_pred.shape[0]):
	# 				if ii == 0 and ss == 0: label_x_traj_pred = r"Predictions"
	# 				else: label_x_traj_pred = None
	# 				hdl_splots_pred.plot(x_traj_pred_all[ss,:,0],x_traj_pred_all[ss,:,1],linestyle="-",color="crimson",lw=0.5,alpha=0.5,label=label_x_traj_pred)


	# 	if plotting_dict["plotting"]:
	# 		if plotting_dict["ref_xt_vec"] is not None:
	# 			hdl_splots_pred.plot(ref_xt_vec[:,0],ref_xt_vec[:,1],marker=".",linestyle="-",color="grey",lw=1.0,markersize=5,label=r"Reference",alpha=0.5)
	# 		if plotting_dict["z_vec"] is not None:
	# 			hdl_splots_pred.plot(z_vec[:,0],z_vec[:,1],linestyle="-",color="navy",lw=2.0,label=r"Real traj - nominal dynamics",alpha=0.3)
	# 		if plotting_dict["z_vec_changed_dyn"] is not None:
	# 			hdl_splots_pred.plot(z_vec_changed_dyn[:,0],z_vec_changed_dyn[:,1],linestyle="-",color="navy",lw=2.0,label=r"Real traj - changed dynamics",alpha=0.7)
	# 		if title_fig != "Predictions || Using prior, no training":
	# 			hdl_splots_pred.legend()
	# 		plt.show(block=block_plot)

	# 	# Return average:
	# 	loss_val_avg = loss_val / (Nchunks)
	# 	logger.info(" * Loss averaged wrt {0:d} chunks: {1:f}".format(Nchunks,np.mean(verbo_loss_val_vec[0:ii])))

	# 	return loss_val_avg, x_traj_pred_chunks

	# # @tf.function
	# def get_loss_debug(self):

	# 	raise NotImplementedError("Deprecated")

	# 	loss_val = 0.0
	# 	for ii in range(self.dim_out):
	# 		loss_val += 3.*self.rrgpMO[ii].log_noise_std**(ii+2) + 5.0
	# 	# for ii in range(self.dim_out):
	# 	# 	loss_val += 3.*self.log_noise_std_vec[ii]**(ii+2) + 5.0
	# 	return loss_val

	# # @tf.function
	# def get_loss_debug_2(self,z_vec_real,u_traj_real,Nhorizon):

	# 	raise NotImplementedError("Deprecated")
		
	# 	self.update_features()

	# 	ii = 0
	# 	x_traj_real = z_vec_real[ii*Nhorizon:(ii+1)*Nhorizon,:]
	# 	x0_tf = x_traj_real[0:1,:] # [Npoints,self.dim_in], with Npoints=1
	# 	u_applied_tf = u_traj_real[ii*Nhorizon:(ii+1)*Nhorizon,:]
	# 	xsample_curr = tf.concat([x0_tf,u_applied_tf[ii:ii+1,:]],axis=1) # [ [Npoints,self.dim_in] , [1,dim_u] ], with Npoints=1

	# 	fx = self.get_sample_path_callable(Nsamples=1)
	# 	state_next = fx(xsample_curr) # xsample_next: [Npoints,self.dim_out,Nsamples], with Npoints=1, Nsamples=1
	# 	loss_val = tf.math.reduce_sum(state_next**2)

	# 	return loss_val



	# # @tf.function
	# def _rollout_model_given_control_sequence_numpy(self,x0,Nsamples,Nrollouts,u_traj,traj_length,sort=False,plotting=False,verbo=False):
	# 	# """

	# 	# Pass two initial latent values
	# 	# x0: [1,dim_x]
	# 	# u_traj: [traj_length,dim_u]
		
	# 	# The GP won't be training during sampling, i.e., we won't call self.train_model()

	# 	# return:
	# 	# xsamples_X: [Npoints,self.dim_out,Nsamples]
	# 	# xsamples_Y: [Npoints,self.dim_out,Nsamples]


		
	# 	# TODO: Refactor as
	# 	# Nsamples_per_particle <- Nsamples
	# 	# Nparticles <- Npoints
	# 	# """

	# 	raise NotImplementedError("Haven't iterated on it in a while; probably needs revision; using self._rollout_model_given_control_sequence_tf()")

	# 	# Parsing arguments:
	# 	if u_traj is not None: # Concatenate the inputs to the model as (x_t,u_t)
	# 		if verbo: logger.info(" * Open-loop model x_{t+1} = f(x_t,u_t)")
	# 		if verbo: logger.info(" * Input to the model: x0, and u_traj")
	# 		assert x0.shape[1] == self.dim_out
	# 		assert x0.shape[0] == 1
	# 		assert self.dim_in == self.dim_out + u_traj.shape[1]
	# 		assert traj_length == -1, "Pass -1 to emphasize that traj_length is inferred from u_traj"
	# 		traj_length = u_traj.shape[0]
	# 	else: # The input to the model is the state, directly
	# 		if verbo: logger.info(" * Closed-loop model x_{t+1} = f(x_t)")
	# 		if verbo: logger.info(" * Input to the model: x0")
	# 		assert x0.shape[1] == self.dim_out
	# 		assert x0.shape[1] == self.dim_in
	# 		assert traj_length > 2

	# 	assert Nsamples > 0

	# 	colors_arr = cm.winter(np.linspace(0,1,Nsamples))

	# 	if plotting:
	# 		hdl_fig, hdl_splots = plt.subplots(2,1,figsize=(12,8),sharex=True)
	# 		hdl_fig.suptitle(r"Kink function simulation $x_{t+1} = f(x_t) + \varepsilon$"+", kernel: {0}".format("kink"),fontsize=fontsize_labels)
	# 		xmin = -6.; xmax = +3.; Ndiv = 201
	# 		xpred = tf.linspace(xmin,xmax,Ndiv)
	# 		xpred = tf.reshape(xpred,(-1,1))


	# 	# """
	# 	# NOTE: 	This whole explanation assumes closed-loop, i.e. ut is not present, or ut=[].
	# 	# 		However, extending xt to cinlude ut is trivial
		
	# 	# We need to distinguish between Nsamples and Npoints
	# 	# 	fx() gets as input argument xt, with dimensions [Npoints,self.dim_in]
	# 	# 	It computes, for each dimension, [Npoints,Nsamples] points,
	# 	# 	i.e., for each input point, we get Nsamples samples
	# 	# 	We repeat such operation for all dimensions.
	# 	# 	Then, we stack together all samples for each dimension
	# 	# 	in a vector x_{t+1} [Npoints,self.dim_out,Nsamples]

	# 	# The naive approach would output Nsamples samples of the x_{t+1} vector
	# 	# for each xt input.
	# 	# x_{t+1} = f(xt)
	# 	# xt: [Npoints,self.dim_out]
	# 	# x_{t+1}: [Npoints,self.dim_out,Nsamples]

	# 	# This poses several isues:
	# 	# 	a) In order to get x_{t+2}, we need to pass x_{t+1} is input; and for that,
	# 	# 	we need to reshape it as [Npoints*Nsamples, self.dim_out].
	# 	# 	This means that x_{t+3} will be [Npoints*Nsamples**2, self.dim_out]
	# 	# 	and that x_{t+H} will be [Npoints*Nsamples**(H-1), self.dim_out]
	# 	# 	b) 

	# 	# We propose to start off with a fixed number of Npoints, and get one independent sample 
	# 	# for each point. To this end we need to:
	# 	# 	1) At iteration=0, we gather Nsamples of x_{t+1}, i.e., 
	# 	# 	x0: [Npoints,self.dim_out] = stack([x0]*Npoints,axis=0) (just copy the initial point for all Npoints)
	# 	# 	2) Set Nsamples=1
	# 	# 	3) Get x1 as x1 = fx(x0), with x1: [Npoints,self.dim_out,Nsamples=1]
	# 	# 	4) Reshape: x1 <- x1 as [Npoints,self.dim_out]
	# 	# 	5) Get x2 as x2 = fx(x1), with x2: [Npoints,self.dim_out,Nsamples=1]
	# 	# 	6) Repeat		

	# 	# """
	# 	# Npoints = Nsamples # This should be viewed as the number of independent particles that we propagate, each of which has the dimension of the state
	# 	# Nsamples = 1 # This should be viewed as "how many samples per point?"
	# 	# fx = self.get_sample_path_callable(Nsamples=Nsamples)
	# 	# xsamples = np.zeros((traj_length,Nsamples,self.dim_out),dtype=np.float32)
	# 	# xsamples = np.zeros((traj_length,self.dim_out,Nsamples),dtype=np.float32)
	# 	# xsamples[0,...] = np.vstack([x0]*Nsamples)

	# 	assert Nsamples == 1, "We need one sample per roll-out"
	# 	xsamples = np.zeros((Nrollouts,traj_length,self.dim_out),dtype=np.float32) # Nrollouts: each roll-out constitutes a different sample

	# 	# pdb.set_trace()
	# 	# xsamples[0,...] = np.stack([x0]*Nsamples,axis=2)
	# 	xsamples[:,0,:] = np.vstack([x0]*Nrollouts) # Same initial condition for all roll-outs

	# 	for ss in range(Nrollouts):

	# 		logger.info("Rollout {0:d} / {1:d}".format(ss+1,Nrollouts))
	# 		fx = self.get_sample_path_callable(Nsamples=Nsamples)

	# 		for ii in range(0,traj_length-1):

	# 			# xsamples_mean = np.mean(xsamples[ii,...],axis=0,keepdims=True)
	# 			# xsamples_mean = np.mean(xsamples[ii:ii+1,...],axis=2)

	# 			if u_traj is None:
	# 				# xsamples_in = xsamples[ii,...]
	# 				# xsamples_in = xsamples_mean
	# 				xsamples_in = xsamples[ss,ii:ii+1,:]
	# 			else:
	# 				u_traj_ii = u_traj[ii:ii+1,:] # [1,dim_u]
	# 				# xsamples_in = np.hstack([xsamples[ii,...],np.vstack([u_traj_ii]*Nsamples)])
	# 				# xsamples_in = np.hstack([xsamples_mean,u_traj_ii])
	# 				xsamples_in = np.concatenate([xsamples[ss,ii:ii+1,:],u_traj_ii],axis=1) # [Npoints,self.dim_in], with Npoints=1

	# 			xsample_tp = tf.convert_to_tensor(value=xsamples_in,dtype=np.float32) # [Npoints,self.dim_in], with Npoints=1


	# 			# Por algun motivo,
	# 			# self.rrgpMO[0].get_predictive_beta_distribution()
	# 			# self.rrgpMO[1].get_predictive_beta_distribution()
	# 			# son diferentes.... The mean is different; the covariance is the same
	# 			# Then, weirdly enough, xsamples_next are all the same values...
	# 			# But are they exactly the same xsamples_next[0] == xsamples_next[0] ??

	# 			# xsamples_next = fx(xsample_tp) # [Nsamples,self.dim_out,Nsamples]
	# 			# xsamples[ii+1,...] = tf.reshape(xsamples_next,(Nsamples,self.dim_out))
	# 			# xsamples[ii+1,...] = fx(xsample_tp) # [Npoints,self.dim_out,Nsamples]

	# 			xsamples_next = fx(xsample_tp) # [Npoints,self.dim_out,Nsamples], with Npoints=1, Nsamples=1
	# 			# xsamples[ss,ii:ii+1,...] = tf.reshape(xsamples_next,(1,self.dim_out))
	# 			xsamples[ss,ii+1:ii+2,:] = tf.reshape(xsamples_next,(1,self.dim_out))

	# 			if plotting and self.dim_out == 1:

	# 				# Plot what's going on at each iteration here:
	# 				MO_mean_pred, std_diag_pred = self.predict_at_locations(xpred)
	# 				hdl_splots[0].cla()
	# 				hdl_splots[0].plot(xpred,MO_mean_pred,linestyle="-",color="b",lw=3)
	# 				hdl_splots[0].fill_between(xpred[:,0],MO_mean_pred[:,0] - 2.*std_diag_pred[:,0],MO_mean_pred[:,0] + 2.*std_diag_pred[:,0],color="cornflowerblue",alpha=0.5)
	# 				for n_samples in range(xsamples.shape[2]):
	# 					hdl_splots[0].plot(xsamples[0:ii,0,n_samples],xsamples[1:ii+1,0,n_samples],marker=".",linestyle="--",color=colors_arr[n_samples,0:3],lw=0.5,markersize=5)
	# 				# hdl_splots[0].set_xlim([xmin,xmax])
	# 				hdl_splots[0].set_ylabel(r"$x_{t+1}$",fontsize=fontsize_labels)
	# 				# pdb.set_trace()
	# 				for n_samples in range(xsamples.shape[2]):
	# 					hdl_splots[0].plot(xsamples[0,0,n_samples],xsamples[1,0,n_samples],marker="o",markersize=10)
	# 				# hdl_splots[0].plot(xpred,yplot_true_fun,marker="None",linestyle="-",color="grey",lw=1)
	# 				# hdl_splots[0].plot(xpred,yplot_sampled_fun,marker="None",linestyle="--",color="r",lw=0.5)

	# 				plt.pause(0.5)
	# 				# input()


	# 	# xsamples_X = xsamples[0:-1,...]
	# 	# xsamples_Y = xsamples[1::,...]

	# 	# pdb.set_trace()

	# 	# xsamples_X = np.reshape(xsamples[:,0:-1,:],(traj_length-1,self.dim_out,Nrollouts),order="F") # [traj_length-1,self.dim_out,Nrollouts]
	# 	# xsamples_Y = np.reshape(xsamples[:,1::,:],(traj_length-1,self.dim_out,Nrollouts),order="F") # [traj_length-1,self.dim_out,Nrollouts]

	# 	xsamples_X = xsamples[:,0:-1,:] # [Nrollouts,traj_length-1,self.dim_out]
	# 	xsamples_Y = xsamples[:,1::,:] # [Nrollouts,traj_length-1,self.dim_out]

	# 	# assert np.all(xsamples_X[:,:,0] == xsamples[0,:,:])

	# 	if sort:
	# 		raise NotImplementedError("Incompatible with xsamples = np.zeros((Nrollouts,traj_length,self.dim_out),dtype=np.float32)")
	# 		assert x0.shape[1] == 1, "This only makes sense in 1D"

	# 		xsamples_X_red = xsamples_X[:,0,:]
	# 		ind_sort = np.argsort(xsamples_X_red,axis=0)
	# 		xsamples_X_red_sorted = xsamples_X_red[ind_sort]
	# 		xsamples_X = xsamples_X_red_sorted[:,0,:]

	# 		xsamples_Y_red = xsamples_Y[:,0,:]
	# 		ind_sort = np.argsort(xsamples_Y_red,axis=0)
	# 		xsamples_Y_red_sorted = xsamples_Y_red[ind_sort]
	# 		xsamples_Y = xsamples_Y_red_sorted[:,0,:]
	# 	elif self.dim_in == 1:
	# 		raise NotImplementedError("Incompatible with xsamples = np.zeros((Nrollouts,traj_length,self.dim_out),dtype=np.float32)")
	# 		xsamples_X = xsamples_X[:,0,:]
	# 		xsamples_Y = xsamples_Y[:,0,:]

	# 	# # Go back to the dataset at it was:
	# 	# self.update_model(X=Xtraining,Y=Ytraining)

	# 	return xsamples_X, xsamples_Y











