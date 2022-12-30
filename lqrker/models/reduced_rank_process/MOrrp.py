import tensorflow as tf
import tensorflow_probability as tfp
import math
import pdb
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from lqrker.models import RRPRegularFourierFeatures, RRPDiscreteCosineFeatures, RRPLinearFeatures
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


class MultiObjectiveReducedRankProcess():

	def __init__(self, dim_in: int, cfg: dict, spectral_density: SpectralDensityBase, Xtrain, Ytrain):

		"""
		
		Initialize "dim" number of RRPRegularFourierFeatures() models, one per output channel		

		We use a single spectral density instance for all the models. For that instance, 
		we compute the needed frequencies and use them throughout all the models

		"""
		self.dim_in = dim_in
		self.dim_out = Ytrain.shape[1]

		assert cfg.gpmodel.which_features in ["RRPLinearFeatures", "RRPDiscreteCosineFeatures", "RRPRegularFourierFeatures", "RRPRandomFourierFeatures"]

		self.rrgpMO = [None]*self.dim_out
		for ii in range(self.dim_out):

			if cfg.gpmodel.which_features == "RRPLinearFeatures":
				self.rrgpMO[ii] = RRPLinearFeatures(dim=self.dim_in,cfg=cfg.gpmodel,spectral_density=spectral_density)
			elif cfg.gpmodel.which_features == "RRPDiscreteCosineFeatures":
				self.rrgpMO[ii] = RRPDiscreteCosineFeatures(dim=self.dim_in,cfg=cfg.gpmodel,spectral_density=spectral_density)


				# self.rrgpMO[ii].update_model(Xtrain,Ytrain[:,ii:ii+1]) # Update model indexing the target outputs at the corresponding dimension
				# self.rrgpMO[ii].get_MLII_loss_gaussian_predictive(xpred=Xtrain)

			elif cfg.gpmodel.which_features == "RRPRegularFourierFeatures":
				self.rrgpMO[ii] = RRPRegularFourierFeatures(dim=self.dim_in,cfg=cfg.gpmodel,spectral_density=spectral_density)
			elif cfg.gpmodel.which_features == "RRPRandomFourierFeatures":
				self.rrgpMO[ii] = RRPRandomFourierFeatures(dim=self.dim_in,cfg=cfg.gpmodel,spectral_density=spectral_density)

			self.rrgpMO[ii].select_output_dimension(dim_out_ind=ii)

		self.update_model(X=Xtrain,Y=Ytrain)

	def update_model(self,X,Y):
		for ii in range(self.dim_out):
			self.rrgpMO[ii].update_model(X,Y[:,ii:ii+1]) # Update model indexing the target outputs at the corresponding dimension

	def get_training_data(self):
		X = self.rrgpMO[0].X
		Y_ii = []
		for ii in range(self.dim_out):
			Y_ii += [self.rrgpMO[ii].Y]
		Y = tf.concat(Y_ii,axis=1)

		return tf.identity(X), tf.identity(Y) # Copy tensor

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
			"""
			Args:
				x: [Npoints,self.dim_in]
			
			Returns:
				function
			"""

			out = []
			for ii in range(self.dim_out):
				fx_ii = self.rrgpMO[ii].get_sample_path_callable(Nsamples,from_prior)
				out_ii = fx_ii(x) # [Npoints,Nsamples]
				out += [out_ii]

			return tf.stack(out,axis=1) # [Npoints,self.dim_out,Nsamples]

		return nonlinfun_sampled_callable

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


	def train_model(self,verbosity=False):
		for ii in range(self.dim_out):
			self.rrgpMO[ii].train_model(verbosity)

	def sample_state_space_from_prior_recursively(self,x0,Nsamples,u_traj,traj_length,sort=False,plotting=False):
		"""

		Pass two initial latent values
		x0: [1,dim_x]
		u_traj: [traj_length,dim_u]
		
		The GP won't be training during sampling, i.e., we won't call self.train_model()

		return:
		xsamples_X: [Npoints,self.dim_out,Nsamples]
		xsamples_Y: [Npoints,self.dim_out,Nsamples]


		
		TODO: Refactor as
		Nsamples_per_particle <- Nsamples
		Nparticles <- Npoints
		"""

		# Parsing arguments:
		if u_traj is not None: # Concatenate the inputs to the model as (x_t,u_t)
			print(" * Open-loop model x_{t+1} = f(x_t,u_t)\n * Input to the model: x0, and u_traj")
			assert x0.shape[1] == self.dim_out
			assert x0.shape[0] == 1
			assert self.dim_in == self.dim_out + u_traj.shape[1]
			assert traj_length == -1, "Pass -1 to emphasize that traj_length is inferred from u_traj"
			traj_length = u_traj.shape[0]
		else: # The input to the model is the state, directly
			print(" * Closed-loop model x_{t+1} = f(x_t)\n * Input to the model: x0")
			assert x0.shape[1] == self.dim_out
			assert x0.shape[1] == self.dim_in
			assert traj_length > 2

		assert Nsamples > 0

		colors_arr = cm.winter(np.linspace(0,1,Nsamples))

		if plotting:
			hdl_fig, hdl_splots = plt.subplots(2,1,figsize=(12,8),sharex=True)
			hdl_fig.suptitle(r"Kink function simulation $x_{t+1} = f(x_t) + \varepsilon$"+", kernel: {0}".format("kink"),fontsize=fontsize_labels)
			xmin = -6.; xmax = +3.; Ndiv = 201
			xpred = tf.linspace(xmin,xmax,Ndiv)
			xpred = tf.reshape(xpred,(-1,1))


		"""
		NOTE: 	This whole explanation assumes closed-loop, i.e. ut is not present, or ut=[].
				However, extending xt to cinlude ut is trivial
		
		We need to distinguish between Nsamples and Npoints
			fx() gets as input argument xt, with dimensions [Npoints,self.dim_in]
			It computes, for each dimension, [Npoints,Nsamples] points,
			i.e., for each input point, we get Nsamples samples
			We repeat such operation for all dimensions.
			Then, we stack together all samples for each dimension
			in a vector x_{t+1} [Npoints,self.dim_out,Nsamples]

		The naive approach would output Nsamples samples of the x_{t+1} vector
		for each xt input.
		x_{t+1} = f(xt)
		xt: [Npoints,self.dim_out]
		x_{t+1}: [Npoints,self.dim_out,Nsamples]

		This poses several isues:
			a) In order to get x_{t+2}, we need to pass x_{t+1} is input; and for that,
			we need to reshape it as [Npoints*Nsamples, self.dim_out].
			This means that x_{t+3} will be [Npoints*Nsamples**2, self.dim_out]
			and that x_{t+H} will be [Npoints*Nsamples**(H-1), self.dim_out]
			b) 

		We propose to start off with a fixed number of Npoints, and get one independent sample 
		for each point. To this end we need to:
			1) At iteration=0, we gather Nsamples of x_{t+1}, i.e., 
			x0: [Npoints,self.dim_out] = stack([x0]*Npoints,axis=0) (just copy the initial point for all Npoints)
			2) Set Nsamples=1
			3) Get x1 as x1 = fx(x0), with x1: [Npoints,self.dim_out,Nsamples=1]
			4) Reshape: x1 <- x1 as [Npoints,self.dim_out]
			5) Get x2 as x2 = fx(x1), with x2: [Npoints,self.dim_out,Nsamples=1]
			6) Repeat		

		"""
		# Npoints = Nsamples # This should be viewed as the number of independent particles that we propagate, each of which has the dimension of the state
		# Nsamples = 1 # This should be viewed as "how many samples per point?"
		fx = self.get_sample_path_callable(Nsamples=Nsamples)
		# xsamples = np.zeros((traj_length,Nsamples,self.dim_out),dtype=np.float32)
		xsamples = np.zeros((traj_length,self.dim_out,Nsamples),dtype=np.float32)
		# xsamples[0,...] = np.vstack([x0]*Nsamples)
		xsamples[0,...] = np.stack([x0]*Nsamples,axis=2)
		for ii in range(0,traj_length-1):

			# xsamples_mean = np.mean(xsamples[ii,...],axis=0,keepdims=True)
			xsamples_mean = np.mean(xsamples[ii:ii+1,...],axis=2)

			if u_traj is None:
				# xsamples_in = xsamples[ii,...]
				xsamples_in = xsamples_mean
			else:
				u_traj_ii = u_traj[ii:ii+1,:] # [1,dim_u]
				# xsamples_in = np.hstack([xsamples[ii,...],np.vstack([u_traj_ii]*Nsamples)])
				# xsamples_in = np.hstack([xsamples_mean,u_traj_ii])
				xsamples_in = np.concatenate([xsamples_mean,u_traj_ii],axis=1)

			xsample_tp = tf.convert_to_tensor(value=xsamples_in,dtype=np.float32)


			# Por algun motivo,
			# self.rrgpMO[0].get_predictive_beta_distribution()
			# self.rrgpMO[1].get_predictive_beta_distribution()
			# son diferentes.... The mean is different; the covariance is the same
			# Then, weirdly enough, xsamples_next are all the same values...
			# But are they exactly the same xsamples_next[0] == xsamples_next[0] ??

			# xsamples_next = fx(xsample_tp) # [Nsamples,self.dim_out,Nsamples]
			# xsamples[ii+1,...] = tf.reshape(xsamples_next,(Nsamples,self.dim_out))
			xsamples[ii+1,...] = fx(xsample_tp) # [Npoints,self.dim_out,Nsamples]

			if plotting and self.dim_out == 1:

				# Plot what's going on at each iteration here:
				MO_mean_pred, std_diag_pred = self.predict_at_locations(xpred)
				hdl_splots[0].cla()
				hdl_splots[0].plot(xpred,MO_mean_pred,linestyle="-",color="b",lw=3)
				hdl_splots[0].fill_between(xpred[:,0],MO_mean_pred[:,0] - 2.*std_diag_pred[:,0],MO_mean_pred[:,0] + 2.*std_diag_pred[:,0],color="cornflowerblue",alpha=0.5)
				for n_samples in range(xsamples.shape[2]):
					hdl_splots[0].plot(xsamples[0:ii,0,n_samples],xsamples[1:ii+1,0,n_samples],marker=".",linestyle="--",color=colors_arr[n_samples,0:3],lw=0.5,markersize=5)
				# hdl_splots[0].set_xlim([xmin,xmax])
				hdl_splots[0].set_ylabel(r"$x_{t+1}$",fontsize=fontsize_labels)
				# pdb.set_trace()
				for n_samples in range(xsamples.shape[2]):
					hdl_splots[0].plot(xsamples[0,0,n_samples],xsamples[1,0,n_samples],marker="o",markersize=10)
				# hdl_splots[0].plot(xpred,yplot_true_fun,marker="None",linestyle="-",color="grey",lw=1)
				# hdl_splots[0].plot(xpred,yplot_sampled_fun,marker="None",linestyle="--",color="r",lw=0.5)

				plt.pause(0.5)
				# input()

		xsamples_X = xsamples[0:-1,...]
		xsamples_Y = xsamples[1::,...]

		if sort:
			assert x0.shape[1] == 1, "This only makes sense in 1D"

			xsamples_X_red = xsamples_X[:,0,:]
			ind_sort = np.argsort(xsamples_X_red,axis=0)
			xsamples_X_red_sorted = xsamples_X_red[ind_sort]
			xsamples_X = xsamples_X_red_sorted[:,0,:]

			xsamples_Y_red = xsamples_Y[:,0,:]
			ind_sort = np.argsort(xsamples_Y_red,axis=0)
			xsamples_Y_red_sorted = xsamples_Y_red[ind_sort]
			xsamples_Y = xsamples_Y_red_sorted[:,0,:]
		elif self.dim_in == 1:
			xsamples_X = xsamples_X[:,0,:]
			xsamples_Y = xsamples_Y[:,0,:]

		# # Go back to the dataset at it was:
		# self.update_model(X=Xtraining,Y=Ytraining)

		return xsamples_X, xsamples_Y

