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

	def __init__(self, dim: int, cfg: dict, spectral_density: SpectralDensityBase, Xtrain, Ytrain):

		"""
		
		Initialize "dim" number of RRPRegularFourierFeatures() models, one per output channel		
		"""

		self.dim_in = dim
		self.dim_out = Ytrain.shape[1] # TODO: pass this as an actual input argument
		# assert dim == Ytrain.shape[1]

		# We use a single spectral density instance for all the models. For that instance, we compute the needed frequencies
		# and use them throughout all the models
		self.spectral_density = spectral_density 

		self.rrgpMO = [None]*self.dim_out
		for ii in range(self.dim_out):
			# raise ValueError("Figure out what features should we use for Van der pole (probably LINEAR, not RRGPRandomFourierFeatures(); because once lifted to the observable space, we're going linear); have a way to specify this in the config file; HARCODED!!!")
			# self.rrgpMO[ii] = RRPRegularFourierFeatures(dim=self.dim_in,cfg=cfg.gpmodel,spectral_density=self.spectral_density)
			# self.rrgpMO[ii] = RRPDiscreteCosineFeatures(dim=self.dim_in,cfg=cfg.gpmodel,spectral_density=self.spectral_density)
			self.rrgpMO[ii] = RRPLinearFeatures(dim=self.dim_in,cfg=cfg.gpmodel,spectral_density=self.spectral_density)
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


	def sample_state_space_from_prior_recursively(self,x0,x1,traj_length,Nsamples,sort=False,plotting=False):
		"""

		Pass two initial latent values
		x0: [1,self.dim]
		x1: [1,self.dim]

		The GP won't be training during sampling, i.e., we won't call self.train_model()


		return:
		xsamples_X: [Npoints,self.dim_out,Nsamples]
		xsamples_Y: [Npoints,self.dim_out,Nsamples]

		"""

		assert traj_length > 2
		assert Nsamples > 0

		xmin = -6.
		xmax = +3.
		Ndiv = 201
		xpred = tf.linspace(xmin,xmax,Ndiv)
		xpred = tf.reshape(xpred,(-1,1))

		
		colors_arr = cm.winter(np.linspace(0,1,Nsamples))

		# yplot_true_fun = self.spectral_density._nonlinear_system_fun(xpred)
		# yplot_sampled_fun = fx(xpred)

		# Herein, we append (x0,x1). Note that these two points come from a different simulated roll-out of that of the points from the training data.
		# That's fine because our model doesn't make any structural assumption about the data coming from a single MDP.
		Xtraining, Ytraining = self.get_training_data()
		Xtraining_and_new = tf.concat([Xtraining,x0],axis=0)
		Ytraining_and_new = tf.concat([Ytraining,x1],axis=0)
		# Xtraining_and_new = Xtraining
		# Ytraining_and_new = Ytraining
		self.update_model(X=Xtraining_and_new,Y=Ytraining_and_new)

		# Xtmp, Ytmp = self.get_training_data()
		# pdb.set_trace()

		fx = self.get_sample_path_callable(Nsamples=Nsamples)

		if plotting:
			hdl_fig, hdl_splots = plt.subplots(2,1,figsize=(12,8),sharex=True)
			hdl_fig.suptitle(r"Kink function simulation $x_{t+1} = f(x_t) + \varepsilon$"+", kernel: {0}".format("kink"),fontsize=fontsize_labels)

		xsamples = np.zeros((traj_length,self.dim_out,Nsamples),dtype=np.float32)

		# pdb.set_trace()


		xsamples[0,...] = np.stack([x0]*Nsamples,axis=2)
		xsamples[1,...] = np.stack([x1]*Nsamples,axis=2)
		for ii in range(1,traj_length-1):

			# print("ii:",ii)

			xsample_tp = tf.convert_to_tensor(value=xsamples[ii:ii+1,:,0],dtype=np.float32)

			# xsamples[ii+1,:] = self.sample_path_from_predictive(xpred=xsample_tp,Nsamples=Nsamples)
			xsamples[ii+1,...] = fx(xsample_tp)

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