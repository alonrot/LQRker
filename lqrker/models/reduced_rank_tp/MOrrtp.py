import tensorflow as tf
import tensorflow_probability as tfp
import math
import pdb
import numpy as np

from lqrker.models import RRTPRegularFourierFeatures, RRTPDiscreteCosineFeatures
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
			# self.rrgpMO[ii] = RRTPRegularFourierFeatures(dim=self.dim_in,cfg=cfg.gpmodel,spectral_density=self.spectral_density)
			self.rrgpMO[ii] = RRTPDiscreteCosineFeatures(dim=self.dim_in,cfg=cfg.gpmodel,spectral_density=self.spectral_density)
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

		# xmin = -6.
		# xmax = +3.
		# Ndiv = 201
		# xpred = tf.linspace(xmin,xmax,Ndiv)
		# xpred = tf.reshape(xpred,(-1,1))

		# yplot_true_fun = self.spectral_density._nonlinear_system_fun(xpred)
		# yplot_sampled_fun = fx(xpred)


		Xtraining = tf.identity(self.X) # Copy tensor
		Ytraining = tf.identity(self.Y) # Copy tensor
		Xtraining_and_new = tf.concat([Xtraining,x0],axis=0)
		Ytraining_and_new = tf.concat([Ytraining,x1],axis=0)
		self.update_model(X=Xtraining_and_new,Y=Ytraining_and_new)

		fx = self.get_sample_path_callable(Nsamples=Nsamples)

		if plotting:
			hdl_fig, hdl_splots = plt.subplots(2,1,figsize=(12,8),sharex=True)
			hdl_fig.suptitle(r"Kink function simulation $x_{t+1} = f(x_t) + \varepsilon$"+", kernel: {0}".format("kink"),fontsize=fontsize_labels)

		xsamples = np.zeros((traj_length,self.dim_out,Nsamples),dtype=np.float32)

		# pdb.set_trace()


		xsamples[0,...] = np.stack([x0]*Nsamples,axis=2)
		xsamples[1,...] = np.stack([x1]*Nsamples,axis=2)
		for ii in range(1,traj_length-1):

			print("ii:",ii)

			xsample_tp = tf.convert_to_tensor(value=xsamples[ii:ii+1,:,0],dtype=np.float32)

			# xsamples[ii+1,:] = self.sample_path_from_predictive(xpred=xsample_tp,Nsamples=Nsamples)
			xsamples[ii+1,...] = fx(xsample_tp)

			# Xnew = tf.convert_to_tensor(value=xsamples[0:ii,:,0],dtype=np.float32)
			# Ynew = tf.convert_to_tensor(value=xsamples[1:ii+1,:,0],dtype=np.float32)

			# Xtraining_and_new = tf.concat([Xtraining,Xnew],axis=0)
			# Ytraining_and_new = tf.concat([Ytraining,Ynew],axis=0)
			# self.update_model(X=Xtraining_and_new,Y=Ytraining_and_new)

			if plotting and self.dim_out in [1,2]:
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

		xsamples_X = xsamples[0:-1,...]
		xsamples_Y = xsamples[1::,...]

		if sort:
			assert x0.shape[1] == 1, "This only makes sense in 1D"
			ind_sort = tf.argsort(xsamples_X[:,0],axis=0)
			# pdb.set_trace()
			xsamples_X = xsamples_X[ind_sort,:]
			xsamples_Y = xsamples_Y[ind_sort,:]

		# # Go back to the dataset at it was:
		# self.update_model(X=Xtraining,Y=Ytraining)

		return xsamples_X, xsamples_Y