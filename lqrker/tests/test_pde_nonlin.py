import tensorflow as tf
import gpflow
import numpy as np
import pdb
import matplotlib.pyplot as plt


class Matern32sinusoidal(gpflow.kernels.IsotropicStationary):
    """
    The Matern 3/2 kernel. Functions drawn from a GP with this kernel are once
    differentiable. The kernel equation is

    k(r) = σ² (1 + √3r) exp{-√3 r}

    where:
    r  is the Euclidean distance between the input points, scaled by the lengthscales parameter ℓ,
    σ² is the variance parameter.
    """

    def K_r(self, r):
        
        return self.variance * ( tf.sin(r) - r*tf.cos(r) )

def test_pde():

	Ndiv_data = 201
	t_evol = np.linspace(0.0, 10.0, Ndiv_data,dtype=np.float64).reshape(Ndiv_data, 1)
	x_evol = np.zeros(Ndiv_data,dtype=np.float64).reshape(Ndiv_data,1)
	x_evol[0,0] = np.pi
	mean_evol = np.zeros(Ndiv_data,dtype=np.float64)
	std_evol = np.zeros(Ndiv_data,dtype=np.float64)

	
	# ## generate test points for prediction
	# Ndiv = 101
	# tt = np.linspace(0.0, 10.0, Ndiv).reshape(Ndiv, 1)  # test points must be of shape (N, D)

	epsi = 0.0
	for k in range(Ndiv_data-1):

		# Local time and state xt
		t_local = np.array([[t_evol[k,0]]],dtype=np.float64)
		x_local = np.array([[x_evol[k,0]]],dtype=np.float64)

		if np.cos(x_evol[k,0]) <= epsi: # Stable
			ker = Matern32sinusoidal()
			# fac = 0.03*np.pi
			fac = 0.0
			# ker = gpflow.kernels.Matern52()
			ls = 1./np.abs(np.cos(x_local[0,0])) # Adjust the lengthscale a bit
		else: # Unstable
			ker = gpflow.kernels.Matern52()
			fac = 0.0
			ls = 1./np.abs(np.cos(x_local[0,0]))
		
		# We do inference in the vicinity of the point
		# t_local_vicinity = np.array([[0.0]])
		m = gpflow.models.GPR(data=(t_local, x_local), kernel=ker, mean_function=None)

		# Update model:
		m.likelihood.variance.assign(0.001)
		m.kernel.lengthscales.assign(ls)
		m.kernel.variance.assign(0.5)

		# Predict for the next time step:
		# t_delta_vicinity = np.array([[(10.0 - 0.0)/(Ndiv_data-1)]])
		t_next = np.array([[t_evol[k+1,0]+fac]],dtype=np.float64)
		
		mean, var = m.predict_f(t_next)

		# pdb.set_trace()

		stddev = tf.sqrt(var)
		std_evol[k+1] = stddev
		mean_evol[k+1] = mean

		# Sample from such Gaussian:
		x_next = tf.random.normal(shape=(1,1),mean=tf.cast(mean,dtype=tf.float32),stddev=tf.cast(stddev,dtype=tf.float32))
		x_evol[k+1,0] = x_next[0,0]


	# pdb.set_trace()

	# ## generate 10 samples from posterior
	# tf.random.set_seed(1)  # for reproducibility
	# samples = m.predict_f_samples(tt, 10)  # shape (10, 100, 1)

	## plot
	plt.figure(figsize=(12, 6))
	plt.errorbar(t_evol, mean_evol, yerr=std_evol, linewidth=2, linestyle="--",color="b")
	plt.plot(t_evol, x_evol, linewidth=2, linestyle="none",color="k",markersize=7,marker="o")
	plt.show(block=True)
	# plt.plot(tt, mean, "C0", lw=2)
	# plt.fill_between(
	#     tt[:, 0],
	#     mean[:, 0] - 1.96 * np.sqrt(var[:, 0]),
	#     mean[:, 0] + 1.96 * np.sqrt(var[:, 0]),
	#     color="C0",
	#     alpha=0.2,
	# )

	# plt.plot(tt, samples[:, :, 0].numpy().T, "C0", linewidth=0.5)
	# _ = plt.xlim(-0.1, 1.1)


def test_Matern32sinusoidal():

	r_vec = np.linspace(0.,5.,101)

	ker = Matern32sinusoidal()
	ker_vec = ker.K_r(r_vec)

	plt.plot(r_vec,ker_vec)
	plt.show(block=True)


if __name__ == "__main__":

	# test_Matern32sinusoidal()

	test_pde()





