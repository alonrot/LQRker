import numpy as np
from lqrker.solve_lqr import SolveLQR
import pdb
import control
import matplotlib.pyplot as plt
np.set_printoptions(linewidth=1000)
import scipy

def test_solve_lqr():

	# np.random.seed(1)

	Q_emp = np.eye(2)
	R_emp = 0.5*np.eye(1)

	dim_state = Q_emp.shape[0]
	dim_control = R_emp.shape[1]

	A = np.random.uniform(size=Q_emp.shape,low=-2.0,high=2.0)
	B = np.random.uniform(size=(dim_state,dim_control),low=-2.0,high=2.0)

	# Approximate discretization:
	# DeltaT = 0.1
	# A = DeltaT * A + np.eye(A.shape[0])
	# B = DeltaT * B

	# Compute controller:
	P, eig, K = control.dare(A, B, Q_emp, R_emp)

	# Closed loop system:
	A_tilde = A - B @ K
	Q_tilde = Q_emp + K.T @ (R_emp @ K)

	# # pdb.set_trace()
	# Adiag = np.random.uniform(size=2,low=-0.99,high=+0.99)
	# A_tilde = np.diag(Adiag)

	print("np.linalg.eigvals(A_tilde):",np.linalg.eigvals(A_tilde))

	# Initial condition:
	mu0 = np.zeros((dim_state,1))
	Sigma0 = np.eye(dim_state)
	# Sigma0_L = np.random.rand(dim_state, dim_state)
	# Sigma0 = Sigma0_L @ Sigma0_L.T
	x0 = np.random.randn(A.shape[0],1)

	# Noise covariance:
	V = np.eye(dim_state)
	
	# Horizon:
	Hor = 2

	# Roll-out simulation:
	xx = x0
	Jcost = 0.0
	for jj in range(Hor):
		Jcost += np.matmul(xx.T,np.matmul(Q_tilde,xx))
		xx = np.matmul(A_tilde,xx)

	print("Jcost:",Jcost)

	# Create joint matrix of the distribution of z = (x0,x1,...,xN)
	SigmaHor = np.zeros([dim_state*(Hor+1)]*2)

	# Get stationary state covariance:
	# See Eq. 7 from [1].
	# [1] Schluter, H., Solowjow, F. and Trimpe, S., 2020. Event-triggered
	# learning for linear quadratic control. IEEE Transactions on Automatic
	# Control.
	XV = scipy.linalg.solve_discrete_lyapunov(A_tilde,V)


	Upper_list = []
	for ii in range(Hor+1):
		Upper_list.append(XV @ (np.linalg.matrix_power(A_tilde, ii).T))

	# Fill first the upper triangular:
	for row in range(Hor+1):
		for col in range(row,Hor+1):
			SigmaHor[row*dim_state:(row+1)*dim_state,col*dim_state:(col+1)*dim_state] = Upper_list[col-row][:,:]


	# assert False, "Make sure P is the same as XV in the paper ..."


	# Compute the entire matrix:
	SigmaHor = (SigmaHor + SigmaHor.T) - np.kron(np.eye(Hor+1),XV)

	eigvals_SigmaHor = np.sort(np.linalg.eigvals(SigmaHor))
	print("eigvals_SigmaHor:",eigvals_SigmaHor)
	if not np.all(eigvals_SigmaHor > 0):#, "Some eigenvalues of SigmaHor are negative ... (!)"
		pdb.set_trace()
		# pass

	# pdb.set_trace()
	QHor = np.kron(np.eye(Hor+1),Q_tilde)

	# Compute spectral decomposition:
	# QHor and SigmaHor must be real symmetric, and SigmaHor > 0
	eigvals, eigvect = np.linalg.eigh(QHor @ SigmaHor)
	eigvals = np.abs(eigvals)
	print("eigvals(QHor @ SigmaHor):",eigvals)
	# print("eigvals:",eigvals)

	# Compute ranks (d.o.f of chi-squared distributions), i.e., multiplicity of eigenvalues:	
	pass

	# Assuming that there are no repeated eigenvalues (all with single
	# multiplicity), we end up having a linear combination of dim_state*(Hor+1)
	# chi-squared variables with the eigenvalues as weights of the linear
	# combination. If E[x0] = 0, then E[z] = 0, with z = (x0,x1,...,xN)
	Nsamples = 5000
	Nels = len(eigvals)
	samples_chi2 = np.random.noncentral_chisquare(df=1,nonc=0.0,size=(Nsamples,Nels))
	sample_cost = np.sum(eigvals[None,:] * samples_chi2,axis=1)

	# print("sample_cost:",sample_cost)

	# Double-check by sampling directly from the MVN distribution:
	z_samples = np.random.multivariate_normal(mean=np.zeros(SigmaHor.shape[0]), cov=SigmaHor, size=(Nsamples)) # [Nsamples, dim_state*(Hor+1)]

	sample_cost_nor = np.diag(z_samples @ ( QHor @ z_samples.T ))
	# print("sample_cost_nor:",sample_cost_nor)

	# pdb.set_trace()

	hdl_fig, hdl_splots = plt.subplots(2,1,figsize=(14,10),sharex=True)
	hdl_splots[0].hist(sample_cost,bins=50)
	hdl_splots[0].set_xlabel("Using Chi2 directly")

	hdl_splots[1].hist(sample_cost_nor,bins=50)
	hdl_splots[1].set_xlabel("Using Normal")

	plt.show(block=True)

	# Some conclusions:
	# It appears that the multiplicity of the eigenvalues is 1 for all of them.
	# This means that the d.o.f of the chi-squared distributions are all one.
	# We are assuming E[x0] = 0, then E[z] = 0, with z = (x0,x1,...,xN), which simplifies things.
	# The eigenvalues are the weights of the linear combination. Such linear
	# combination does not follow a chi-squared distribution and its density cannot be written analytically.
	# Its distribution is also not known and can be approximated with Laguerre polynomials [1] or an infinite Gamma series [2]
	# [1]https://www.ine.pt/revstat/pdf/rs130301.pdf
	# [2] https://www.sciencedirect.com/science/article/pii/089812218490066X#:~:text=The%20distribution%20function%20of%20a%20linear%20combination%20of%20independent%20central,a%20sufficient%20degree%20of%20accuracy.
	# For some reason, the sample_cost_nor does NOT coincide with sample_cost ... Why?

if __name__ == "__main__":

	test_solve_lqr()