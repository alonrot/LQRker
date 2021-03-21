from scipy import linalg as la
from scipy import stats as sts
import pdb
import numpy as np
import control

class SolveLQR:
	"""

	This class simulates forward experiments of a linear time-invariant system,
	when the cost is quadratic, parametrized with (Q_emp,R_emp).

	For each feedback controller K, the function get_cost() returns the corresponding
	cost.

	Each feedback controller K is computed based on an underlying design cost
	parametrized with (Q_des,R_des). It is computed by solving the Riccatti equation:
	K = DLQR(A,B,Q_des,R_des)

	This class uses the control library python-control
	https://python-control.readthedocs.io/en/0.9.0/

	TODO: Find a way to have this library in tensorflow. Maybe reimplement the needed functions...
	"""

	def __init__(self,Q_emp,R_emp,mu0,Sigma0):

		assert Q_emp.ndim == 2
		assert R_emp.ndim == 2
		assert mu0.ndim == 2, "Pass mu0 as a 2D column vector"
		assert mu0.shape[1] == 1
		assert Sigma0.ndim == 2

		self.Q_emp = Q_emp
		self.R_emp = R_emp
		self.mu0 = mu0
		self.Sigma0 = Sigma0

		# # Debug:
		# self.DBG_P_trace_list = []


	def _get_controller(self, A, B, Q_des, R_des):
		"""
		
		return:
		P: solution to the Ricatti equation for discrete time invariant linear systems
		eig: eigenvalues of the closed loop A - BK
		K: feedabck gain assuming u = -Kx
		"""

		P, eig, K = control.dare(A, B, Q_des, R_des)

		return K

	def get_Lyapunov_solution(self, A, B, Q_des, R_des):
		"""

		We consider here an infinite horizon LQR with stochastic initial condition
		TODO: consider the LQG case
		"""

		K = self._get_controller(A, B, Q_des, R_des)

		A_tilde = A + np.matmul(B,-K) # Flip sign of the controller, as we assume u = Kx

		# TODO: This check will be eliminated in the future
		eig = la.eigvals(A_tilde)
		assert np.all(np.absolute(eig) <= 1.0), "The eigenvalues must be inside the unit circle"

		A_tilde_inv = np.linalg.inv(A_tilde)
		Q_tilde = self.Q_emp + np.matmul(K.T,np.matmul(self.R_emp,K))

		# Too large numbers in large dimensions:
		A_syl = -A_tilde.T
		B_syl = A_tilde_inv
		Q_syl = np.matmul(Q_tilde,A_tilde_inv)


		P = la.solve_discrete_lyapunov(A_tilde,Q_tilde)

		return P

	def forward_simulation_with_random_initial_condition(self, A, B, Q_des, R_des, Nsamples=1):
		"""

		With a random initial condition, we obtain a random sample of the LQR cost.
		Because we are considering a LTI system with infinite horizon, a forward
		simulation of the dynamics can be emulated by simply solving the Lyapunov
		equation.

		The resulting samples follow a distribution which density does not have a
		closed-form solution. However, such distribution is equivalent to the
		distribution of a linear combination of independent chi-squared random
		variates.

		return: [Nsamples,]
		"""

		P = self.get_Lyapunov_solution(A,B,Q_des,R_des)

		x0_vec = np.random.multivariate_normal(mean=self.mu0.reshape(-1), cov=self.Sigma0, size=(Nsamples)) # [Nsamples,dim]

		# Get cost samples:
		Jsamples = np.diag(x0_vec @ P @ x0_vec.T)

		return Jsamples
		

	def forward_simulation_expected_value(self, A, B, Q_des, R_des):
		"""

		We compute the expected value of the LQR cost, with x0 ~ N(mu0,Sigma0).
		Because we are considering a LTI system with infinite horizon, a forward
		simulation of the dynamics can be emulated by simply solving the Lyapunov
		equation.

		return: [Nsamples,]
		"""
		P = self.get_Lyapunov_solution(A,B,Q_des,R_des)

		J = np.trace( P @ self.Sigma0 ) + self.mu0.T @ P @ self.mu0

		# assert J.shape == (1,1)

		# self.DBG_P_trace_list.append(np.trace(np.matmul(P,self.Sigma0)))
		
		return J[0,0]
		
# class GenerateLQRData():
# 	"""

# 	NOTE: Deprecated. Will be removed.


# 	Given a matrix-normal distribution over the linear model matrices (A,B)
# 	and a distribution over possible controller designs theta = (Q_des,R_des),
# 	this class generates forward simulations of the sampled systems (A_j,B_j) with 
# 	controller designs theta_j. For each forward simulation, we compute the
# 	resulting quadratic cost, according to the empirical weights (Q_emp,R_emp)

# 	The class assumes quadratic empirical cost, defined by (Q_emp,R_emp)
# 	"""

# 	def __init__(self,Q_emp,R_emp,mu0,Sigma0,Nsys,Ncon,check_controllability=True):

# 		# Number of systems to sample:
# 		self.Nsys = Nsys

# 		# Number of controller designs to sample, for each samples system:
# 		self.Ncon = Ncon

# 		self.dim_state = Q_emp.shape[0]
# 		self.dim_control = R_emp.shape[0]

# 		self.solve_lqr = SolveLQR(Q_emp,R_emp,mu0,Sigma0)

# 		self.check_controllability = check_controllability

# 	def _sample_systems(self,Nsamples):

# 		M = np.zeros((self.dim_state,self.dim_state))
# 		V = np.eye(self.dim_state)
# 		U = 0.5*np.eye(self.dim_state)
# 		A_samples = self._sample_matrix_normal_distribution(M,V,U,Nsamples)

# 		M = np.zeros((self.dim_state,self.dim_control))
# 		V = np.eye(self.dim_control)
# 		U = 0.5*np.eye(self.dim_state)
# 		B_samples = self._sample_matrix_normal_distribution(M,V,U,Nsamples)

# 		if self.check_controllability:
# 			for ii in range(Nsamples):
# 				self._check_controllability(A_samples[ii,:,:], B_samples[ii,:,:])

# 		return A_samples, B_samples


# 	def _check_controllability(self,A,B):
		
# 		# # assert A.shape[0] > 1, "This function is not designed for scalar systems"
# 		# if A.shape[0] == 1:
# 		# 	return

# 		# Controlability:
# 		ctrb = B
# 		AB_mult = B
# 		for ii in range(1,self.dim_state):
# 			AB_mult = np.matmul(A,AB_mult)
# 			ctrb = np.hstack((ctrb,AB_mult))

# 		rank = np.linalg.matrix_rank(ctrb)

# 		assert rank == A.shape[0], "The generated system is not controllable"
# 		if rank != A.shape[0]:
# 			pdb.set_trace()

# 	def _sample_controller_design_parameters(self,Nsamples_controller):

# 		# https://en.wikipedia.org/wiki/Gamma_distribution
# 		alpha = 1.0 # shape
# 		beta = 1.0 # rate
# 		theta_pars = sts.gamma.rvs(a=alpha,loc=0,scale=1/beta,size=(self.dim_state + self.dim_control,Nsamples_controller))

# 		Q_des_samples = np.zeros((self.dim_state,self.dim_state,Nsamples_controller))
# 		R_des_samples = np.zeros((self.dim_control,self.dim_control,Nsamples_controller))

# 		# pdb.set_trace()
# 		for jj in range(Nsamples_controller):

# 			Q_des_samples[:,:,jj] = np.diag(theta_pars[0:self.dim_state,jj])
# 			R_des_samples[:,:,jj] = np.diag(theta_pars[self.dim_control::,jj])

# 		return Q_des_samples, R_des_samples, theta_pars

# 	def _sample_matrix_normal_distribution(self,M,V,U,Nsamples):
# 		mat_samples = np.random.multivariate_normal(mean=M.ravel(), cov=np.kron(V, U),size=Nsamples).reshape( [-1,M.shape[0],M.shape[1]] )
# 		return mat_samples

# 	def compute_cost_for_each_controller(self):
# 		"""
# 		We sample Nsys linear time-invariant systems living within the system uncertainty and, 
# 		for each system, we sample Ncon controller designs (Q_des,R_des)
# 		For each system, and each design, an optimal infinite-horizon LQR controller is computed.
# 		This controller is executed on the simulated system, and the quadratic cost J, computed with
# 		the empirical weights (Q_emp,R_emp) is returned.

# 		returns:
# 		cost_values_all: [Nsys x Ncon] # One cost value for each system and each controller
# 		theta_pars_all: [(dim_state + dim_control) x  ]
# 		"""

# 		cost_values_all = np.zeros((self.Nsys,self.Ncon))
# 		theta_pars_all = np.zeros((self.dim_state + self.dim_control,self.Nsys,self.Ncon))
		
# 		A_samples, B_samples = self._sample_systems(Nsamples=self.Nsys)
# 		for ii in range(self.Nsys):

# 			Q_des_samples, R_des_samples, theta_pars = self._sample_controller_design_parameters(self.Ncon)
# 			theta_pars_all[:,ii,:] = theta_pars[:,:]
# 			for jj in range(self.Ncon):

# 				Q_des = Q_des_samples[:,:,jj]
# 				R_des = R_des_samples[:,:,jj]

# 				cost_values_all[ii,jj] = self.solve_lqr.forward_simulation_expected_value(A_samples[ii,:,:], B_samples[ii,:,:], Q_des, R_des)

# 		# pdb.set_trace()

# 		return cost_values_all, theta_pars_all




# 		# Approximate discretization (just in case):
# 		# DeltaT = 0.01
# 		# A = A*DeltaT + np.eye(A.shape[0])
# 		# B = B*DeltaT



# 		# Q_tilde_des = Q_des + np.matmul(K.T,np.matmul(R_des,K))
# 		# print("np.linalg.eigvals(Q_tilde_des):",np.linalg.eigvals(Q_tilde_des))
# 		# print("np.linalg.eigvals(Q_tilde):",np.linalg.eigvals(Q_tilde))
# 		# print("np.absolute(eig):",np.absolute(eig))



# 		# # Doesn't work: Gives a negative trace...
# 		# A_syl = A_tilde.T
# 		# B_syl = A_tilde
# 		# Q_syl = -Q_tilde


# 		# P = la.solve_sylvester(A_syl,B_syl,Q_syl) # A_syl*P + P*B_syl = Q_syl


# 		# # # Check:
# 		# # check1 = A_syl @ P + P @ B_syl

# 		# # P = la.solve_discrete_are(A,B,Q,R)

# 		# # Q_new = Q_tilde
# 		# # R_new = np.zeros((2,2))
# 		# # pdb.set_trace()
# 		# # Plib,_,_ = control.dare(A_tilde, np.zeros((2,2)), Q_new, R_new)

# 		# print("A**100:\n",np.linalg.matrix_power(A_tilde,100)[0:3,0:3])
		
# 		# Nrep = 100
# 		# Hor = 1000
# 		# Jcost_vec = np.zeros(Nrep)
# 		# for ii in range(Nrep):
# 		# 	x0 = np.random.randn(A.shape[0],1)
# 		# 	xx = x0
# 		# 	for jj in range(Hor):
# 		# 		Jcost_vec[ii] += np.matmul(xx.T,np.matmul(Q_tilde,xx))
# 		# 		xx = np.matmul(A_tilde,xx)
# 		# 		# if jj == Hor//10:
# 		# 		# 	print("Jcost_vec[ii]:",Jcost_vec[ii])


# 		# Jcost = np.mean(Jcost_vec/Hor)

# 		# print("Jcost:",Jcost)
# 		# print("trace(P):",np.trace(P))

# 		# # # Compute P (not correct)
# 		# # P = np.zeros(A.shape)
# 		# # Hor_red = 10 # The number of terms inside the sum grows exponentially. We end up with 2**Hor_red terms
# 		# # P[:,:] = Q_tilde[:,:]
# 		# # for _ in range(Hor):
# 		# # 	# pdb.set_trace()
# 		# # 	P[:,:] = P + np.matmul(A_tilde.T,np.matmul(P,A_tilde))

# 		# # Jnew = np.trace(np.matmul(P,self.Sigma0)) + np.matmul(self.mu0.T,np.matmul(P,self.mu0))

# 		# # print("Jnew:",Jnew)

# 		# P = Q_tilde
# 		# Hor_red = 1000 # The number of terms inside the sum grows exponentially. We end up with 2**Hor_red terms
# 		# for ii in range(Hor_red):
			
# 		# 	A_tilde_pow = np.linalg.matrix_power(A_tilde,ii+1)
# 		# 	P += np.matmul(A_tilde_pow.T,np.matmul(Q_tilde,A_tilde_pow))
# 		# 	# if ii == Hor_red // 1000 or ii == Hor_red // 100 or ii == Hor_red // 10:
# 		# 	# 	print("trace(P):",np.trace(P))
# 		# 	# if ii == Hor_red-10:
# 		# 	# 	pdb.set_trace()

# 		# Jnew2 = np.trace(np.matmul(P,self.Sigma0)) + np.matmul(self.mu0.T,np.matmul(P,self.mu0))

# 		# print("Jnew2:",Jnew2/Hor_red)

# 		# pdb.set_trace()


# 		# J = np.trace(np.matmul(P,self.Sigma0)) + np.matmul(self.mu0.T,np.matmul(P,self.mu0))
# 		# assert J.shape == (1,1)
		
# 		# return J[0,0]

