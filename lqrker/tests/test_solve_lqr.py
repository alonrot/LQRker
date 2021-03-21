import numpy as np
from lqrker.utils.solve_lqr import SolveLQR
from lqrker.utils.generate_linear_systems import GenerateLinearSystems
import pdb
import matplotlib.pyplot as plt

def test_solve_lqr():
	"""

	Testing the SolveLQR class
	"""

	Q_emp = np.eye(4)
	R_emp = 0.1*np.eye(1)

	dim_state = Q_emp.shape[0]
	dim_control = R_emp.shape[1]

	generate_linear_systems =  GenerateLinearSystems(dim_state,dim_control,Nsys=1,check_controllability=True)
	A_samples, B_samples = generate_linear_systems()
	A = A_samples[0,:,:]
	B = B_samples[0,:,:]

	mu0 = np.zeros((dim_state,1))
	Sigma0 = np.eye(dim_state)

	solve_lqr = SolveLQR(Q_emp,R_emp,mu0,Sigma0)

	J_mean = solve_lqr.forward_simulation_expected_value(A,B,Q_des=Q_emp,R_des=R_emp)
	J_samples = solve_lqr.forward_simulation_with_random_initial_condition(A,B,Q_des=Q_emp,R_des=R_emp,Nsamples=1000)
	P = solve_lqr.get_Lyapunov_solution(A,B,Q_des=Q_emp,R_des=R_emp)
	print("P = ",P)

	hdl_fig, hdl_splots = plt.subplots(1,1,figsize=(14,10),sharex=True)
	hdl_splots.hist(J_samples,bins=50)
	hdl_splots.axvline(x=J_mean,color="red")
	plt.show(block=True)

if __name__ == "__main__":

	test_solve_lqr()