import numpy as np
from lqrker.solve_lqr import SolveLQR, GenerateLQRData
import pdb


def test_generate_lqr_data():

	Q_emp = np.array([[1.0,0.0],[0.0,1.0]])
	R_emp = np.array([[0.1,0.0],[0.0,0.1]])

	dim_state = Q_emp.shape[0]
	dim_control = R_emp.shape[1]
	
	mu0 = np.zeros((dim_state,1))
	Sigma0 = np.eye(dim_state)

	generate_lqr_data = GenerateLQRData(Q_emp,R_emp,mu0,Sigma0)

	cost_values_all, theta_pars_all = generate_lqr_data.compute_cost_for_each_controller()

	pdb.set_trace()

def test_solve_lqr():


	Q_emp = np.array([[1.0,0.0],[0.0,1.0]])
	R_emp = np.array([[0.1,0.0],[0.0,0.1]])

	dim_state = Q_emp.shape[0]
	dim_control = R_emp.shape[1]

	A = np.random.uniform(size=Q_emp.shape,low=-2.0,high=2.0)
	B = np.random.uniform(size=(dim_state,dim_control),low=-2.0,high=2.0)

	# Discrete system:

	# Controlability:
	ctrb = np.hstack((B,np.matmul(A,B)))
	rank = np.linalg.matrix_rank(ctrb)
	assert rank == Q_emp.shape[0], "The generated system is not controllable"

	mu0 = np.zeros((dim_state,1))
	Sigma0 = np.eye(dim_state)

	solve_lqr = SolveLQR(Q_emp,R_emp,mu0,Sigma0)

	# pdb.set_trace()

	J = solve_lqr.forward_simulation(A,B,Q_des=Q_emp,R_des=R_emp)
	print("J = ",J)

if __name__ == "__main__":

	# test_solve_lqr()

	test_generate_lqr_data()