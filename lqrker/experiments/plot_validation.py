import matplotlib
# matplotlib.use('TkAgg') # Solves a no-plotting issue for macOS users
import signal
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import rc
from matplotlib.ticker import FormatStrFormatter
import yaml
import numpy as np
import pdb


def sigquit_handler(signum, frame):
	print('SIGQUIT received; exiting')
	sys.exit(os.EX_SOFTWARE)

def load_data(which_model):

	if which_model == "RRTP":
		path2data = "./outputs/2021-03-12/01-56-28" # RRTP
	elif which_model == "GPFLOW":
		path2data = "./outputs/2021-03-12/02-17-45" # GPFLOW
	else:
		raise ValueError()
	
	Nexps = 100
	smse_vec = np.zeros(Nexps)
	msll_vec = np.zeros(Nexps)
	ind_sel = np.ones(Nexps,dtype=bool)
	for nr_exp in range(Nexps):

		# Open corresponding file to the wanted results (we assume only one experiment has been made):
		path2file = "{0:s}/data_{1:s}_{2:d}.yaml".format(path2data,which_model,nr_exp)
		print("Loading {0:s} ...".format(path2file))
		try:
			stream = open(path2file, "r")
			my_node = yaml.load(stream,Loader=yaml.Loader)
			smse_vec[nr_exp] = my_node["smse"]
			msll_vec[nr_exp] = my_node["msll"]
			stream.close()
		except Exception as inst:
			print("type: {0:s} | args: {1:s}".format(str(type(inst)),str(inst.args)))
			ind_sel[nr_exp] = False


	if np.all(ind_sel == False):
		raise ValueError("All files failed to load...")
	smse_vec = smse_vec[ind_sel]
	msll_vec = msll_vec[ind_sel]

	return smse_vec, msll_vec

def main():

	Nbins = 20
	smse_rrtp_vec, msll_rrtp_vec = load_data(which_model="RRTP")
	smse_gpflow_vec, msll_gpflow_vec = load_data(which_model="GPFLOW")

	hdl_fig, hdl_splots = plt.subplots(2,2,figsize=(14,10))
	# hdl_fig.suptitle("Reduced-rank Student-t process")
	hdl_splots[0,0].hist(smse_rrtp_vec,Nbins)
	hdl_splots[0,1].hist(msll_rrtp_vec,Nbins)
	# hdl_splots[0,0].set_xlabel("SMSE")
	# hdl_splots[0,1].set_xlabel("MSLL")
	hdl_splots[0,0].set_title("RRTP")
	hdl_splots[0,1].set_title("RRTP")

	smse_gpflow_vec = smse_gpflow_vec[smse_gpflow_vec < 120000]
	msll_gpflow_vec = msll_gpflow_vec[msll_gpflow_vec < 120000]

	hdl_splots[1,0].hist(smse_gpflow_vec,Nbins)
	hdl_splots[1,1].hist(msll_gpflow_vec,Nbins)
	hdl_splots[1,0].set_xlabel("SMSE")
	hdl_splots[1,1].set_xlabel("MSLL")
	hdl_splots[1,0].set_title("GPFLOW")
	hdl_splots[1,1].set_title("GPFLOW")
	# hdl_splots[1,0].set_xlim([0,10000])
	# hdl_splots[1,1].set_xlim([0,10000])



	plt.show(block=True)



if __name__ == "__main__":

	# Handle signal that kills the program:
	signal.signal(signal.SIGQUIT, sigquit_handler)

	main()