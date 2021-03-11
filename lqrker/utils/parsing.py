import logging
import yaml

def save_data(node2write: dict, which_model: str, rep_nr: int) -> None:

	file2save = "./data_{0:s}_{1:d}.yaml".format(which_model,rep_nr)

	print("\nSaving in {0:s} ...".format(file2save))

	with open(file2save, "w") as stream_write:
		yaml.dump(node2write, stream_write)

def get_logger(name,level=logging.INFO):

	logger = logging.getLogger(name)
	ch = logging.StreamHandler()
	ch.setLevel(level)
	formatter = logging.Formatter('[%(name)s] %(message)s')
	ch.setFormatter(formatter)
	logger.addHandler(ch)
	logger.setLevel(level)

	return logger