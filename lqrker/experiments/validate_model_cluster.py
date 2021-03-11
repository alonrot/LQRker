import hydra
import pdb
from lqrker.utils.parsing import get_logger, save_data
logger = get_logger(__name__)

@hydra.main(config_path=".",config_name="config.yaml")
def validate(cfg: dict) -> None:

	if cfg.cluster.which_model == "RRTP":
		from validate_model import validate_rrtp_for_func as validate
	elif cfg.cluster.which_model == "GPFLOW":
		from validate_model import validate_gpflow_for_func as validate
	else:
		raise ValueError("")

	# Validate model:
	rep_nr = cfg.cluster.rep_nr
	which_model = cfg.cluster.which_model
	smse, msll = validate(cfg,rep_nr)
	node2write = dict(smse=smse,msll=msll)

	logger.info("smse_rrtp: {0:f}".format(smse))
	logger.info("msll_rrtp: {0:f}".format(msll))

	save_data(node2write,which_model,rep_nr)

if __name__ == "__main__":

	validate()