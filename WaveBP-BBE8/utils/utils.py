import numpy as np

def read_human_loc(human_loc_dict, record):    
	userID, testDate, testID = record
	return human_loc_dict[f"{userID}_{testDate}_{testID}"]
