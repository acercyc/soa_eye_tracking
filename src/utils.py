# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
import os
import re
from scipy.stats import gaussian_kde


# %%


class Data:
    path_raw = "/mnt/sdb1/SynologyDrive/WENLAB-HP-EYE/eye tracker task/sence of agency/data"

    # e.g. a01_phase2_202306221103.csv
    # a01_phase2_202306221103.csv

    def findFile(subjID, phase, filetype="tsv"):
        # find file name and ignore date in the filename using regular expression
        pattern = re.compile("a{:02d}_phase{}.*{}".format(subjID, phase, filetype), re.IGNORECASE)
        for file in os.listdir(Data.path_raw):
            if pattern.match(file):
                return os.path.join(Data.path_raw, file)
        return None

    def loadEyeData(subjID, phase):
        # Read file into list of lines
        filename = Data.findFile(subjID, phase, filetype="tsv")
        with open(filename, "r") as f:
            lines = f.readlines()

        # Extract metadata
        metadata = {}
        for line in lines:
            if line.startswith("Recording"):
                key, value = line.strip().split("\t")
                metadata[key] = value

        # Identify start and end of gaze data and event data
        gaze_data_start = lines.index("Session Start\n") + 1
        gaze_data_end = lines.index("TimeStamp\tEvent\n")
        event_data_start = gaze_data_end
        event_data_end = lines.index("Session End\n")

        # Convert gaze data and event data sections to pa|ndas dataframes
        gaze_data_lines = lines[gaze_data_start:gaze_data_end]
        event_data_lines = lines[event_data_start:event_data_end]

        gaze_data = pd.read_csv(StringIO("\n".join(gaze_data_lines)), delimiter="\t")
        event_data = pd.read_csv(StringIO("\n".join(event_data_lines)), delimiter="\t")

        # Convert 'nan' to np.nan
        gaze_data.replace("nan", np.nan, inplace=True)

        return metadata, gaze_data, event_data

    def loadTrainInfo(subjID, phase):
        filename = Data.findFile(subjID, phase, filetype="csv")
        df_trial = pd.read_csv(filename, delimiter=",", index_col=0)
        df_trial.index.name = "trial"
        df_trial.reset_index(inplace=True)
        return df_trial

