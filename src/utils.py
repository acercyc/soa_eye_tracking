# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
import os
import re
from scipy.stats import gaussian_kde


class Data:
    path_raw = "/mnt/sdb1/SynologyDrive/WENLAB-HP-EYE/eye tracker task/sence of agency/data"
    path_data = "../data"

    def __init__(self):
        pass
    
    @staticmethod
    def findFile(subjID, phase, filetype="tsv"):
        # find file name and ignore date in the filename using regular expression
        pattern = re.compile("a{:02d}_phase{}.*{}".format(subjID, phase, filetype), re.IGNORECASE)
        for file in os.listdir(Data.path_raw):
            if pattern.match(file):
                return os.path.join(Data.path_raw, file)
        return None

    @staticmethod
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

    @staticmethod
    def loadTrainInfo(subjID, phase):
        filename = Data.findFile(subjID, phase, filetype="csv")
        df_trial = pd.read_csv(filename, delimiter=",", index_col=0)
        df_trial.index.name = "trial"
        df_trial.reset_index(inplace=True)
        return df_trial
    
    @staticmethod
    def createSavePath(folder):
        save_path = os.path.join(Data.path_data, folder)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        return save_path
    
    @staticmethod
    def createFilePath(folder, filename):
        save_path = Data.createSavePath(folder)
        return os.path.join(save_path, filename)

class Preprocessing:
    def __init__(self) -> None:
        pass

    def epochGazeData(gaze_data, event_data):
        gaze_data_epoch = []
        for iEvent in range(int(len(event_data) / 2)):
            onset = event_data["TimeStamp"][2 * iEvent]
            offset = event_data["TimeStamp"][2 * iEvent + 1]
            onset_ = np.searchsorted(gaze_data["TimeStamp"], onset, side="left")
            offset_ = np.searchsorted(gaze_data["TimeStamp"], offset, side="left")
            gaze_data_epoch.append(gaze_data[onset_:offset_])
        return gaze_data_epoch


class Visualisation:
    def __init__(self):
        pass
    
    @staticmethod
    def plotTrajectory(x, y, fig=None, ax=None):
        '''x and y are 1D numpy array'''
        import matplotlib as mpl
        # find the first non-nan in both x and y 
        nonNaN = np.logical_and(~np.isnan(x), ~np.isnan(y))
        idx = np.argmax(nonNaN)
        x_start = x[idx]
        y_start = y[idx]
                
        if fig is None:
            fig, ax = plt.subplots()
        colors = np.linspace(0, 1, len(x))
        ax.plot(x, y, "-k", alpha=0.2)
        ax.scatter(x, y, c=colors, cmap="turbo", s=2)
        ax.plot(x_start, y_start, "Dr", label="start", markersize=4)
        ax.set_aspect("equal", adjustable="box")
        norm = mpl.colors.Normalize(vmin=0, vmax=len(x))
        cbar = fig.colorbar(mpl.cm.ScalarMappable(cmap="turbo", norm=norm), ax=ax)
        cbar.set_label("Time step")
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])        
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend()
        return fig, ax
    
    @staticmethod
    def plotTrajectory_epoch(epoch, **kwargs):
        x = epoch["GazePointX"].to_numpy()
        y = epoch["GazePointY"].to_numpy()
        return Visualisation.plotTrajectory(x, y, **kwargs)
    
    # @staticmethod
    # def plotTrajectory_epoches(epoches, **kwargs):
    #     nEpoch = len(epoches)
    #     # 5 subplot columns
    #     nRow = int(np.ceil(nEpoch / 5))
    #     fig, axes = plt.subplots(nRow, 5, figsize=(30, 5 * nRow))
    #     for iEpoch, ax in enumerate(axes.flatten()):
    #         if iEpoch < nEpoch:
    #             Visualisation.plotTrajectory_epoch(epoches[iEpoch], fig=fig, ax=ax)
    #             ax.set_title("Epoch {}".format(iEpoch))
    #         else:
    #             ax.axis("off")
    #     return fig, axes

    @staticmethod
    def mapPlotEpoches(f_plot_epoch, epoches, sortVar=None, sortVarName=None, **kwargs):
        '''Map a plotting function to a list of epoches
        sortVar: a list of values ordered by trial order for sorting epoches
        '''
        nEpoch = len(epoches)
        
        if sortVarName is None:
            sortVarName = "value"
        if sortVar is not None:
            # create sorting index
            sortIdx = np.argsort(sortVar)
            
        
        # 5 subplot columns
        nRow = int(np.ceil(nEpoch / 5))
        fig, axes = plt.subplots(nRow, 5, figsize=(30, 5 * nRow))
        for iEpoch, ax in enumerate(axes.flatten()):
            if iEpoch < nEpoch:
                if sortVar is not None:
                    f_plot_epoch(epoches[sortIdx[iEpoch]], fig=fig, ax=ax, **kwargs)
                    ax.set_title(f"Trial {sortIdx[iEpoch]}, {sortVarName}={sortVar[sortIdx[iEpoch]]}")

                else:
                    f_plot_epoch(epoches[iEpoch], fig=fig, ax=ax, **kwargs)
                    ax.set_title(f"Trial {iEpoch}")
            else:
                ax.axis("off")
        return fig, axes
    
    @staticmethod
    def qPlot_trajectory(subjID, phase, sortVarName='delay'):
        _, gaze_data, event_data = Data.loadEyeData(subjID, phase)
        gaze_data_epoch = Preprocessing.epochGazeData(gaze_data, event_data)
        sortVar = Data.loadTrainInfo(subjID, phase)[sortVarName].to_numpy()
        fig, axes = Visualisation.mapPlotEpoches(Visualisation.plotTrajectory_epoch, gaze_data_epoch, sortVar=sortVar, sortVarName=sortVarName)
        return fig, axes




# %%
# metadata, gaze_data, event_data = Data.loadEyeData(1, 2)
# gaze_data_epoch = Preprocessing.epochGazeData(gaze_data, event_data)
# fig, axes = Visualisation.mapPlotEpoches(Visualisation.plotTrajectory_epoch, gaze_data_epoch)

# sortVar = Data.loadTrainInfo(1, 2)['delay'].to_numpy()
# fig, axes = Visualisation.mapPlotEpoches(Visualisation.plotTrajectory_epoch,  gaze_data_epoch, sortVar=sortVar, sortVarName="delay")