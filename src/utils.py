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
    def loadEyeData_epoched(subjID, phase, epochOnly=True):
        metadata, gaze_data, event_data = Data.loadEyeData(subjID, phase)
        gaze_data_epoch = Preprocessing.epochGazeData(gaze_data, event_data)
        if epochOnly:
            return gaze_data_epoch
        else:
            return gaze_data_epoch, metadata, event_data

    @staticmethod
    def loadTrainInfo(subjID, phase, addSubjID=False, addPhase=False):
        filename = Data.findFile(subjID, phase, filetype="csv")
        df_trial = pd.read_csv(filename, delimiter=",", index_col=0)
        df_trial.index.name = "trial"
        df_trial.reset_index(inplace=True)
        if addSubjID:
            df_trial["subjID"] = subjID
        if addPhase:
            df_trial["phase"] = phase
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

    @staticmethod
    def epochGazeData(gaze_data, event_data):
        gaze_data_epoch = []
        for iEvent in range(int(len(event_data) / 2)):
            onset = event_data["TimeStamp"][2 * iEvent]
            offset = event_data["TimeStamp"][2 * iEvent + 1]
            onset_ = np.searchsorted(gaze_data["TimeStamp"], onset, side="left")
            offset_ = np.searchsorted(gaze_data["TimeStamp"], offset, side="left")
            gaze_data_epoch.append(gaze_data[onset_:offset_])
        return gaze_data_epoch
    
    @staticmethod
    def gaze_data2xy(gaze_data, isPickSide=True):
        '''Pick the side with more valid data points'''
        
        if isPickSide:
            if gaze_data['ValidityLeft'].sum() > gaze_data['ValidityRight'].sum():
                x = gaze_data['GazePointXLeft'].to_numpy()
                y = gaze_data['GazePointYLeft'].to_numpy()
            else:
                x = gaze_data['GazePointXRight'].to_numpy()
                y = gaze_data['GazePointYRight'].to_numpy()
        else:
            x = gaze_data['GazePointX'].to_numpy()
            y = gaze_data['GazePointY'].to_numpy()
        return x, y
    
    @staticmethod
    def createDelayDataframe(df, delay):
        '''Create a dataframe with delay (s) in the TimeStamp column'''
        delay = delay * 1000
        df['TimeRelative'] = df['TimeStamp'] - df['TimeStamp'].iloc[0]
        df_delay = df.copy()
        df_delay = df_delay[df_delay['TimeRelative'] > delay]
        df = df.iloc[:len(df_delay)]
        return df, df_delay

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
        x, y = Preprocessing.gaze_data2xy(epoch)
        # x = epoch["GazePointX"].to_numpy()
        # y = epoch["GazePointY"].to_numpy()
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
    def mapPlotEpoches(f_plot_epoch, epoches, sortIdx=None, sortVal=None, sortVarName=None, **kwargs):
        '''Map a plotting function to a list of epoches
        sortVal: a list of index ordered by trial order for sorting epoches
        '''
        nEpoch = len(epoches)
        
        if sortVarName is None:
            sortVarName = "value"            
        
        nRow = int(np.ceil(nEpoch / 5))   # 5 subplot columns
        fig, axes = plt.subplots(nRow, 5, figsize=(30, 5 * nRow))
        for iEpoch, ax in enumerate(axes.flatten()):
            if iEpoch < nEpoch:
                if sortIdx is not None:
                    f_plot_epoch(epoches[sortIdx[iEpoch]], fig=fig, ax=ax, **kwargs)
                    ax.set_title(f"Trial {sortIdx[iEpoch]}, {sortVarName}={sortVal[iEpoch]}")

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
        df = Data.loadTrainInfo(subjID, phase).sort_values(by=[sortVarName, 'trial'], inplace=False)
        sortIdx = df.index.to_numpy()
        # sortVar = Data.loadTrainInfo(subjID, phase)[sortVarName].to_numpy()
        fig, axes = Visualisation.mapPlotEpoches(Visualisation.plotTrajectory_epoch, gaze_data_epoch, 
                                                 sortIdx=sortIdx, 
                                                 sortVal=df[sortVarName].to_numpy(),
                                                 sortVarName=sortVarName)
        return fig, axes

    @staticmethod
    def animateXY(x, y, savepath=None, nTrail=30, fs=600, downsample=30, fps=30):
        """
        Animates the x and y coordinates of a point over time, along with a trail of previous positions.

        Parameters:
        x (array-like): The x-coordinates of the point over time.
        y (array-like): The y-coordinates of the point over time.
        nTrail (int, optional): The number of previous positions to include in the trail. Default is 30.
        fs (int, optional): The sampling frequency of the input data. Default is 60.
        downsample (int, optional): The factor by which to downsample the input data. Default is 30.
        fps (int, optional): The frames per second of the output animation. Default is 30.
        """
        import matplotlib.animation as animation
        import matplotlib.colors as mcolors

        # downsample the data from "fs" to "downsample"
        if downsample is not None:
            x = x[::int(fs/downsample)]
            y = y[::int(fs/downsample)]
            fs = downsample

        # compute time for each sample
        ts = np.arange(len(x)) / fs

        # Create a figure and axis for the animation
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)    

        # Create colormap and normalization instances
        colormap = plt.cm.YlOrBr_r
        norm = mcolors.Normalize(vmin=0, vmax=nTrail-1)

        # Initial plot setup
        trail_points = [ax.plot([], [], 'o', 
                                markersize=6, 
                                alpha=0.5,
                                markerfacecolor=colormap(norm(i)),
                                markeredgecolor='blue',
                                color=colormap(norm(i)))[0] for i in range(nTrail)]
        point, = ax.plot([], [], 'o', 
                        markersize=8, 
                        markerfacecolor=None, 
                        markeredgecolor='red', 
                        markeredgewidth=2)
        timestamp_text = ax.annotate("", xy=(0.02, 0.98), 
                                    xycoords='axes fraction', 
                                    verticalalignment='top', 
                                    fontsize=10)

        # Initialization function for the animation
        def init():
            point.set_data([], [])
            [tp.set_data([], []) for tp in trail_points]
            timestamp_text.set_text("")
            return [point, timestamp_text] + trail_points

        # Animation update function
        def update(frame):
            # update the current point
            if not np.isnan(x[frame]) and not np.isnan(y[frame]):
                point.set_data(x[frame], y[frame])
            else:
                point.set_data([], [])
            
            # update the trail points
            for i, trail_point in enumerate(trail_points):
                if (frame-1-i) >= 0:
                    if not np.isnan(x[frame-1-i]) and not np.isnan(y[frame-1-i]):
                        trail_point.set_data(x[frame-1-i], y[frame-1-i])
                    else:
                        trail_point.set_data([], [])
                else:
                    trail_point.set_data([], [])

            # Update the displayed timestamp
            # convert seconds to readable format
            minutes, seconds = divmod(ts[frame], 60)
            if minutes:
                t_string = f"{int(minutes)}:{seconds:05.2f}"
            else:
                t_string = f"{seconds:05.2f}"

            timestamp_text.set_text(f"Frame: {frame}, Time: {t_string}")            
            return [point, timestamp_text] + trail_points

        # Create the animation object
        ani = animation.FuncAnimation(fig, update, frames=len(x), init_func=init, blit=False, repeat=True, interval=1000/30)
        
        if savepath is None:
            plt.show()
        else:
            ani.save(savepath, writer='ffmpeg', fps=fps)
        plt.close(fig)   
        return ani
    
    @staticmethod
    def qAnimateXY(subjID, iTrial, phase=2, **kwargs):
        folder = Data.createSavePath("ReplayVideo")
        # check if iTrial is list 
        if not isinstance(iTrial, list):
            iTrial = [iTrial]
        
        gaze_data_epoch = Data.loadEyeData_epoched(subjID, phase=phase)
        for i in iTrial:
            x, y = Preprocessing.gaze_data2xy(gaze_data_epoch[i])
            filename = Data.createFilePath(folder, f"subj{subjID}_phase{phase}_trial{i}.mp4")
            Visualisation.animateXY(x, y, savepath=filename, **kwargs)
    
   
Visualisation.qAnimateXY(1, [1, 2, 3]) 
# gaze_data_epoch = Data.loadEyeData_epoched(1, 2)
# x, y = Preprocessing.gaze_data2xy(gaze_data_epoch[0])
# data = np.array(xy).transpose()
# np.save("data.npy", data)




# %%
