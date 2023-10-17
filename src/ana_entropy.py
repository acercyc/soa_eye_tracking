

# %%
def find_nearest(array, x):
    array = np.asarray(array)
    idx = (np.abs(array - x)).argmin()
    return idx

# Use gaze_data['TimeStamp'] to segment gaze_data by event_data['TimeStamp']
def epochGazeData(gaze_data, event_data):
    gaze_data_epoch = []
    for iEvent in range(int(len(event_data) / 2)):
        onset = event_data["TimeStamp"][2 * iEvent]
        offset = event_data["TimeStamp"][2 * iEvent + 1]
        onset_nearest = find_nearest(gaze_data["TimeStamp"], onset)
        offset_nearest = find_nearest(gaze_data["TimeStamp"], offset) + 1
        gaze_data_epoch.append(gaze_data[onset_nearest:offset_nearest])
    return gaze_data_epoch


def compute_density(samples, area=(-1, 1, -1, 1), kernel_bandwidth=None):
    """Compute the density of samples on a 2D grid"""
    from scipy.stats import gaussian_kde

    x, y = samples.T
    if kernel_bandwidth is None:
        kde = gaussian_kde(samples.T)
    else:
        kde = gaussian_kde(kernel_bandwidth)

    x_grid = np.linspace(area[0], area[1], 100)
    y_grid = np.linspace(area[2], area[3], 100)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = kde.evaluate(np.vstack([X.ravel(), Y.ravel()]))
    Z = Z / Z.sum()
    Z = Z.reshape(X.shape)
    return Z, X, Y


def plot_density_estimation(ax, Z, x, y, entropy=None, colorbar=False):
    im = ax.imshow(Z, origin="lower", aspect="auto", extent=[-1, 1, -1, 1], cmap="viridis")
    ax.scatter(x, y, s=2, color="k", alpha=0.1)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    if entropy is not None:
        ax.set_title("Density estimation")
    else:
        ax.set_title(f"Entropy = {entropy:.2f}")
    if colorbar:
        plt.colorbar(im, ax=ax)
    ax.set_aspect("equal", adjustable="box")


def compute_entropy(Z):
    # prevent log(0)
    Z = Z + 1e-10
    entropy = -np.sum(Z * np.log2(Z))
    return entropy


#%%
subjIDs = range(1, 11)
df_trial_all = []
for subjID in subjIDs:
    df_trial = Data.loadTrainInfo(subjID, 2)
    # add subjID to df_trial
    df_trial["subject"] = subjID
    metadata, gaze_data, event_data = Data.loadEyeData(subjID, 2)
    gaze_data_epoch = epochGazeData(gaze_data, event_data)
    df_trial_ = df_trial.sort_values(by=["delay", "trial"]).reset_index(inplace=False, drop=True)

    entropys = []
    for i in range(len(df_trial_)):
        print(i)
        data = gaze_data_epoch[df_trial_["trial"][i]][["GazePointX", "GazePointY"]].values
        data = data[~np.isnan(data).any(axis=1)]
        Z, X, Y = compute_density(data)
        entropy = compute_entropy(Z.flatten())
        entropys.append(entropy)

    df_trial_["entropy"] = entropys
    df_trial_all.append(df_trial_)

df_trial_all = pd.concat(df_trial_all, axis=0)


# %%
# save df_trial_all to csv file
df_trial_all.to_csv('../data/ana_kde/kde.csv', index=False)


# %%
# plot delay against entropy for each subject 
import seaborn as sns
sns.set_theme(style="whitegrid")
g = sns.FacetGrid(df_trial_all, col="subject", col_wrap=3, sharey=False)
g.map(sns.lineplot, "delay", "entropy", errorbar='sd', markers=True, dashes=False, marker="o", err_style="bars")


# %%
# Correlation between rating and entropy
np.corrcoef(df_trial_all["rating"], df_trial_all["entropy"])

# %%
# average entropy for each subject
df_mean = df_trial_all.groupby(["subject", "delay"], as_index=False)["entropy"].mean()

# ungroup df_mean
df_mean = df_mean.reset_index(inplace=False, drop=True)

# plot delay against entropy 
sns.lineplot(x="delay", y="entropy", data=df_mean, errorbar='se', markers=True, dashes=False, marker="o")

# %%
# save df_mean to csv file
df_mean.to_csv('../data/ana_kde/kde_mean.csv', index=False)

# %%
# pivot df_mean: expend delay to column; remove index name
df_mean_pivot = df_mean.pivot(index="subject", columns="delay", values="entropy")
# remove column name
df_mean_pivot.columns.name = None

# remove hierarchical index
df_mean_pivot = df_mean_pivot.reset_index(inplace=False, drop=True)
df_mean_pivot

# save df_mean_pivot to csv file
df_mean_pivot.to_csv('../data/ana_kde/kde_mean_pivot.csv', index=False)


# %%

#%%

# plot average delay against average entropy for all subjects 
import seaborn as sns
sns.lineplot(x="delay", y="entropy", data=df_trial_all, errorbar='se', markers=True, dashes=False, marker="o")


# %%

df_trial = Data.loadTrainInfo(2, 2)
metadata, gaze_data, event_data = Data.loadEyeData(2, 2)
gaze_data_epoch = epochGazeData(gaze_data, event_data)


df_trial_ = df_trial.sort_values(by=["delay", "trial"]).reset_index(inplace=False, drop=True)

entropys = []
for i in range(len(df_trial_)):
    print(i)
    data = gaze_data_epoch[df_trial_["trial"][i]][["GazePointX", "GazePointY"]].values
    data = data[~np.isnan(data).any(axis=1)]
    Z, X, Y = compute_density(data)
    entropy = compute_entropy(Z.flatten())
    entropys.append(entropy)

df_trial_["entropy"] = entropys

# %%

entropys = []
fig, axs = plt.subplots(4, 5, figsize=(30, 30))
for i, ax in enumerate(axs.flatten()):
    print(i)
    data = gaze_data_epoch[df_trial_["trial"][i]][["GazePointX", "GazePointY"]].values
    data = data[~np.isnan(data).any(axis=1)]
    x, y = data.T
    Z, X, Y = compute_density(data)
    entropy = compute_entropy(Z.flatten())
    entropys.append(entropy)
    plot_density_estimation(ax, Z, X, Y, entropy, colorbar=False)
    ax.scatter(x, y, s=2, color="k", alpha=0.02)
    ax.set_title(f'delay = {df_trial_["delay"][i]}, trial = {df_trial_["trial"][i]}, entropy = {entropy:.2f}')
    # limit the axis to [-1, 1]
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    plt.xlabel("X")
    plt.ylabel("Y")
    ax.set_aspect("equal", adjustable="box")



# %%



gaze_data_epoch = epochGazeData(gaze_data, event_data)


# %%
# 'GazePointX', 'GazePointY' with x and y limit from 0 to 1
iTrial = 14
df_ = gaze_data_epoch[iTrial]
df_.plot(x="GazePointX", y="GazePointY", style="o-", xlim=[-1, 1], ylim=[-1, 1], markersize=2)
plt.gca().set_aspect("equal", adjustable="box")

# %%
# length of a data frame
len(gaze_data_epoch[14])


# %%
colors = np.arange(len(df_))
plt.scatter(df_["GazePointX"], df_["GazePointY"], c=colors, cmap="viridis", s=2)
plt.plot(df_["GazePointX"], df_["GazePointY"], color="gray", alpha=0.5)

# Add colorbar
cbar = plt.colorbar()
cbar.set_label("Sample Index")


# %%
# load csv file with pandas, the the cell [0, 0] is empty. Replace cell [0, 0] with 'trial'
subjID = 2
# filename = phase2_tsv_files[subjID]
# filename = '../data/pilot02_Phase2_202305251417.tsv'
# filename = '../data/pilot03_Phase2_202305291538.tsv'
metadata, gaze_data, event_data = loadEyeData(phase2_tsv_files[subjID])
gaze_data_epoch = epochGazeData(gaze_data, event_data)

# filename = '../data/pilot01_phase2_202305251314.csv'
# filename = '../data/pilot02_phase2_202305251417.csv'
# filename = '../data/pilot03_phase2_202305291538.csv'
df_trial = loadTrainInfo(phase2_csv_files[subjID])


# %%
# count the number of trial delays in the df_trial
df_trial["delay"].value_counts()

# sort argument delay
df_trial_ = df_trial.sort_values(by=["delay", "trial"]).reset_index(inplace=False, drop=True)


# %%
# Plot trajectory of trials based on the trial delay with subplots
# delays = np.sort(df_trial['delay'].unique())
fig, axs = plt.subplots(4, 5, figsize=(20, 20))
for i, ax in enumerate(axs.flatten()):
    df_ = gaze_data_epoch[df_trial_["trial"][i]]
    colors = np.arange(len(df_))
    ax.plot(df_["GazePointX"], df_["GazePointY"], color="gray", alpha=0.5)
    ax.scatter(df_["GazePointX"], df_["GazePointY"], c=colors, cmap="viridis", s=2)

    # ax.plot(df_['GazePointX'], df_['GazePointY'], 'o-', markersize=2)
    ax.set_title(f'delay = {df_trial_["delay"][i]}, trial = {df_trial_["trial"][i]}')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_aspect("equal", adjustable="box")


# %%
# add subject column and combine df_trial data across subjects
# filename = '../data/pilot01_phase2_202305251314.csv'
# filename = '../data/pilot02_phase2_202305251417.csv'
# filename = '../data/pilot03_phase2_202305291538.csv'
filenames = phase2_csv_files
df_trial = pd.DataFrame()
for filename in filenames:
    df_trial_ = loadTrainInfo(filename)
    df_trial_["subject"] = filename.split("_")[0].split("/")[-1]
    df_trial = pd.concat([df_trial, df_trial_], axis=0)


# %%
# compute mean of df_trial['rating] based on df_trial['delay']
rating = df_trial.groupby(["subject", "delay"], as_index=False)["delay", "rating"].mean()

# plot mean rating across subjects with dot and line
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
# sns.scatterplot(x='delay', y='rating', data=rating, hue='subject', ax=ax)
sns.lineplot(x="delay", y="rating", data=rating, hue="subject", ax=ax, style="subject", markers=True, dashes=False)
# plt.show()


# %%
rating = df_trial.groupby(["subject", "delay"], as_index=False)["delay", "rating"].mean()
# make a table with subject as row and delay as column
rating_table = rating.pivot(index="subject", columns="delay", values="rating")
rating_table


# %%
# compute entropy by subject and epoch
df_trial_all = []
for i in range(len(phase2_tsv_files)):
    metadata, gaze_data, event_data = loadEyeData(phase2_tsv_files[i])
    gaze_data_epoch = epochGazeData(gaze_data, event_data)
    df_trial = loadTrainInfo(phase2_csv_files[i])
    df_trial["subject"] = i
    entropys = []
    for iEpoch in range(len(gaze_data_epoch)):
        data = gaze_data_epoch[iEpoch][["GazePointX", "GazePointY"]].values
        # clean data with nan
        data = data[~np.isnan(data).any(axis=1)]
        Z, X, Y = compute_density(data)
        entropys.append(compute_entropy(Z.flatten()))
        print(f"subj {i}, epoch {iEpoch}, entropy = {entropys[-1]}")

    df_trial["entropy"] = entropys
    df_trial_all.append(df_trial)

df_trial_all = pd.concat(df_trial_all, axis=0)


# %%
# save df_trial_all to csv file
# df_trial_all.to_csv('../data/df_trial_all.csv', index=False)


# %%
# load df_trial_all from csv file
df_trial_all = pd.read_csv("../data/df_trial_all.csv")
df_trial_all["subject"] = df_trial_all["subject"] + 1


# %%
# compute mean entropy
rating = df_trial_all.groupby(["subject", "delay"], as_index=False)["delay", "entropy"].mean()
# make a table with subject as row and delay as column
rating_table = rating.pivot(index="subject", columns="delay", values="entropy")
rating_table

# %%
# Plot entropy into subplot with subject as column using seaborn
grid = sns.FacetGrid(data=df_trial_all, col="subject", sharey=False)
grid.map(sns.lineplot, "delay", "entropy", errorbar=None, markers=True, dashes=False, marker="o")
plt.show()

# %%
#
data = gaze_data_epoch[0][["GazePointX", "GazePointY"]].values
x, y = data.T
x


# %%

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Here is an example 2D dataset
data = gaze_data_epoch[8][["GazePointX", "GazePointY"]].values

# clean data with nan
data = data[~np.isnan(data).any(axis=1)]

# compute the point density
Z, X, Y = compute_density(data)
fig, ax = plt.subplots(figsize=(8, 8))
plot_density_estimation(ax, Z, x, y, entropy)
plt.show()

# # Create a 2D Gaussian KDE
# kde = gaussian_kde(data.T)
# # kde = gaussian_kde(data.T, bw_method=0.1)

# # Create a regular grid of values to evaluate the KDE on
# x_grid = np.linspace(-1, 1, 100)
# y_grid = np.linspace(-1, 1, 100)
# X, Y = np.meshgrid(x_grid, y_grid)

# # Evaluate the KDE on the grid
# Z = kde.evaluate(np.vstack([X.ravel(), Y.ravel()]))

# # normalize Z
# Z = Z / Z.sum()


# %%

# Example usage:


# %%
# compute entropy


# Plot the result
plt.figure(figsize=(8, 8))
plt.imshow(Z, origin="lower", aspect="auto", extent=[-1, 1, -1, 1], cmap="viridis")
# plt.colorbar(label="Density")
plt.scatter(x, y, s=2, color="k", alpha=0.1)
plt.xlabel("X")
plt.ylabel("Y")

# add entropy to the title
plt.title(f"2D Gaussian Kernel Density Estimation\nEntropy = {entropy:.2f}")

# plt.title('2D Gaussian Kernel Density Estimation')
# equal aspect ratio
plt.gca().set_aspect("equal", adjustable="box")
plt.show()


# %%
iSubj = 0

filename = phase2_tsv_files[iSubj]
metadata, gaze_data, event_data = loadEyeData(filename)
gaze_data_epoch = epochGazeData(gaze_data, event_data)

filename = phase2_csv_files[iSubj]
df_trial = loadTrainInfo(filename)


# count the number of trial delays in the df_trial
df_trial["delay"].value_counts()
# sort argument delay
df_trial_ = df_trial.sort_values(by=["delay", "trial"]).reset_index(inplace=False, drop=True)


# %%
# Plot trajectory of trials based on the trial delay with subplots
# delays = np.sort(df_trial['delay'].unique())
fig, axs = plt.subplots(4, 5, figsize=(40, 40))
for i, ax in enumerate(axs.flatten()):
    print(i)
    data = gaze_data_epoch[df_trial_["trial"][i]][["GazePointX", "GazePointY"]].values
    data = data[~np.isnan(data).any(axis=1)]
    x, y = data.T
    Z, X, Y = compute_density(data)
    entropy = compute_entropy(Z.flatten())
    plot_density_estimation(ax, Z, X, Y, entropy, colorbar=False)
    ax.scatter(x, y, s=2, color="k", alpha=0.1)
    ax.set_title(f'delay = {df_trial_["delay"][i]}, trial = {df_trial_["trial"][i]}, entropy = {entropy:.2f}')
    plt.xlabel("X")
    plt.ylabel("Y")
    ax.set_aspect("equal", adjustable="box")


# %%
# %%
# iSubj = 0

for iSubj in range(5):
    filename = phase2_tsv_files[iSubj]
    metadata, gaze_data, event_data = loadEyeData(filename)
    gaze_data_epoch = epochGazeData(gaze_data, event_data)

    filename = phase2_csv_files[iSubj]
    df_trial = loadTrainInfo(filename)

    # count the number of trial delays in the df_trial
    df_trial["delay"].value_counts()
    # sort argument delay
    df_trial_ = df_trial.sort_values(by=["delay", "trial"]).reset_index(inplace=False, drop=True)

    # Plot trajectory of trials based on the trial delay with subplots
    # delays = np.sort(df_trial['delay'].unique())
    fig, axs = plt.subplots(4, 5, figsize=(30, 30))
    for i, ax in enumerate(axs.flatten()):
        print(i)
        data = gaze_data_epoch[df_trial_["trial"][i]][["GazePointX", "GazePointY"]].values
        data = data[~np.isnan(data).any(axis=1)]
        x, y = data.T
        Z, X, Y = compute_density(data)
        entropy = compute_entropy(Z.flatten())
        plot_density_estimation(ax, Z, X, Y, entropy, colorbar=False)
        ax.scatter(x, y, s=2, color="k", alpha=0.02)
        ax.set_title(f'delay = {df_trial_["delay"][i]}, trial = {df_trial_["trial"][i]}, entropy = {entropy:.2f}')
        # limit the axis to [-1, 1]
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        plt.xlabel("X")
        plt.ylabel("Y")
        ax.set_aspect("equal", adjustable="box")
        # plt.show()
