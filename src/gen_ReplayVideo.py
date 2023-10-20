from utils import Visualisation
# change the current root directory to the this directory
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

subjIDs = range(1,11)

for subjID in subjIDs:
    print(f"SubjID: {subjID}")
    Visualisation.qAnimateXY(subjID, list(range(20)))
    print(f"Subject {subjID} done")

print(f"Done")