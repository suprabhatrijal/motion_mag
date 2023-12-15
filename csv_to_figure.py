import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

cam_df = pd.read_csv('./pixel_tracking_results/Camera_A/Camera_A_data.csv')
data = pd.read_csv('./Camera_A.csv')

cam_A_y = cam_df.loc[:,'y_movement/pixel']
cam_A_min = np.min(cam_A_y)
cam_A_max = np.max(cam_A_y)
cam_A_y = (cam_A_y-cam_A_min)/ (cam_A_max - cam_A_min)
data_y = data.loc[:,'Position (in)']
data_min = np.min(data_y)
data_max = np.max(data_y)
data_y = (data_y-data_min)/ (data_max - data_min)

print(data.columns)
time = cam_df['time/seconds']
data_time = data['Time']


plt.figure()
plt.xlabel("time/sec(Offset by 0.05 seconds)")
plt.ylabel("postion/pixel(Normalized)")
plt.plot(time,cam_A_y)
plt.plot(data_time,data_y)
plt.savefig("comparision.png")

