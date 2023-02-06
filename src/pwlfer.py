import pwlf
import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib


# assign the thresholds
cruise_thresh = 0.1
cruse_rmse_thresh = 0.25
accel_rmse_thresh = 0.25
decel_rmse_thresh = 0.25
stopped_rmse_thresh = 0
seconds = 3


def pwlf_fit(x, y, n=5):
    # fit the piecewise linear model
    my_pwlf = pwlf.PiecewiseLinFit(x, y)
    breaks = my_pwlf.fit(n)
    return my_pwlf, breaks


def process_trajectory(group, time_col="time", speed_col="speed", abs_time_col="time"):
    name, group = group
    x, y = group[time_col].values, group[speed_col].values
    x = x - x[0]
    my_pwlf, breaks = pwlf_fit(x, y, n=5)
    slopes = my_pwlf.calc_slopes()
    yHat = my_pwlf.predict(x)
    break_list = []
    for i, (b0, b1) in enumerate(
        zip(
            breaks[:-1],
            breaks[1:],
        )
    ):
        slicer = (x >= b0) & (x < b1)
        x_sliced = x[slicer]
        if len(x_sliced) < 3:
            continue
        yHat_break = yHat[slicer]
        yAct_break = y[slicer]
        break_list.append(
            {
                "break": i,
                "y0": my_pwlf.intercepts[i],
                "speed": np.mean(yHat_break),
                "slope": slopes[i],
                "r2": r2_score(yAct_break, yHat_break),
                "rmse": np.sqrt(mean_squared_error(yAct_break, yHat_break)),
                "start": b0,
                "end": b1,
            }
        )
    return {
            "vehicle_id": name,
            "mean_time": group[abs_time_col].mean(),
            "total_seconds": x.max(),
            "line_fits": break_list
        }
    

# with tqdm_joblib(desc="process", total=fcd_df.id.nunique()) as progress_bar:
#     res = Parallel(n_jobs=4)(delayed(process_trajectory)(group) for group in fcd_df.groupby("id"))

def process_trajectories(df, time_col="time", speed_col="speed", abs_time_col="time", id_col="id", n_jobs=4):
    with tqdm_joblib(desc="process", total=df[id_col].nunique()) as progress_bar:
        return Parallel(n_jobs=n_jobs)(delayed(process_trajectory)(
            group, 
            time_col=time_col,
            speed_col=speed_col,
            abs_time_col=abs_time_col,
        ) for group in df.groupby(id_col))

def classify(data):
    if (abs(data["slope"]) < cruise_thresh) and (data["rmse"] < cruse_rmse_thresh):
        # assign type to cruise
        return "cruise"
    elif (data["slope"] > 0) and (data["rmse"] < accel_rmse_thresh):
        # assign type to accel
        return "accel"
    elif (data["slope"] < 0) and (data["rmse"] < decel_rmse_thresh):
        # assign type to decel
        return "decel"
    elif (data["slope"] < 0) and (data["rmse"] < stopped_rmse_thresh):
        # assign type to stopped
        return "stopped"

    return "undefined"
