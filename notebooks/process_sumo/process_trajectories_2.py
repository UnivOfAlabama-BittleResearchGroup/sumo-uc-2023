import os
import sys
from pathlib import Path


def recursive_root(path: str, find="sumo-uc-23"):
    if os.path.split(path)[-1] == find:
        return Path(path)
    return recursive_root(os.path.split(path)[0], find=find)


ROOT = recursive_root(os.path.abspath("."))
sys.path.append(str(ROOT))

from src.walk_configs import walk_configs

"""

This is nasty code but its just a jupyter export so I don't care.

"""
dt_col = "time"
velocity_col = "speed"
id_col = "id"
time_col = "time"
speed_col = "speed"


experiment_path = Path(
    # load the path from the command line
    sys.argv[1]
)


configs = list(walk_configs(experiment_path))


import pandas as pd


def load_fcd(config):
    fcd_df = pd.read_parquet(config.Blocks.XMLConvertConfig.target)
    fcd_df[["x", "y", "time", "speed", "pos"]] = fcd_df[
        ["x", "y", "time", "speed", "pos"]
    ].astype(float)
    return fcd_df


from shapely.geometry import Polygon, Point
from sumolib.shapes import polygon


polys = polygon.read(str(ROOT / "sumo-xml" / "detectors" / "radar_boxes.xml"))
polygon_dict = {
    poly.id: Polygon(poly.shape)
    for poly in polys
    if poly.id
    in [
        "Radar137_East_thru",
    ]
}


def label_polygons(fcd_df):
    fcd_df["box"] = ""
    for poly_id, poly in polygon_dict.items():
        fcd_df.loc[fcd_df["box"] == "", "box"] = (
            fcd_df.loc[fcd_df["box"] == "", ["x", "y"]]
            .apply(lambda x: poly.contains(Point(*x)), axis=1, raw=True)
            .map({True: poly_id, False: ""})
        )
    return fcd_df


from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from src.lowess import lowess
import numpy as np
import pwlf


def get_xy(
    df,
):
    x = df[dt_col] - df[dt_col].min()
    y = df[velocity_col]
    a = y < 0.1
    zero_sum = a.cumsum() - a.cumsum().where(~a).ffill().fillna(0).astype(int)
    # keep only where zero_sum is < 3
    slicer = (zero_sum < 2) | (zero_sum.shift(1).fillna(1) > 0)
    return x.loc[slicer].values, y.loc[slicer].values.astype(float)


def get_lowess(x, y):
    return lowess(x=x, y=y, f=1 / 5, n_iter=1)


def get_acceleration(x, y):
    return np.gradient(y, x)


def get_splits(accel):
    # find consecutive accel/decel
    return np.where(np.diff(np.signbit(accel)))


def split_array(array, indices):
    return np.split(array, indices[0] + 1)


def get_lowess_acceleration_chunks(
    x,
    y,
):
    lowess_y = get_lowess(x, y)
    lowess_y[lowess_y < 0] = 0
    accel = get_acceleration(x, lowess_y)
    splits = get_splits(accel)
    times = split_array(x, splits)
    speed = split_array(lowess_y, splits)
    accel = split_array(accel, splits)
    for return_times, return_accel, speed in zip(times, accel, speed):
        time_diff = return_times[-1] - return_times[0]
        yield return_times[0], return_times[
            -1
        ], return_accel.mean(), time_diff, speed.mean(), 1, 1


def get_pwlf_acceleration_chunks(
    x,
    y,
):
    pwlf_model = pwlf.PiecewiseLinFit(x, y)
    breaks = pwlf_model.fit(5)
    slopes = pwlf_model.calc_slopes()
    yHat = pwlf_model.predict(x)
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
        r2 = r2_score(yAct_break, yHat_break)
        mse = mean_squared_error(
            yAct_break,
            yHat_break,
        )
        yield b0, b1, slopes[i], b1 - b0, yHat_break.mean(), r2, mse


def process_trajectory_pwlf(
    group_name,
    group,
):
    x, y = get_xy(group)
    if len(x) < 3:
        return []
    try:
        return [
            {
                "vehicle_id": group_name,
                "start": start,
                "end": end,
                "accel": accel,
                "time_diff": time_diff,
                "speed": speed,
                "r2": r2,
                "mse": mse,
            }
            for start, end, accel, time_diff, speed, r2, mse in get_pwlf_acceleration_chunks(
                x, y
            )
        ]
    except np.linalg.LinAlgError:
        return []


from scipy import interpolate


def find_headway(fcd_df_copy):
    times = []
    measure_distances = [90, 60, 40]
    for lane in fcd_df_copy.lane.unique():
        _df = fcd_df_copy.loc[fcd_df_copy["lane"] == lane]
        for vehicle_id, group in _df.groupby("id"):
            if (group.pos.max() < 100) or (group.pos.min() > 40):
                continue

            f = interpolate.interp1d(group.pos, group.time, copy=False, kind="linear")
            times.extend(
                {"veh": vehicle_id, "lane": lane, "distance": x, "time": y}
                for x, y in zip(measure_distances, f(measure_distances))
            )

    headway_df = pd.DataFrame(times)
    headway_df = headway_df.sort_values("time")

    # find headway
    leader = None
    headway_times = []
    headway_df["headway"] = None
    for lane in headway_df["lane"].drop_duplicates():
        lane_df = headway_df.loc[headway_df.lane == lane].copy()
        for veh in lane_df["veh"].drop_duplicates():
            veh_times = lane_df.loc[lane_df.veh == veh].copy()
            if leader is None:
                leader = veh_times
                continue

            veh_times["headway"] = veh_times.time.values - leader.time.values
            veh_times["leader"] = leader.veh.values
            headway_times.append(veh_times.copy())
            leader = veh_times.copy()

    # calculate headway
    timed_headway_df = pd.concat(headway_times)
    timed_headway_df = timed_headway_df.loc[timed_headway_df.headway > 0]
    timed_headway_df.head()

    # calculate mean headway per vehicle
    dist = (
        timed_headway_df.groupby(["veh"])["headway"]
        .agg(
            [
                "mean",
            ]
        )
        .reset_index()
    )
    return dist.loc[dist["mean"] < 5].copy()


from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
from joblib import Parallel, delayed



for config in configs:

    fcd_df = load_fcd(config)
    fcd_df = label_polygons(fcd_df)
    # drop all rows without a box
    fcd_df = fcd_df[fcd_df["box"] != ""].copy()

    for box_id, box_df in fcd_df.groupby("box"):
        lowess_data = []
        for veh_id, veh_df in tqdm(box_df.groupby("id")):
            x, y = get_xy(veh_df)
            if len(x) < 3:
                continue
            lowess_data.extend(
                {
                    "vehicle_id": veh_id,
                    "start_time": start_time,
                    "end_time": end_time,
                    "accel": accel,
                    "time_diff": time_diff,
                    "speed": speed,
                    "r2": r2,
                    "mse": mse,
                }
                for start_time, end_time, accel, time_diff, speed, r2, mse in get_lowess_acceleration_chunks(
                    x, y
                )
            )

        lowess_df = pd.DataFrame(lowess_data)
        lowess_df.to_parquet(
            Path(config.Metadata.cwd) / f"{box_id}_lowess_summary_df.parquet"
        )

        # find headway
        headway_df = find_headway(box_df)
        headway_df.to_parquet(
            Path(config.Metadata.cwd) / f"{box_id}_headway_df.parquet"
        )

        # do pwlf
        with tqdm_joblib(
            desc="process", total=box_df[id_col].nunique()
        ) as progress_bar:
            pwlf_accels = Parallel(n_jobs=-1)(
                delayed(process_trajectory_pwlf)(
                    *group,
                )
                for group in box_df.groupby(id_col)
            )
            pwlf_accels = [item for sublist in pwlf_accels for item in sublist]

        pwlf_df = pd.DataFrame(pwlf_accels)
        pwlf_df.to_parquet(
            Path(config.Metadata.cwd) / f"{box_id}_pwlf_summary_df.parquet"
        )

