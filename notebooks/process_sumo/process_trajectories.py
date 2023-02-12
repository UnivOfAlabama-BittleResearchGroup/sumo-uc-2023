import os
import sys
from typing import Tuple
from datetime import datetime
from pathlib import Path

import pandas as pd
from shapely.geometry import Polygon, Point
from sumolib.shapes import polygon



def recursive_root(path: str, find="sumo-uc-2023"):
    if os.path.split(path)[-1] == find:
        return Path(path)
    return recursive_root(os.path.split(path)[0], find=find)

ROOT = recursive_root(os.path.abspath("."))
sys.path.append(str(ROOT))

from src.walk_configs import walk_configs
from src.pwlfer import process_trajectories, classify
from tqdm import tqdm


experiment_path = Path(
    # load the path from the command line
    sys.argv[1]
)
print(f"Processing experiment: {experiment_path}")
polys = polygon.read("/home/max/Development/sumo-uc-2023/sumo-xml/detectors/radar_boxes.xml")
configs = list(walk_configs(experiment_path))


def load_fcd(config):
    fcd_df = pd.read_parquet(config.Blocks.XMLConvertConfig.target)
    fcd_df[["x", "y", "time", "speed"]] = fcd_df[["x", "y", "time", "speed"]].astype(
        float
    )
    return fcd_df


polygon_dict = {
    poly.id: Polygon(poly.shape)
    for poly in polys
    if poly.id in ["Radar137_East_thru", "Radar136_West_thru"]
}

def label_polygons(fcd_df):
    fcd_df['box'] = ''
    for poly_id, poly in polygon_dict.items():
        fcd_df.loc[fcd_df['box'] == '', 'box'] = fcd_df.loc[fcd_df['box'] == '', ['x', 'y']].apply(lambda x: poly.contains(Point(*x)), axis=1, raw=True).map({True: poly_id, False: ''})
    return fcd_df

for config in configs:
    if all((Path(config.Metadata.cwd) / f"{poly_id}_summary_df.parquet").exists() for poly_id in polygon_dict.keys()):
        # print(f"skipping: {config.Metadata.cwd}")
        continue
    
    fcd_df = load_fcd(config)
    fcd_df = label_polygons(fcd_df)
    # drop all rows without a box
    fcd_df = fcd_df[fcd_df['box'] != ''].copy()
    
    for box_group in fcd_df.groupby('box'):
        box = box_group[0]
        box_df = box_group[1]
        res = process_trajectories(box_df, time_col="time", speed_col="speed", n_jobs=-1)
        summary_df = pd.DataFrame([{**v, **l} for v in res for l in v.pop("line_fits")])
        summary_df["type"] = summary_df.apply(classify, axis=1)
        summary_df["minimum_time"] = summary_df["end"] - summary_df["start"]
        # save summary_df
        summary_df.to_parquet(Path(config.Metadata.cwd) / f"{box}_summary_df.parquet")


