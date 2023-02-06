from datetime import datetime
from os import PathLike


import numpy as np
from innosent_api.utils.parse_kml import RadarTransform
from pandarallel import pandarallel
import pandas as pd
from shapely.geometry import Point
import utm


class AnalysisRadar(RadarTransform):
    def __init__(self, kml_file_path: PathLike, name: str, stay_gps: bool = False, parallel: bool = True, *args, **kwargs) -> None:
        print(name)
        super().__init__(kml_file_path, name, stay_gps, *args, **kwargs)
        self.df: pd.DataFrame = None
        if parallel:
            pandarallel.initialize(progress_bar=True)

    def read_csv(self, path_2_csv: str) -> None:
        self.df = pd.read_csv(path_2_csv, on_bad_lines='skip')

    def radar_xy_2_latlon(
        self, x_col: str = "f32_positionX_m", y_col: str = "f32_positionY_m"
    ) -> None:
        x, y = (self.df[[x_col, y_col]].values @ self.rotation_matrix).T
        self.df["lat"], self.df["lon"] = utm.to_latlon(
            x + self.easting, y + self.northing, self.zone_number, self.zone_letter
        )

    def apply_rw_time(
        self,
    ) -> None:

        self.df["dt"] = self.df["epoch_time"].apply(
            lambda x: datetime.fromtimestamp(x)
        )
        self.df.sort_values("dt", inplace=True)
    

    def label_boxes(self, parallel: bool = True) -> None:
        def _label_box_fn(row: np.array) -> str:
            for approach in self:
                p = Point(*row)
                if approach[1].contains(p):
                    return approach[0]
            return ""
        if parallel:
            self.df['box'] = self.df[["f32_positionX_m", "f32_positionY_m"]].parallel_apply(_label_box_fn, axis=1, raw=True)
        else:
            self.df['box'] = self.df[["f32_positionX_m", "f32_positionY_m"]].apply(_label_box_fn, axis=1, raw=True)
