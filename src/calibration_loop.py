import math
import random
import subprocess
import sys
import os
from pathlib import Path
import json5 as json
from typing import List, Tuple, Union
from dataclasses import MISSING, dataclass, field
from datetime import datetime, timedelta
import itertools

import sumolib
from sumolib.net import Net
import pandas as pd
import click
from omegaconf import OmegaConf
from sumo_pipelines.config import (
    MetaData,
    SimulationConfig,
    open_config,
    Pipeline,
    create_consumers,
)


@dataclass
class IteratorConfig:
    edge_data: str
    turn_data: str
    edge_file: str
    turn_file: str
    agg_intervals: List[float]
    iterations: int
    seed: int


@dataclass
class RouteSamplerConfig:
    random_route_file: str
    edge_file: str
    turn_file: str
    output_file: str
    seed: int
    prefix: str = MISSING
    mode: str = "poisson"


@dataclass
class RandomRouteConfig:
    output_file: str
    seed: int
    net_file: str


@dataclass
class Config:
    Metadata: MetaData
    SimulationConfig: SimulationConfig
    IteratorConfig: IteratorConfig
    RouteSamplerConfig: RouteSamplerConfig
    RandomRouteConfig: RandomRouteConfig
    Pipeline: Pipeline


def write_turn_xml(
    config: IteratorConfig,
    df: pd.DataFrame,
    write_zeros: bool,
    mode: str = 'edge'
):
    df = df.reset_index(drop=True)
    df.loc[df["start"] < 0, "start"] = 0
    df = df.loc[df["start"] < df["end"]]
    output_path = config.get(f"{mode}_file")

    with open(output_path, "w") as f_:
        f_.write("<additional >\n")
        for i, row in df.iterrows():
            if row["start"] == row["end"]:
                continue
            f_.write(
                f"""\t<interval id="interval_{str(row['start'])}" begin="{str(row['start'])}" end="{str(row['end'])}" >\n"""
            )

            for col in row.index.difference(["start", "end"]):
                if not write_zeros and row[col] <= 0:
                    continue

                if isinstance(col, tuple):
                    f_.write(
                        f"""\t\t<edgeRelation from="{col[0]}" to="{col[1]}" count="{str(row[col])}" />\n"""
                    )
                else:
                    f_.write(f"""\t\t<edge id="{col}" entered="{str(row[col])}" />\n""")
            f_.write("\t</interval>\n")
        f_.write("</additional >\n")


def call_route_sampler(config: RouteSamplerConfig):
    
    rs_output = Path(config.output_file).parent / "route_sampler_output.txt"

    with open(rs_output, 'w') as f:
        r = subprocess.run(
            [
                "python",
                str(Path(os.environ["SUMO_HOME"]) / "tools" / "routeSampler.py"),
                "-r",
                str(config.random_route_file),
                "-o",
                str(config.output_file),
                "--optimize",
                "full",
                "-a",
                'departSpeed="max" type="vehDist" departLane="best"',
                "--threads",
                "4",
                "--weighted",
                # "--minimize-vehicles",
                # "1",
                "--write-flows",
                config.mode,
                "--prefix",
                config.prefix,
                "--seed",
                str(config.seed),
                "--edgedata-files",
                str(config.edge_file),
                "--turn-files",
                str(config.turn_file),
            ],
            stdout=f,
            stderr=f,
        )


# def parse_detector_output(
#     file: Path,
#     output_path: Path,
# ):

#     csv_df = pd.read_xml(
#         file,
#         # xpath="./detector//*",
#         namespaces={"detector": "http://sumo.dlr.de/xsd/det_e2_file.xsd"},
#         parser="etree",
#     )

#     csv_df.columns = [f"interval_{c}" for c in csv_df.columns]
#     csv_df.to_csv(output_path)
#     return csv_df

# def calculate_difference(
#     sumo_df: pd.DataFrame,
#     resample_interval: int,
#     shift: int,
#     multiplier: float,
#     obj_handler: ObjectHandler,
# ) -> ObjectHandler:
#     diff = (
#         obj_handler.sql_df_raw.resample(
#             f"{resample_interval}S", offset=f"{shift}S"
#         ).sum()
#         * multiplier
#     ) - sumo_df.resample(f"{resample_interval}S", offset=f"{shift}S").sum()
#     diff.fillna(0, inplace=True)
#     return diff


def call_random_routes(config: RandomRouteConfig):
    r = subprocess.run(
        [
            "python",
            str(Path(os.environ["SUMO_HOME"]) / "tools" / "randomTrips.py"),
            "--fringe-factor",
            "max",
            "-r",
            str(config.output_file),
            "-n",
            config.net_file,
            "--min-distance",
            "30",
            "--lanes",
            "-e",
            "3600",
            "--period",
            "1",
            "--seed",
            str(config.seed),
        ]
    )

    print(r)


# def broad_file_intervals(
#     agg_interval: int,
#     shift: int,
#     obj_handler: ObjectHandler,
# ) -> ObjectHandler:

#     broaden_agg_interval(
#         obj_handler.route_files[-1],
#         obj_handler.route_files[-1],
#         agg_interval,
#         shift,
#     )

#     return obj_handler


# def number_to_probability(
#     obj_handler: ObjectHandler,
# ) -> ObjectHandler:

#     flow_number_2_prob(
#         obj_handler.route_files[-1],
#         obj_handler.route_files[-1],
#         poisson=True
#     )

#     return obj_handler


def prepare_df(interval, offset, df):
    min_time = df.index.min()
    df = df.resample(f"{interval}S", offset=f"{offset}S").sum().copy()
    df["start"] = (df.index - min_time).total_seconds()
    df["end"] = df["start"] + interval
    df.fillna(0, inplace=True)
    return df


@click.command()
@click.argument("settings_file", type=click.Path(exists=True))
def main(settings_file):
    c: Config = open_config(settings_file, structured=OmegaConf.structured(Config))

    consumers = create_consumers(
        [
            (p.function, f"Pipeline.pipeline[0].producers[{i}].config")
            for i, p in enumerate(c.Pipeline.pipeline[0].producers)
        ],
        parallel=False,
    )

    random.seed(c.Metadata.random_seed)
    total_iterations = len(c.IteratorConfig.agg_intervals) * c.IteratorConfig.iterations

    edge_df = pd.read_parquet(c.IteratorConfig.edge_data)
    turn_df = pd.read_parquet(c.IteratorConfig.turn_data)
    # split the columns into tuples
    turn_df.columns = [tuple(c.split("~~")) for c in turn_df.columns]

    # set the step length to 1
    c.SimulationConfig.step_length = 1

    for i, (interval, j) in enumerate(
        itertools.product(
            c.IteratorConfig.agg_intervals, range(c.IteratorConfig.iterations)
        )
    ):
        c.Metadata.run_id = f"{i}"
        # create the output directory
        Path(c.Metadata.cwd).mkdir(parents=True, exist_ok=True)

        c.IteratorConfig.seed = random.randint(0, 1000000)
        offset = (c.IteratorConfig.agg_intervals[0] // total_iterations) * i
        multiplier = (1 / total_iterations) * (i + 1)

        for mode, _df in [("edge", edge_df), ("turn", turn_df)]:
            
            _df = prepare_df(interval, offset, _df)

            write_turn_xml(
                c.IteratorConfig,
                _df,
                write_zeros=True,
                mode=mode
            )


        call_random_routes(c.RandomRouteConfig)
        call_route_sampler(c.RouteSamplerConfig)
        
        if i == 0:
            c.SimulationConfig.route_files = []
        c.SimulationConfig.route_files.append(c.RouteSamplerConfig.output_file)

        # run the simulation
        consumers(c)


    # o = ObjectHandler(
    #     settings=Settings(settings_file),
    #     seed=seed,
    # )

    # # seed the random number generator
    # random.seed(o.seed)

    # # make the output directory
    # Path(o.settings.OUTPUT_ROOT).mkdir(parents=True, exist_ok=True)

    # print(o.settings.OUTPUT_ROOT)

    # o.load_detectors()
    # o.create_detector_location()

    # # load the volume object
    # o = load_volume(o.settings.ROUTE_GENERATION_PARAMS.SQL_DATA, o)

    # # load the turn ratio info
    # o = load_turn_ratios(o.settings.TURN_RATIOS, o)

    # difference_df = None
    # iterations = iterations
    # intervals = intervals

    # total_iterations = len(intervals) * iterations

    # for i, (interval, j) in enumerate(itertools.product(intervals, range(iterations))):
    #     # set the seed
    #     o.seed = random.randint(0, 1000000)

    #     offset = (intervals[0] // total_iterations) * i
    #     multiplier = (1 / total_iterations) * (i + 1)

    #     print(f"iteration {i} offset {offset} multiplier {multiplier}")

    #     if i > 0:
    #         # load the correction object
    #         o = load_corrections(difference_file=difference_df, obj_handler=o)
    #         # switch the target to the error df
    #         o.target_df = "corr_df"

    #     o = agg_n_shift(
    #         agg_period=interval,
    #         shift=offset,
    #         # multiplier=(1 / iterations) * (i + 1),  # only take 1/iterations every time
    #         multiplier=multiplier,
    #         obj_handler=o,
    #     )
    #     o = create_detectors_relations(create_turn_relations(o))
    #     o = create_edge_df(o)
    #     o = cleanup_dfs(o)
    #     o = write_turn_xml(
    #         output_path=Path(o.settings.OUTPUT_ROOT) / f"edge.{i}.xml",
    #         df="edge_df",
    #         write_zeros=True,
    #         obj_handler=o,
    #     )
    #     o = write_turn_xml(
    #         output_path=Path(o.settings.OUTPUT_ROOT) / f"turn.{i}.xml",
    #         df="relationship_df",
    #         write_zeros=True,
    #         obj_handler=o,
    #     )

    #     # call the random route file
    #     o = call_random_routes(
    #         output_file=Path(o.settings.OUTPUT_ROOT) / f"random.routes.{i}.xml",
    #         obj_handler=o,
    #     )

    #     o = call_route_sampler(
    #         edge_file=Path(o.settings.OUTPUT_ROOT) / f"edge.{i}.xml",
    #         turn_file=Path(o.settings.OUTPUT_ROOT) / f"turn.{i}.xml",
    #         random_route_file=Path(o.settings.OUTPUT_ROOT) / f"random.routes.{i}.xml",
    #         output_file=Path(o.settings.OUTPUT_ROOT) / f"route.{i}.xml",
    #         route_prefix=f"{i}_",
    #         obj_handler=o,
    #     )

    #     if route_mode != "number":

    #         o = broad_file_intervals(
    #             agg_interval=interval,
    #             shift=offset,
    #             obj_handler=o,
    #         )

    #         print('possion replace')
    #         o = number_to_probability(o)

    #     o = run_sumo(
    #         obj_handler=o,
    #         sim_step=1 if i < (total_iterations - 1) else 0.1,
    #         gui=(i >= (total_iterations - 1)) and (o.settings.GUI),
    #         # gui=i > 0,
    #         fcd_output=Path(o.settings.OUTPUT_ROOT) / "fcd.final.out.xml"
    #         if i >= (total_iterations - 1)
    #         else None,
    #     )

    #     # load detector file
    #     detector_counts = parse_detector_output(
    #         o.settings.DETECTOR_OUTPUT,
    #         output_path=Path(o.settings.OUTPUT_ROOT) / "detector.final.out.csv"
    #         if i >= (total_iterations - 1)
    #         else (Path(o.settings.OUTPUT_ROOT) / f"detector.{i}.out.csv"),
    #     )

    #     if i < (total_iterations - 1):
    #         detector_counts = prepare_detector_output(detector_counts, o)
    #         difference_df = calculate_difference(
    #             sumo_df=detector_counts,
    #             resample_interval=10,
    #             shift=offset,
    #             multiplier=1,
    #             obj_handler=o
    #         )

    # o.settings.save(Path(o.settings.OUTPUT_ROOT) / "settings.json")


if __name__ == "__main__":
    main()
    # o = ObjectHandler(
    #     settings=Settings(Path(root.ROOT) / "sim-settings" / "baseline.json")
    # )

    # # seed the random number generator
    # random.seed(o.seed)

    # # make the output directory
    # Path(o.settings.OUTPUT_ROOT).mkdir(parents=True, exist_ok=True)

    # print(o.settings.OUTPUT_ROOT)

    # o.load_detectors()
    # o.create_detector_location()

    # # load the volume object
    # o = load_volume(o.settings.ROUTE_GENERATION_PARAMS.SQL_DATA, o)

    # # load the turn ratio info
    # o = load_turn_ratios(o.settings.TURN_RATIOS, o)

    # difference_df = None
    # iterations = 2
    # intervals = [200, 600]

    # total_iterations = len(intervals) * iterations

    # for i, (interval, j) in enumerate(itertools.product(intervals, range(iterations))):
    #     # set the seed
    #     o.seed = random.randint(0, 1000000)

    #     offset = (intervals[0] // total_iterations) * i
    #     multiplier = (1 / total_iterations) * (i + 1)

    #     print(f"iteration {i} offset {offset} multiplier {multiplier}")

    #     if i > 0:
    #         # load the correction object
    #         o = load_corrections(difference_file=difference_df, obj_handler=o)
    #         # switch the target to the error df
    #         o.target_df = "corr_df"

    #     o = agg_n_shift(
    #         agg_period=interval,
    #         shift=offset,
    #         # multiplier=(1 / iterations) * (i + 1),  # only take 1/iterations every time
    #         multiplier=multiplier,
    #         obj_handler=o,
    #     )
    #     o = create_detectors_relations(create_turn_relations(o))
    #     o = create_edge_df(o)
    #     o = cleanup_dfs(o)
    #     o = write_turn_xml(
    #         output_path=Path(o.settings.OUTPUT_ROOT) / f"edge.{i}.xml",
    #         df="edge_df",
    #         write_zeros=True,
    #         obj_handler=o,
    #     )
    #     o = write_turn_xml(
    #         output_path=Path(o.settings.OUTPUT_ROOT) / f"turn.{i}.xml",
    #         df="relationship_df",
    #         write_zeros=True,
    #         obj_handler=o,
    #     )

    #     # call the random route file
    #     o = call_random_routes(
    #         output_file=Path(o.settings.OUTPUT_ROOT) / f"random.routes.{i}.xml",
    #         obj_handler=o,
    #     )

    #     o = call_route_sampler(
    #         edge_file=Path(o.settings.OUTPUT_ROOT) / f"edge.{i}.xml",
    #         turn_file=Path(o.settings.OUTPUT_ROOT) / f"turn.{i}.xml",
    #         random_route_file=Path(o.settings.OUTPUT_ROOT) / f"random.routes.{i}.xml",
    #         output_file=Path(o.settings.OUTPUT_ROOT) / f"route.{i}.xml",
    #         route_prefix=f"{i}_",
    #         obj_handler=o,
    #     )

    #     # o = broad_file_intervals(
    #     #     agg_interval=interval,
    #     #     shift=offset,
    #     #     obj_handler=o,
    #     # )

    #     o = run_sumo(
    #         obj_handler=o,
    #         sim_step=1 if i < (total_iterations - 1) else 0.1,
    #         gui=(i >= (total_iterations - 1)) and (o.settings.GUI),
    #         # gui=i > 0,
    #         fcd_output=Path(o.settings.OUTPUT_ROOT) / "fcd.final.out.xml"
    #         if i >= (total_iterations - 1)
    #         else None,
    #     )

    #     # load detector file
    #     detector_counts = parse_detector_output(
    #         o.settings.DETECTOR_OUTPUT,
    #         output_path=Path(o.settings.OUTPUT_ROOT) / "detector.final.out.csv"
    #         if i >= (total_iterations - 1)
    #         else (Path(o.settings.OUTPUT_ROOT) / f"detector.{i}.out.csv"),
    #     )

    #     if i < (total_iterations - 1):

    #         detector_counts = prepare_detector_output(detector_counts, o)

    #         # offset = (interval // iterations) * (i + 2)

    #         # find the differences
    #         difference_df = calculate_difference(
    #             sumo_df=detector_counts,
    #             resample_interval=10,
    #             shift=offset,
    #             multiplier=1,
    #             obj_handler=o
    #         )

    # o.settings.save(Path(o.settings.OUTPUT_ROOT) / "settings.json")
