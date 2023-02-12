from pathlib import Path
from sumo_pipelines.config import open_completed_config


def walk_configs(configs_dir, config_name: str = 'config.yaml', replace_cwd: bool = True):
    """Walks through all completed configs in a directory and yields them.

    Args:
        configs_dir: The directory to walk.
        config_name: The name of the config to open.

    Yields:
        The next config.
    """
    for config_path in configs_dir.glob(f'*/{config_name}'):
        c = open_completed_config(config_path, validate=False)
        if replace_cwd:
            c.Metadata.cwd = str(config_path.parent)
            c.Metadata.output = str(configs_dir)
        yield c
