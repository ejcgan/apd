"""Run spd on a TMS model.

Note that the first instance index is fixed to the identity matrix. This is done so we can compare
the losses of the "correct" solution during training.
"""

import time
from pathlib import Path
from tempfile import TemporaryDirectory

import fire
import torch
import wandb
import yaml

from spd.log import logger
from spd.models.tms_models import TMSSPDModel
from spd.run_spd import Config, TMSConfig, optimize
from spd.scripts.tms.train_tms import TMSModel
from spd.utils import (
    init_wandb,
    load_config,
    set_seed,
)

wandb.require("core")


def get_run_name(config: Config, task_config: TMSConfig) -> str:
    """Generate a run name based on the config."""
    if config.wandb_run_name:
        run_suffix = config.wandb_run_name
    else:
        run_suffix = (
            f"lr{config.lr}_"
            f"topk{config.topk}_"
            f"sp{config.max_sparsity_coeff}_"
            f"bs{config.batch_size}_"
            f"ft{task_config.n_features}_"
            f"hid{task_config.n_hidden}"
        )
    return config.wandb_run_name_prefix + run_suffix


def main(
    config_path_or_obj: Path | str | Config, sweep_config_path: Path | str | None = None
) -> None:
    config = load_config(config_path_or_obj, config_model=Config)
    task_config = config.task_config
    assert isinstance(task_config, TMSConfig)

    if config.wandb_project:
        config = init_wandb(config, config.wandb_project, sweep_config_path)
        # Save the config to wandb
        with TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "final_config.yaml"
            with open(config_path, "w") as f:
                yaml.dump(config.model_dump(mode="json"), f, indent=2)
            wandb.save(str(config_path), policy="now", base_path=tmp_dir)
            # Unfortunately wandb.save is async, so we need to wait for it to finish before
            # continuing, and wandb python api provides no way to do this.
            # TODO: Find a better way to do this.
            time.sleep(1)

    set_seed(config.seed)
    logger.info(config)

    run_name = get_run_name(config, task_config)
    if config.wandb_project:
        assert wandb.run, "wandb.run must be initialized before training"
        wandb.run.name = run_name
    out_dir = Path(__file__).parent / "out" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TMSSPDModel(
        n_instances=task_config.n_instances,
        n_features=task_config.n_features,
        n_hidden=task_config.n_hidden,
        k=task_config.k,
        feature_probability=task_config.feature_probability,
        train_bias=task_config.train_bias,
        bias_val=task_config.bias_val,
        device=device,
    )

    pretrained_model = None
    if task_config.pretrained_model_path:
        pretrained_model = TMSModel(
            n_instances=task_config.n_instances,
            n_features=task_config.n_features,
            n_hidden=task_config.n_hidden,
            device=device,
        )
        pretrained_model.load_state_dict(
            torch.load(task_config.pretrained_model_path, map_location=device)
        )
        pretrained_model.eval()

    # TODO: Make dataloader independent of the model
    class TMSDataLoader:
        def __init__(self, model: TMSSPDModel, batch_size: int):
            self.model = model
            self.batch_size = batch_size

        def __iter__(self):
            while True:
                batch = self.model.generate_batch(self.batch_size)
                yield batch, batch

    dataloader = TMSDataLoader(model, config.batch_size)
    optimize(
        model=model,
        config=config,
        out_dir=out_dir,
        device=device,
        dataloader=dataloader,  # type: ignore
        pretrained_model=pretrained_model,
    )

    if config.wandb_project:
        wandb.finish()


if __name__ == "__main__":
    fire.Fire(main)
