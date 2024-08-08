"""Linear decomposition script."""

import time
from collections.abc import Callable
from pathlib import Path
from tempfile import TemporaryDirectory

import fire
import torch
import wandb
import yaml
from jaxtyping import Float
from torch import Tensor
from torch.utils.data import DataLoader

from spd.log import logger
from spd.models.piecewise_models import (
    PiecewiseFunctionSPDTransformer,
    PiecewiseFunctionTransformer,
)
from spd.run_spd import Config, PiecewiseConfig, optimize
from spd.scripts.multilayer_functions.multilayer_functions_dataset import PiecewiseDataset
from spd.utils import (
    init_wandb,
    load_config,
    set_seed,
)

wandb.require("core")


def generate_trig_functions(
    num_trig_functions: int,
) -> list[Callable[[Float[Tensor, " n_inputs"]], Float[Tensor, " n_inputs"]]]:
    def create_trig_function(
        a: float, b: float, c: float, d: float, e: float, f: float, g: float
    ) -> Callable[[Float[Tensor, " n_inputs"]], Float[Tensor, " n_inputs"]]:
        return lambda x: a * torch.sin(b * x + c) + d * torch.cos(e * x + f) + g

    trig_functions = []
    for _ in range(num_trig_functions):
        a = torch.rand(1).item() * 2 - 1  # Uniform(-1, 1)
        b = torch.exp(torch.rand(1) * 4 - 1).item()  # exp(Uniform(-1, 3))
        c = torch.rand(1).item() * 2 * torch.pi - torch.pi  # Uniform(-π, π)
        d = torch.rand(1).item() * 2 - 1  # Uniform(-1, 1)
        e = torch.exp(torch.rand(1) * 4 - 1).item()  # exp(Uniform(-1, 3))
        f = torch.rand(1).item() * 2 * torch.pi - torch.pi  # Uniform(-π, π)
        g = torch.rand(1).item() * 2 - 1  # Uniform(-1, 1)
        trig_functions.append(create_trig_function(a, b, c, d, e, f, g))
    return trig_functions


def get_run_name(config: Config) -> str:
    """Generate a run name based on the config."""
    if config.wandb_run_name:
        run_suffix = config.wandb_run_name
    else:
        run_suffix = (
            f"sp{config.max_sparsity_coeff}_"
            f"lr{config.lr}_"
            f"p{config.pnorm}_"
            f"topk{config.topk}_"
            f"bs{config.batch_size}_"
        )
    return config.wandb_run_name_prefix + run_suffix


def main(
    config_path_or_obj: Path | str | Config, sweep_config_path: Path | str | None = None
) -> None:
    config = load_config(config_path_or_obj, config_model=Config)

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

    run_name = get_run_name(config)
    if config.wandb_project:
        assert wandb.run, "wandb.run must be initialized before training"
        wandb.run.name = run_name
    out_dir = Path(__file__).parent / "out" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    assert isinstance(config.task_config, PiecewiseConfig)
    assert config.task_config.k is not None

    functions = generate_trig_functions(config.task_config.n_functions)

    piecewise_model = PiecewiseFunctionTransformer.from_handcoded(
        functions=functions,
        neurons_per_function=config.task_config.neurons_per_function,
        n_layers=config.task_config.n_layers,
    ).to(device)

    piecewise_model_spd = PiecewiseFunctionSPDTransformer(
        n_inputs=piecewise_model.n_inputs,
        d_mlp=piecewise_model.d_mlp,
        n_layers=piecewise_model.n_layers,
        k=config.task_config.k,
    ).to(device)

    dataset = PiecewiseDataset(
        n_inputs=piecewise_model.n_inputs,
        functions=functions,
        prob_one=0.2,
        range_min=config.task_config.range_min,
        range_max=config.task_config.range_max,
    )
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)

    optimize(
        model=piecewise_model_spd,
        config=config,
        out_dir=out_dir,
        device=device,
        pretrained_model=piecewise_model,
        dataloader=dataloader,
    )

    if config.wandb_project:
        wandb.finish()


if __name__ == "__main__":
    fire.Fire(main)
