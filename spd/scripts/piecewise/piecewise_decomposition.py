"""Linear decomposition script."""

import json
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import fire
import torch
import wandb
import yaml
from torch.utils.data import DataLoader

from spd.log import logger
from spd.run_spd import Config, PiecewiseConfig, calc_recon_mse, optimize
from spd.scripts.piecewise.models import (
    PiecewiseFunctionSPDTransformer,
    PiecewiseFunctionTransformer,
)
from spd.scripts.piecewise.piecewise_dataset import PiecewiseDataset
from spd.scripts.piecewise.trig_functions import generate_trig_functions
from spd.utils import (
    init_wandb,
    load_config,
    set_seed,
)

wandb.require("core")


def get_run_name(config: Config) -> str:
    """Generate a run name based on the config."""
    if config.wandb_run_name:
        run_suffix = config.wandb_run_name
    else:
        assert isinstance(config.task_config, PiecewiseConfig)
        run_suffix = (
            f"sp{config.max_sparsity_coeff}_"
            f"l2{config.topk_l2_coeff}_"
            f"lay{config.task_config.n_layers}_"
            f"lr{config.lr}_"
            f"p{config.pnorm}_"
            f"topk{config.topk}_"
            f"topkl2{config.topk_l2_coeff}_"
            f"bs{config.batch_size}"
        )
        if config.task_config.handcoded_AB:
            run_suffix += "_hAB"
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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    assert isinstance(config.task_config, PiecewiseConfig)
    assert config.task_config.k is not None

    functions, function_params = generate_trig_functions(config.task_config.n_functions)

    out_dir = Path(__file__).parent / "out" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "function_params.json", "w") as f:
        json.dump(function_params, f, indent=4)
    logger.info(f"Saved function params to {out_dir / 'function_params.json'}")

    piecewise_model = PiecewiseFunctionTransformer.from_handcoded(
        functions=functions,
        neurons_per_function=config.task_config.neurons_per_function,
        n_layers=config.task_config.n_layers,
        range_min=config.task_config.range_min,
        range_max=config.task_config.range_max,
        seed=config.seed,
    ).to(device)
    piecewise_model.eval()

    input_biases = [
        piecewise_model.mlps[i].input_layer.bias.detach().clone()
        for i in range(piecewise_model.n_layers)
    ]
    piecewise_model_spd = PiecewiseFunctionSPDTransformer(
        n_inputs=piecewise_model.n_inputs,
        d_mlp=piecewise_model.d_mlp,
        n_layers=piecewise_model.n_layers,
        k=config.task_config.k,
        input_biases=input_biases,
    )
    if config.task_config.handcoded_AB:
        logger.info("Setting handcoded A and B matrices (!)")
        piecewise_model_spd.set_handcoded_AB(piecewise_model)
    piecewise_model_spd.to(device)

    # Set requires_grad to False for all embeddings and all input biases
    for i in range(piecewise_model_spd.n_layers):
        piecewise_model_spd.mlps[i].bias1.requires_grad_(False)
    piecewise_model_spd.W_E.requires_grad_(False)
    piecewise_model_spd.W_U.requires_grad_(False)

    dataset = PiecewiseDataset(
        n_inputs=piecewise_model.n_inputs,
        functions=functions,
        feature_probability=config.task_config.feature_probability,
        range_min=config.task_config.range_min,
        range_max=config.task_config.range_max,
    )
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)

    # Evaluate the hardcoded model on 5 batches to get the labels
    n_batches = 5
    loss = 0
    for i, (batch, labels) in enumerate(dataloader):
        if i >= n_batches:
            break
        hardcoded_out = piecewise_model(batch.to(device))
        loss += calc_recon_mse(hardcoded_out, labels.to(device))
    loss /= n_batches
    logger.info(f"Loss of hardcoded model on 5 batches: {loss}")

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
