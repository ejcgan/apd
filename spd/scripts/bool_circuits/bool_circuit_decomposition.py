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
from spd.models.bool_circuit_models import BoolCircuitSPDTransformer, BoolCircuitTransformer
from spd.run_spd import BoolCircuitModelConfig, Config, optimize
from spd.scripts.bool_circuits.bool_circuit_dataset import BooleanCircuitDataset
from spd.scripts.bool_circuits.bool_circuit_utils import form_circuit
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
    print(f"Using device: {device}")
    model_config = config.torch_model_config
    assert isinstance(model_config, BoolCircuitModelConfig)

    dl_model = BoolCircuitTransformer.from_pretrained(model_config.pretrained_model_path).to(device)

    dlc_model = BoolCircuitSPDTransformer(
        n_inputs=dl_model.n_inputs,
        d_embed=dl_model.d_embed,
        d_mlp=dl_model.d_mlp,
        n_layers=dl_model.n_layers,
        k=model_config.k,
        n_outputs=dl_model.n_outputs,
    ).to(device)

    with open(model_config.pretrained_model_path.parent / "circuit_repr.json") as f:
        circuit_repr = json.load(f)

    dataset = BooleanCircuitDataset(circuit=form_circuit(circuit_repr), n_inputs=dl_model.n_inputs)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    optimize(
        model=dlc_model,
        config=config,
        out_dir=out_dir,
        device=device,
        pretrained_model=dl_model,
        dataloader=dataloader,
    )

    if config.wandb_project:
        wandb.finish()


if __name__ == "__main__":
    fire.Fire(main)
