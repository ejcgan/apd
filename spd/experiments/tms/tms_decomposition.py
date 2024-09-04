"""Run spd on a TMS model.

Note that the first instance index is fixed to the identity matrix. This is done so we can compare
the losses of the "correct" solution during training.
"""

from pathlib import Path

import fire
import matplotlib.pyplot as plt
import torch
import wandb
from tqdm import tqdm

from spd.experiments.tms.models import TMSSPDModel
from spd.experiments.tms.train_tms import TMSModel
from spd.experiments.tms.utils import TMSDataset, plot_A_matrix
from spd.log import logger
from spd.run_spd import Config, TMSConfig, optimize
from spd.utils import (
    DatasetGeneratedDataLoader,
    init_wandb,
    load_config,
    permute_to_identity,
    save_config_to_wandb,
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
            f"p{config.pnorm}_"
            f"topk{config.topk}_"
            f"topkrecon{config.topk_recon_coeff}_"
            f"lpsp{config.lp_sparsity_coeff}_"
            f"topkl2_{config.topk_l2_coeff}_"
            f"bs{config.batch_size}_"
            f"ft{task_config.n_features}_"
            f"hid{task_config.n_hidden}"
        )
    return config.wandb_run_name_prefix + run_suffix


def plot_perumated_A(model: TMSSPDModel, step: int, out_dir: Path, **_) -> dict[str, plt.Figure]:
    permuted_A_T_list: list[torch.Tensor] = []
    for i in range(model.n_instances):
        permuted_matrix = permute_to_identity(model.A[i].T.abs())
        permuted_A_T_list.append(permuted_matrix)
    permuted_A_T = torch.stack(permuted_A_T_list, dim=0)

    fig = plot_A_matrix(permuted_A_T, pos_only=True)
    fig.savefig(out_dir / f"A_{step}.png")
    plt.close(fig)
    tqdm.write(f"Saved A matrix to {out_dir / f'A_{step}.png'}")
    return {"A": fig}


def main(
    config_path_or_obj: Path | str | Config, sweep_config_path: Path | str | None = None
) -> None:
    config = load_config(config_path_or_obj, config_model=Config)
    task_config = config.task_config
    assert isinstance(task_config, TMSConfig)

    if config.wandb_project:
        config = init_wandb(config, config.wandb_project, sweep_config_path)
        save_config_to_wandb(config)
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

    dataset = TMSDataset(
        n_instances=task_config.n_instances,
        n_features=task_config.n_features,
        feature_probability=task_config.feature_probability,
        device=device,
    )
    dataloader = DatasetGeneratedDataLoader(dataset, batch_size=config.batch_size)

    optimize(
        model=model,
        config=config,
        out_dir=out_dir,
        device=device,
        dataloader=dataloader,
        pretrained_model=pretrained_model,
        plot_results_fn=plot_perumated_A,
    )

    if config.wandb_project:
        wandb.finish()


if __name__ == "__main__":
    fire.Fire(main)
