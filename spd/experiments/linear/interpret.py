import json
from pathlib import Path

import torch

from spd.experiments.linear.models import (
    DeepLinearComponentFullRankModel,
    DeepLinearComponentModel,
    DeepLinearModel,
)
from spd.experiments.linear.plotting import make_linear_plots
from spd.run_spd import Config, DeepLinearConfig
from spd.utils import REPO_ROOT


def main():
    pretrained_path = REPO_ROOT / "spd/experiments/linear/demo_spd_model/model_10000.pth"
    with open(pretrained_path.parent / "config.json") as f:
        config_dict = json.load(f)
        config = Config(**config_dict)

    assert isinstance(config.task_config, DeepLinearConfig)
    if config.task_config.pretrained_model_path is not None:
        dl_model = DeepLinearModel.from_pretrained(config.task_config.pretrained_model_path)
        assert (
            config.task_config.n_features is None
            and config.task_config.n_layers is None
            and config.task_config.n_instances is None
        ), "n_features, n_layers, and n_instances must not be set if pretrained_model_path is set"
        n_features = dl_model.n_features
        n_layers = dl_model.n_layers
        n_instances = dl_model.n_instances
    else:
        assert config.out_recon_coeff is not None, "Only out recon loss allows no pretrained model"
        dl_model = None
        n_features = config.task_config.n_features
        n_layers = config.task_config.n_layers
        n_instances = config.task_config.n_instances
        assert (
            n_features is not None and n_layers is not None and n_instances is not None
        ), "n_features, n_layers, and n_instances must be set"

    if config.spd_type == "full_rank":
        dlc_model = DeepLinearComponentFullRankModel(
            n_features=n_features,
            n_layers=n_layers,
            n_instances=n_instances,
            k=config.task_config.k,
        )
    else:
        dlc_model = DeepLinearComponentModel(
            n_features=n_features,
            n_layers=n_layers,
            n_instances=n_instances,
            k=config.task_config.k,
        )

    dlc_model.load_state_dict(torch.load(pretrained_path, weights_only=True, map_location="cpu"))

    out_dir = Path(__file__).parent / "out/figures" / pretrained_path.parent.name
    out_dir.mkdir(parents=True, exist_ok=True)

    make_linear_plots(model=dlc_model, step=None, out_dir=out_dir, device="cpu", n_instances=1)


if __name__ == "__main__":
    main()
