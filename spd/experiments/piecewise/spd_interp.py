# %%
import json
from pathlib import Path

import torch
import yaml

from spd.experiments.piecewise.models import (
    PiecewiseFunctionTransformer,
)
from spd.experiments.piecewise.piecewise_decomposition import get_model_and_dataloader
from spd.experiments.piecewise.plotting import (
    plot_components_rank_penalty,
    plot_model_functions,
    plot_piecewise_network,
)
from spd.experiments.piecewise.trig_functions import create_trig_function
from spd.plotting import plot_subnetwork_correlations
from spd.run_spd import (
    Config,
    PiecewiseConfig,
)
from spd.utils import handle_deprecated_config_keys

# pretrained_path = REPO_ROOT / "spd/experiments/piecewise/demo_spd_model/model_50000.pth"
pretrained_path = Path(
    "/data/apollo/spd/4f_2l_rp_p1.00e+00_topk2.22e-01_topkrecon5.00e+00_schatten5.00e+00_sd0_attr-gra_lr3.00e-03_bs10000_lay2/spd_model_200000.pth"
)
with open(pretrained_path.parent / "final_config.yaml") as f:
    config_dict = yaml.safe_load(f)
    config_dict = handle_deprecated_config_keys(config_dict)
    config = Config(**config_dict)

with open(pretrained_path.parent / "function_params.json") as f:
    function_params = json.load(f)
functions = [create_trig_function(*param) for param in function_params]

device = "cuda" if torch.cuda.is_available() else "cpu"

assert isinstance(config.task_config, PiecewiseConfig)

hardcoded_model, spd_model, dataloader, test_dataloader = get_model_and_dataloader(
    config, device, out_dir=None
)
assert isinstance(hardcoded_model, PiecewiseFunctionTransformer)
spd_model.load_state_dict(torch.load(pretrained_path, weights_only=True, map_location="cpu"))

fig_dict = plot_components_rank_penalty(model=spd_model, step=-1, out_dir=None, slow_images=True)


if config.topk is not None:
    fig_dict.update(
        **plot_subnetwork_correlations(
            dataloader,
            target_model=hardcoded_model,
            spd_model=spd_model,
            config=config,
            device=device,
        )
    )
    fig_dict.update(**plot_piecewise_network(spd_model))
    fig_dict.update(
        **plot_model_functions(
            spd_model=spd_model,
            target_model=hardcoded_model,
            attribution_type=config.attribution_type,
            device=device,
            start=config.task_config.range_min,
            stop=config.task_config.range_max,
            print_info=True,
        )
    )
out_path = Path(__file__).parent / "out/attribution_scores" / pretrained_path.parent.name
out_path.mkdir(parents=True, exist_ok=True)
for k, v in fig_dict.items():
    out_file = out_path / f"{k}.png"
    v.savefig(out_file)
    print(f"Saved plot to {out_file}")
