# %%
import json
from pathlib import Path

import torch

from spd.experiments.piecewise.models import (
    PiecewiseFunctionSPDFullRankTransformer,
    PiecewiseFunctionSPDTransformer,
    PiecewiseFunctionTransformer,
)
from spd.experiments.piecewise.piecewise_decomposition import get_model_and_dataloader
from spd.experiments.piecewise.plotting import (
    plot_components,
    plot_components_fullrank,
    plot_model_functions,
    plot_piecewise_network,
    plot_subnetwork_correlations,
)

# plot_subnetwork_correlations,
from spd.experiments.piecewise.trig_functions import create_trig_function
from spd.run_spd import (
    Config,
    PiecewiseConfig,
)
from spd.utils import REPO_ROOT

pretrained_path = REPO_ROOT / "spd/experiments/piecewise/demo_spd_model/model_50000.pth"
with open(pretrained_path.parent / "config.json") as f:
    config_dict = json.load(f)
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
assert isinstance(
    spd_model, PiecewiseFunctionSPDTransformer | PiecewiseFunctionSPDFullRankTransformer
)
spd_model.load_state_dict(torch.load(pretrained_path, weights_only=True, map_location="cpu"))

# To test handcoded AB, uncomment the following line
# spd_model.set_handcoded_AB(hardcoded_model)


if config.full_rank:
    assert isinstance(spd_model, PiecewiseFunctionSPDFullRankTransformer)
    fig_dict = plot_components_fullrank(model=spd_model, step=-1, out_dir=None, slow_images=True)
else:
    assert isinstance(spd_model, PiecewiseFunctionSPDTransformer)
    fig_dict = plot_components(
        model=spd_model, step=-1, out_dir=None, device=device, slow_images=True
    )

if config.topk is not None:
    fig_dict.update(**plot_subnetwork_correlations(dataloader, spd_model, config, device))
    fig_dict.update(**plot_piecewise_network(spd_model))
    fig_dict.update(
        **plot_model_functions(
            spd_model=spd_model,
            target_model=hardcoded_model,
            full_rank=config.full_rank,
            ablation_attributions=config.ablation_attributions,
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
