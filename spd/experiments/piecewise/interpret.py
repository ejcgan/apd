# %%
import json
from pathlib import Path

import torch

from spd.experiments.piecewise.models import (
    PiecewiseFunctionSPDTransformer,
    PiecewiseFunctionTransformer,
)
from spd.experiments.piecewise.piecewise_decomposition import plot_components
from spd.experiments.piecewise.trig_functions import create_trig_function
from spd.run_spd import Config, PiecewiseConfig

pretrained_path = Path("demo_spd_model/model_50000.pth")
with open(pretrained_path.parent / "config.json") as f:
    config = Config(**json.load(f))

with open(pretrained_path.parent / "function_params.json") as f:
    function_params = json.load(f)
functions = [create_trig_function(*param) for param in function_params]

device = "cuda" if torch.cuda.is_available() else "cpu"

assert isinstance(config.task_config, PiecewiseConfig)
hardcoded_model = PiecewiseFunctionTransformer.from_handcoded(
    functions=functions,
    neurons_per_function=config.task_config.neurons_per_function,
    n_layers=config.task_config.n_layers,
    range_min=config.task_config.range_min,
    range_max=config.task_config.range_max,
    seed=config.seed,
).to(device)
hardcoded_model.eval()

model = PiecewiseFunctionSPDTransformer(
    n_inputs=hardcoded_model.n_inputs,
    d_mlp=hardcoded_model.d_mlp,
    n_layers=hardcoded_model.n_layers,
    k=config.task_config.k,
    d_embed=hardcoded_model.d_embed,
)
model.load_state_dict(torch.load(pretrained_path, weights_only=True, map_location="cpu"))
model.to(device)

topk = config.topk
batch_topk = config.batch_topk
fig = plot_components(model=model, step=-1, out_dir=None, device=device, slow_images=True)

out_path = Path(__file__).parent / "out/attribution_scores" / pretrained_path.parent.name
out_path.mkdir(parents=True, exist_ok=True)
out_file = out_path / "attribution_scores.png"
fig.savefig(out_file)
print(f"Saved plot to {out_file}")
