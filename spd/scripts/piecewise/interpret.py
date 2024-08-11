# %%
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from jaxtyping import Float
from torch import Tensor

from spd.models.piecewise_models import (
    PiecewiseFunctionSPDTransformer,
    PiecewiseFunctionTransformer,
)
from spd.run_spd import Config, PiecewiseConfig
from spd.scripts.piecewise.trig_functions import create_trig_function

# %%
if __name__ == "__main__":
    pretrained_path = Path(
        # "/root/spd/spd/scripts/piecewise/out/sp1.0_lr0.01_pNone_topk4_bs2048_/model_19999.pth"
        # "/root/spd/spd/scripts/piecewise/out/test_sp1.0_lr0.01_pNone_topk4_bs2048_/model_20000.pth"
        # "/root/spd/spd/scripts/piecewise/out/test2_sp1.0_lr0.01_pNone_topk4_bs2048_/model_15999.pth"
        "out/sp1.0_lr0.01_pNone_topk4_bs2048_/model_20000.pth"
    )

    with open(pretrained_path.parent / "config.json") as f:
        config = Config(**json.load(f))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(pretrained_path.parent / "function_params.json") as f:
        function_params = json.load(f)
    functions = [create_trig_function(*param) for param in function_params]

    assert isinstance(config.task_config, PiecewiseConfig)
    hardcoded_model = PiecewiseFunctionTransformer.from_handcoded(
        functions=functions,
        neurons_per_function=config.task_config.neurons_per_function,
        n_layers=config.task_config.n_layers,
        range_min=config.task_config.range_min,
        range_max=config.task_config.range_max,
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

    # %%
    # NOTE: This is currently broken. I think our hardcoded network is different than the one used
    # in the training script. Maybe there is more randomness that happens at initialization?
    # Check the param match between the two models
    hardcoded_weights = hardcoded_model.all_decomposable_params()
    param_match_loss = torch.zeros(1, device=device)
    for i, (A, B) in enumerate(zip(model.all_As(), model.all_Bs(), strict=True)):
        normed_A = A / A.norm(p=2, dim=-2, keepdim=True)
        AB = torch.einsum("...fk,...kg->...fg", normed_A, B)
        param_match_loss = param_match_loss + ((AB - hardcoded_weights[i]) ** 2).mean(dim=(-2, -1))
    param_match_loss = param_match_loss / model.n_param_matrices
    print(f"Param match loss: {param_match_loss}")

    # %%
    # Step 1: Create a batch of inputs with different control bits active
    x_val = torch.tensor(2.5)
    batch_size = len(functions)
    true_labels = torch.tensor([f(x_val) for f in functions])
    print(f"Input: {x_val}, True labels: {true_labels}")

    x = torch.zeros(batch_size, hardcoded_model.n_inputs).to(device)
    x[:, 0] = x_val.item()
    x[torch.arange(batch_size), torch.arange(1, batch_size + 1)] = 1

    # Step 2: Forward pass on the spd model
    out, layer_acts, inner_acts = model(x)
    print(f"out: {out}")
    # Also check the hardcoded model
    out_hardcoded = hardcoded_model(x)
    print(f"out_hardcoded: {out_hardcoded}")

    # Step 3: Backward pass on the spd model.
    # all_grads: [Float[Tensor, "batch k"], Float[Tensor, "batch k"], ...] # Total = n_param_matrices
    all_grads = [torch.zeros_like(inner_acts[i]) for i in range(model.n_param_matrices)]
    for feature_idx in range(out.shape[-1]):
        grads = torch.autograd.grad(out[..., feature_idx].sum(), inner_acts, retain_graph=True)
        for param_matrix_idx in range(model.n_param_matrices):
            all_grads[param_matrix_idx] += grads[param_matrix_idx]

    assert len(inner_acts) == len(all_grads) == model.n_param_matrices
    all_grads_stacked = torch.stack(all_grads, dim=0)
    inner_acts_stacked = torch.stack(inner_acts, dim=0)
    attribution_scores: Float[Tensor, "batch k"] = (inner_acts_stacked * all_grads_stacked).sum(
        dim=0
    )
    print(f"Attribution scores: {attribution_scores}")
    # Plot a matshow of the attribution scores
    # Each row should have it's own color scale
    # Normalize each row to have mean 0 and std 1
    attribution_scores_normed = (
        attribution_scores - attribution_scores.mean(dim=1, keepdim=True)
    ) / attribution_scores.std(dim=1, keepdim=True)

    # Find the max absolute value in the attribution scores
    max_abs_value = attribution_scores_normed.abs().max()
    # matshow
    plt.matshow(
        attribution_scores_normed.detach().cpu().numpy(),
        cmap="coolwarm",
        vmin=-max_abs_value,
        vmax=max_abs_value,
    )
    # plt.matshow(attribution_scores_normed.detach().cpu().numpy(), cmap="coolwarm")
    # ylabel should be the function index
    plt.ylabel("Function index")
    plt.xlabel("subnetwork index")
    # Add title saying it's the attribution scores
    plt.title("Attribution scores")
    # Use coolwarm colormap
    plt.colorbar()
    plt.show()

    # Get the top 4 attribution abs values for each function
    top_k_indices = attribution_scores.abs().topk(4, dim=-1).indices
    # print(f"Top-k indices: {top_k_indices}")

    # Do a forward_topk pass
    out_topk, layer_acts_topk, inner_acts_topk = model.forward_topk(x, top_k_indices)
    print(f"Top-k output: {out_topk}")

# %%
