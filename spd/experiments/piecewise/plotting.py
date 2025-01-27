from pathlib import Path
from typing import Literal

import einops
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
import numpy as np
import torch
from einops import einsum
from jaxtyping import Float, Int
from matplotlib.colors import CenteredNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch import Tensor
from tqdm import tqdm

from spd.experiments.piecewise.models import (
    PiecewiseFunctionSPDTransformer,
    PiecewiseFunctionTransformer,
)
from spd.models.components import ParamComponents
from spd.run_spd import calc_recon_mse
from spd.utils import run_spd_forward_pass


def get_weight_matrix(
    general_param_components: ParamComponents,
) -> Float[Tensor, "k i j"]:
    a: Float[Tensor, "k i m"] = general_param_components.A
    b: Float[Tensor, "k m j"] = general_param_components.B
    weight: Float[Tensor, "k i j"] = einsum(a, b, "k i m, k m j -> k i j")
    return weight


def plot_matrix(
    ax: plt.Axes,
    matrix: torch.Tensor,
    title: str,
    xlabel: str,
    ylabel: str,
    colorbar_format: str = "%.1f",
    norm: plt.Normalize | None = None,
) -> None:
    # Useful to have bigger text for small matrices
    fontsize = 8 if matrix.numel() < 50 else 4
    norm = norm if norm is not None else CenteredNorm()
    im = ax.matshow(matrix.detach().cpu().numpy(), cmap="coolwarm", norm=norm)
    # If less than 500 elements, show the values
    if matrix.numel() < 500:
        for (j, i), label in np.ndenumerate(matrix.detach().cpu().numpy()):
            ax.text(i, j, f"{label:.2f}", ha="center", va="center", fontsize=fontsize)
    ax.set_xlabel(xlabel)
    if ylabel != "":
        ax.set_ylabel(ylabel)
    else:
        ax.set_yticklabels([])
    ax.set_title(title)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=0.1, pad=0.05)
    fig = ax.get_figure()
    assert fig is not None
    fig.colorbar(im, cax=cax, format=tkr.FormatStrFormatter(colorbar_format))
    if ylabel == "Function index":
        n_functions = matrix.shape[0]
        ax.set_yticks(range(n_functions))
        ax.set_yticklabels([f"{L:.0f}" for L in range(1, n_functions + 1)])


def plot_components(
    model: PiecewiseFunctionSPDTransformer,
    out_dir: Path | None,
    slow_images: bool,
    show_bias: bool | None = None,
) -> dict[str, plt.Figure]:
    decompose_bias = len(model.mlps[0].linear1.bias.shape) > 1
    show_bias = show_bias if show_bias is not None else decompose_bias
    n_layers = model.n_layers
    _, n_neurons, d_embed = model.mlps[0].linear1.subnetwork_params.shape
    ncols_per_layer = 2 + show_bias
    ncols = ncols_per_layer * n_layers
    if slow_images:
        nrows = model.k + 1
        fig, axs = plt.subplots(
            nrows,
            ncols,
            # 1/5 inch per dimension, 0.5 w_inches for colorbar, 4 inches for labels etc.
            figsize=(4 + (d_embed / 5 + 0.5) * ncols, 4 + n_neurons / 5 * nrows),
            constrained_layout=True,
        )
    else:
        nrows = 1
        fig, axs_row = plt.subplots(
            nrows,
            ncols,
            figsize=(4 + (d_embed / 5 + 0.5) * ncols, 4 + n_neurons / 5 * nrows),
            constrained_layout=True,
        )
        axs = np.array([axs_row])

    # For Piecewise, we want shared normalization across layers, but separate for W_in/bias/W_out
    def max_val(matrices: list[Float[Tensor, "..."]]) -> float:
        return max(matrix.abs().max().item() for matrix in matrices)

    colors = matplotlib.colors  # type: ignore
    norm_scale_w_in = max_val([model.mlps[i].linear1.subnetwork_params for i in range(n_layers)])
    norm_w_in = colors.Normalize(vmin=-norm_scale_w_in, vmax=norm_scale_w_in)
    norm_scale_bias = max_val([model.mlps[i].linear1.bias for i in range(n_layers)])
    norm_bias = colors.Normalize(vmin=-norm_scale_bias, vmax=norm_scale_bias)
    norm_scale_w_out = max_val([model.mlps[i].linear2.subnetwork_params for i in range(n_layers)])
    norm_w_out = colors.Normalize(vmin=-norm_scale_w_out, vmax=norm_scale_w_out)

    assert isinstance(axs, np.ndarray)
    for lay in range(n_layers):
        plot_matrix(
            axs[0, ncols_per_layer * lay],
            einops.einsum(model.mlps[lay].linear1.subnetwork_params, "k ... -> ..."),
            f"Layer {lay} W_in",
            "Neuron index",
            "Embedding index, sum over k" if lay == 0 else "",
            norm=norm_w_in,
        )
        if show_bias:
            plot_matrix(
                axs[0, ncols_per_layer * lay + 1],
                torch.einsum("kd->d", model.mlps[lay].linear1.bias).unsqueeze(0)
                if decompose_bias
                else model.mlps[lay].linear1.bias.unsqueeze(0),
                f"Layer {lay} Bias" if decompose_bias else f"Layer {lay} Bias (not decomposed)",
                "Neuron index",
                "",
                norm=norm_bias,
            )
        plot_matrix(
            axs[0, ncols_per_layer * lay + 1 + show_bias],
            einops.einsum(model.mlps[lay].linear2.subnetwork_params, "k ... -> ...").T,
            f"Layer {lay} W_out.T",
            "Neuron index",
            "",
            norm=norm_w_out,
        )
    if slow_images:
        for k in range(model.k):
            for lay in range(n_layers):
                mlp = model.mlps[lay]
                W_in_k = mlp.linear1.subnetwork_params[k]
                ax = axs[k + 1, ncols_per_layer * lay]  # type: ignore
                plot_matrix(
                    ax,
                    W_in_k,
                    "",
                    "",
                    f"Embedding index, k={k}" if lay == 0 else "",
                    norm=norm_w_in,
                )
                if decompose_bias and show_bias:
                    bias_k = mlp.linear1.bias[None, k]
                    ax = axs[k + 1, ncols_per_layer * lay + 1]  # type: ignore
                    plot_matrix(
                        ax,
                        bias_k,
                        "",
                        "",
                        "",
                        norm=norm_bias,
                    )
                elif show_bias:
                    # Remove the frame
                    axs[k + 1, ncols_per_layer * lay + 1].axis("off")
                W_out_k = mlp.linear2.subnetwork_params[k].T
                ax = axs[k + 1, ncols_per_layer * lay + 1 + show_bias]  # type: ignore
                plot_matrix(
                    ax,
                    W_out_k,
                    "",
                    "",
                    "",
                    norm=norm_w_out,
                )
    return {"matrices_layer0": fig}


def plot_model_functions(
    spd_model: PiecewiseFunctionSPDTransformer,
    target_model: PiecewiseFunctionTransformer,
    attribution_type: Literal["gradient", "ablation", "activation"],
    device: str,
    start: float,
    stop: float,
    print_info: bool = False,
    distil_from_target: bool = False,
) -> dict[str, plt.Figure]:
    fig, axes = plt.subplots(nrows=3, figsize=(12, 12), constrained_layout=True)
    assert isinstance(axes, np.ndarray)
    [ax, ax_attrib, ax_inner] = axes
    # For these tests we run with unusual data where there's always 1 control bit active, which
    # might differ from training. Thus we manually set topk to 1. Note that this should work for
    # both, batch and non-batch topk.
    topk = 1
    # Disable batch_topk to rule out errors caused by batch -- non-batch is an easier task for SPD
    # in the toy setting used for plots.
    batch_topk = False
    fig.suptitle(
        f"Model outputs for each control bit. (Plot with topk={topk} & batch_topk={batch_topk}.\n"
        "This should differ from the training settings, topk>=1 ought to work for plotting.)"
    )
    # Get model outputs for simple example data. Create input array with 10_000 rows, 1000
    # rows for each function. Set the 0th column to be linspace(0, 5, 1000) repeated. Set the
    # control bits to [0,1,0,0,...] for the first 1000 rows, [0,0,1,0,...] for the next 1000 rows,
    # etc.
    n_samples = 1000
    n_functions = spd_model.num_functions
    # Set the control bits
    input_array = torch.eye(spd_model.n_inputs)[-n_functions:, :]
    input_array = input_array.repeat_interleave(n_samples, dim=0)
    input_array = input_array.to(device)
    # Set the 0th input to x_space
    x_space = torch.linspace(start, stop, n_samples)
    input_array[:, 0] = x_space.repeat(n_functions)

    spd_outputs = run_spd_forward_pass(
        spd_model=spd_model,
        target_model=target_model,
        input_array=input_array,
        attribution_type=attribution_type,
        batch_topk=batch_topk,
        topk=topk,
        distil_from_target=distil_from_target,
    )
    model_output_hardcoded = spd_outputs.target_model_output
    model_output_spd = spd_outputs.spd_model_output
    out_topk = spd_outputs.spd_topk_model_output
    inner_acts = spd_outputs.inner_acts
    attribution_scores = spd_outputs.attribution_scores
    topk_mask = spd_outputs.topk_mask

    if print_info:
        # Check if, ever, there are cases where the control bit is 1 but the topk_mask is False.
        # We check this by calculating whether topk_mask is True OR control bit is 0.
        control_bits = input_array[:, 1:].cpu().detach().numpy()
        topk_mask = topk_mask.cpu().detach().numpy()
        topk_mask_control_bits = topk_mask | (control_bits == 0)
        print(
            f"How often is topk_mask True or control_bits == 0: {topk_mask_control_bits.mean():.3%}"
        )
        # Calculate recon loss
        topk_recon_loss = calc_recon_mse(out_topk, model_output_hardcoded, has_instance_dim=False)
        print(f"Topk recon loss: {topk_recon_loss:.4f}")

    # Convert stuff to numpy
    model_output_spd = model_output_spd[:, 0].cpu().detach().numpy()
    model_output_hardcoded = model_output_hardcoded[:, 0].cpu().detach().numpy()
    out_topk = out_topk.cpu().detach().numpy()
    input_xs = input_array[:, 0].cpu().detach().numpy()

    # Plot for every k
    tab20b_colors = plt.get_cmap("tab20b").colors  # type: ignore
    tab20c_colors = plt.get_cmap("tab20c").colors  # type: ignore
    colors = [*tab20b_colors, *tab20c_colors]
    # cb stands for control bit which is active there; this differs from k due to permutation
    for cb in range(n_functions):
        color0 = colors[(4 * cb + 0) % len(colors)]
        color1 = colors[(4 * cb + 1) % len(colors)]
        color2 = colors[(4 * cb + 2) % len(colors)]
        color3 = colors[(4 * cb + 3) % len(colors)]
        s = slice(cb * n_samples, (cb + 1) * n_samples)
        ax.plot(input_xs[s], out_topk[s], ls="--", color=color0)
        assert target_model.controlled_resnet is not None
        ax.plot(input_xs[s], model_output_hardcoded[s], label=f"cb={cb}", color=color1)
        ax.plot(
            x_space,
            target_model.controlled_resnet.functions[cb](x_space),
            ls=":",
            color=color2,
        )
        ax.plot(input_xs[s], model_output_spd[s], ls="-.", color=color3)
        k_cb = attribution_scores[s].mean(dim=0).argmax()
        # ax_attrib
        for k in range(n_functions):
            # Find permutation
            if k == k_cb:
                ax_attrib.plot(
                    input_xs[s],
                    attribution_scores[s][:, k],
                    color=color0,
                    label=f"cb={cb}, k_cb={k}",
                )
            else:
                ax_attrib.plot(input_xs[s], attribution_scores[s][:, k], color=color0, alpha=0.2)
        # ax_inner
        if len(inner_acts) <= 2:
            for k in range(n_functions):
                # Find permutation
                if k == k_cb:
                    assert len(inner_acts) <= 2, "Didn't implement more than 2 SPD 'layers' yet"
                    for j, layer_name in enumerate(inner_acts.keys()):
                        ls = ["-", "--"][j]
                        ax_inner.plot(
                            input_xs[s],
                            inner_acts[layer_name].cpu().detach()[s][:, k],
                            color=color0,
                            ls=ls,
                            label=f"cb={cb}, k_cb={k}" if j == 0 else None,
                        )
                else:
                    for j, layer_name in enumerate(inner_acts.keys()):
                        ls = ["-", "--"][j]
                        ax_inner.plot(
                            input_xs[s],
                            inner_acts[layer_name].cpu().detach()[s][:, k],
                            color="k",
                            ls=ls,
                            lw=0.2,
                        )
        ax_inner.plot([], [], color=colors[0], label="W_in", ls="-")
        ax_inner.plot([], [], color=colors[0], label="W_out", ls="--")
        ax_inner.plot([], [], color="k", label="k!=k_cb", ls="-", lw=0.2)
    else:
        tqdm.write("Skipping inner acts plot for more than 2 SPD 'layers'")
    # Add some additional (blue) legend lines explaining the different line styles
    ax.plot([], [], ls="--", color=colors[0], label="spd model topk")
    ax.plot([], [], ls="-", color=colors[1], label="target model")
    ax.plot([], [], ls=":", color=colors[2], label="true function")
    ax.plot([], [], ls="-.", color=colors[3], label="spd model")
    ax.legend(ncol=3, loc="lower left")
    ax_attrib.legend(ncol=3, loc="lower left")
    if attribution_scores.min() > 0:
        ax_attrib.set_yscale("log")
        ax_attrib.set_ylabel("attribution_scores (log)")
    ax_attrib.set_xlabel("x (model input dim 0)")
    ax_attrib.set_title(
        "Attributions of each subnetwork for every control bit case (k=k_cb in bold)"
    )
    ax_inner.legend(ncol=3, loc="lower left")
    ax_inner.set_ylabel("inner acts (symlog)")
    ax_inner.set_title("'inner acts', coloured for the top subnetwork, black for the others")
    ax_inner.set_xlabel("x (model input dim 0)")
    ax.set_xlabel("x (model input dim 0)")
    ax.set_ylabel("f(x) (model output dim 0)")
    return {"model_functions": fig}


def plot_model_functions_paper(
    spd_model: PiecewiseFunctionSPDTransformer,
    target_model: PiecewiseFunctionTransformer,
    attribution_type: Literal["gradient", "ablation", "activation"],
    device: str,
    start: float,
    stop: float,
    print_info: bool = False,
    distil_from_target: bool = False,
) -> dict[str, plt.Figure]:
    k = spd_model.k
    fig, axs = plt.subplots(ncols=k, figsize=(10, 3), constrained_layout=True, sharey=True)
    assert isinstance(axs, np.ndarray)
    # For these tests we run with unusual data where there's always 1 control bit active, which
    # might differ from training. Thus we manually set topk to 1. Note that this should work for
    # both, batch and non-batch topk.
    topk = 1
    # Disable batch_topk to rule out errors caused by batch -- non-batch is an easier task for SPD
    # in the toy setting used for plots.
    batch_topk = False
    # Get model outputs for simple example data. Create input array with 10_000 rows, 1000
    # rows for each function. Set the 0th column to be linspace(0, 5, 1000) repeated. Set the
    # control bits to [0,1,0,0,...] for the first 1000 rows, [0,0,1,0,...] for the next 1000 rows,
    # etc.
    n_samples = 1000
    n_functions = spd_model.num_functions
    # Set the control bits
    input_array = torch.eye(spd_model.n_inputs)[-n_functions:, :]
    input_array = input_array.repeat_interleave(n_samples, dim=0)
    input_array = input_array.to(device)
    # Set the 0th input to x_space
    x_space = torch.linspace(start, stop, n_samples)
    input_array[:, 0] = x_space.repeat(n_functions)

    spd_outputs = run_spd_forward_pass(
        spd_model=spd_model,
        target_model=target_model,
        input_array=input_array,
        attribution_type=attribution_type,
        batch_topk=batch_topk,
        topk=topk,
        distil_from_target=distil_from_target,
    )
    model_output_hardcoded = spd_outputs.target_model_output
    out_topk = spd_outputs.spd_topk_model_output

    # Convert stuff to numpy
    model_output_hardcoded = model_output_hardcoded[:, 0].cpu().detach().numpy()
    out_topk = out_topk.cpu().detach().numpy()
    input_xs = input_array[:, 0].cpu().detach().numpy()

    # Plot for every k
    colors = plt.get_cmap("tab20").colors  # type: ignore
    # cb stands for control bit which is active there; this differs from k due to permutation
    for cb in range(n_functions):
        color0 = colors[(2 * cb + 0) % len(colors)]
        color1 = colors[(2 * cb + 1) % len(colors)]
        s = slice(cb * n_samples, (cb + 1) * n_samples)
        ax = axs[cb]
        ax.plot(input_xs[s], model_output_hardcoded[s], color=color0, label="Target model")
        ax.plot(input_xs[s], out_topk[s], ls="--", color=color1, label="APD top-k")
        ax.legend(loc="lower left")
        assert target_model.controlled_resnet is not None
        ax.set_xlabel("Input x")
        ax.set_title(f"Feature {cb}")
    axs[0].set_ylabel("Output F(x)")
    return {"model_functions_paper": fig}


def plot_single_network(
    ax: plt.Axes,
    weights: list[dict[str, Float[Tensor, "i j"]]],
    colors: list[dict[str, Int[Tensor, "i j"]]] | None = None,
    show_resid_shades: bool = False,
) -> None:
    n_layers = len(weights)
    d_embed = weights[0]["W_in"].shape[0]
    d_mlp = weights[0]["W_in"].shape[1]
    assert all(W["W_in"].shape == (d_embed, d_mlp) for W in weights)
    assert all(W["W_out"].shape == (d_mlp, d_embed) for W in weights)

    # Define node positions
    x_embed = np.linspace(0.05, 0.45, d_embed)
    x_mlp = np.linspace(0.55, 0.95, d_mlp)

    # Plot nodes
    for lay in range(n_layers + 1):
        ax.scatter(x_embed, [2 * lay] * d_embed, s=100, color="grey", edgecolors="k", zorder=3)
    for lay in range(n_layers):
        ax.scatter(x_mlp, [2 * lay + 1] * d_mlp, s=100, color="grey", edgecolors="k", zorder=3)

    # Plot edges
    cmap = plt.get_cmap("RdBu")
    max_in_norm = max(weights[lay]["W_in"].abs().max().item() for lay in range(n_layers))
    max_out_norm = max(weights[lay]["W_out"].abs().max().item() for lay in range(n_layers))
    for lay in range(n_layers):
        # Normalize weights
        W_in_norm = weights[lay]["W_in"] / max_in_norm
        W_out_norm = weights[lay]["W_out"] / max_out_norm
        colors_in = colors[lay]["W_in"].int() if colors is not None else None
        colors_out = colors[lay]["W_out"].int() if colors is not None else None
        plot_lay = n_layers - (lay + 1)
        for i in range(d_embed):
            for j in range(d_mlp):
                weight = W_in_norm[i, j].item()
                if colors_in is not None:
                    color = f"C{colors_in[i, j]}" if colors_in[i, j] != -1 else "k"
                else:
                    color = cmap(0.5 * (weight + 1))
                ax.plot(
                    [x_embed[i], x_mlp[j]],
                    [2 * plot_lay + 2, 2 * plot_lay + 1],
                    color=color,
                    linewidth=2 * abs(weight),
                )
                weight = W_out_norm[j, i].item()
                if colors_out is not None:
                    color = f"C{colors_out[j, i]}" if colors_out[j, i] != -1 else "k"
                else:
                    color = cmap(0.5 * (weight + 1))
                ax.plot(
                    [x_mlp[j], x_embed[i]],
                    [2 * plot_lay + 1, 2 * plot_lay],
                    color=color,
                    linewidth=2 * abs(weight),
                )
    # Draw residual steam
    for i in range(d_embed):
        ax.plot([x_embed[i], x_embed[i]], [0, 2 * n_layers], color="grey", linewidth=0.5, zorder=-1)
    if show_resid_shades:
        ax.add_patch(
            plt.Rectangle(
                (0.05, 0.05),
                0.45,
                0.9,
                fill=True,
                color="grey",
                alpha=0.25,
                transform=ax.transAxes,
                zorder=-2,
            )
        )
        # Draw MLP branching off
        y1 = np.linspace(0, 2 * n_layers, 100)
        x1a = (1 + np.cos((y1 + 1) * 2 * np.pi / n_layers)) / 4 - 0.00
        x1b = (1 + np.cos((y1 + 1) * 2 * np.pi / n_layers)) / 4 + 0.50
        ax.fill_betweenx(y1, x1a, x1b, color="tan", alpha=0.5, zorder=-1)
    # Turn off axes
    ax.axis("off")


def get_weight_colors_single(
    weights: dict[str, Float[Tensor, "i j"]],
    target_weights: dict[str, Float[Tensor, "i j"]],
) -> dict[str, Int[Tensor, "i j"]]:
    d_resid, d_mlp = weights["W_in"].shape
    assert weights["W_out"].shape == (d_mlp, d_resid)
    assert target_weights["W_in"].shape == (d_mlp, d_resid), target_weights["W_in"].shape
    assert target_weights["W_out"].shape == (d_resid, d_mlp), target_weights["W_out"].shape
    colors = {}
    colors["W_in"] = torch.ones_like(weights["W_in"], dtype=torch.int64) * (-2)
    colors["W_out"] = torch.ones_like(weights["W_out"], dtype=torch.int64) * (-2)
    for j in range(d_mlp):
        f = torch.where(target_weights["W_in"].T[1:, j] != 0)[0]
        colors["W_in"][1 + f, j] = f
        colors["W_in"][0, j] = f
        colors["W_out"][j, d_resid - 1] = f
    # Set colors to -1 for weights that should be 0 (belong to no feature)
    mask_in = target_weights["W_in"].T == 0
    mask_out = target_weights["W_out"].T == 0
    colors["W_in"][mask_in] = d_resid - 2
    colors["W_out"][mask_out] = d_resid - 2
    # -2 was placeholder value and should be gone now
    assert torch.all(colors["W_in"] != -2)
    assert torch.all(colors["W_out"] != -2)
    return colors


def get_subnetwork_colors(
    subnetworks: dict[int, list[dict[str, Float[Tensor, "i j"]]]],
    target_network: list[dict[str, Float[Tensor, "i j"]]],
) -> dict[int, list[dict[str, Int[Tensor, "i j"]]]]:
    colors = {}
    for key in subnetworks:
        colors[key] = []
        for lay in range(len(subnetworks[key])):
            colors[key].append(get_weight_colors_single(subnetworks[key][lay], target_network[lay]))
    return colors


def plot_piecewise_network(
    model: PiecewiseFunctionSPDTransformer,
    hardcoded_model: PiecewiseFunctionTransformer,
    first_column: Literal["Full SPD model", "Target model"] = "Target model",
) -> dict[str, plt.Figure]:
    n_components = model.k
    mlps: torch.nn.ModuleList = model.mlps
    n_layers = len(mlps)

    W_ins = [get_weight_matrix(mlps[lay].linear1) for lay in range(n_layers)]
    W_outs = [get_weight_matrix(mlps[lay].linear2) for lay in range(n_layers)]
    subnetworks = {}
    subnetworks[-1] = []
    target_network = []
    for lay in range(n_layers):
        target_network.append(
            {
                "W_in": hardcoded_model.mlps[lay].input_layer.weight,
                "W_out": hardcoded_model.mlps[lay].output_layer.weight,
            }
        )

    if first_column == "Full SPD model":
        for lay in range(n_layers):
            subnetworks[-1].append({"W_in": W_ins[lay].sum(0), "W_out": W_outs[lay].sum(0)})
    else:
        for lay in range(n_layers):
            subnetworks[-1].append(
                {"W_in": target_network[lay]["W_in"].T, "W_out": target_network[lay]["W_out"].T}
            )

    for k in range(n_components):
        subnetworks[k] = []
        for lay in range(n_layers):
            subnetworks[k].append({"W_in": W_ins[lay][k, :], "W_out": W_outs[lay][k, :]})

    colors = get_subnetwork_colors(subnetworks, target_network)

    fig, axs = plt.subplots(
        1,
        n_components + 1,
        figsize=(2.5 * (n_components + 1), 2.5 * n_layers),
        constrained_layout=True,
    )
    axs = np.array(axs)
    for i, k in enumerate(np.arange(-1, n_components)):
        axs[i].set_title(f"Component {k}")
        plot_single_network(axs[i], subnetworks[k], colors[k])  # type: ignore
    axs[0].set_title(first_column)
    axs[0].text(0.275, 0.01, "Outputs", ha="center", va="center", transform=axs[0].transAxes)
    axs[0].text(0.275, 0.985, "Inputs", ha="center", va="center", transform=axs[0].transAxes)
    for lay in range(n_layers):
        axs[0].text(1, 2 * lay + 1, "MLP", ha="left", va="center")
    legend = fig.legend(
        [plt.Line2D([0], [0], color=f"C{k}") for k in range(n_components + 1)],
        [*[f"Feature {k} weights" for k in range(n_components)], "Not in target weights"],
        ncol=n_components + 1,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.05),
        bbox_transform=fig.transFigure,
    )
    legend.set_in_layout(True)
    return {"subnetworks_graph_plots": fig}
