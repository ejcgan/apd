# %%


from spd.experiments.resid_mlp.models import ResidualMLPModel, ResidualMLPSPDRankPenaltyModel
from spd.experiments.resid_mlp.resid_mlp_dataset import ResidualMLPDataset
from spd.run_spd import ResidualMLPConfig, calc_recon_mse
from spd.utils import run_spd_forward_pass, set_seed

# %%

if __name__ == "__main__":
    # Set up device and seed
    device = "cpu"
    print(f"Using device: {device}")
    set_seed(0)  # You can change this seed if needed

    wandb_path = (
        "spd-resid-linear/runs/dr4jw84y"  # Solves it for 5 features but only topk_recon=0.01
    )
    # local_path = "spd/experiments/resid_mlp/out/fr_seed0_topk1.10e+00_topkrecon1.00e+00_topkl2_1.00e-02_lr1.00e-02_bs1024_ft5_lay1_resid5_mlp5/model_10000.pth"

    # Load the pretrained SPD model
    model, config, label_coeffs = ResidualMLPSPDRankPenaltyModel.from_wandb(wandb_path)

    assert isinstance(config.task_config, ResidualMLPConfig)
    # Path must be local
    target_model, target_config_dict, target_label_coeffs = ResidualMLPModel.from_pretrained(
        config.task_config.pretrained_model_path
    )
    assert target_label_coeffs == label_coeffs

    dataset = ResidualMLPDataset(
        n_instances=model.n_instances,
        n_features=model.n_features,
        feature_probability=config.task_config.feature_probability,
        device=device,
        calc_labels=False,  # Our labels will be the output of the target model
        data_generation_type=config.task_config.data_generation_type,
    )
    batch, labels = dataset.generate_batch(config.batch_size)
    # Print some basic information about the model
    # print(f"Model structure:\n{model}")
    print(f"Number of features: {model.n_features}")
    print(f"Embedding dimension: {model.d_embed}")
    print(f"MLP dimension: {model.d_mlp}")
    print(f"Number of layers: {model.n_layers}")
    print(f"Number of subnetworks (k): {model.k}")

    assert config.topk is not None
    spd_outputs = run_spd_forward_pass(
        spd_model=model,
        target_model=target_model,
        input_array=batch,
        attribution_type=config.attribution_type,
        spd_type=config.spd_type,
        batch_topk=config.batch_topk,
        topk=config.topk,
        distil_from_target=config.distil_from_target,
    )
    # Topk recon (Note that we're using true labels not the target model output)
    topk_recon_loss = calc_recon_mse(spd_outputs.spd_topk_model_output, labels)
    print(f"Topk recon loss: {topk_recon_loss}")
    print(f"batch:\n{batch[:10]}")
    print(f"labels:\n{labels[:10]}")
    print(f"spd_outputs.spd_topk_model_output:\n{spd_outputs.spd_topk_model_output[:10]}")

    in_matrix = model.W_E @ target_model.layers[0].input_layer.weight.T
    print(f"target in_matrix:\n{in_matrix}")
    for k in range(model.k):
        in_matrix_subnet_k = model.W_E @ model.layers[0].linear1.subnetwork_params[k]
        print(f"in_matrix_subnet{k}:\n{in_matrix_subnet_k}")

    out_matrix = target_model.layers[0].output_layer.weight.T
    print(f"target out_matrix:\n{out_matrix}")
    for k in range(model.k):
        out_matrix_subnet_k = model.layers[0].linear2.subnetwork_params[k]
        print(f"out_matrix_subnet{k}:\n{out_matrix_subnet_k}")

# %%
import torch

from spd.utils import REPO_ROOT

# path = REPO_ROOT / "spd/experiments/resid_mlp/out/resid_mlp_identity_n-features5_d-resid5_d-mlp4_n-layers2_seed0/target_model.pth"
path = (
    REPO_ROOT
    # / "spd/experiments/resid_mlp/out/resid_mlp_identity_n-features5_d-resid5_d-mlp4_n-layers2_seed0_wd0.05/target_model.pth"
    # / "spd/experiments/resid_mlp/out/resid_mlp_identity_n-features5_d-resid5_d-mlp4_n-layers2_seed0_wd0.5/target_model.pth"
    # / "spd/experiments/resid_mlp/out/resid_mlp_identity_n-features5_d-resid5_d-mlp4_n-layers2_seed0_l2-0.001/target_model.pth"
    / "spd/experiments/resid_mlp/out/resid_mlp_identity_n-features5_d-resid5_d-mlp5_n-layers2_seed0_l2-0.001/target_model.pth"
)
with open(path, "rb") as f:
    params = torch.load(f)

# %%
