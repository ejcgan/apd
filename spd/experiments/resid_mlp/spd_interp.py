# %%


from spd.experiments.resid_mlp.models import ResidualMLPModel, ResidualMLPSPDRankPenaltyModel
from spd.experiments.resid_mlp.resid_mlp_dataset import ResidualMLPDataset
from spd.run_spd import ResidualMLPTaskConfig, calc_recon_mse
from spd.utils import run_spd_forward_pass, set_seed

# %%

if __name__ == "__main__":
    # Set up device and seed
    device = "cpu"
    print(f"Using device: {device}")
    set_seed(0)  # You can change this seed if needed

    wandb_path = (
        "wandb:spd-resid-linear/runs/dr4jw84y"  # Solves it for 5 features but only topk_recon=0.01
    )
    # local_path = "spd/experiments/resid_mlp/out/fr_seed0_topk1.10e+00_topkrecon1.00e+00_topkl2_1.00e-02_lr1.00e-02_bs1024_ft5_lay1_resid5_mlp5/model_10000.pth"

    # Load the pretrained SPD model
    model, config, label_coeffs = ResidualMLPSPDRankPenaltyModel.from_pretrained(wandb_path)

    assert isinstance(config.task_config, ResidualMLPTaskConfig)
    # Path must be local
    target_model, target_config_dict, target_label_coeffs = ResidualMLPModel.from_pretrained(
        config.task_config.pretrained_model_path
    )
    assert target_label_coeffs == label_coeffs

    dataset = ResidualMLPDataset(
        n_instances=model.config.n_instances,
        n_features=model.config.n_features,
        feature_probability=config.task_config.feature_probability,
        device=device,
        calc_labels=False,  # Our labels will be the output of the target model
        label_type=None,
        act_fn_name=None,
        label_fn_seed=None,
        label_coeffs=None,
        data_generation_type=config.task_config.data_generation_type,
    )
    batch, labels = dataset.generate_batch(config.batch_size)
    # Print some basic information about the model
    # print(f"Model structure:\n{model}")
    print(f"Number of features: {model.config.n_features}")
    print(f"Embedding dimension: {model.config.d_embed}")
    print(f"MLP dimension: {model.config.d_mlp}")
    print(f"Number of layers: {model.config.n_layers}")
    print(f"Number of subnetworks (k): {model.config.k}")

    assert config.topk is not None
    spd_outputs = run_spd_forward_pass(
        spd_model=model,
        target_model=target_model,
        input_array=batch,
        attribution_type=config.attribution_type,
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
