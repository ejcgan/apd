"""Trains a residual linear model on one-hot input vectors."""

import torch
import wandb

from spd.experiments.resid_mlp.train_resid_mlp import (
    ResidMLPTrainConfig,
    ResidualMLPConfig,
    run_train,
)
from spd.utils import set_seed

wandb.require("core")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = ResidMLPTrainConfig(
        wandb_project="spd-train-resid-mlp",
        seed=0,
        resid_mlp_config=ResidualMLPConfig(
            n_instances=1,
            n_features=100,
            d_embed=100,
            d_mlp=50,
            n_layers=1,
            act_fn_name="relu",
            apply_output_act_fn=False,
            in_bias=False,
            out_bias=False,
        ),
        label_fn_seed=0,
        label_type="act_plus_resid",
        loss_type="resid",
        use_trivial_label_coeffs=True,
        feature_probability=0.01,
        importance_val=1,
        data_generation_type="at_least_zero_active",
        batch_size=2048,
        steps=1000,
        print_freq=100,
        lr=3e-3,
        lr_schedule="cosine",
        fixed_random_embedding=True,
        fixed_identity_embedding=False,
        n_batches_final_losses=10,
    )
    # config = Config(
    #     seed=0,
    #     label_fn_seed=0,
    #     fixed_random_embedding=True,
    #     fixed_identity_embedding=False,
    #     n_instances=3,
    #     n_features=100,
    #     d_embed=1000,
    #     d_mlp=50,
    #     n_layers=1,
    #     act_fn_name="relu",
    #     apply_output_act_fn=False,
    #     label_type="act_plus_resid",
    #     data_generation_type="at_least_zero_active",
    #     use_trivial_label_coeffs=True,
    #     in_bias=False,
    #     out_bias=False,
    #     feature_probability=0.01,
    #     importance_val=1,
    #     batch_size=2048,
    #     steps=10000,
    #     print_freq=100,
    #     lr=3e-3,
    #     lr_schedule="cosine",
    # )
    # config = Config(
    #     seed=0,
    #     label_fn_seed=0,
    #     fixed_random_embedding=True,
    #     fixed_identity_embedding=False,
    #     n_instances=8,
    #     n_features=20,
    #     d_embed=200,
    #     d_mlp=10,
    #     n_layers=1,
    #     act_fn_name="relu",
    #     apply_output_act_fn=False,
    #     label_type="act_plus_resid",
    #     data_generation_type="at_least_zero_active",
    #     use_trivial_label_coeffs=True,
    #     in_bias=False,
    #     out_bias=False,
    #     feature_probability=0.05,
    #     importance_val=1,
    #     batch_size=2048,
    #     steps=10000,
    #     print_freq=100,
    #     lr=3e-3,
    #     lr_schedule="cosine",
    # )

    set_seed(config.seed)

    run_train(config, device)
