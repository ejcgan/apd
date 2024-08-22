import torch

from spd.scripts.tms.train_tms import BatchedDataLoader, Config, TMSDataset, TMSModel, optimize


def test_tms_training():
    # Set up a small configuration
    config = Config(n_features=3, n_hidden=2, n_instances=2, feature_probability=0.1, batch_size=32)

    # Initialize model, dataset, and dataloader
    device = "cpu"
    model = TMSModel(
        n_instances=config.n_instances,
        n_features=config.n_features,
        n_hidden=config.n_hidden,
        device=device,
    )
    dataset = TMSDataset(
        n_instances=config.n_instances,
        n_features=config.n_features,
        feature_probability=config.feature_probability,
        device=device,
    )
    dataloader = BatchedDataLoader(dataset, batch_size=config.batch_size)

    # Calculate initial loss
    batch, labels = next(iter(dataloader))
    initial_out = model(batch)
    initial_loss = torch.mean((labels.abs() - initial_out) ** 2)

    # Run optimize function
    optimize(model, dataloader, steps=10, print_freq=1000)

    # Calculate final loss
    final_out = model(batch)
    final_loss = torch.mean((labels.abs() - final_out) ** 2)

    # Assert that the final loss is lower than the initial loss
    assert (
        final_loss < initial_loss
    ), f"Final loss ({final_loss:.4f}) is not lower than initial loss ({initial_loss:.4f})"
