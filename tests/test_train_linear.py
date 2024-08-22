import torch

from spd.scripts.linear.linear_dataset import DeepLinearDataset
from spd.scripts.linear.models import DeepLinearModel
from spd.scripts.linear.train_linear import Config, train
from spd.utils import BatchedDataLoader


def test_train_linear_happy_path() -> None:
    device = "cpu"
    config = Config(
        n_features=2,
        n_layers=2,
        n_instances=2,
        batch_size=2,
        steps=3,  # Run only a few steps
        print_freq=100,
        lr=0.01,
        out_file=None,
    )

    model = DeepLinearModel(config.n_features, config.n_layers, config.n_instances).to(device)
    dataset = DeepLinearDataset(config.n_features, config.n_instances)
    dataloader = BatchedDataLoader(dataset, batch_size=config.batch_size, shuffle=False)

    # Calculate initial loss
    batch, labels = next(iter(dataloader))
    initial_out = model(batch.to(device))
    initial_loss = torch.mean((labels.to(device) - initial_out) ** 2).item()

    # assert True
    # Train the model
    train(config, model, dataloader, device)

    # Calculate final loss
    final_out = model(batch.to(device))
    final_loss = torch.mean((labels.to(device) - final_out) ** 2).item()

    assert (
        final_loss < initial_loss
    ), f"Expected final loss to be lower than initial loss, but got {final_loss} >= {initial_loss}"
