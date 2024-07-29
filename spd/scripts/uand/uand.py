# %%

import torch
from einops import einsum
from jaxtyping import Float
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import LinearLR
from tqdm import tqdm

# Set default torch type to bfloat16
torch.set_default_dtype(torch.bfloat16)


class UANDMLP(nn.Module):
    def __init__(self, d: int, n_mlp: int = 2, p: float = 0.1):
        super().__init__()
        self.d = d  # == d_resid == n_features
        self.n_mlp = n_mlp
        self.p = p
        self.n_out = d * (d - 1) // 2

        self.W1 = nn.Parameter(self._initialize_W1())
        self.b1 = nn.Parameter(torch.ones(self.n_mlp) * -1.0)

        self.W2 = nn.Parameter(self._initialize_W2(self.W1))
        # self.W2 = nn.Parameter(torch.empty(self.n_mlp, self.n_out))
        # Initialize with kaiming normal
        # nn.init.kaiming_normal_(self.W2)
        self.b2 = nn.Parameter(torch.zeros(self.n_out))

    def _initialize_W1(self) -> Float[Tensor, "d n_mlp"]:
        # p = (np.log(self.d) ** 2) / np.sqrt(self.d)  # Is the square here correct?
        # p = 10 / np.sqrt(self.d)
        W1 = torch.zeros(self.d, self.n_mlp)
        mask = torch.rand(self.d, self.n_mlp) < self.p
        W1[mask] = 1.0
        return W1

    # def _initialize_W2(self, W1: Float[Tensor, "d n_mlp"]) -> Float[Tensor, "n_mlp n_out"]:
    #     W2 = torch.zeros(self.n_mlp, self.n_out)
    #     counter = 0
    #     for in_idx_1 in range(self.d):
    #         for in_idx_2 in range(in_idx_1 + 1, self.d):
    #             S_intersection = W1[in_idx_1, :] * W1[in_idx_2, :]
    #             W2[:, counter] = S_intersection / S_intersection.sum()
    #             counter += 1
    #     return W2

    def _initialize_W2(self, W1: Float[Tensor, "d n_mlp"]) -> Float[Tensor, "n_mlp n_out"]:
        # Create indices for all pairs
        indices = torch.triu_indices(self.d, self.d, offset=1)

        # Compute S_intersection for all pairs at once
        S_intersection = W1[indices[0], :] * W1[indices[1], :]

        # Normalize S_intersection
        W2 = S_intersection / S_intersection.sum(dim=1, keepdim=True)

        return W2.T

    def forward(self, x: Float[Tensor, "batch d"]) -> Float[Tensor, "batch n_out"]:
        h: Float[Tensor, "batch n_mlp"] = F.relu(
            einsum(x, self.W1, "batch d, d n_mlp -> batch n_mlp") + self.b1
        )
        out: Float[Tensor, "batch n_out"] = F.relu(
            einsum(h, self.W2, "batch n_mlp, n_mlp n_out -> batch n_out") + self.b2
        )
        return out


def get_ground_truth(x: Float[Tensor, "batch d"]) -> Float[Tensor, "... n_out"]:
    n_out = x.shape[-1] * (x.shape[-1] - 1) // 2
    labels = torch.zeros(x.shape[:-1] + (n_out,), device=x.device)
    counter = 0
    for i in range(x.shape[-1]):
        for j in range(i + 1, x.shape[-1]):
            labels[..., counter] = x[..., i] * x[..., j]
            counter += 1
    return labels


def run_model_for_various_d(
    d_values: list[int], batch_size: int = 1000, device: str = "cpu"
) -> list[float]:
    losses = []
    for d in tqdm(d_values, desc="Testing different d values"):
        uand = UANDMLP(d).to(device)
        indices = torch.randint(0, d, (batch_size, 2), device=device)
        x = torch.zeros(batch_size, d, device=device)
        x[torch.arange(batch_size), indices[:, 0]] = 1
        x[torch.arange(batch_size), indices[:, 1]] = 1

        out = uand(x)
        gt = get_ground_truth(x)
        mse_loss = F.mse_loss(out, gt).item()
        losses.append(mse_loss)
        tqdm.write(f"d: {d}, MSE loss: {mse_loss}")

    return losses


def train_uand(
    model: UANDMLP,
    d: int,
    batch_size: int = 1000,
    n_batches: int = 100,
    learning_rate: float = 0.01,
    warmup_steps: int = 100,
    device: str = "cpu",
) -> list[float]:
    optimizer = torch.optim.AdamW([model.W2, model.b2], lr=learning_rate)

    # Create a composite scheduler: warmup followed by linear decay
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps),
            LinearLR(
                optimizer, start_factor=1.0, end_factor=0.1, total_iters=n_batches - warmup_steps
            ),
        ],
        milestones=[warmup_steps],
    )

    losses = []

    progress_bar = tqdm(range(n_batches), desc="Training UAND")
    for _ in progress_bar:
        # Generate random two-hot vectors
        indices = torch.randint(0, d, (batch_size, 2), device=device)
        x = torch.zeros(batch_size, d, device=device)
        x[torch.arange(batch_size), indices[:, 0]] = 1
        x[torch.arange(batch_size), indices[:, 1]] = 1

        # Forward pass
        out = model(x)
        gt = get_ground_truth(x)
        loss = (out - gt).pow(2).sum(dim=-1).mean()

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())
        progress_bar.set_postfix(
            {"Loss": f"{loss.item():.4f}", "LR": f"{scheduler.get_last_lr()[0]:.6f}"}
        )

    return losses


# %%
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 1
    d = 100
    # n_mlp = 20**2
    n_mlp = 100
    # d = 5
    # n_mlp = 10
    p = 0.25
    uand = UANDMLP(d, n_mlp, p).to(device)
    print(f"W1 ({uand.W1.shape}):\n {uand.W1}")
    print(f"W2 ({uand.W2.shape}):\n {uand.W2}")
    # print(f"b1 ({uand.b1.shape}):\n {uand.b1}")
    # print(f"b2 ({uand.b2.shape}):\n {uand.b2}")
    # Get two random indices in d to use as two-hot vectors
    indices = torch.randint(0, d, (batch_size, 2))
    # Set the input as all zeros and ones in the indices
    x = torch.zeros(batch_size, d, device=device)
    x[torch.arange(batch_size), indices[:, 0]] = 1
    x[torch.arange(batch_size), indices[:, 1]] = 1

    out = uand(x)
    print(f"out ({out.shape}):\n {out}")
    gt = get_ground_truth(x).to(device)
    print(f"x ({x.shape}):\n {x}")
    # print(f"gt ({gt.shape}):\n {gt}")

    # mse_loss = F.mse_loss(out, gt)
    mse_loss = (out - gt).pow(2).sum(dim=-1).mean()
    print(f"mse_loss: {mse_loss}")

    # # Add training loop
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # output_dir = Path("out")
    # output_dir.mkdir(exist_ok=True)
    # batch_size = 10000
    # d = 1000

    # lr = 1e-2
    # n_batches = 1000
    # uand = UANDMLP(d).to(device)
    # # Set requires_grad to false on W1 and b1
    # uand.W1.requires_grad = False
    # uand.b1.requires_grad = False
    # print("Training UAND model...")
    # train_losses = train_uand(
    #     uand,
    #     d,
    #     batch_size=batch_size,
    #     n_batches=n_batches,
    #     learning_rate=lr,
    #     warmup_steps=100,
    #     device=device,
    # )

    # # Plot training loss
    # plt.figure(figsize=(10, 6))
    # plt.plot(train_losses)
    # plt.xlabel("Epoch")
    # plt.ylabel("MSE Loss")
    # plt.title("Training Loss over Epochs")
    # plt.grid(True)
    # plt.savefig(output_dir / "training_loss.png")
    # plt.close()

    # print(f"Training loss plot saved to {output_dir / 'training_loss.png'}")

    # # Evaluate final model performance
    # final_loss = train_losses[-1]
    # print(f"Final MSE loss after training: {final_loss:.4f}")

# %%
