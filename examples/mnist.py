"""
MNIST Digit Classification using Echo State Network (ESN)

This script demonstrates how to use the PyTorch ESN library with larger datasets
that require batching to fit in memory, using MNIST as an example.

Key Features:
- Shows ESN usage with datasets too large to fit entirely in memory
- Demonstrates proper batching techniques for ESN training and inference
- Uses PyTorch DataLoader for efficient memory management
- Implements batch-wise processing for both training and evaluation
- Handles memory constraints through configurable batch sizes

The script processes MNIST's 60,000 training samples in batches, showing how ESNs
can scale to larger datasets without loading everything into memory at once.

Functions:
    Accuracy_Correct(y_pred, y_true): Calculates the number of correct predictions
    one_hot(y, output_dim): Converts integer labels to one-hot encoded vectors
    reshape_batch(batch): Reshapes image batches for sequential processing

Parameters:
    batch_size (int): Number of samples per batch - tune according to available memory
    input_size (int): Dimensionality of input features (784)
    hidden_size (int): Number of neurons in the ESN reservoir (1000)
    output_size (int): Number of output classes (10 for digits 0-9)
    washout_rate (float): Fraction of initial timesteps to discard (0.2)

NOTE: THIS IS A DEMONSTRATION OF LIBRARY USAGE WITH LARGER DATASETS REQUIRING BATCHING.
THE HYPERPARAMETERS ARE NOT OPTIMIZED FOR ACCURACY.
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import argparse
import torch
from torchvision import datasets, transforms
from torchesn.nn import ESN
import matplotlib.pyplot as plt
import time


def save_figure(fig, output_path_base):
    output_dir = os.path.dirname(output_path_base)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(f"{output_path_base}.{ext}", bbox_inches="tight")


def Accuracy_Correct(y_pred, y_true):
    labels = torch.argmax(y_pred, 1)
    correct = (labels == y_true).sum().item()
    return correct


def one_hot(y, output_dim):
    onehot = torch.zeros(y.size(0), output_dim, device=y.device)

    for i in range(output_dim):
        onehot[y == i, i] = 1

    return onehot


def resolve_sequence_params(sequence_mode, time_steps, image_height, image_width):
    flattened = image_height * image_width
    if sequence_mode == "flat":
        return 1, flattened
    if sequence_mode == "rows":
        return image_height, image_width
    if sequence_mode == "cols":
        return image_width, image_height
    if sequence_mode == "time_steps":
        if time_steps is None:
            raise ValueError("time_steps must be set when sequence_mode='time_steps'.")
        if flattened % time_steps != 0:
            raise ValueError(
                f"time_steps={time_steps} must divide {flattened} for MNIST."
            )
        return time_steps, flattened // time_steps
    raise ValueError(f"Unsupported sequence_mode: {sequence_mode}")


def reshape_batch(batch, sequence_mode, time_steps=None):
    batch = batch.view(batch.size(0), 28, 28)
    if sequence_mode == "flat":
        batch = batch.view(batch.size(0), -1)
        return batch.unsqueeze(0)
    if sequence_mode == "rows":
        seq = batch
    elif sequence_mode == "cols":
        seq = batch.transpose(1, 2)
    elif sequence_mode == "time_steps":
        flattened = batch.view(batch.size(0), -1)
        seq = flattened.view(batch.size(0), time_steps, -1)
    else:
        raise ValueError(f"Unsupported sequence_mode: {sequence_mode}")
    return seq.permute(1, 0, 2).contiguous()


def aggregate_output(output, output_steps):
    if output_steps == 'mean':
        return output.mean(dim=0)
    if output_steps == 'last':
        return output[-1]
    raise ValueError(f"Unsupported output_steps for classification: {output_steps}")


def build_arg_parser():
    parser = argparse.ArgumentParser(description="MNIST ESN classifier")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-size", type=int, default=1000)
    parser.add_argument("--washout-rate", type=float, default=0.2)
    parser.add_argument("--eval-output-steps", choices=["mean", "last"], default="mean")
    parser.add_argument(
        "--sequence-mode",
        choices=["flat", "rows", "cols", "time_steps"],
        default="flat",
    )
    parser.add_argument("--time-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--readout-training", default="cholesky")
    parser.add_argument("--output-steps", choices=["all", "mean", "last"], default="mean")
    parser.add_argument("--spectral-radius", type=float, default=0.9)
    parser.add_argument("--leaking-rate", type=float, default=1.0)
    parser.add_argument("--w-ih-scale", type=float, default=1.0)
    parser.add_argument("--lambda-reg", type=float, default=0.0)
    parser.add_argument("--density", type=float, default=1.0)
    parser.add_argument("--w-io", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    return parser

if __name__ == "__main__":
    args = build_arg_parser().parse_args()

    if args.device == "auto":
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    dtype = torch.float
    torch.set_default_dtype(dtype)
    loss_fcn = Accuracy_Correct

    torch.manual_seed(args.seed)

    image_height = 28
    image_width = 28
    output_size = 10
    seq_len, input_size = resolve_sequence_params(
        args.sequence_mode, args.time_steps, image_height, image_width
    )
    data_path = args.data_path or os.path.join(os.path.dirname(__file__), 'datasets')
    train_iter = torch.utils.data.DataLoader(
        datasets.MNIST(data_path, train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))])),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        pin_memory=args.pin_memory or torch.cuda.is_available())

    test_iter = torch.utils.data.DataLoader(
        datasets.MNIST(data_path, train=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))])),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        pin_memory=args.pin_memory or torch.cuda.is_available())

    start = time.time()

    # Training
    model = ESN(
        input_size,
        args.hidden_size,
        output_size,
        output_steps=args.output_steps,
        readout_training=args.readout_training,
        w_io=args.w_io,
        spectral_radius=args.spectral_radius,
        leaking_rate=args.leaking_rate,
        w_ih_scale=args.w_ih_scale,
        lambda_reg=args.lambda_reg,
        density=args.density,
    )
    model.to(device)

    # Fit the model
    for batch in train_iter:
        x, y = batch
        x = x.to(device)
        y = y.to(device)

        x = reshape_batch(x, args.sequence_mode, args.time_steps)
        target = one_hot(y, output_size)
        washout_list = [int(args.washout_rate * x.size(0))] * x.size(1)

        model(x, washout_list, None, target)

    model.fit()

    # Evaluate on training set (optional, can be slow on CPU)
    if os.environ.get("EVAL_TRAINING") == "1":
        tot_correct = 0
        tot_obs = 0

        for batch in train_iter:
            x, y = batch
            x = x.to(device)
            y = y.to(device)

            x = reshape_batch(x, args.sequence_mode, args.time_steps)
            washout_list = [int(args.washout_rate * x.size(0))] * x.size(1)

            output, hidden = model(x, washout_list)
            logits = aggregate_output(output, args.eval_output_steps)
            tot_obs += x.size(1)
            tot_correct += loss_fcn(logits, y)

        print("Training accuracy:", tot_correct / tot_obs)

    # Test
    tot_correct = 0
    tot_obs = 0
    viz_images = None
    viz_labels = None
    viz_preds = None

    for batch in test_iter:
        x, y = batch
        x = x.to(device)
        y = y.to(device)

        x_images = x.detach().cpu()
        x_seq = reshape_batch(x, args.sequence_mode, args.time_steps)
        washout_list = [int(args.washout_rate * x_seq.size(0))] * x_seq.size(1)

        output, hidden = model(x_seq, washout_list)
        logits = aggregate_output(output, args.eval_output_steps)
        tot_obs += x_seq.size(1)
        tot_correct += loss_fcn(logits, y)
        if viz_images is None:
            viz_images = x_images
            viz_labels = y.detach().cpu()
            viz_preds = torch.argmax(logits.detach().cpu(), dim=1)

    print("Test accuracy:", tot_correct / tot_obs)

    print("Ended in", time.time() - start, "seconds.")

    if viz_images is not None:
        num_images = min(16, viz_images.size(0))
        fig, axes = plt.subplots(4, 4, figsize=(6, 6))
        for idx in range(num_images):
            ax = axes[idx // 4, idx % 4]
            image = viz_images[idx].squeeze().numpy()
            ax.imshow(image, cmap="gray")
            ax.set_title(f"P:{viz_preds[idx].item()} T:{viz_labels[idx].item()}")
            ax.axis("off")
        plt.suptitle("MNIST Predictions (P: Predicted, T: True)")
        plt.tight_layout()
        save_figure(fig, os.path.join(os.path.dirname(__file__), "figures", "mnist_predictions"))
        if os.environ.get("SHOW_PLOTS") == "1":
            plt.show()
