"""
Mackey-Glass Time Series Prediction using Echo State Network (ESN).

This script demonstrates the use of an Echo State Network for predicting the Mackey-Glass
chaotic time series. The Mackey-Glass equation is a well-known benchmark problem in
nonlinear time series prediction and chaos theory.

The script performs the following operations:
1. Loads the Mackey-Glass dataset (mg17.csv) containing input-output pairs
2. Splits the data into training (first 5000 samples) and test sets
3. Creates and trains an ESN model with specified parameters
4. Evaluates the model performance on both training and test data
5. Reports training and test errors along with execution time

Parameters:
    - input_size: Dimension of input data (1 for univariate time series)
    - hidden_size: Number of reservoir neurons (500)
    - output_size: Dimension of output prediction (1)
    - washout: Number of initial samples to discard during training (500)
    - device: Computation device (CPU)
    - dtype: Data type for tensors (torch.double)

The ESN uses a washout period to allow the reservoir dynamics to stabilize before
training begins. The model is trained using the fit() method and evaluated using
Mean Squared Error (MSE) loss.

Expected output:
    - Training error: MSE on training data after washout period
    - Test error: MSE on test data
    - Execution time in seconds

Dataset:
    The mg17.csv file should contain two columns representing the input and target
    values of the Mackey-Glass time series with tau=17.
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import argparse
import torch
import torch.nn
import numpy as np
import matplotlib.pyplot as plt
from torchesn.nn import ESN
from torchesn import utils
import time


def save_figure(fig, output_path_base):
    output_dir = os.path.dirname(output_path_base)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(f"{output_path_base}.{ext}", bbox_inches="tight")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Mackey-Glass Time Series Prediction using ESN"
    )
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--train-size", type=int, default=5000)
    parser.add_argument("--washout", type=int, default=500)
    parser.add_argument("--hidden-size", type=int, default=500)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--dtype", choices=["float", "double"], default="double")
    parser.add_argument("--spectral-radius", type=float, default=0.9)
    parser.add_argument("--leaking-rate", type=float, default=1.0)
    parser.add_argument("--w-ih-scale", type=float, default=1.0)
    parser.add_argument("--lambda-reg", type=float, default=0.0)
    parser.add_argument("--density", type=float, default=1.0)
    parser.add_argument("--readout-training", default="svd")
    parser.add_argument("--output-steps", choices=["all", "mean", "last"], default="all")
    parser.add_argument("--show-plots", action="store_true")
    args = parser.parse_args()

    device = (
        torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if args.device == "auto"
        else torch.device(args.device)
    )
    dtype = torch.double if args.dtype == "double" else torch.float
    torch.set_default_dtype(dtype)

    data_path = args.data_path or os.path.join(os.path.dirname(__file__), 'datasets/mg17.csv')
    if dtype == torch.double:
        data = np.loadtxt(data_path, delimiter=',', dtype=np.float64)
    elif dtype == torch.float:
        data = np.loadtxt(data_path, delimiter=',', dtype=np.float32)
    X_data = np.expand_dims(data[:, [0]], axis=1)
    Y_data = np.expand_dims(data[:, [1]], axis=1)
    X_data = torch.from_numpy(X_data).to(device)
    Y_data = torch.from_numpy(Y_data).to(device)

    if args.train_size <= 0 or args.train_size >= X_data.size(0):
        raise ValueError("train-size must be > 0 and < total samples.")

    trX = X_data[:args.train_size]
    trY = Y_data[:args.train_size]
    tsX = X_data[args.train_size:]
    tsY = Y_data[args.train_size:]

    washout = [args.washout]
    input_size = output_size = 1
    loss_fcn = torch.nn.MSELoss()

    start = time.time()

    # Training
    trY_flat = utils.prepare_target(trY.clone(), [trX.size(0)], washout)

    model = ESN(
        input_size,
        args.hidden_size,
        output_size,
        output_steps=args.output_steps,
        readout_training=args.readout_training,
        spectral_radius=args.spectral_radius,
        leaking_rate=args.leaking_rate,
        w_ih_scale=args.w_ih_scale,
        lambda_reg=args.lambda_reg,
        density=args.density,
    )
    model.to(device)

    model(trX, washout, None, trY_flat)
    model.fit()
    output, hidden = model(trX, washout)
    print("Training error:", loss_fcn(output, trY[washout[0]:]).item())

    # Test
    output, hidden = model(tsX, [0], hidden)
    print("Test error:", loss_fcn(output, tsY).item())
    print("Ended in", time.time() - start, "seconds.")

    predictions = output.detach().cpu().squeeze().numpy()
    targets = tsY.detach().cpu().squeeze().numpy()

    fig = plt.figure(figsize=(10, 4))
    plt.plot(targets, label="Target")
    plt.plot(predictions, label="Prediction")
    plt.title("Mackey-Glass Time Series Prediction")
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    save_figure(fig, os.path.join(os.path.dirname(__file__), "figures", "mackey_glass_prediction"))
    if args.show_plots:
        plt.show()
