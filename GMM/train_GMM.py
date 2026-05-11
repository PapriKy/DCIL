import argparse
from data import envgen, training_eval, testing
import torch
import torch.nn as nn
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import random

mse = torch.nn.MSELoss(reduction="none")
torch.set_default_dtype(torch.float64)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def get_dropout_mask(net, dropout_prob=0.3, scale=1):
    first_layer = net.input_layer
    weight_shape = first_layer.weight.shape
    device = first_layer.weight.device
    probs_matrix = torch.full(weight_shape, scale * dropout_prob, device=device)  # 确保在正确设备上
    # Detach to avoid tracking gradients
    probs_matrix = probs_matrix.detach()
    # Generate dropout mask
    mask = torch.bernoulli(1.0 - probs_matrix)
    mask = mask / (1.0 - probs_matrix)  # Inverted dropout scaling

    return mask


class Network_gmm(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth, device='cpu'):
        super(Network_gmm, self).__init__()
        self.device = torch.device(device) if isinstance(device, str) else device
        self.input_layer = nn.Linear(obs_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(hidden_depth - 1)])
        self.output_layer = nn.Linear(hidden_dim, action_dim)
        self.activation = nn.Sigmoid()
        self._init_weights()

    def forward(self, x, mask=None):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.device, dtype=torch.float64)
        else:
            x = x.to(self.device, dtype=torch.float64)

        if mask is not None:
            weighted_input = self.input_layer.weight * mask
        else:
            weighted_input = self.input_layer.weight
        x = torch.matmul(x, weighted_input.t()) + self.input_layer.bias
        x = self.activation(x)
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        q = self.output_layer(x)
        return q

    def _init_weights(self):
        for layer in [self.input_layer] + list(self.hidden_layers) + [self.output_layer]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)


def DCIL(net, loss):
    target_norm = 1
    epsilon = 1e-8
    cc_loss = 0
    cc_loss_e = 0
    phi_grad = torch.autograd.grad(loss, net.parameters(), create_graph=True)
    first_layer = net.input_layer
    first_layer_params = list(first_layer.parameters())
    phi_grad_first_layer = phi_grad[:len(first_layer_params)]

    for i, phi in enumerate(first_layer_params):
        grad = phi_grad_first_layer[i]
        if i == 0:
            phi_norms = torch.norm(phi, p=2, dim=0)
            scaling_factors = (target_norm / (phi_norms + epsilon)).detach()
            adjusted_norms = phi_norms * scaling_factors
            cc_loss_e += torch.mean(torch.square(grad * adjusted_norms))
        else:
            cc_loss_e += torch.mean(torch.square(phi * grad))
    cc_loss += torch.sqrt(cc_loss_e)
    return cc_loss_e


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GMM')
    parser.add_argument('--N', type=int, default=1000, help='data in one component')
    parser.add_argument('--K', type=int, default=5, help='number of component')
    parser.add_argument('--sigma', type=float, default=1., help='std of GMM')
    parser.add_argument('--lmbd', type=float, default=30.0, help='weight for coco term')
    parser.add_argument('--lmbd_irm', type=float, default=100.0, help='weight for irm')
    parser.add_argument('--lmbd_rex', type=float, default=10000.0, help='weight for rex')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--n_env', type=int, default=2)
    parser.add_argument('--steps', type=int, default=5001)
    parser.add_argument('--spurious', type=bool, default=True)
    parser.add_argument('--seed', type=int, default=3, help='Random seed')
    parser.add_argument('--path', default='results/', help='The path results to be saved.')
    parser.add_argument('--method', default='DCIL', help='ERM, IRM, REx, CoCo,DCIL')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    path = os.path.join(args.path, f'gmm_{args.method}_{args.seed}')
    if not os.path.exists(path):
        os.makedirs(path)
    dim_in = args.K + (args.K // 2 + 1) if args.spurious else args.K
    dim_out = args.K
    dims = [dim_in, 10, 10, dim_out]
    net = Network_gmm(dim_in, dim_out, hidden_dim=10, hidden_depth=2, device=device).to(device)  # 移到 GPU

    if args.method == 'IRM':
        dummy_w = torch.nn.Parameter(torch.tensor([1.], device=device))
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    envs = envgen(args)

    test_r = []
    train_r = []
    iter_r = []

    for epoch in range(args.steps):
        if ((args.method == 'CoCo') or (args.method == 'ERM')):
            risk = 0
            coco_loss = 0  # over envs
            for [inputs, labels] in envs:
                inputs = torch.tensor(inputs, device=device)
                labels = torch.as_tensor(labels, dtype=torch.long, device=device)
                outputs = net(inputs)
                risk_e = criterion(outputs, labels)
                risk += risk_e
                phi_grad = torch.autograd.grad(risk_e, net.parameters(), create_graph=True)
                coco_loss_e = 0
                for i, phi in enumerate(net.parameters()):
                    coco_loss_e += torch.mean(torch.square(phi * phi_grad[i]))
                coco_loss += torch.sqrt(coco_loss_e)
            risk = risk / len(envs)
            coco_loss = coco_loss / len(envs)

            optimizer.zero_grad()
            if args.method == 'CoCo':
                lmbd_risk = 1 if epoch < (args.steps / 2) else 0.1
                tot_loss = lmbd_risk * risk + args.lmbd * coco_loss if args.spurious else risk
            if args.method == 'ERM':
                tot_loss = risk
            tot_loss.backward()
            optimizer.step()

        if (args.method == 'IRM'):
            risk = 0
            loss = 0
            for [inputs, labels] in envs:
                inputs = torch.tensor(inputs, device=device)
                labels = torch.as_tensor(labels, dtype=torch.long, device=device)
                outputs = net(inputs)

                risk_e = criterion(outputs * dummy_w, labels)
                w_grad = torch.autograd.grad(risk_e, \
                                             dummy_w, create_graph=True)[0]
                loss_e = torch.square(w_grad)
                loss += loss_e
                risk += risk_e

            optimizer.zero_grad()
            risk = risk / len(envs)
            loss = loss / len(envs)

            lmbd_risk = 1 if epoch < (args.steps / 2) else 0.1
            tot_loss = risk * lmbd_risk + args.lmbd_irm * loss
            tot_loss.backward()
            optimizer.step()

        if (args.method == 'REx'):
            risk = 0
            loss = 0
            risks = []
            for [inputs, labels] in envs:
                inputs = torch.tensor(inputs, device=device)
                labels = torch.as_tensor(labels, dtype=torch.long, device=device)
                outputs = net(inputs)

                risk_e = criterion(outputs, labels)
                risks.append(risk_e)
                risk += risk_e
            optimizer.zero_grad()
            risk = risk / len(envs)
            loss = torch.stack(risks).var()

            lmbd_risk = 1 if epoch < (args.steps / 2) else 0.1
            tot_loss = lmbd_risk * risk + args.lmbd_rex * loss
            tot_loss.backward()
            optimizer.step()

        if (args.method == 'DCIL'):
            risk = 0
            cc_loss = 0  # over envs
            scale = 1 - epoch / args.steps
            mask = get_dropout_mask(net)
            for [inputs, labels] in envs:
                inputs = torch.tensor(inputs, device=device)
                labels = torch.as_tensor(labels, dtype=torch.long, device=device)

                outputs = net(inputs, mask)
                risk_e = criterion(outputs, labels)
                risk += risk_e
                cc_loss_e = DCIL(net, risk_e)
                cc_loss += torch.sqrt(cc_loss_e)

            risk = risk / len(envs)
            cc_loss = cc_loss / len(envs)

            optimizer.zero_grad()
            if args.method == 'DCIL':
                lmbd_risk = 1 if epoch < (args.steps / 2) else 0.1
                tot_loss = lmbd_risk * risk + 0.5 * args.lmbd * cc_loss
            tot_loss.backward()
            optimizer.step()

        if epoch % 100 == 0:
            print('epoch', epoch, '########################', flush=True)
            test_perform = testing(net, args)
            train_perform = training_eval(net, envs)
            test_r.append([np.mean(test_perform), np.std(test_perform)])
            train_r.append([np.mean(train_perform), np.std(train_perform)])
            iter_r.append(epoch)
    # Save both mean and standard deviation
    test_r = np.array(test_r)
    train_r = np.array(train_r)
    iter_r = np.array(iter_r)

    # Save to pickle file
    pickle.dump([iter_r, train_r, test_r], open(os.path.join(path, 'gmm_' + str(args.seed) + '.pkl'), 'wb'))

    # Load the saved data for plotting
    with open(os.path.join(path, 'gmm_' + str(args.seed) + '.pkl'), 'rb') as f:
        iter_r, train_r, test_r = pickle.load(f)

    # Prepare data for Seaborn
    data = {
        'Epoch': np.concatenate([iter_r, iter_r]),
        'Mean': np.concatenate([train_r[:, 0], test_r[:, 0]]),
        'Std': np.concatenate([train_r[:, 1], test_r[:, 1]]),
        'Dataset': ['Train'] * len(iter_r) + ['Test'] * len(iter_r)
    }
    df = pd.DataFrame(data)

    # Plot using Seaborn
    plt.rcParams.update({
        'font.family': 'Times New Roman',
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12
    })

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='Epoch', y='Mean', hue='Dataset', style='Dataset', markers=True, dashes=False)
    # Add error bars manually
    for dataset in df['Dataset'].unique():
        subset = df[df['Dataset'] == dataset]
        plt.fill_between(subset['Epoch'], subset['Mean'] - subset['Std'], subset['Mean'] + subset['Std'], alpha=0.2)

    plt.title('Training and Test Performance Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Performance (Mean ± Std)')
    plt.grid(True)
    plt.savefig(os.path.join(path, 'performance_plot.png'))
    # plt.close()

    weights = net.input_layer.weight.data.cpu().numpy()
    print("Weight matrix shape:", weights.shape)  # Should be 10x8

    # Calculate L2 norm of each column
    column_norms = np.linalg.norm(weights, axis=0)
    print("L2 norms of weight matrix columns:", column_norms)

    plt.figure(figsize=(5, 4))
    sns.heatmap(weights, vmin=-2, vmax=2, cmap="viridis", annot=False, cbar=True)
    plt.xlabel("Input layer")
    plt.ylabel("First hidden layer")
    plt.title("Heatmap of First Hidden Layer Weights")
    # Save as .jpg
    jpg_path = os.path.join(path, f'weights_heatmap_seed{args.seed}.jpg')
    plt.savefig(jpg_path, dpi=300)
    print("Saved heatmap as JPG to:", jpg_path)

    # Save as vector graphic (e.g., PDF or SVG)
    vector_path = os.path.join(path, f'weights_heatmap_seed{args.seed}.svg')
    plt.savefig(vector_path)
    print("Saved heatmap as vector graphic to:", vector_path)