import os
import pickle
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def spurious_correlations_matrix(dims_c, dims_mu, xi_e, seed=None):
    if seed is not None:
        np.random.seed(seed)  # Set random seeds for reproducibility
    matrix = np.random.normal(loc=xi_e, scale=1.0, size=(dims_c, dims_mu))
    matrix = np.clip(matrix, xi_e - 3, xi_e + 3)
    return matrix


def generate_obs_traj(state, multiplicative_matrix, spur_corr_init, noise_mean=0, seed=None):
    if seed is not None:
        np.random.seed(seed)  # Set random seeds for reproducibility
    dim_c, dim_mu = multiplicative_matrix.shape
    obs_traj = []

    for i in range(len(state)):
        state_dim = len(state[i][:dim_c])
        obs_noise = np.random.randn(state_dim) * noise_mean  # Generate noise

        # Use state from previous time step (t-1) to compute spur_corr
        if i == 0:
            # For the first step, spur_corr could be zero or based on some initial state
            # Here, we assume spur_corr is just spur_corr_init (no state[-1] yet)
            spur_corr = np.matmul(spur_corr_init[:dim_c], multiplicative_matrix)
        else:
            # Use the last dim_c elements of the previous state
            spur_corr = np.matmul(state[i - 1][:dim_c], multiplicative_matrix)

        # Form observation: state + noise, spur_corr_init + spur_corr
        obs = np.concatenate([state[i] + obs_noise, spur_corr])
        obs_traj.append(obs)

    return obs_traj


def soft_update(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data +
                                (1 - tau) * target_param.data)


def hard_update(source, target):
    for param, target_param in zip(source.parameters(), target.parameters()):
        target_param.data.copy_(param.data)


def average_dicts(dict1, dict2):
    return {key: 1 / 2 * (dict1.get(key, 0) + dict2.get(key, 0))
            for key in set(dict1) | set(dict2)}


def load_data(file_base_path):

    npy_path = f'{file_base_path}.npy'
    pkl_path = f'{file_base_path}.pkl'

    if os.path.exists(npy_path):
        print(f'Loading from {npy_path}')
        data = np.load(npy_path, allow_pickle=True)[()]
    elif os.path.exists(pkl_path):
        print(f'Loading from {pkl_path}')
        with open(pkl_path, 'rb') as file:
            data = pickle.load(file)
    else:
        raise FileNotFoundError(f'Neither {npy_path} nor {pkl_path} exists.')

    return data


def save_experiment_results(args, rewards, agent, rewards_dir):
    # Create base directory if it doesn't exist
    base_reward_dir = os.path.join(os.getcwd(), rewards_dir)
    os.makedirs(base_reward_dir, exist_ok=True)

    # Create subdirectory with experiment metadata
    subfolder_name = f"{args.env.name}_Envnum{args.n_env}_spurdims{args.extra_dims}_traj{args.buffer.traj_num}_K{args.K}"
    subfolder_path = os.path.join(base_reward_dir, subfolder_name)
    os.makedirs(subfolder_path, exist_ok=True)

    # Save rewards
    reward_path = os.path.join(subfolder_path, "reward.npy")
    np.save(reward_path, np.array(rewards))

    # Save agent network parameters
    if args.agent.name == "softq":
        critic_path = os.path.join(subfolder_path, "critic_net.pth")
        torch.save(agent.causal_net.state_dict(), critic_path)


def plot_heatmap(weights, cmap='viridis'):
    # Set seaborn style for aesthetics
    sns.set(style="whitegrid", palette="muted")

    # Create the heatmap plot
    plt.figure(figsize=(5, 5))
    sns.heatmap(weights, vmin=-0.3, vmax=0.3, cmap=cmap, annot=False, cbar=True)

    # Axis labels and title
    plt.xlabel("Input Layer")
    plt.ylabel("First Hidden Layer")
    plt.title("Heatmap of First Hidden Layer Weights")
    plt.tight_layout()
    plt.show()

