import numpy as np


class Buffer:
    def __init__(self, state_dim, action_dim, total_size, batch_size, extend, seed=None):
        """
        Experience replay buffer for offline RL or imitation learning.

        Args:
            state_dim (int): Dimension of state vector.
            action_dim (int): Dimension of action vector.
            total_size (int): Maximum number of transitions to store.
            batch_size (int): Number of samples per training batch.
            extend (bool): Whether to dynamically grow buffer size.
            seed (int, optional): Random seed.
        """
        self.state_buf = np.zeros([total_size, state_dim], dtype=np.float32)
        self.action_buf = np.zeros([total_size, action_dim], dtype=np.float32)
        self.reward_buf = np.zeros([total_size], dtype=np.float32)
        self.next_state_buf = np.zeros([total_size, state_dim], dtype=np.float32)
        self.done_buf = np.zeros(total_size, dtype=np.float32)
        self.env_buf = np.zeros(total_size, dtype=np.float32)
        self.traj_num_buf = np.zeros(total_size, dtype=np.float32)

        self.total_size = total_size
        self.batch_size = batch_size
        self.extend = extend
        self.ptr = 0
        self.size = 0

        self.rng = np.random.default_rng(seed)

    def store(self, state, action, reward, next_state, done, env, traj_num):
        """
        Store a single transition in the buffer.
        """
        self.state_buf[self.ptr] = state
        self.action_buf[self.ptr] = action
        self.reward_buf[self.ptr] = reward
        self.next_state_buf[self.ptr] = next_state
        self.done_buf[self.ptr] = done
        self.env_buf[self.ptr] = env
        self.traj_num_buf[self.ptr] = traj_num

        if self.extend:
            self.ptr += 1
            self.size += 1
            self.total_size = max(self.size + 1, self.total_size)
        else:
            self.ptr = (self.ptr + 1) % self.total_size
            self.size = min(self.size + 1, self.total_size)

    def sample(self):
        """
        Sample a batch of transitions from the buffer.
        """
        if self.size >= self.batch_size:
            indices = self.rng.choice(self.size, size=self.batch_size, replace=False)
        else:
            indices = np.arange(self.size)
        return self.take_from(indices)

    def sample_all(self):
        """
        Return all stored transitions in the buffer.
        """
        return self.take_from(slice(None, None, None))

    def take_from(self, indices):
        """
        Fetch data by indices.
        """
        return dict(
            state=self.state_buf[indices],
            action=self.action_buf[indices],
            reward=self.reward_buf[indices],
            next_state=self.next_state_buf[indices],
            done=self.done_buf[indices],
            env=self.env_buf[indices],
            traj_num=self.traj_num_buf[indices],
        )

    def __len__(self):
        return self.size


def fill_buffer(
    trajs_path,
    total_size,
    batch_size,
    select_all,
    subsample_frequency,
    seed=None,
    trajs_num=None,
    env_traj_spec=None
):
    """
    Fill a buffer with trajectories loaded from disk.

    Args:
        trajs_path (str): Path to expert trajectory file (.npy).
        total_size (int): Total number of transitions the buffer can hold.
        batch_size (int): Number of samples per batch.
        select_all (bool): Whether to include all trajectories.
        subsample_frequency (int): Sample every n-th step in trajectory.
        seed (int, optional): Random seed.
        trajs_num (list[int], optional): Specific trajectory indices to include.
        env_traj_spec (dict[int, int], optional): Mapping from env ID to number of trajectories to include.
                      For example: {0: 5, 1: 3, 2: 7}
                      means load
                      5 trajectories from env 0,
                      3 from env 1,
                      7 from env 2.

    Returns:
        Buffer: Filled replay buffer.
    """
    trajs = np.load(trajs_path, allow_pickle=True)[()]
    rng = np.random.default_rng(seed)

    if select_all:
        selected_indices = np.arange(len(trajs['states']))
    elif trajs_num is not None:
        selected_indices = np.array(trajs_num)
    elif env_traj_spec is not None:
        selected_indices = []
        for env_id, num_trajs in env_traj_spec.items():
            env_candidates = [i for i in range(len(trajs['states'])) if trajs['Env_id'][i] == env_id]
            num_trajs = min(num_trajs, len(env_candidates))
            if num_trajs > 0:
                temp_indices = []
                available_candidates = env_candidates.copy()
                for _ in range(num_trajs):
                    if available_candidates:
                        chosen_idx = rng.choice(available_candidates)
                        temp_indices.append(chosen_idx)
                        available_candidates.remove(chosen_idx)
                selected_indices.extend(temp_indices)
        selected_indices = np.array(selected_indices)

    state_dim = trajs['states'][0][0].shape[0]
    action_dim = trajs['actions'][0][0].shape[0]
    buffer = Buffer(state_dim, action_dim, total_size, batch_size, extend=False, seed=seed)

    for i in selected_indices:
        traj_length = trajs['lengths'][i]
        sampled_indices = rng.choice(traj_length, size=traj_length // subsample_frequency, replace=False)
        for j in sorted(sampled_indices):
            buffer.store(
                state=trajs['states'][i][j],
                action=trajs['actions'][i][j],
                reward=trajs['rewards'][i][j],
                next_state=trajs['next_states'][i][j],
                done=trajs['dones'][i][j],
                env=trajs['Env_id'][i],
                traj_num=i,
            )

    return buffer
