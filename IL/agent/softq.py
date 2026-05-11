import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import Categorical
from network.Drop_Net import DropNetwork


class SoftQ(object):
    def __init__(self, obs_dim, action_dim, args):
        self.gamma = args.gamma
        self.device = torch.device(args.device)
        self.args = args
        self.actor = None
        self.critic_tau = args.agent.critic_tau

        self.critic_target_update_frequency = args.agent.critic_target_update_frequency
        self.log_alpha = torch.tensor(np.log(args.agent.init_temp)).to(self.device)

        self.q_net = DropNetwork(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=args.agent.hidden_dim,
            hidden_depth=args.agent.hidden_depth,
            args=args
        ).to(self.device)
        self.target_net = DropNetwork(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=args.agent.hidden_dim,
            hidden_depth=args.agent.hidden_depth,
            args=args
        ).to(self.device)

        self.target_net.load_state_dict(self.q_net.state_dict())
        self.critic_optimizer = Adam(self.q_net.parameters(), lr=args.agent.critic_lr,
                                     betas=args.agent.critic_betas)
        self.train()
        self.target_net.train()

    def train(self, training=True):
        self.training = training
        self.q_net.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    @property
    def critic_net(self):
        return self.q_net

    @property
    def critic_target_net(self):
        return self.target_net

    @property
    def causal_net(self):
        return self.q_net

    def getV(self, obs, mask_matrix):
        q = self.q_net(obs, mask_matrix)
        v = self.alpha * \
            torch.logsumexp(q / self.alpha, dim=1, keepdim=True)
        return v

    def critic(self, obs, action, mask_matrix):
        q = self.q_net(obs, mask_matrix)
        return q.gather(1, action.long())

    def get_targetV(self, obs, mask_matrix):
        q = self.target_net(obs, mask_matrix)
        target_v = self.alpha * \
                   torch.logsumexp(q / self.alpha, dim=1, keepdim=True)
        return target_v

    def choose_action(self, state,sample=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        with torch.no_grad():
            self.q_net.eval()
            q = self.q_net(state)
            dist = F.softmax(q / self.alpha, dim=1)
            if sample:
                dist = Categorical(dist)
                action = dist.sample()
            else:
                action = torch.argmax(dist, dim=1)
            self.q_net.train()
        return action.detach().cpu().numpy()[0]

    def get_value_estimates(self, expert_obs, expert_next_obs, expert_action, mask_matrix):
        args = self.args
        current_V = self.getV(expert_obs, mask_matrix)

        if args.train.use_target:
            with torch.no_grad():
                next_V = self.get_targetV(expert_next_obs, mask_matrix)
        else:
            next_V = self.getV(expert_next_obs, mask_matrix)

        current_Q = self.critic(expert_obs, expert_action, mask_matrix)

        return current_V, next_V, current_Q
