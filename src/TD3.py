# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
from __future__ import print_function
import torch
import torch.nn.functional as F
from ModularActor import ActorGraphPolicy
from ModularCritic import CriticGraphPolicy
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TD3(object):

    def __init__(self, args):

        self.args = args
        self.actor = ActorGraphPolicy(args.limb_obs_size, 1,
                                      args.msg_dim, args.batch_size,
                                      args.max_action, args.max_children,
                                      args.disable_fold, args.td, args.bu).to(device)
        self.actor_target = ActorGraphPolicy(args.limb_obs_size, 1,
                                             args.msg_dim, args.batch_size,
                                             args.max_action, args.max_children,
                                             args.disable_fold, args.td, args.bu).to(device)
        self.critic = CriticGraphPolicy(args.limb_obs_size, 1,
                                        args.msg_dim, args.batch_size,
                                        args.max_children, args.disable_fold,
                                        args.td, args.bu).to(device)
        self.critic_target = CriticGraphPolicy(args.limb_obs_size, 1,
                                               args.msg_dim, args.batch_size,
                                               args.max_children, args.disable_fold,
                                               args.td, args.bu).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.lr)

    def change_morphology(self, graph):
        self.actor.change_morphology(graph)
        self.actor_target.change_morphology(graph)
        self.critic.change_morphology(graph)
        self.critic_target.change_morphology(graph)

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        with torch.no_grad():
            action = self.actor(state, 'inference').cpu().numpy().flatten()
            return action

    def train_single(self, replay_buffer, iterations, batch_size=100, discount=0.99,
                tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):

        for it in range(iterations):

            # sample replay buffer
            x, y, u, r, d = replay_buffer.sample(batch_size)
            state = torch.FloatTensor(x).to(device)
            next_state = torch.FloatTensor(y).to(device)
            action = torch.FloatTensor(u).to(device)
            reward = torch.FloatTensor(r).to(device)
            done = torch.FloatTensor(1 - d).to(device)

            # select action according to policy and add clipped noise
            with torch.no_grad():
                noise = torch.FloatTensor(u).data.normal_(0, policy_noise).to(device)
                noise = noise.clamp(-noise_clip, noise_clip)
                next_action = self.actor_target(next_state) + noise
                next_action = next_action.clamp(-self.args.max_action, self.args.max_action)

                # Qtarget = reward + discount * min_i(Qi(next_state, pi(next_state)))
                target_Q1, target_Q2 = self.critic_target(next_state, next_action)
                target_Q = torch.min(target_Q1, target_Q2)
                target_Q = reward + (done * discount * target_Q)

            # get current Q estimates
            current_Q1, current_Q2 = self.critic(state, action)

            # compute critic loss
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
            # optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # delayed policy updates
            if it % policy_freq == 0:

                # compute actor loss
                actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

                # optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # update the frozen target models
                for param, target_param in zip(self.critic.parameters(),
                                               self.critic_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                for param, target_param in zip(self.actor.parameters(),
                                               self.actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def train(self, replay_buffer_list, iterations_list, batch_size=100, discount=0.99,
              tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2, graphs=None, envs_train_names=None):
        per_morph_iter = sum(iterations_list) // len(envs_train_names)
        for env_name in envs_train_names:
            replay_buffer = replay_buffer_list[env_name]
            self.change_morphology(graphs[env_name])
            self.train_single(replay_buffer, per_morph_iter, batch_size=batch_size, discount=discount,
                tau=tau, policy_noise=policy_noise, noise_clip=noise_clip, policy_freq=policy_freq)

    def save(self, fname):
        torch.save(self.actor.state_dict(), '%s_actor.pth' % fname)
        torch.save(self.critic.state_dict(), '%s_critic.pth' % fname)

    def load(self, fname):
        self.actor.load_state_dict(torch.load('%s_actor.pth' % fname))
        self.critic.load_state_dict(torch.load('%s_critic.pth' % fname))
