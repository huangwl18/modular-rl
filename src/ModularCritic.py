from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import MLPBase
import torchfold
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CriticVanilla(nn.Module):
    """a vanilla critic module that outputs a node's q-values given only its observation and action(no message between nodes)"""
    def __init__(self, state_dim, action_dim):
        super(CriticVanilla, self).__init__()
        self.baseQ1 = MLPBase(state_dim + action_dim, 1)
        self.baseQ2 = MLPBase(state_dim + action_dim, 1)

    def forward(self, x, u):
        xu = torch.cat([x, u], -1)
        x1 = self.baseQ1(xu)
        x2 = self.baseQ2(xu)
        return x1, x2

    def Q1(self, x, u):
        xu = torch.cat([x, u], -1)
        x1 = self.baseQ1(xu)
        return x1


class CriticUp(nn.Module):
    """a bottom-up module used in bothway message passing that only passes message to its parent"""
    def __init__(self, state_dim, action_dim, msg_dim, max_children):
        super(CriticUp, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 64)
        self.fc2 = nn.Linear(64 + msg_dim * max_children, 64)
        self.fc3 = nn.Linear(64, msg_dim)

    def forward(self, x, u, *m):
        m = torch.cat(m, dim=-1)
        xu = torch.cat([x, u], dim=-1)
        xu = self.fc1(xu)
        xu = F.normalize(xu, dim=-1)
        xum = torch.cat([xu, m], dim=-1)
        xum = torch.tanh(xum)
        xum = self.fc2(xum)
        xum = torch.tanh(xum)
        xum = self.fc3(xum)
        xum = F.normalize(xum, dim=-1)
        msg_up = xum

        return msg_up


class CriticUpAction(nn.Module):
    """a bottom-up module used in bottom-up-only message passing that passes message to its parent and outputs q-values"""
    def __init__(self, state_dim, action_dim, msg_dim, max_children):
        super(CriticUpAction, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 64)
        self.fc2 = nn.Linear(64 + msg_dim * max_children, 64)
        self.fc3 = nn.Linear(64, msg_dim)
        self.baseQ1 = MLPBase(state_dim + action_dim + msg_dim * max_children, 1)
        self.baseQ2 = MLPBase(state_dim + action_dim + msg_dim * max_children, 1)

    def forward(self, x, u, *m):
        m = torch.cat(m, dim=-1)
        xum = torch.cat([x, u, m], dim=-1)

        x1 = self.baseQ1(xum)
        x2 = self.baseQ2(xum)

        xu = torch.cat([x, u], dim=-1)
        xu = self.fc1(xu)
        xu = F.normalize(xu, dim=-1)
        xum = torch.cat([xu, m], dim=-1)
        xum = torch.tanh(xum)
        xum = self.fc2(xum)
        xum = torch.tanh(xum)
        xum = self.fc3(xum)
        xum = F.normalize(xum, dim=-1)
        msg_up = xum

        return msg_up, x1, x2

    def Q1(self, x, u, *m):
        m = torch.cat(m, dim=-1)
        xum = torch.cat([x, u, m], dim=-1)
        x1 = self.baseQ1(xum)

        xu = torch.cat([x, u], dim=-1)
        xu = self.fc1(xu)
        xu = F.normalize(xu, dim=-1)
        xum = torch.cat([xu, m], dim=-1)
        xum = torch.tanh(xum)
        xum = self.fc2(xum)
        xum = torch.tanh(xum)
        xum = self.fc3(xum)
        xum = F.normalize(xum, dim=-1)
        msg_up = xum

        return msg_up, x1


class CriticDownAction(nn.Module):
    """a top-down module used in bothway message passing that passes messages to children and outputs q-values"""
    # input dim is state dim if only using top down message passing
    # if using bottom up and then top down, it is the node's outgoing message dim
    def __init__(self, self_input_dim, action_dim, msg_dim, max_children):
        super(CriticDownAction, self).__init__()
        self.baseQ1 = MLPBase(self_input_dim + action_dim + msg_dim, 1)
        self.baseQ2 = MLPBase(self_input_dim + action_dim + msg_dim, 1)
        self.msg_base = MLPBase(self_input_dim + msg_dim, msg_dim * max_children)

    def forward(self, x, u, m):
        xum = torch.cat([x, u, m], dim=-1)
        x1 = self.baseQ1(xum)
        x2 = self.baseQ2(xum)
        xm = torch.cat([x, m], dim=-1)
        xm = torch.tanh(xm)
        msg_down = self.msg_base(xm)
        msg_down = F.normalize(msg_down, dim=-1)

        return x1, x2, msg_down

    def Q1(self, x, u, m):
        xum = torch.cat([x, u, m], dim=-1)
        x1 = self.baseQ1(xum)
        xm = torch.cat([x, m], dim=-1)
        xm = torch.tanh(xm)
        msg_down = self.msg_base(xm)
        msg_down = F.normalize(msg_down, dim=-1)

        return x1, msg_down


class CriticGraphPolicy(nn.Module):
    """a weight-sharing dynamic graph policy that changes its structure based on different morphologies and passes messages between nodes"""
    def __init__(self, state_dim, action_dim, msg_dim, batch_size, max_children, disable_fold, td, bu):
        super(CriticGraphPolicy, self).__init__()
        self.num_limbs = 1
        self.x1 = [None] * self.num_limbs
        self.x2 = [None] * self.num_limbs
        self.input_state = [None] * self.num_limbs
        self.input_action = [None] * self.num_limbs
        self.msg_down = [None] * self.num_limbs
        self.msg_up = [None] * self.num_limbs
        self.msg_dim = msg_dim
        self.batch_size = batch_size
        self.max_children = max_children
        self.disable_fold = disable_fold
        self.state_dim = state_dim
        self.action_dim = action_dim

        assert self.action_dim == 1
        self.td = td
        self.bu = bu
        if self.bu:
            # bottom-up then top-down
            if self.td:
                self.sNet = nn.ModuleList([CriticUp(state_dim, action_dim, msg_dim, max_children)] * self.num_limbs).to(device)
            # bottom-up only
            else:
                self.sNet = nn.ModuleList([CriticUpAction(state_dim, action_dim, msg_dim, max_children)] * self.num_limbs).to(device)
            if not self.disable_fold:
                for i in range(self.num_limbs):
                    setattr(self, "sNet" + str(i).zfill(3), self.sNet[i])
        # we pass msg_dim as first argument because in both-way message-passing, each node takes in its passed-up message as 'state'
        if self.td:
            # bottom-up then top-down
            if self.bu:
                self.critic = nn.ModuleList([CriticDownAction(msg_dim, action_dim, msg_dim, max_children)] * self.num_limbs).to(device)
            # top-down only
            else:
                self.critic = nn.ModuleList([CriticDownAction(state_dim, action_dim, msg_dim, max_children)] * self.num_limbs).to(device)
            if not self.disable_fold:
                for i in range(self.num_limbs):
                    setattr(self, "critic" + str(i).zfill(3), self.critic[i])

        # no message passing
        if not self.bu and not self.td:
            self.critic = nn.ModuleList([CriticVanilla(state_dim, action_dim)] * self.num_limbs).to(device)
            if not self.disable_fold:
                for i in range(self.num_limbs):
                    setattr(self, "critic" + str(i).zfill(3), self.critic[i])

        if not self.disable_fold:
            for i in range(self.max_children):
                setattr(self, 'get_{}'.format(i), self.addFunction(i))

    def forward(self, state, action):
        self.clear_buffer()
        if not self.disable_fold:
            self.fold = torchfold.Fold()
            self.fold.cuda()
            self.zeroFold_td = self.fold.add("zero_func_td")
            self.zeroFold_bu = self.fold.add("zero_func_bu")
            self.x1_fold, self.x2_fold = [], []
        assert state.shape[1] == self.state_dim * self.num_limbs, 'state.shape[1] expects {} but got {} with num_limbs being {} and state_dim being {}'.format(self.state_dim * self.num_limbs, state.shape[1], self.num_limbs, self.state_dim)
        for i in range(self.num_limbs):
            self.input_state[i] = state[:, i * self.state_dim:(i + 1) * self.state_dim]
            self.input_action[i] = action[:, i]
            self.input_action[i] = torch.unsqueeze(self.input_action[i], -1)
            if not self.disable_fold:
                self.input_state[i] = torch.unsqueeze(self.input_state[i], 0)
                self.input_action[i] = torch.unsqueeze(self.input_action[i], 0)

        if self.bu:
            # bottom up transmission by recursion
            for i in range(self.num_limbs):
                self.bottom_up_transmission(i)

        if self.td:
            # top down transmission by recursion
            for i in range(self.num_limbs):
                self.top_down_transmission(i)

        if not self.bu and not self.td:
            for i in range(self.num_limbs):
                if not self.disable_fold:
                    self.x1[i], self.x2[i] = self.fold.add('critic' + str(0).zfill(3), self.input_state[i], self.input_action[i]).split(2)
                else:
                    self.x1[i], self.x2[i] = self.critic[i](self.input_state[i], self.input_action[i])

        if not self.disable_fold:
            if self.bu and not self.td:
                self.x1_fold = self.x1_fold + [self.x1]
                self.x2_fold = self.x2_fold + [self.x2]
            else:
                self.x1_fold = self.x1_fold + self.x1
                self.x2_fold = self.x2_fold + self.x2
            self.x1, self.x2 = self.fold.apply(self, [self.x1_fold, self.x2_fold])
            self.x1 = torch.transpose(self.x1, 0, 1)
            self.x2 = torch.transpose(self.x2, 0, 1)
            self.fold = None
        else:
            self.x1 = torch.stack(self.x1, dim=-1)  # (bs,num_limbs,1)
            self.x2 = torch.stack(self.x2, dim=-1)

        return torch.sum(self.x1, dim=-1), torch.sum(self.x2, dim=-1)

    def Q1(self, state, action):
        self.clear_buffer()
        if not self.disable_fold:
            self.fold = torchfold.Fold()
            self.fold.cuda()
            self.zeroFold_td = self.fold.add("zero_func_td")
            self.zeroFold_bu = self.fold.add("zero_func_bu")
            self.x1_fold = []

        for i in range(self.num_limbs):
            self.input_state[i] = state[:, i * self.state_dim:(i + 1) * self.state_dim]
            self.input_action[i] = action[:, i]
            self.input_action[i] = torch.unsqueeze(self.input_action[i], -1)
            if not self.disable_fold:
                self.input_state[i] = torch.unsqueeze(self.input_state[i], 0)
                self.input_action[i] = torch.unsqueeze(self.input_action[i], 0)

        if self.bu:
            # bottom up transmission by recursion
            for i in range(self.num_limbs):
                self.bottom_up_transmission(i)

        if self.td:
            # top down transmission by recursion
            for i in range(self.num_limbs):
                self.top_down_transmission(i)

        if not self.bu and not self.td:
            for i in range(self.num_limbs):
                if not self.disable_fold:
                    self.x1[i] = self.fold.add('critic' + str(0).zfill(3), self.input_state[i], self.input_action[i])
                else:
                    self.x1[i] = self.critic[i](self.input_state[i], self.input_action[i])

        if not self.disable_fold:
            if self.bu and not self.td:
                self.x1 = [self.x1]
            self.x1_fold = self.x1_fold + self.x1
            self.x1 = self.fold.apply(self, [self.x1_fold])[0]
            if not self.bu and not self.td:
                self.x1 = self.x1[0]
            self.x1 = torch.transpose(self.x1, 0, 1)
            self.fold = None
        else:
            self.x1 = torch.stack(self.x1, dim=-1)  # (bs,num_limbs,1)

        return torch.sum(self.x1, dim=-1)

    def bottom_up_transmission(self, node):

        if node < 0:
            if not self.disable_fold:
                return self.zeroFold_bu
            else:
                return torch.zeros((self.batch_size, self.msg_dim), requires_grad=True).to(device)

        if self.msg_up[node] is not None:
            return self.msg_up[node]

        state = self.input_state[node]
        action = self.input_action[node]

        children = [i for i, x in enumerate(self.parents) if x == node]
        assert (self.max_children - len(children)) >= 0
        children += [-1] * (self.max_children - len(children))
        msg_in = [None] * self.max_children
        for i in range(self.max_children):
            msg_in[i] = self.bottom_up_transmission(children[i])

        if not self.disable_fold:
            if self.td:
                self.msg_up[node] = self.fold.add('sNet' + str(0).zfill(3), state, action, *msg_in)
            else:
                self.msg_up[node], self.x1, self.x2 = self.fold.add('sNet' + str(0).zfill(3), state, action, *msg_in).split(3)
        else:
            if self.td:
                self.msg_up[node] = self.sNet[node](state, action, *msg_in)
            else:
                self.msg_up[node], self.x1, self.x2 = self.sNet[node](state, action, *msg_in)

        return self.msg_up[node]

    def top_down_transmission(self, node):

        if node < 0:
            if not self.disable_fold:
                return self.zeroFold_td
            else:
                return torch.zeros((self.batch_size, self.msg_dim * self.max_children), requires_grad=True).to(device)

        elif self.msg_down[node] is not None:
            return self.msg_down[node]

        # in both-way message-passing, each node takes in its passed-up message as 'state'
        if self.bu:
            state = self.msg_up[node]
        else:
            state = self.input_state[node]

        action = self.input_action[node]
        parent_msg = self.top_down_transmission(self.parents[node])

        # find self children index (first child of parent, second child of parent, etc)
        # by finding the number of previous occurences of parent index in the list
        self_children_idx = self.parents[:node].count(self.parents[node])

        # if the structure is flipped, flip message order at the root
        if self.parents[0] == -2 and node == 1:
            self_children_idx = (self.max_children - 1) - self_children_idx

        if not self.disable_fold:
            msg_in = self.fold.add('get_{}'.format(self_children_idx), parent_msg)
        else:
            msg_in = self.msg_slice(parent_msg, self_children_idx)

        if not self.disable_fold:
            self.x1[node], self.x2[node], self.msg_down[node] = self.fold.add('critic' + str(0).zfill(3), state, action, msg_in).split(3)
        else:
            self.x1[node], self.x2[node], self.msg_down[node] = self.critic[node](state, action, msg_in)

        return self.msg_down[node]

    def zero_func_td(self):
        return torch.zeros((1, self.batch_size, self.msg_dim * self.max_children), requires_grad=True).to(device)

    def zero_func_bu(self):
        return torch.zeros((1, self.batch_size, self.msg_dim), requires_grad=True).to(device)

    # an ugly way to define functions in a for loop (for torchfold only)
    def addFunction(self, n):
        def f(x):
            return torch.split(x, x.shape[-1] // self.max_children, dim=-1)[n]
        return f

    def msg_slice(self, x, idx):
        return torch.split(x, x.shape[-1] // self.max_children, dim=-1)[idx]

    def clear_buffer(self):
        self.x1 = [None] * self.num_limbs
        self.x2 = [None] * self.num_limbs
        self.input_state = [None] * self.num_limbs
        self.input_action = [None] * self.num_limbs
        self.msg_down = [None] * self.num_limbs
        self.msg_up = [None] * self.num_limbs
        self.zeroFold_td = None
        self.zeroFold_bu = None
        self.fold = None

    def change_morphology(self, parents):
        if not self.disable_fold:
            if self.bu:
                for i in range(1, self.num_limbs):
                    delattr(self, "sNet" + str(i).zfill(3))
            if not (self.bu and not self.td):
                for i in range(1, self.num_limbs):
                    delattr(self, "critic" + str(i).zfill(3))
        self.parents = parents
        self.num_limbs = len(parents)
        self.msg_down = [None] * self.num_limbs
        self.msg_up = [None] * self.num_limbs
        self.action = [None] * self.num_limbs
        self.input_state = [None] * self.num_limbs
        if self.bu:
            self.sNet = nn.ModuleList([self.sNet[0]] * self.num_limbs)
            if not self.disable_fold:
                for i in range(self.num_limbs):
                    setattr(self, "sNet" + str(i).zfill(3), self.sNet[i])
        if not (self.bu and not self.td):
            self.critic = nn.ModuleList([self.critic[0]] * self.num_limbs)
            if not self.disable_fold:
                for i in range(self.num_limbs):
                    setattr(self, "critic" + str(i).zfill(3), self.critic[i])
