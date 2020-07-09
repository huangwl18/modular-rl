from __future__ import print_function
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import xmltodict
import wrappers
import gym
from gym.envs.registration import register
from shutil import copyfile
from config import *


def makeEnvWrapper(env_name, obs_max_len=None, seed=0):
    """return wrapped gym environment for parallel sample collection (vectorized environments)"""
    def helper():
        e = gym.make("environments:%s-v0" % env_name)
        e.seed(seed)
        return wrappers.ModularEnvWrapper(e, obs_max_len)
    return helper


def findMaxChildren(env_names, graphs):
    """return the maximum number of children given a list of env names and their corresponding graph structures"""
    max_children = 0
    for name in env_names:
        most_frequent = max(graphs[name], key=graphs[name].count)
        max_children = max(max_children, graphs[name].count(most_frequent))
    return max_children


def registerEnvs(env_names, max_episode_steps, custom_xml):
    """register the MuJoCo envs with Gym and return the per-limb observation size and max action value (for modular policy training)"""
    # get all paths to xmls (handle the case where the given path is a directory containing multiple xml files)
    paths_to_register = []
    # existing envs
    if not custom_xml:
        for name in env_names:
            paths_to_register.append(os.path.join(XML_DIR, "{}.xml".format(name)))
    # custom envs
    else:
        if os.path.isfile(custom_xml):
            paths_to_register.append(custom_xml)
        elif os.path.isdir(custom_xml):
            for name in sorted(os.listdir(custom_xml)):
                if '.xml' in name:
                    paths_to_register.append(os.path.join(custom_xml, name))
    # register each env
    for xml in paths_to_register:
        env_name = os.path.basename(xml)[:-4]
        env_file = env_name
        # create a copy of modular environment for custom xml model
        if not os.path.exists(os.path.join(ENV_DIR, '{}.py'.format(env_name))):
            # create a duplicate of gym environment file for each env (necessary for avoiding bug in gym)
            copyfile(BASE_MODULAR_ENV_PATH, '{}.py'.format(os.path.join(ENV_DIR, env_name)))
        params = {'xml': os.path.abspath(xml)}
        # register with gym
        register(id=("%s-v0" % env_name),
                 max_episode_steps=max_episode_steps,
                 entry_point="environments.%s:ModularEnv" % env_file,
                 kwargs=params)
        env = wrappers.IdentityWrapper(gym.make("environments:%s-v0" % env_name))
        # the following is the same for each env
        limb_obs_size = env.limb_obs_size
        max_action = env.max_action
    return limb_obs_size, max_action


def quat2expmap(q):
    """
    Converts a quaternion to an exponential map
    Matlab port to python for evaluation purposes
    https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/quat2expmap.m#L1
    Args
    q: 1x4 quaternion
    Returns
    r: 1x3 exponential map
    Raises
    ValueError if the l2 norm of the quaternion is not close to 1
    """
    if (np.abs(np.linalg.norm(q)-1)>1e-3):
        raise(ValueError, "quat2expmap: input quaternion is not norm 1")

    sinhalftheta = np.linalg.norm(q[1:])
    coshalftheta = q[0]
    r0 = np.divide( q[1:], (np.linalg.norm(q[1:]) + np.finfo(np.float32).eps));
    theta = 2 * np.arctan2( sinhalftheta, coshalftheta )
    theta = np.mod( theta + 2*np.pi, 2*np.pi )
    if theta > np.pi:
        theta =  2 * np.pi - theta
        r0    = -r0
    r = r0 * theta
    return r

# replay buffer: expects tuples of (state, next_state, action, reward, done)
# modified from https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
class ReplayBuffer(object):
    def __init__(self, max_size=1e6, slicing_size=None):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0
        # maintains slicing info for [obs, new_obs, action, reward, done]
        if slicing_size:
            self.slicing_size = slicing_size
        else:
            self.slicing_size = None

    def add(self, data):
        if self.slicing_size is None:
            self.slicing_size = [data[0].size, data[1].size, data[2].size, 1, 1]
        data = np.concatenate([data[0], data[1], data[2], [data[3]], [data[4]]])
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        x, y, u, r, d = [], [], [], [], []

        for i in ind:
            data = self.storage[i]
            X = data[:self.slicing_size[0]]
            Y = data[self.slicing_size[0]:self.slicing_size[0] + self.slicing_size[1]]
            U = data[self.slicing_size[0] + self.slicing_size[1]:self.slicing_size[0] + self.slicing_size[1] + self.slicing_size[2]]
            R = data[self.slicing_size[0] + self.slicing_size[1] + self.slicing_size[2]:self.slicing_size[0] + self.slicing_size[1] + self.slicing_size[2] + self.slicing_size[3]]
            D = data[self.slicing_size[0] + self.slicing_size[1] + self.slicing_size[2] + self.slicing_size[3]:]
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))

        return (np.array(x), np.array(y), np.array(u),
                    np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1))


class MLPBase(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(MLPBase, self).__init__()
        self.l1 = nn.Linear(num_inputs, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, num_outputs)

    def forward(self, inputs):
        x = F.relu(self.l1(inputs))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


def getGraphStructure(xml_file):
    """Traverse the given xml file as a tree by pre-order and return the graph structure as a parents list"""
    def preorder(b, parent_idx=-1):
        self_idx = len(parents)
        parents.append(parent_idx)
        if 'body' not in b:
            return
        if not isinstance(b['body'], list):
            b['body'] = [b['body']]
        for branch in b['body']:
            preorder(branch, self_idx)
    with open(xml_file) as fd:
        xml = xmltodict.parse(fd.read())
    parents = []
    try:
        root = xml['mujoco']['worldbody']['body']
        assert not isinstance(root, list), 'worldbody can only contain one body (torso) for the current implementation, but found {}'.format(root)
    except:
        raise Exception("The given xml file does not follow the standard MuJoCo format.")
    preorder(root)
    # signal message flipping for flipped walker morphologies
    if 'walker' in os.path.basename(xml_file) and 'flipped' in os.path.basename(xml_file):
        parents[0] = -2
    return parents


def getGraphJoints(xml_file):
    """Traverse the given xml file as a tree by pre-order and return all the joints defined as a list of tuples (body_name, joint_name1, ...) for each body"""
    """Used to match the order of joints defined in worldbody and joints defined in actuators"""
    def preorder(b):
        if 'joint' in b:
            if isinstance(b['joint'], list) and b['@name'] != 'torso':
                raise Exception("The given xml file does not follow the standard MuJoCo format.")
            elif not isinstance(b['joint'], list):
                b['joint'] = [b['joint']]
            joints.append([b['@name']])
            for j in b['joint']:
                joints[-1].append(j['@name'])
        if 'body' not in b:
            return
        if not isinstance(b['body'], list):
            b['body'] = [b['body']]
        for branch in b['body']:
            preorder(branch)
    with open(xml_file) as fd:
        xml = xmltodict.parse(fd.read())
    joints = []
    try:
        root = xml['mujoco']['worldbody']['body']
    except:
        raise Exception("The given xml file does not follow the standard MuJoCo format.")
    preorder(root)
    return joints


def getMotorJoints(xml_file):
    """Traverse the given xml file as a tree by pre-order and return the joint names in the order of defined actuators"""
    """Used to match the order of joints defined in worldbody and joints defined in actuators"""
    with open(xml_file) as fd:
        xml = xmltodict.parse(fd.read())
    joints = []
    motors = xml['mujoco']['actuator']['motor']
    if not isinstance(motors, list):
        motors = [motors]
    for m in motors:
        joints.append(m['@joint'])
    return joints
