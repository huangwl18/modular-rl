import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from utils import *
import os

class ModularEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, xml):
        self.xml = xml
        mujoco_env.MujocoEnv.__init__(self, xml, 4)
        utils.EzPickle.__init__(self)

    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        reward_ctrl = - 0.1 * np.square(action).sum()
        reward_run = (xposafter - xposbefore)/self.dt
        reward = reward_ctrl + reward_run
        done = False
        ob = self._get_obs()
        return ob, reward, done, dict(reward_run=reward_run, reward_ctrl=reward_ctrl)

    def _get_obs(self):
        def _get_obs_per_limb(b):
            if b == 'torso':
                limb_type_vec = np.array((1, 0, 0, 0))
            elif 'thigh' in b:
                limb_type_vec = np.array((0, 1, 0, 0))
            elif 'shin' in b:
                limb_type_vec = np.array((0, 0, 1, 0))
            elif 'foot' in b:
                limb_type_vec = np.array((0, 0, 0, 1))
            else:
                limb_type_vec = np.array((0, 0, 0, 0))
            torso_x_pos = self.data.get_body_xpos('torso')[0]
            xpos = self.data.get_body_xpos(b)
            xpos[0] -= torso_x_pos
            q = self.data.get_body_xquat(b)
            expmap = quat2expmap(q)
            obs = np.concatenate([xpos, np.clip(self.data.get_body_xvelp(b), -10, 10), \
                self.data.get_body_xvelr(b), expmap, limb_type_vec])
            # include current joint angle and joint range as input
            if b == 'torso':
                angle = 0.
                joint_range = [0., 0.]
            else:
                body_id = self.sim.model.body_name2id(b)
                jnt_adr = self.sim.model.body_jntadr[body_id]
                qpos_adr = self.sim.model.jnt_qposadr[jnt_adr] # assuming each body has only one joint
                angle = np.degrees(self.data.qpos[qpos_adr]) # angle of current joint, scalar
                joint_range = np.degrees(self.sim.model.jnt_range[jnt_adr]) # range of current joint, (2,)
                # normalize
                angle = (angle - joint_range[0]) / (joint_range[1] - joint_range[0])
                joint_range[0] = (180. + joint_range[0]) / 360.
                joint_range[1] = (180. + joint_range[1]) / 360.
            obs = np.concatenate([obs, [angle], joint_range])
            return obs

        full_obs = np.concatenate([_get_obs_per_limb(b) for b in self.model.body_names[1:]])

        return full_obs.ravel()

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5