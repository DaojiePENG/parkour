# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from time import time
from warnings import WarningMessage
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import Tensor
from typing import Tuple, Dict
from copy import copy

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.terrain.terrain import Terrain
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float
from legged_gym.utils.helpers import class_to_dict
from .legged_robot_config import LeggedRobotCfg

class LeggedRobot(BaseTask):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = getattr(self.cfg.viewer, "debug_viz", False) # 这个参数决定是否在仿真时绘制和显示高度测量点, bool，默认为 False； debug_viz 在 play.py 文件中会改为False；
        self.init_done = False
        self._parse_cfg(self.cfg) # 将config文件中的相关变量拿出来作为类的 self. 局部变量；
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)

        if not self.headless:
            '''如果不是 headless 的情况，就按照 config 文件中 viewer 定义的参数放置观查相机镜头；'''
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat) # 相机的设置需要两个参数，相机的位置和摆放的角度；
        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        self.pre_physics_step(actions)
        # step physics and render each frame
        self.render()
        for dec_i in range(self.cfg.control.decimation):
            '''
            这里 decimation=4；
            这意味着这个 torque 计算->传入仿真->运行仿真->刷新状态张量 的过程需要执行4次；然后才进行后续物理处理；
            '''
            self.torques = self._compute_torques(self.actions).view(self.torques.shape) # 根据输入的动作指令，计算出相应的扭矩，准备传入仿真空间；
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques)) # 将计算出来的扭矩传入仿真系统中；用这个函数来驱动仿真空间的里面的机器人！
            self.gym.simulate(self.sim)# 进行一次仿真，让机器人按照给定的扭矩在仿真空间中进行演化；
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim) # 刷新 gym 仿真空间中的各个自由度状态张量；
            self.post_decimation_step(dec_i)
        self.post_physics_step() # 主要是检查终止条件、计算观测值、计算奖励；重新随机采样新的指令、计算测量到的高度值、添加运动干扰（随机推动机器人）；

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def pre_physics_step(self, actions):
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)

    def post_decimation_step(self, dec_i):
        '''每次数据抽取后的操作
        substep_xxx 比 xxx 多了 dec_i 这个维度；
        主要被用来提供监测和计算奖励；
        '''
        self.substep_torques[:, dec_i, :] = self.torques
        self.substep_dof_vel[:, dec_i, :] = self.dof_vel
        self.substep_exceed_dof_pos_limits[:, dec_i, :] = (self.dof_pos < self.dof_pos_limits[:, 0]) | (self.dof_pos > self.dof_pos_limits[:, 1])

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10]) # 转化后的相对于本体的线速度，m/s
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13]) # 转化后的相对于本体的角速度，m/s
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec) # 转化后的相对于本体的重力投影；实际中应该由IMU中的重力传感器提供；

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten() # 返回非零项标号，一维；
        self.reset_idx(env_ids)
        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]
        self.last_torques[:] = self.torques[:]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            '''绘制和显示高度测量点'''
            self._draw_debug_vis()

    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length==0):
            self.update_command_curriculum(env_ids)
        
        self._fill_extras(env_ids)

        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        self._resample_commands(env_ids)
        self._reset_buffers(env_ids)

    
    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            '''
            _prepare_reward_function 已经将奖励函数和函数名打包好了；
            奖励函数的命名需遵循： func_name = '_reward_' + name
            '''
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name] # 直接通过索引就可以调用相应的函数了，这归功于 getattr(self, func_name)； 另外这里处理了奖励的权重；
            self.rew_buf += rew # 将所有的奖励相加进行累积；
            self.episode_sums[name] += rew # 将每一片段的奖励按名称分别进行相加累积；
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales: # 为什么要 clip 后再添加终止奖励呢？
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew
    
    def compute_observations(self):
        '''在这里定义了 observation 的结构，也为它的内存 obs_buf 进行了赋值。'''
        """ Computes observations
        """
        self.obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel, # 基体线速度 3 ；从仿真空间中计算得出的归一化的速度 self.base_lin_vel 乘上速度缩放 config 中定义的 self.obs_scales.lin_vel
                                    self.base_ang_vel  * self.obs_scales.ang_vel, # 基体角速度 3 ； 解析同上；这里的速度跟实际速度是如何对应的，这有单位吗？看了官方API没说单位，那就默认是标准单位 m/s；
                                    self.projected_gravity, # 基体重力 3 ；实际上这个 self.obs_scales.xxx 应该只在输入神经网络的时候用到了；
                                    self.commands[:, :3] * self.commands_scale, # 控制指令 3 ， 前两列是线速度，第三列是角速度；
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos, # 节点位置 12 ；
                                    self.dof_vel * self.obs_scales.dof_vel, # 节点速度 12；
                                    self.actions # 上次的动作目标 12 ；
                                    ),dim=-1) # 这样基础的观测输入就是 48 ， 如果要进一步增加输入就需要下面的内容了。
        # add perceptive inputs if not blind
        if self.cfg.terrain.measure_heights:
            '''
            在有地形感知的情况下添加高度感知信息；

            这里把高度测量值限制在 [-1.，1.]之间；
            这里应该是默认机器人高度是 0.5m 所以直接减了个 0.5 变成以接触到的地面为起点；
            然后减去测量到的高度值得到相对于机器人的地表凹凸情况（也就是机器人自身雷达传感器可以的观测数据）；
            '''
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)

        if not self.num_privileged_obs is None:
            '''
            在考虑特权观测的情况下，降特权观测信息和观测信息结合起来；
            '''
            min_shape = min(self.obs_buf.shape[1], self.privileged_obs_buf.shape[1])
            self.privileged_obs_buf[:, :min_shape] = self.obs_buf[:, :min_shape] # copy content
        if self.num_obs == 48:
            self.obs_buf = self.obs_buf[:, :48]
        
        # add noise if needed
        if self.add_noise:
            '''如果想拓展感知信息数量，也需要同时考虑这部分；'''
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

        if not self.cfg.env.use_lin_vel:
            self.obs_buf[:, :3] = 0.

    def create_sim(self):
        '''该函数由 base_task 的初始化函数直接调用；'''
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_terrain()
        self._create_envs()

    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    #------------- Callbacks --------------
    def _process_rigid_shape_props(self, props, env_id):
        '''
        实现对刚体性质的调控；
        
        这仅实现了刚体摩擦力的随机化，这里的实现并没有区分具体是哪个刚体，而是对所有的都随机设置了；
        实现方式很巧妙，使用了 bucket 来作为容器，然后实体的参数从 bucket 里面选：
            也就是说用 bucket_ids 的数来选择 friction_buckets 里面的数； 
            这样做就避免了对于上千个实例需要生成上千个随机数的需求，总共随机数也就64个就够了；
        '''
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            if env_id==0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets,1), device='cpu')
                self.friction_coeffs = friction_buckets[bucket_ids]

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]
        return props

    def _process_dof_props(self, props, env_id):
        '''
        props 来自于 gym.get_asset_dof_properties(robot_asset) 返回的来自导入的 urdf 中的特性信息；

        没有看到改变这个 props 的代码，怎么做到的改变其返回值的？
        实际上就是没改变，只是将 urdf 中导入的信息给保存了下来，变成类的局部变量来给其他地方用（创建了节点软约束，用来奖励的计算）；
        所以想要随机这些特性的话需要自己另写代码实现；
        '''
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id==0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
        return props

    def _process_rigid_body_props(self, props, env_id):
        '''给机器人添加随机重量负载'''
        # if env_id==0:
        #     sum = 0
        #     for i, p in enumerate(props):
        #         sum += p.mass
        #         print(f"Mass of body {i}: {p.mass} (before randomization)")
        #     print(f"Total mass {sum} (before randomization)")
        # randomize base mass
        if self.cfg.domain_rand.randomize_base_mass:
            rng = self.cfg.domain_rand.added_mass_range
            props[0].mass += np.random.uniform(rng[0], rng[1])
        return props
    
    def _post_physics_step_callback(self):
        '''
        这个函数干三件事，也分别对应着三个函数：
            重新随机采样新的指令 _resample_commands(env_ids)；
            计算测量到的高度值 _get_heights()；
            添加运动干扰，随机推动机器人 _push_robots()；
        '''
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        # 
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.5*wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)

        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
        if self.cfg.domain_rand.push_robots and  (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()

    def _resample_commands(self, env_ids):
        '''
        该函数在 _post_physics_step_callback(self) 和 reset_idx(self, env_ids) 函数中被调用；
        '''
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)

        # set small commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)
        self.commands[env_ids, 2] *= (torch.abs(self.commands[env_ids, 2]) > 0.1)

    def _compute_torques(self, actions):
        '''
        这里实现了个PD控制器，选择电机的控制方式“P,V,T”，根据控制方式和 actions 目标，计算应该输出的扭矩数值，然后将扭矩数值传入仿真空间中进行仿真；
        '''
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        #pd controller
        if isinstance(self.cfg.control.action_scale, (tuple, list)):
            self.cfg.control.action_scale = torch.tensor(self.cfg.control.action_scale, device= self.sim_device)
        actions_scaled = actions * self.cfg.control.action_scale
        control_type = self.cfg.control.control_type
        if control_type=="P":
            torques = self.p_gains*(actions_scaled + self.default_dof_pos - self.dof_pos) - self.d_gains*self.dof_vel
        elif control_type=="V":
            torques = self.p_gains*(actions_scaled - self.dof_vel) - self.d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
        elif control_type=="T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _reset_dofs(self, env_ids):
        '''这个函数的实现与原来有较大不同，关注一下；
        '''
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        if getattr(self.cfg.domain_rand, "init_dof_pos_ratio_range", None) is not None:
            '''对节点初始化位置进行随机化处理'''
            self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(
                self.cfg.domain_rand.init_dof_pos_ratio_range[0],
                self.cfg.domain_rand.init_dof_pos_ratio_range[1],
                (len(env_ids), self.num_dof),
                device=self.device,
            )
        else:
            self.dof_pos[env_ids] = self.default_dof_pos
        # self.dof_vel[env_ids] = 0. # history init method
        dof_vel_range = getattr(self.cfg.domain_rand, "init_dof_vel_range", [-3., 3.])
        # 按照输入张量的形状生成 dof_vel_range 范围内的随机张量；
        self.dof_vel[env_ids] = torch.rand_like(self.dof_vel[env_ids]) * abs(dof_vel_range[1] - dof_vel_range[0]) + min(dof_vel_range)

        # Each env has multiple actors. So the actor index is not the same as env_id. But robot actor is always the first.
        dof_idx = env_ids * self.all_root_states.shape[0] / self.num_envs # 没看明白？？？
        dof_idx_int32 = dof_idx.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.all_dof_states),
                                              gymtorch.unwrap_tensor(dof_idx_int32), len(dof_idx_int32))
    def _reset_root_states(self, env_ids):
        '''这个函数的实现与原来有较大不同，关注一下；
        '''
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        if self.custom_origins:
            '''
            这里实现了随机化的初始化位置，使得在一个 小terrain 中的 env 相互之间不是完全重合的；不过方向都是一样的；
            我可以试一下把初始化的方向也改成随机的，这样看起来就很乱了；
            '''
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            if hasattr(self.cfg.domain_rand, "init_base_pos_range"):
                self.root_states[env_ids, 0:1] += torch_rand_float(*self.cfg.domain_rand.init_base_pos_range["x"], (len(env_ids), 1), device=self.device)
                self.root_states[env_ids, 1:2] += torch_rand_float(*self.cfg.domain_rand.init_base_pos_range["y"], (len(env_ids), 1), device=self.device)
            else:
                self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device) # xy position within 1m of the center
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        # base rotation (roll and pitch)
        if hasattr(self.cfg.domain_rand, "init_base_rot_range"):
            base_roll = torch_rand_float(
                *self.cfg.domain_rand.init_base_rot_range["roll"],
                (len(env_ids), 1),
                device=self.device,
            )[:, 0]
            base_pitch = torch_rand_float(
                *self.cfg.domain_rand.init_base_rot_range["pitch"],
                (len(env_ids), 1),
                device=self.device,
            )[:, 0]
            base_quat = quat_from_euler_xyz(base_roll, base_pitch, torch.zeros_like(base_roll))
            self.root_states[env_ids, 3:7] = base_quat
        # base velocities
        if getattr(self.cfg.domain_rand, "init_base_vel_range", None) is None:
            base_vel_range = (-0.5, 0.5)
        else:
            base_vel_range = self.cfg.domain_rand.init_base_vel_range
        if isinstance(base_vel_range, (tuple, list)):
            self.root_states[env_ids, 7:13] = torch_rand_float(
                *base_vel_range,
                (len(env_ids), 6),
                device=self.device,
            ) # [7:10]: lin vel, [10:13]: ang vel
        elif isinstance(base_vel_range, dict):
            self.root_states[env_ids, 7:8] = torch_rand_float(
                *base_vel_range["x"],
                (len(env_ids), 1),
                device=self.device,
            )
            self.root_states[env_ids, 8:9] = torch_rand_float(
                *base_vel_range["y"],
                (len(env_ids), 1),
                device=self.device,
            )
            self.root_states[env_ids, 9:10] = torch_rand_float(
                *base_vel_range["z"],
                (len(env_ids), 1),
                device=self.device,
            )
            self.root_states[env_ids, 10:11] = torch_rand_float(
                *base_vel_range["roll"],
                (len(env_ids), 1),
                device=self.device,
            )
            self.root_states[env_ids, 11:12] = torch_rand_float(
                *base_vel_range["pitch"],
                (len(env_ids), 1),
                device=self.device,
            )
            self.root_states[env_ids, 12:13] = torch_rand_float(
                *base_vel_range["yaw"],
                (len(env_ids), 1),
                device=self.device,
            )
        else:
            raise NameError(f"Unknown base_vel_range type: {type(base_vel_range)}")
        
        # Each env has multiple actors. So the actor index is not the same as env_id. But robot actor is always the first.
        actor_idx = env_ids * self.all_root_states.shape[0] / self.num_envs
        actor_idx_int32 = actor_idx.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.all_root_states),
                                                     gymtorch.unwrap_tensor(actor_idx_int32), len(actor_idx_int32))

    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        self.root_states[:, 7:9] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device) # lin vel x/y
        '''
        也就是说从外部强制给它赋值一个初速度；但是要是本身 root_states 就不为 0 的情况呢？ 应该是在原有的速度上加上一个额外的推动吧？
        或者说这默认机器人是在静止状态被推动的？估计是。
        '''
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.all_root_states))

    def _update_terrain_curriculum(self, env_ids):
        '''
        1. 完成任务较好的机器人升级，送到更困难的地形上去；
        2. 完成任务太差的机器人降级，送到更简单的地形上去；
        3. 完成最高登记难度的机器人，随机送到基础上的位置；

        具体的实现相关的数据结构还需要进一步理解一下；
        '''
        """ Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum
        if not self.init_done:
            # don't change on initial reset
            return
        move_up, move_down = self._get_terrain_curriculum_move(env_ids) # 返回该升级还是降级；
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # Robots that solve the last level are sent to a random one
        self.terrain_levels[env_ids] = torch.where(self.terrain_levels[env_ids]>=self.max_terrain_level,
                                                   torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
                                                   torch.clip(self.terrain_levels[env_ids], 0)) # (the minumum level is zero)
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]

    def _get_terrain_curriculum_move(self, env_ids):
        distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1) # 计算机器人到原点的距离，离得越远说明任务完成的越好；
        # robots that walked far enough progress to harder terains
        move_up = distance > self.terrain.env_length / 2 # 距离大于小地形长度的一半，说明已经完成的很好了，跑到了别的地形去了；
        # robots that walked less than half of their required distance go to simpler terrains
        move_down = (distance < torch.norm(self.commands[env_ids, :2], dim=1)*self.max_episode_length_s*0.5) * ~move_up
        return move_up, move_down
    
    def update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel"]:
            '''
            奖励计算是采用e指数的，速度误差越小越接近1，因此这里选择 0.8；
            torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length 计算的是每一轮中平均每个片段的奖励；
            self.max_episode_length是每一轮训练中刷新的总次数；

            command_ranges["lin_vel_x"][0]/[1]分别代表x方向速度的最大值和最小值；
            下面就是在更新x方向速度随机取值的范围；
            '''
            self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.5, -self.cfg.commands.max_curriculum, 0.)
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.5, 0., self.cfg.commands.max_curriculum)


    def _get_noise_scale_vec(self, cfg):
        '''
        关于给感知信息添加噪声的定义都在这里完成；

        注：如果要拓展感知信息量，也要同时拓展相关的向量；
        '''
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        self._write_proprioception_noise(noise_vec[:48])
        if self.cfg.terrain.measure_heights:
            self._write_height_measurements_noise(noise_vec[48:235])
        return noise_vec

    def _write_proprioception_noise(self, noise_vec):
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[6:9] = noise_scales.gravity * noise_level
        noise_vec[9:12] = 0. # commands
        noise_vec[12:24] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[24:36] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[36:48] = 0. # previous actions

    def _write_height_measurements_noise(self, noise_vec):
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:] = noise_scales.height_measurements * noise_level * self.obs_scales.height_measurements

    #----------------------------------------
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.all_root_states = gymtorch.wrap_tensor(actor_root_state)
        self.root_states = self.all_root_states.view(self.num_envs, -1, 13)[:, 0, :] # (num_envs, 13)
        self.all_dof_states = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_state = self.all_dof_states.view(self.num_envs, -1, 2)[:, :self.num_dof, :] # (num_envs, 2)
        self.dof_pos = self.dof_state.view(self.num_envs, -1, 2)[..., :self.num_dof, 0]
        self.dof_vel = self.dof_state.view(self.num_envs, -1, 2)[..., :self.num_dof, 1]
        self.base_quat = self.root_states[:, 3:7]

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1)) # 定义重力方向 up_axis_idx=1 为y轴方向；up_axis_idx=2 为z轴方向；这里的 -1 不知道是不是指的反方向的意思；
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1)) # 定义前进方向，这里是 x 轴；
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.last_torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False,) # TODO change this
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()  # 高度数据赋值就是 height_points[:,:,2] ；
        self.measured_heights = 0
        self.substep_torques = torch.zeros(self.num_envs, self.cfg.control.decimation, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.substep_dof_vel = torch.zeros(self.num_envs, self.cfg.control.decimation, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.substep_exceed_dof_pos_limits = torch.zeros(self.num_envs, self.cfg.control.decimation, self.num_dof, dtype=torch.bool, device=self.device, requires_grad=False)

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            '''
            将 config 中定义的节点初始化位置赋值给gym中的节点；
            节点的名称来自 create_envs 函数导入到 gym 空间的提取名称；
            因此在 config 中定义的初始化节点名称要与之对应；
            '''
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    '''给节点定义刚度和阻尼'''
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0) # 这个转换后用于计算无命令情况的奖励；

    def _reset_buffers(self, env_ids):
        if getattr(self.cfg.init_state, "zero_actions", False):
            self.actions[env_ids] = 0.
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.last_torques[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key) 
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)
    
    def _create_heightfield(self):
        """ Adds a heightfield terrain to the simulation, sets parameters based on the cfg.
        """
        hf_params = gymapi.HeightFieldParams()
        hf_params.column_scale = self.terrain.cfg.horizontal_scale
        hf_params.row_scale = self.terrain.cfg.horizontal_scale
        hf_params.vertical_scale = self.terrain.cfg.vertical_scale
        hf_params.nbRows = self.terrain.tot_cols
        hf_params.nbColumns = self.terrain.tot_rows 
        hf_params.transform.p.x = -self.terrain.cfg.border_size 
        hf_params.transform.p.y = -self.terrain.cfg.border_size
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self.cfg.terrain.static_friction
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        hf_params.restitution = self.cfg.terrain.restitution

        self.gym.add_heightfield(self.sim, self.terrain.heightsamples, hf_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_trimesh(self):
        '''
        这种类型的定义应该是可以使得地图斜边更平滑；而高度图形式就比较粗糙，如果像素分辨率低的话可能会看到一个个小像素柱子；这可能是为什么倾向于使用 trimesh ;
        trimesh 的相关知识可以到 API 或者网上了解一下；
        '''
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
        # """
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size 
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'), self.terrain.triangles.flatten(order='C'), tm_params)   
        # 这里是直接从terrain实例中获取到的高度采样点。这些高度点跟仿真空间里的实体对应是怎样的？
        # 如果我想获取仿真空间中沿着某一方向的数据应该怎么做？
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_sensors(self, env_handle= None, actor_handle= None):
        """ attach necessary sensors for each actor in each env
        Considering only one robot in each environment, this method takes only one actor_handle.
        Args:
            env_handle: env_handle from gym.create_env
            actor_handle: actor_handle from gym.create_actor
        Return:
            sensor_handle_dict: a dict of sensor_handles with key as sensor name (defined in cfg["sensor"])
        """
        return dict()

    def _create_npc(self, env_handle, env_idx):
        """ create additional opponent for each environment such as static objects, random agents
        or turbulance.
        """
        return dict()

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions() # 英伟达 isaac gym 提供的关于仿真对象的可选设置参数接口；
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        # 在这里载入 urdf 文件到 gym 空间并获取相应的自由度节点数量及其名称
        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            '''从 config 文件中提取定义的惩罚碰撞条件，这里是名称定义，下面（本函数内）有 gym 中的实现'''
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            '''从 config 文件中提取定义的终止碰撞条件，这里是名称定义，下面（本函数内）有 gym 中的实现'''
            termination_contact_names.extend([s for s in body_names if name in s])

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform() # 按照 config 文件中的设置来初始化内容 base 主体姿态；施加空间转换的变换；
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.npc_handles = [] # surrounding actors or objects or oppoents in each environment.
        self.sensor_handles = []
        self.actor_handles = []
        self.envs = []
        for i in range(self.num_envs):
            '''
            这部分按照设定的参数来创建 envs 然后返回相应的 env_handle 句柄到 envs ；
            同时也用 env_handle 创建了 actor 并返回相应的 actor_handle 句柄到 actor_handles ；

            1. 实现了对 env 的创建和初始化其在地图上的位置设定，实际的位置分配在 self._get_env_origins() 函数中分配，关键变量为 self.env_origins；
            2. 实现了对 env、actor 的特性调整，比如摩擦力随机化（同时也从仿真空间中获取相应参数值，为计算奖励做准备）； 
            '''
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone() # 将第 i 个地形的中心 x,y 坐标值设置为第 i 个 env 的初始化起点；
            pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1) # 初始化的具体位置进行一定的随机化，这也就是为什么初始会看到一簇一簇的机器人；
            start_pose.p = gymapi.Vec3(*pos) # 用上面的地形相关的位置来初始化 base 的位置；这个岂不是会把上面的设置给覆盖掉？
            # 所以上面那句 start_pose.p = gymapi.Vec3(*self.base_init_state[:3]) 没起作用呀？
            # 注意这里 start_pose 是 gymapi.Transform() 类型的，所以每次赋值实际上就是施加一次变换，不冲突；感觉又有点不太对，差看了下就是直接赋值的；
                
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i) # 1. 由于前面将机器人实体导入仿真时已经获得了 rigid_shape_props_asset 信息，因此这里直接对其加工处理；
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props) # 2. 然后设置回仿真；
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i) # 1. 由于前面将机器人实体导入仿真时已经获得了 dof_props_asset 信息，因此这里直接对其加工处理；
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props) # 2. 然后设置回仿真；
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle) # 1. 获取第 i 个实体物理信息；
            body_props = self._process_rigid_body_props(body_props, i) # 2. 对其中的参数进行随机化，此版本函数实现了对重力参数进行随机化；
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True) # 3. 将处理后的实体物理参数设置回仿真；
            sensor_handle_dict = self._create_sensors(env_handle, actor_handle) # 添加机身传感特性；具体用法和定义？转到self.sensor_handles
            npc_handle_dict = self._create_npc(env_handle, i) # 添加非玩家角色，non-player character；具体用法和定义？
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)
            self.sensor_handles.append(sensor_handle_dict) # 每一个 env 都保存一个传感器句柄，后面可以按照env id进行访问使用；
            self.npc_handles.append(npc_handle_dict)

        # 下面几个 for 循环提供了一些需要计算奖励的特殊实体的索引标号，比如脚部、会被惩罚的接触实体、会导致训练重启的实体名等；
        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            '''
            调用 gym 接口，检查每一个腿部接触条件（由 feet_names 定义），
            并将标号分别保存在 feet_indices 的1维空间中，
            随后在若干函数中被使用，如计算脚部摆动、脚部悬空时间、脚部接触力奖励等等；

            注：这里只是提供了索引，并非实际值。实际值需要从 gym state 中获取；
            '''
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            '''
            调用 gym 接口，检查每一个惩罚条件（由 penalized_contact_names 定义），
            并将标号分别保存在 penalised_contact_indices 的1维空间中，
            随后在 _reward_collision(self) 函数中被使用；

            注：这里只是提供了索引，并非实际值。实际值需要从 gym state 中获取；
            '''
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            '''
            调用 gym 接口，检查每一个终止条件（由 termination_contact_names 定义），
            并将标号分别保存在 termination_contact_indices 的1维空间中，
            随后在 check_termination(self) 函数中被使用；

            注：这里只是提供了索引，并非实际值。实际值需要从 gym state 中获取；
            '''
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])

    def _create_terrain(self):
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)
        if mesh_type=='plane':
            self._create_ground_plane()
        elif mesh_type=='heightfield':
            self._create_heightfield()
        elif mesh_type=='trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")

    def _get_env_origins(self):
        '''
        重要变量： env_origins ，形状为 num_envs 行，3列； env 初始化时会按照它定义的位置在仿真环境中初始化；
        有自定义地形的情况下以自定义的地形中心为取值准；没有自定义地形时，均匀的网格化取值布局；这应该是我想实现的确定不同 env 所处的具体地形的关键；
        '''
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # put robots at the origins defined by the terrain
            max_init_level = self.cfg.terrain.max_init_terrain_level # 有课程时指定的最大初始化等级；
            if not self.cfg.terrain.curriculum: max_init_level = self.cfg.terrain.num_rows - 1 # 如果没设置课程，那就每个小地形快都放置一点实体；
            self.terrain_levels = torch.randint(0, max_init_level+1, (self.num_envs,), device=self.device) # 随机生成了初始化难度；形状为 num_envs 行，1 列；
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (self.num_envs/self.cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)
            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float) # 从地形定义类中获取了 num_rows 行，num_cols 列，3 的小地形中点；
            '''self.terrain_levels： 行数选择； self.terrain_types 列数选择；
            最终返回值是 num_envs 个 3 维坐标向量；
            我想关注的是实体在第几列；只需要在指令采样的时候对 env_idx 的编号做一个判断就可以直到它在哪一列地形 (terrain_idx in [0, num_cols-1]) 了，
            即做如下索引： terrain_idx = self.terrain_types[env_idx]。然后就可以针对不同的地形做出相应的处理了。'''
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = self.cfg.env.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt # 决定训练的刷新频率，如文献上说是 50 Hz； sim_params 是从 legged_robot 父类初始化来的；
        self.obs_scales = copy(self.cfg.normalization.obs_scales)
        self.reward_scales = class_to_dict(self.cfg.rewards.scales) # 这是一个 rewards 类， 里面包含了一些列奖励的缩放；这样变成一个字典，可以用来索引奖励函数；
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        if self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s # 以秒为单位的一轮训练时间长度，默认为 20 秒；
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt) # 一轮训练中刷新的次数；由总时长/单位时长 计算得到；

        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt) # 从 config 中定义新的随机推动机器人变量（转换到仿真片段量）；

    def _draw_debug_vis(self):
        '''
        用于向 Isaacgym 仿真平台内绘制图形；
        这里实现的是实时显示机器人周围的环境测量点；
        '''
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        # draw height lines
        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        if not self.terrain.cfg.measure_heights:
            return
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0)) # 定义绘制点的形状、大小、颜色等特性；位置属性将在下面定义；
        for i in range(self.num_envs):
            base_pos = (self.root_states[i, :3]).cpu().numpy()
            heights = self.measured_heights[i].cpu().numpy() # 取测量得到的像素级高度值的第 i 行；
            height_points = quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]), self.height_points[i]).cpu().numpy() # 随所有的高度坐标点进行旋转和平移操作； 形状： 1 行， num_point 列，每个元素有(x,y,z)三个值；
            for j in range(heights.shape[0]):
                x = height_points[j, 0] + base_pos[0] # 第 j 列的 0 就是 x 坐标； 1 就是 y 坐标； 2 就是 z 坐标，但由于没赋值因此用别的代替；
                y = height_points[j, 1] + base_pos[1]
                z = heights[j] # 直接使用之前处理过的高度数据，第 j 个就行了。整体就是 第 i 行的第 j 个位置上的 z 值；
                sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None) # 将 cpu 下的参数转化为 gpu 下的格式；
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose) 

    def _init_height_points(self):
        '''
        初始化高度点；
        通过 config 中定义的 measured_points_y, measured_points_x 的 mesh 化来生成测量网表的基本映射基础；
        '''
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def _get_heights(self, env_ids=None):
        '''
        返回值是一个 mesh 化的 z 测量值，也就是二维网表（形状：横坐标为 envs 行, 纵坐标为 num_points 列，值：不过这里只是第三维的 z 分量）；
        也就是同时返回所有或者选定的 env 的 num_points 个高度测量值； 对于单个 env 来说，结果排成了一排， 也即 flatten 化了；
        '''
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            # 当是平地的时候高度测量结果都为零
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            # 这里为什么要添加 yaw 呢？ points 需要经过转换，最基本的是需要结合每个 env 自己的空间坐标位置和旋转姿态进行调整；
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points), self.height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            # 默认情况下为所有的 envs 测量索求地面高度数据， self.height_points 是相对于机器人坐标系定义的目标测量点（到此为止仅含 x,y 的 mesh 结果部分，z 值下面将计算）；
            # ------------- base_quat 重复 num_height_points 次后并行作用在每个高度点上（旋转操作）----------------加上所有 env 的当前坐标（位移操作）---------------------
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (self.root_states[:, :3]).unsqueeze(1)

        points += self.terrain.cfg.border_size # 因为像素点级的地形图数据不含边界信息，所以要对 x,y 的点都进行相应的偏移；到此为止还都是以 m 为单位的；
        points = (points/self.terrain.cfg.horizontal_scale).long() # 将 m 为单位的长度转换为像素点数量；
        px = points[:, :, 0].view(-1) # 提取第三维的 x 分量；
        py = points[:, :, 1].view(-1) # 提取第三维的 y 分量；
        px = torch.clip(px, 0, self.height_samples.shape[0]-2) # 将 px 的值限制在 0 到 self.height_samples.shape[0]-2 之间；
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        # 为什么要有下面的操作呢？# 这里的 height_samples 就是昨天看的关于 terrain 部分的整个大地图的像素级 z 高度值；
        heights1 = self.height_samples[px, py] # 这个输出就是一个 mesh 化的 z 测量值，也就是二维网表（形状：横坐标为 envs 行, 纵坐标为 num_points 列，值：不过这里只是第三维的 z 分量）；
        heights2 = self.height_samples[px+1, py] # 因此这里的 px, py 都需要考虑到边界来计算在大地图中的绝对像素点位置；
        heights3 = self.height_samples[px, py+1] # 这是测量三个近点，然后取最小值作为测量到的高度值；
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale
    
    def _fill_extras(self, env_ids):
        '''填充需要导出显示到终端的内容'''
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s # 将片段对env_ids数求均值，再对片段长度求均值，得到平均片段奖励值；
            self.extras["episode"]['rew_frame_' + key] = torch.nanmean(self.episode_sums[key][env_ids] / self.episode_length_buf[env_ids])
            self.episode_sums[key][env_ids] = 0.
        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
            if len(env_ids) > 0:
                self.extras["episode"]["terrain_level_max"] = torch.max(self.terrain_levels[env_ids].float())
                self.extras["episode"]["terrain_level_min"] = torch.min(self.terrain_levels[env_ids].float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # log whether the episode ends by timeout or dead, or by reaching the goal
        self.extras["episode"]["timeout_ratio"] = self.time_out_buf.float().sum() / self.reset_buf.float().sum()
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

    #------------ reward functions----------------
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])
    
    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
    
    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        return torch.square(base_height - self.cfg.rewards.base_height_target)
    
    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_delta_torques(self):
        return torch.sum(torch.square(self.torques - self.last_torques), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)
    
    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
    
    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)
    
    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
    
    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf
    
    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)
    
    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw) 
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)

    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        self.feet_air_time *= ~contact_filt
        return rew_airTime
    
    def _reward_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             5 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
        
    def _reward_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)
