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

import sys
from isaacgym import gymapi
from isaacgym import gymutil
import numpy as np
import torch

# Base class for RL tasks
class BaseTask():

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        self.gym = gymapi.acquire_gym()

        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.sim_device = sim_device
        sim_device_type, self.sim_device_id = gymutil.parse_device_str(self.sim_device)
        self.headless = headless

        # env device is GPU only if sim is on GPU and use_gpu_pipeline=True, otherwise returned tensors are copied to CPU by physX.
        if sim_device_type=='cuda' and sim_params.use_gpu_pipeline:
            self.device = self.sim_device
        else:
            self.device = 'cpu'

        # graphics device for rendering, -1 for no rendering
        self.graphics_device_id = self.sim_device_id
        if self.headless == True and self.cfg.sim.no_camera:
            self.graphics_device_id = -1

        self.num_envs = cfg.env.num_envs # 这个参数定义了仿真训练会例化多少个机器人进行并行训练；
        self.num_obs = cfg.env.num_observations # 这个定义了有多少输入观测值（整个策略的输入参数数量），比如控制指令、自身感知、外部感知等。
        self.num_privileged_obs = cfg.env.num_privileged_obs
        self.num_actions = cfg.env.num_actions # 这个定义了机器人可以活动的关节数量，也就是整个策略最终输出控制目标的数量，比如机械狗一般是 12 ；

        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        # allocate buffers
        self.obs_buf = torch.zeros(self.num_envs, self.num_obs, device=self.device, dtype=torch.float)
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        if self.num_privileged_obs is not None:
            self.privileged_obs_buf = torch.zeros(self.num_envs, self.num_privileged_obs, device=self.device, dtype=torch.float)
        else: 
            self.privileged_obs_buf = None
            # self.num_privileged_obs = self.num_obs

        self.extras = {}

        # create envs, sim and viewer
        self.create_sim()
        self.gym.prepare_sim(self.sim)

        # todo: read from config
        self.enable_viewer_sync = True
        self.viewer = None

        # if running with a viewer, set up keyboard shortcuts and camera
        if self.headless == False:
            '''
            当不用 headless 模式时，
            这里调用 Isaacgym 的接口来创建观测图形窗口，并且检测键盘的‘退出键’和‘V键’事件来控制 QUIT 和 toggle_viewer_sync；
            这些功能都是 Isaacgym 内部接口实现的；
            '''
            # subscribe to keyboard shortcuts
            self.viewer = self.gym.create_viewer(
                self.sim, gymapi.CameraProperties())
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_ESCAPE, "QUIT")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_V, "toggle_viewer_sync")

    def get_observations(self):
        '''
        这个参数定义了有多少输入观测值（整个策略的输入参数数量），比如控制指令、自身感知、外部感知等。
        所有的输入都放在这个 obs_buf 里面，用它作为神经网络的输入，就可以获得动作控制的输出值。
        因此 obs_buf 的内部信息格式值得关注，也即诸如控制指令、自身感知、外部感知这些信息的赋值格式、空间占用大小和排列结构。

        这个 obs_buf 将在 LeggedRobot 类中的 compute_observations 函数中进行定义。
        '''
        return self.obs_buf
    
    def get_privileged_observations(self):
        '''
        这个特权观测信息，跟上面的观测信息作用一样。不过特权信息往往只能在仿真中使用，作为训练 teacher 策略的额外输入信息。
        部署到实际机器人的策略不具备特权观测信息的输入，它的策略通常是模仿 teacher 策略而训练的 student 策略。
        student 策略采用监督学习的方式进行训练，输入为实际机器人可获得的观测信息，比较对象是 teacher 策略。
        '''
        return self.privileged_obs_buf

    def reset_idx(self, env_ids):
        """Reset selected robots"""
        raise NotImplementedError

    def reset(self):
        """ Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs, privileged_obs, _, _, _ = self.step(torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        return obs, privileged_obs

    def step(self, actions):
        '''
        参数 actions 的形状就是 num_envs 行， num_actions 列的矩阵；

        这个函数将在 LeggedRobot 类里面进行具体实现。
        它的实现包括提供仿真观测窗口、仿真信息的更新、扭矩计算、奖励计算、终止条件检测和处理、观测值测量等。

        step函数的返回值应该是包含： 
            self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras
            执行完动作之后的新测量到的观测量；
            执行完动作之后的新得到的特权观测量；
            执行完动作之后获得的奖励；
            执行完动作之后需要被重置的机器人表；
            其它信息；
        '''
        raise NotImplementedError

    def render(self, sync_frame_time=True):
        if self.viewer:
            # check for window closed
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()

            # check for keyboard events
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync

            # fetch results
            if self.device != 'cpu':
                self.gym.fetch_results(self.sim, True)

            # step graphics
            if self.enable_viewer_sync:
                '''
                如果 enable_viewer_sync 则在这里调用 gym 库提供画面更新；不然就保持不变；
                '''
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)
                if sync_frame_time:
                    self.gym.sync_frame_time(self.sim)
            else:
                self.gym.poll_viewer_events(self.viewer)