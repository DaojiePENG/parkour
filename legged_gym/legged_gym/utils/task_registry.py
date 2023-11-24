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

import os
from datetime import datetime
from typing import Tuple
import torch
import numpy as np
import json
import shutil

from rsl_rl.env import VecEnv
from rsl_rl.runners import build_runner, OnPolicyRunner

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from .helpers import get_args, update_cfg_from_args, class_to_dict, get_load_path, set_seed, parse_sim_params
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class TaskRegistry():
    '''
    该类在本文件最后进行初始化，创造了全局类变量 task_registry 。
    该类在 __init__.py (legged_gym/legged_gym/envs) 中被调用来注册待训练的机器人及其配置信息；
    该类在 play.py tain.py test_env.py 中进行了调用使用 make_env 和 make_alg_runner 来作为参数来创建环境和运行算法；

    可以说这个类是整个 legged_gym 库里面最上层的类了，在它之上只有 play.py tain.py test_env.py 运行脚本的封装了。
    '''
    def __init__(self):
        self.task_classes = {}
        self.env_cfgs = {}
        self.train_cfgs = {}
    
    def register(self, name: str, task_class: VecEnv, env_cfg: LeggedRobotCfg, train_cfg: LeggedRobotCfgPPO):
        '''
        这个函数是整个 TaskRegistry 类的核心函数，外部函数会通过调用它来实现注册功能；
        它会将机器人的任务类、训练类、环境类都放在字典里面；
        '''
        self.task_classes[name] = task_class
        self.env_cfgs[name] = env_cfg
        self.train_cfgs[name] = train_cfg
    
    def get_task_class(self, name: str) -> VecEnv:
        return self.task_classes[name]
    
    def get_cfgs(self, name) -> Tuple[LeggedRobotCfg, LeggedRobotCfgPPO]:
        '''
        这个函数只在 play.py 和 test_env.py 使用到了，应该是从 name 关键名称提取相应的配置。
        也就是说这个 train_cfgs[name] 是一个保存了各个机器人配置的字典。它包括了所有用 task_registry 注册过的机器人配置的配置。
        
        实际使用时会根据提供的 name 从 train_cfgs[name] 中提取特定的机器人配置 train_cfg 进行赋值。
        '''
        train_cfg = self.train_cfgs[name]
        env_cfg = self.env_cfgs[name]
        # copy seed
        env_cfg.seed = train_cfg.seed
        return env_cfg, train_cfg
    
    def make_env(self, name, args=None, env_cfg=None) -> Tuple[VecEnv, LeggedRobotCfg]:
        """ Creates an environment either from a registered namme or from the provided config file.

        Args:
            name (string): Name of a registered env.
            args (Args, optional): Isaac Gym comand line arguments. If None get_args() will be called. Defaults to None.
            env_cfg (Dict, optional): Environment config file used to override the registered config. Defaults to None.

        Raises:
            ValueError: Error if no registered env corresponds to 'name' 

        Returns:
            isaacgym.VecTaskPython: The created environment
            Dict: the corresponding config file
        """
        # if no args passed get command line arguments
        if args is None:
            args = get_args()
        # check if there is a registered env with that name
        if name in self.task_classes:
            task_class = self.get_task_class(name)
        else:
            raise ValueError(f"Task with name: {name} was not registered")
        if env_cfg is None:
            # load config files
            env_cfg, _ = self.get_cfgs(name)
        # override cfg from args (if specified)
        env_cfg, _ = update_cfg_from_args(env_cfg, None, args)
        set_seed(env_cfg.seed)
        # parse sim params (convert to dict first)
        sim_params = {"sim": class_to_dict(env_cfg.sim)}
        sim_params = parse_sim_params(args, sim_params)
        env = task_class(   cfg=env_cfg,
                            sim_params=sim_params,
                            physics_engine=args.physics_engine,
                            sim_device=args.sim_device,
                            headless=args.headless)
        return env, env_cfg

    def make_alg_runner(self, env, name=None, args=None, train_cfg=None, log_root="default", env_cfg=None, save_cfg= True) -> Tuple[OnPolicyRunner, LeggedRobotCfgPPO]:
        """ Creates the training algorithm  either from a registered namme or from the provided config file.

        Args:
            env (isaacgym.VecTaskPython): The environment to train (TODO: remove from within the algorithm)
            name (string, optional): Name of a registered env. If None, the config file will be used instead. Defaults to None.
            args (Args, optional): Isaac Gym comand line arguments. If None get_args() will be called. Defaults to None.
            train_cfg (Dict, optional): Training config file. If None 'name' will be used to get the config file. Defaults to None.
            log_root (str, optional): Logging directory for Tensorboard. Set to 'None' to avoid logging (at test time for example). 
                                      Logs will be saved in <log_root>/<date_time>_<run_name>. Defaults to "default"=<path_to_LEGGED_GYM>/logs/<experiment_name>.
            env_cfg (Dict, optional): for logging

        Raises:
            ValueError: Error if neither 'name' or 'train_cfg' are provided
            Warning: If both 'name' or 'train_cfg' are provided 'name' is ignored

        Returns:
            PPO: The created algorithm
            Dict: the corresponding config file
        """
        # if no args passed get command line arguments
        if args is None:
            args = get_args()
        # if config files are passed use them, otherwise load from the name
        if train_cfg is None:
            if name is None:
                raise ValueError("Either 'name' or 'train_cfg' must be not None")
            # load config files
            _, train_cfg = self.get_cfgs(name)
        else:
            if name is not None:
                print(f"'train_cfg' provided -> Ignoring 'name={name}'")
        # override cfg from args (if specified)
        _, train_cfg = update_cfg_from_args(None, train_cfg, args)

        if log_root=="default":
            log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name)
            log_dir = os.path.join(log_root, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + train_cfg.runner.run_name)
        elif log_root is None:
            log_dir = None
        else:
            log_dir = os.path.join(log_root, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + train_cfg.runner.run_name)

        log_cfg_dict = dict()
        train_cfg_dict = class_to_dict(train_cfg)
        log_cfg_dict.update(train_cfg_dict)
        if not env_cfg is None:
            env_cfg_dict = class_to_dict(env_cfg)
            log_cfg_dict.update(env_cfg_dict)
        '''
        所有与 rsl_rl 库联系的内容集中在下面这句代码上了。
        它调用 build_runner/OnPolicyRunner 完成了神经网络的创建和优化算法的选择，这两者的具体实现在rsl_rl库中。
        '''
        runner = build_runner(
            getattr(train_cfg, "runner_class_name", "OnPolicyRunner"),
            env,
            train_cfg_dict,
            log_dir,
            device=args.rl_device,
        )
        
        if save_cfg:
            os.makedirs(log_dir, exist_ok= True)
            with open(os.path.join(log_dir, "config.json"), "w") as f:
                json.dump(log_cfg_dict, f, indent= 4)
        resume = train_cfg.runner.resume
        if resume:
            '''
            这个就是当训练好了模型后，希望载入训练好的模型参数来直接使用的情况。
            也是返回一个可调用的 runner ，也就是一个包含很多函数的神经网络类，只不过这时只会使用其中获取动作控制目标值的函数。
            实际使用的时候调用它的 get_inference_policy 函数就可以了。如 play.py 中的使用一样。
            '''
            # load previously trained model
            resume_path = get_load_path(log_root, load_run=train_cfg.runner.load_run, checkpoint=train_cfg.runner.checkpoint)
            print(f"Loading model from: {resume_path}")
            if save_cfg:
                shutil.copyfile(resume_path, os.path.join(log_dir, os.path.basename(resume_path)))
            runner.load(resume_path)
        return runner, train_cfg

# make global task registry
task_registry = TaskRegistry()