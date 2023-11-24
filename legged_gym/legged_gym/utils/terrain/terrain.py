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

import numpy as np
from numpy.random import choice
from scipy import interpolate

from isaacgym import terrain_utils
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg

class Terrain:
    '''
    地形定义，它需要两个参数：
    config.terrain：config文件中关于地形的定义；
    num_robots：机器人的数量；

    它由 legged_robot.py 文件调用， LeggedRobot 类的 create_sim 方法中被例化。
    '''
    def __init__(self, cfg: LeggedRobotCfg.terrain, num_robots) -> None:

        self.cfg = cfg
        self.num_robots = num_robots
        self.type = cfg.mesh_type
        if self.type in ["none", 'plane']:
            return
        self.env_length = cfg.terrain_length
        self.env_width = cfg.terrain_width
        self.proportions = [np.sum(cfg.terrain_proportions[:i+1]) for i in range(len(cfg.terrain_proportions))] # 作用是将基本元素累计保存； [0.1, 0.1, 0.35, 0.25, 0.2]-->[0.1, 0.2, 0.55, 0.8, 1.0]

        self.cfg.num_sub_terrains = cfg.num_rows * cfg.num_cols # num_sub_terrains 表示有多少小块地形；地形的总数是地形的行数与地形的列数的乘积；
        self.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3)) # 前两维标号地形，3 用来储存 3 维的中心坐标；

        self.width_per_env_pixels = int(self.env_width / cfg.horizontal_scale) # 每个小地形的宽度/水平缩放，获得每个像素点对应的宽度；比如：8m/0.1m=80，每个像素点代表0.1m，表示8米需要80个像素点；
        self.length_per_env_pixels = int(self.env_length / cfg.horizontal_scale) # 每个小地形的长度/水平缩放，获得每个像素点对应的长度；

        self.border = int(cfg.border_size/self.cfg.horizontal_scale) # 这是边界的大小计算；25m/0.1=250，每个像素点代表0.1m，表示25米需要250个像素点；
        self.tot_cols = int(cfg.num_cols * self.width_per_env_pixels) + 2 * self.border # 20*80+2*150=1600+300=1900列；整个地形图的像素列数；
        self.tot_rows = int(cfg.num_rows * self.length_per_env_pixels) + 2 * self.border # 10*80+2*150=800+300=1200列；整个地形图的像素行数；

        self.height_field_raw = np.zeros((self.tot_rows , self.tot_cols), dtype=np.int16) # 用整个地形图的行数和列数初始化地图大小；1200*1900裸图大小；
        if cfg.curriculum: # 在 config 文件的 command下定义的，默认为 True；
            self.curiculum()
        elif cfg.selected: # 在 config 文件的 terrain 下定义的，默认为 False；
            self.selected_terrain()
        else:    
            self.randomized_terrain() # 两者都没定义的情况下采用随机的地形组合；
        
        self.heightsamples = self.height_field_raw
        if self.type=="trimesh":
            self.vertices, self.triangles = terrain_utils.convert_heightfield_to_trimesh(   self.height_field_raw,
                                                                                            self.cfg.horizontal_scale,
                                                                                            self.cfg.vertical_scale,
                                                                                            self.cfg.slope_treshold)
    
    def randomized_terrain(self):
        '''
        循环遍历地生成每个地形，一共生成 num_sub_terrains 个小地形；

        困难程度会从给定的数据中随机选择；
        每次循环最后生成地形并添加到地图中；
        '''
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            choice = np.random.uniform(0, 1) # 因为 choice 是从[0,1]中选择的，所以有效的 proportions 和应该要求是 1 ；
            difficulty = np.random.choice([0.5, 0.75, 0.9]) # 困难度会从提供的这三个数中随机挑选；
            terrain = self.make_terrain(choice, difficulty)
            self.add_terrain_to_map(terrain, i, j)
        
    def curiculum(self):
        '''
        按照一定的规律生成地形；

        每一行的困难度逐步上升，由0直到1；
        每一列的具体地形由0.001到1.001的概率逐步过渡；
        每次循环最后将生成的地形添加到地图中；
        '''
        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                difficulty = i / self.cfg.num_rows
                choice = j / self.cfg.num_cols + 0.001

                terrain = self.make_terrain(choice, difficulty)
                self.add_terrain_to_map(terrain, i, j)

    def selected_terrain(self):
        '''
        这个根据在 config 文件中的 terrain类中 terrain_kwargs 字典中定义的地形名来创建地形；只能创建选定的一种地形；

        使用时需要启用 selected = True ，并且给 terrain_kwargs 字典传入要调用的地形函数关键字及其相应的配置，
        这些关键字来源于自己定义的或者 Isaacgym 的 terrain_utils.py 文件中定义的地形函数；
        
        举例：
        如果要实现这样的调用 terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.) ，则需要定义这样的字典： 
        tpye=terrain_utils.pyramid_sloped_terrain;
        slope=slope;
        platform_size=3.;

        '''
        terrain_type = self.cfg.terrain_kwargs.pop('type')
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            terrain = terrain_utils.SubTerrain("terrain",
                              width=self.width_per_env_pixels,
                              length=self.width_per_env_pixels,
                              vertical_scale=self.vertical_scale,
                              horizontal_scale=self.horizontal_scale)

            eval(terrain_type)(terrain, **self.cfg.terrain_kwargs.terrain_kwargs) # 用选择的地形关键字来控制生成的地形；
            # 上面的代码相当于这样的调用： terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
            self.add_terrain_to_map(terrain, i, j)
    
    def make_terrain(self, choice, difficulty):
        '''
        这个函数是为随机地形情况采用的 randomized_terrain 函数的地形生成服务的；

        choice: 取值范围为[0,1]，是一个概率数；
        difficulty: 取值可以是任意浮点数，如：[0.5, 0.75, 0.9]；
        '''
        terrain = terrain_utils.SubTerrain(   "terrain",
                                width=self.width_per_env_pixels,
                                length=self.width_per_env_pixels,
                                vertical_scale=self.cfg.vertical_scale,
                                horizontal_scale=self.cfg.horizontal_scale)
        slope = difficulty * 0.4
        step_height = 0.05 + 0.18 * difficulty
        discrete_obstacles_height = 0.05 + difficulty * 0.2 # 离散随机方块障碍的方块高度；
        stepping_stones_size = 1.5 * (1.05 - difficulty)
        stone_distance = 0.05 if difficulty==0 else 0.1
        gap_size = 1. * difficulty # 修改降低难度
        pit_depth = 1. * difficulty # 修改降低难度
        if choice < self.proportions[0]:# choice < 0.1
            '''金字塔形斜坡地形'''
            if choice < self.proportions[0]/ 2:# choice < 0.05
                '''金字塔形斜坡地形-下坡'''
                slope *= -1 # 如果选择更小，那么坡度就变成负的；
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
        elif choice < self.proportions[1]: # choice < 0.2
            '''金字塔形斜坡上坡、随机均匀地形'''
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
            terrain_utils.random_uniform_terrain(terrain, min_height=-0.05, max_height=0.05, step=0.005, downsampled_scale=0.2)
        elif choice < self.proportions[3]: # choice < 0.8
            '''上下楼梯地形'''
            if choice<self.proportions[2]: # choice < 0.55
                '''上下楼梯地形-下楼梯'''
                step_height *= -1
            terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.31, step_height=step_height, platform_size=3.)
        elif choice < self.proportions[4]: # choice < 1.0
            '''20个随机方块儿作为障碍物构成的地形'''
            num_rectangles = 20  # 随机方块儿的数量
            rectangle_min_size = 1.
            rectangle_max_size = 2.
            terrain_utils.discrete_obstacles_terrain(terrain, discrete_obstacles_height, rectangle_min_size, rectangle_max_size, num_rectangles, platform_size=3.)
        elif choice < self.proportions[5]: # choice < ? # 不知道为什么这里并没有选项也不会报错；
            '''垫脚石地形'''
            terrain_utils.stepping_stones_terrain(terrain, stone_size=stepping_stones_size, stone_distance=stone_distance, max_height=0., platform_size=4.)
        elif choice < self.proportions[6]: # choice < ?
            '''缺口地形'''
            gap_terrain(terrain, gap_size=gap_size, platform_size=3.)
        else:
            '''矿坑地形'''
            pit_terrain(terrain, depth=pit_depth, platform_size=4.)
        
        return terrain

    def add_terrain_to_map(self, terrain, row, col):
        '''
        将单独生成的 subterrain 按照标号添加到整幅地图中；用传入的地形高度数值给裸图赋值；
        同时也准备了各个小地形图的中心坐标位置以供 env 初始化和重置时作为原点使用；

        首先根据传入的 i, j 标号计算地形图对应在整图的像素点起始位置；
        然后依据起始像素点选择，将调用 isaacgym 库生成的地形区赋值到相应的位置上；
        最后计算每个小地形图的中心位置作为 env 初始化和重置时的原点使用；

        作用结果：
            self.height_field_raw-地形图，值为像素点高度，用x,y像素标号索引；
            self.env_origins-地形图原点，值为中心x,y,z坐标，用小地形图标号索引；
        '''
        i = row
        j = col
        # map coordinate system
        start_x = self.border + i * self.length_per_env_pixels
        end_x = self.border + (i + 1) * self.length_per_env_pixels
        start_y = self.border + j * self.width_per_env_pixels
        end_y = self.border + (j + 1) * self.width_per_env_pixels
        self.height_field_raw[start_x: end_x, start_y:end_y] = terrain.height_field_raw # 这一步给裸图赋值；用传入的地形高度数值给裸图赋值；

        env_origin_x = (i + 0.5) * self.env_length # 第n个小地形的中心位置就是：n个小地形长度加上半个小地形长度；单位 m；
        env_origin_y = (j + 0.5) * self.env_width # 宽度同理；
        x1 = int((self.env_length/2. - 1) / terrain.horizontal_scale)
        x2 = int((self.env_length/2. + 1) / terrain.horizontal_scale)
        y1 = int((self.env_width/2. - 1) / terrain.horizontal_scale)
        y2 = int((self.env_width/2. + 1) / terrain.horizontal_scale)
        env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2])*terrain.vertical_scale # z 方向的高度直接从小地形图的中点得到，再乘上像素缩放转换成 m 为单位；
        self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z] # 以 m 为单位的地形图中心坐标点；（以地图为边界计算，不含 border 定义的边界）

def gap_terrain(terrain, gap_size, platform_size=1.):
    '''
    这里自定义了缺口地形；
    gap_size：缺口大小；
    platform_size：是整个小地形块儿中平地的大小。这在 isaacgym/python/isaacgym/terrain_utils.py 中定义这些地形函数时有说明；

    首先将中心范围内 x2=x1+gap_size 的地图赋极小值-1000；
    然后再将中心范围内 x1 的地图再赋值回 0 ；
    最终得到一个宽度为 gap_size 的凹陷的方形环；
    '''
    gap_size = int(gap_size / terrain.horizontal_scale) # 距离转化成像素点数；
    platform_size = int(platform_size / terrain.horizontal_scale)

    center_x = terrain.length // 2 # terrain.length config文件中定义的小地形块儿的长度；center_x 就是地形块儿的长度中点；
    center_y = terrain.width // 2 # terrain.length config文件中定义的小地形块儿的宽度；center_x 就是地形块儿的宽度中点；
    x1 = (terrain.length - platform_size) // 2
    x2 = x1 + gap_size
    y1 = (terrain.width - platform_size) // 2
    y2 = y1 + gap_size
   
    terrain.height_field_raw[center_x-x2 : center_x + x2, center_y-y2 : center_y + y2] = -1000 # 将m转化成像素点数；
    terrain.height_field_raw[center_x-x1 : center_x + x1, center_y-y1 : center_y + y1] = 0

def pit_terrain(terrain, depth, platform_size=1.):
    '''
    这里定义了矿坑地形；
    depth： 矿坑深度；

    首先计算 platform 的半宽；
    然后将 platform 设置为矿坑的深度 depth ；
    '''
    depth = int(depth / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale / 2)
    x1 = terrain.length // 2 - platform_size
    x2 = terrain.length // 2 + platform_size
    y1 = terrain.width // 2 - platform_size
    y2 = terrain.width // 2 + platform_size
    terrain.height_field_raw[x1:x2, y1:y2] = -depth
