from abc import abstractmethod, ABC
import gymnasium as gym
import numpy as np
from gymnasium.utils import seeding
from gymnasium import spaces
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
import math
from termcolor import colored
import os
from joystick_controller import JoystickSimulator
from env_obs_strategy import ObsNone

class EnvironmentBase(gym.Env, ABC):
    def __init__(self, field_file_path, render_mode=None,
                 random_start_and_target=False, swap_start_and_target=False, random_t0=False,
                 swim_vel=0.9, dxdy_norm_scale=8, seed=None):

        self.swim_vel = swim_vel
        self.render_mode = render_mode
        self.random_start_and_target = random_start_and_target
        self.random_t0 = random_t0
        self.swap_start_and_target = swap_start_and_target
        self.dxdy_norm_scale = dxdy_norm_scale
        self.seed(seed)

        # 仿真相关参数
        self.env_name = 'need override'  # need override
        self.x_min, self.x_max, self.y_min, self.y_max = None, None, None, None  # need override
        self.dt = None  # need override
        self.t_step_max = 400  # may need override
        self.rand_t0_within = 0  # need override
        self.rand_start_target_r = 0  # need override
        self.target_r = None  # need override
        self.default_start, self.default_target = None, None  # need override
        self.default_t0 = 0
        self.env_cycle = np.inf  # may need override
        self.render_quiver_x = 21  # need override
        self.render_quiver_y = 11  # need override
        self.render_vel_max = 3  # need override
        self.ts_per_step = 1  # step()每次前进几个timestep
        self.frame_pause = 0.01

        # else
        self.agent_pos_history = []
        self.start, self.target = self.default_start, self.default_target
        self.action = 0
        self.agent_pos = np.array([0, 0])
        self.prev_dist_to_start = 0
        self.t0 = 0
        self.t_step = 0
        self.d_2_target = 0
        self.strategy = None
        self.save_rendering = False  # 是否保存每步渲染仿真

        # 读取流场数据
        self.field_data = loadmat(field_file_path)
        self.x_grid = self.field_data['x_grid']
        self.y_grid = self.field_data['y_grid']
        self.u_grid = self.field_data['u_grid']
        self.v_grid = self.field_data['v_grid']
        self.o_grid = self.field_data['o_grid']
        self.Vel_grid = self.field_data['Vel_grid']

    def apply(self, strategy):
        self.strategy = strategy
        self.strategy.apply(self)  # 向self注入or更改必要变量

    def _get_obs(self, input_info=None):
        if self.strategy is None:
            raise AttributeError("请指定构建state_space的strategy！ e.g. env.apply( *Obs strategy* )")
        # 具体的get_obs由不同的state_space策略实现
        return self.strategy._get_obs(self, input_info)

    @staticmethod
    def _normalize(_in, _min, _max, scale=1.0):
        return scale * (2 * (_in - _min) / (_max - _min) - 1)

    @staticmethod
    def _rand_in_circle(origin, r):
        d = math.sqrt(np.random.uniform(0, r ** 2))
        theta = np.random.uniform(0, 2 * math.pi)
        return origin + np.array([d * math.cos(theta), d * math.sin(theta)])

    def _env_cycle(self, t_step):
        return int(t_step % self.env_cycle)

    @abstractmethod
    def step(self, input_action):
        pass

    def set_state_space(self, box):
        """
        Need to be like gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        :param box: gym.space.Box()
        :return: None
        """
        self.observation_space = box

    def set_action_space(self, box):
        """
        Need to be like gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        :param box: gym.space.Box()
        :return: None
        """
        self.action_space = box

    def set_dxdy_norm_scale(self, k):
        """
        设置归一化时候的dxdy归一化值，虽然在实例化时可以传参，但先写在这儿
        :param k:
        :return:
        """
        self.dxdy_norm_scale = k

    def reward_shaping(self, sth):
        # TODO: doesn't quite sure how to do this
        pass

    def set_ts_per_step(self, i):
        self.ts_per_step = i
        self.dt = self.dt * i

    def set_default_start(self, default_start):
        self.default_start = np.array(default_start)

    def set_default_target(self, default_target):
        self.default_target = np.array(default_target)

    def set_default_t0(self, default_t0):
        self.default_t0 = np.array(default_t0)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t0 = self.default_t0
        self.t_step = 0
        if self.random_t0:
            self.t0 = np.random.randint(0, self.rand_t0_within) * self.dt
        if self.random_start_and_target:
            self.start = self._rand_in_circle(self.default_start, self.rand_start_target_r)
            self.target = self._rand_in_circle(self.default_target, self.rand_start_target_r)
            if self.swap_start_and_target:
                if np.random.uniform() > 0.5:
                    self.start = self._rand_in_circle(self.default_target, self.rand_start_target_r)
                    self.target = self._rand_in_circle(self.default_start, self.rand_start_target_r)
        else:
            self.start = self.default_start
            self.target = self.default_target

        self.agent_pos = self.start
        self.d_2_target = np.linalg.norm(self.agent_pos - self.target)
        self.agent_pos_history = [self.agent_pos]

        observation = self._get_obs()
        info = self._get_info()
        if self.render_mode == 'human':
            self._render_frame()
        return observation, info

    def _get_info(self):
        delta = self.target - self.agent_pos
        t_index = self._env_cycle(self.t0 / self.dt + self.t_step)
        x_index = int((self.agent_pos[0] - self.x_grid[0, 0]) / (self.x_grid[1, 0] - self.x_grid[0, 0]))
        y_index = int((self.agent_pos[1] - self.y_grid[0, 0]) / (self.y_grid[0, 1] - self.y_grid[0, 0]))
        u = self.u_grid[t_index, x_index, y_index]
        v = self.v_grid[t_index, x_index, y_index]
        o = self.o_grid[t_index, x_index, y_index]
        Vel = math.sqrt(u ** 2 + v ** 2)
        if self.t_step != 0:
            u_last = self.u_grid[t_index - 1, x_index, y_index]
            v_last = self.v_grid[t_index - 1, x_index, y_index]
            _du = u - u_last
            _dv = v - v_last
            du = _du / self.dt if _du != 0 else 0
            dv = _dv / self.dt if _du != 0 else 0
        else:
            du = dv = 0

        return {"dx": delta[0], "dy": delta[1], "u": u, "v": v, "du": du, "dv": dv, "o": o, "Vel": Vel,
                "x": self.agent_pos[0], "y": self.agent_pos[1], "t_index": t_index}

    def _get_uv_at(self, t_step, x, y):
        if not (self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max):
            return 0, 0
        t_index = self._env_cycle(self.t0 / self.dt + t_step)
        x_index = int((x - self.x_grid[0, 0]) / (self.x_grid[1, 0] - self.x_grid[0, 0]))
        y_index = int((y - self.y_grid[0, 0]) / (self.y_grid[0, 1] - self.y_grid[0, 0]))
        u = self.u_grid[t_index, x_index, y_index]
        v = self.v_grid[t_index, x_index, y_index]
        return u, v

    def render(self):
        self._render_frame()

    def close(self):
        plt.close('all')

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _render_frame(self):
        if len(self.agent_pos_history) == 0:
            return
        plt.figure(self.env_name, figsize=(10, 6))
        plt.clf()
        cur_time = self.t0 + self.t_step * self.dt

        plt.imshow(np.rot90(self.Vel_grid[self._env_cycle(int(cur_time / self.dt))]), origin='upper',
                   extent=(self.x_min, self.x_max, self.y_min, self.y_max),
                   cmap='coolwarm', aspect='equal', vmin=0, vmax=self.render_vel_max)
        plt.colorbar()
        X, Y = np.meshgrid(np.linspace(self.x_min, self.x_max, self.render_quiver_x),
                           np.linspace(self.y_min, self.y_max, self.render_quiver_y))
        UU = np.zeros_like(X)
        VV = np.zeros_like(Y)
        t_index = self._env_cycle(int(cur_time / self.dt))
        for i in range(self.render_quiver_x):
            for j in range(self.render_quiver_y):
                x_index = int((X[j, i] - self.x_grid[0, 0]) / (self.x_grid[1, 0] - self.x_grid[0, 0]))
                y_index = int((Y[j, i] - self.y_grid[0, 0]) / (self.y_grid[0, 1] - self.y_grid[0, 0]))
                UU[j, i] = self.u_grid[t_index, x_index, y_index]
                VV[j, i] = self.v_grid[t_index, x_index, y_index]
        plt.quiver(X, Y, UU, VV)
        agent_positions = np.array(self.agent_pos_history)
        plt.plot(agent_positions[:, 0], agent_positions[:, 1], 'b-')
        plt.scatter(agent_positions[-1, 0], agent_positions[-1, 1], c='blue', label='Agent')
        plt.scatter(*self.target, c='red', label='Target')
        plt.scatter(*self.start, c='green', label='Start')
        if self.action is not None:
            t = max(self.y_max - self.y_min, self.x_max - self.x_min) * 0.075
            plt.arrow(float(self.agent_pos[0]), float(self.agent_pos[1]), t * math.cos(self.action),
                      t * math.sin(self.action), shape='full', color='blue', linewidth=2, head_width=0.25 * t,
                      length_includes_head=True)

        self.strategy.render_more(self)

        plt.legend()

        plt.xlim(self.x_min, self.x_max)
        plt.ylim(self.y_min, self.y_max)
        plt.title(f't={self.t0:.2f} + {self.t_step * self.dt:.2f}', fontsize=14)
        plt.tight_layout()
        plt.draw()
        plt.pause(self.frame_pause)

        if self.save_rendering:
            save_path = './temp_save_folder'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            # TODO: 更改渲染保存选项
            file_name = f'frame_{self.t_step}_pred.png'
            file_path = os.path.join(save_path, file_name)
            plt.savefig(file_path)

        # plt.pause(self.frame_pause)

    def render_with_compared_path(self, x_list, y_list):
        if not (x_list is None or y_list is None):
            plt.figure(self.env_name, figsize=(10, 6))
            plt.plot(x_list, y_list, 'm--', label='optimal traj')
            plt.legend()
            # plt.draw()
            plt.pause(self.frame_pause)

    def render_frame_at_ts(self, time_step):
        if not isinstance(time_step, int):
            raise ValueError("time_step must be an integer.")

        t_index = time_step % self.env_cycle

        if time_step >= self.env_cycle:
            print(f"Time step {time_step} exceeds environment cycle. Rendering at time step {t_index} instead.")

        plt.clf()
        plt.imshow(np.rot90(self.Vel_grid[t_index]), origin='upper',
                   extent=(self.x_min, self.x_max, self.y_min, self.y_max),
                   cmap='coolwarm', aspect='equal', vmin=0, vmax=self.render_vel_max)
        plt.colorbar()

        X, Y = np.meshgrid(np.linspace(self.x_min, self.x_max, self.render_quiver_x),
                           np.linspace(self.y_min, self.y_max, self.render_quiver_y))
        UU = np.zeros_like(X)
        VV = np.zeros_like(Y)

        for i in range(self.render_quiver_x):
            for j in range(self.render_quiver_y):
                x_index = int((X[j, i] - self.x_grid[0, 0]) / (self.x_grid[1, 0] - self.x_grid[0, 0]))
                y_index = int((Y[j, i] - self.y_grid[0, 0]) / (self.y_grid[0, 1] - self.y_grid[0, 0]))
                UU[j, i] = self.u_grid[t_index, x_index, y_index]
                VV[j, i] = self.v_grid[t_index, x_index, y_index]

        plt.quiver(X, Y, UU, VV)

        plt.xlim(self.x_min, self.x_max)
        plt.ylim(self.y_min, self.y_max)
        plt.title(f'Time Step = {time_step}', fontsize=14)
        plt.draw()
        plt.pause(0.01)

    def render_frames_and_save(self, end_time_step, save_path=f'./rendered_frames'):
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for time_step in range(0, end_time_step + 1):
            self.render_frame_at_ts(time_step)
            file_name = f'frame_{time_step}.png'
            file_path = os.path.join(save_path, file_name)
            plt.savefig(file_path)
            print(f'Saved frame {time_step} to {file_path}')

        print("Rendering and saving frames complete.")


class Cylinder2D_Re400(EnvironmentBase):

    def __init__(self, field_file_path='../flow_field/cylinder2D_Re400_ref_grid_-3.0to15.0of361_-6.0to6.0of241.mat',
                 render_mode=None, random_start_and_target=True, swap_start_and_target=False, random_t0=True,
                 swim_vel=0.9, dxdy_norm_scale=8, seed=None):

        super().__init__(field_file_path, render_mode, random_start_and_target, swap_start_and_target, random_t0,
                         swim_vel, dxdy_norm_scale, seed)

        self.env_name = 'cylinder2D_Re400'
        self.x_min, self.x_max = -3, 15
        self.y_min, self.y_max = -6, 6
        self.dt = 0.15
        self.t_step_max = 400
        self.frame_pause = 0.01
        self.default_start = np.array([5.0, -2.1])
        self.default_target = np.array([5.0, 2.1])
        self.start = self.default_start
        self.target = self.default_target
        self.target_r = 1 / 6
        self.rand_t0_within = 38  # reset时，在0至多少tp中随机初始时间t0
        self.rand_start_target_r = 2  # reset时，在默认起终点半径多少内随机起终点
        self.env_cycle = 66  # 仿真环境周期*2，一个周期大约33
        self.render_quiver_x = 20  # 渲染时横轴速度箭头数量
        self.render_quiver_y = 24  # 渲染时纵轴速度箭头数量
        self.render_vel_max = 1.6

        self.action_space = gym.spaces.Box(low=0, high=2 * np.pi, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=np.array([-6, -6, 0.6, -0.4]),
                                                high=np.array([6, 6, 1.4, 0.4]),
                                                shape=(4,), dtype=np.float32)

    def step(self, input_action):
        input_action = input_action[0]
        pos_cur = self.agent_pos
        action_xy = np.array([self.swim_vel * math.cos(input_action), self.swim_vel * math.sin(input_action)])
        self.action = input_action
        _terminated = False
        _truncated = False
        _reward = 0
        _info = self._get_info()
        _observation = self._get_obs(input_info=_info)

        # 更新位置
        u = _info["u"]
        v = _info["v"]
        new_pos = self.agent_pos + (np.array([u, v]) + action_xy) * self.dt

        self.agent_pos = new_pos
        self.agent_pos_history.append(new_pos.copy())
        self.t_step += self.ts_per_step

        d = np.linalg.norm(new_pos - self.target)

        # 检查出界
        if not (self.x_min <= new_pos[0] <= self.x_max and self.y_min <= new_pos[1] <= self.y_max):
            _truncated = True
            _reward = -0
            print(colored("**** Episode Finished **** Hit Boundary.", 'red'))
        else:
            # 检查超时
            if self.t_step >= self.t_step_max:
                _truncated = True
                _reward = -0
                print(colored("**** Episode Finished **** Reaches Env Time limit.", 'blue'))
            # 正常情况
            else:
                # 检查终点
                if d <= self.target_r * 2.5:
                    mid = 0.5 * (new_pos + pos_cur)
                    d_mid = np.linalg.norm(mid - self.target)
                    if d_mid <= self.target_r or d <= self.target_r:
                        _terminated = True
                        _reward = 200
                        print(colored("**** Episode Finished **** SUCCESS.", 'green'))
                # 运行途中
                # TODO: Cylinder Re400 Reward Shaping
                else:
                    _reward = -self.dt - 20 * (d - self.d_2_target) / self.swim_vel

        self.d_2_target = d

        if self.render_mode == 'human':
            self._render_frame()

        return _observation, _reward, _terminated, _truncated, _info

    def generate_training_data(self, num_episodes=100, save_path='./', file_name='training_data_CF_Re400.mat'):
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        file_path = os.path.join(save_path, file_name)

        # 读取MAT文件
        if os.path.exists(file_path):
            existing_data = loadmat(file_path)
            inputs = existing_data['inputs']
            outputs = existing_data['outputs']
        else:
            inputs = None
            outputs = None

        for episode in range(num_episodes):
            self.reset()
            episode_data = []
            terminated = False
            truncated = False
            input_stack = []

            while not (terminated or truncated):

                # Todo: select action
                # action = np.array([joy_stick.get_angle()])
                action = [self.action + 0.1 * np.pi * np.random.randn()]
                # action = np.clip([self.action + 0.1 * np.pi * np.random.randn()], 0.9*3.14, 1.1*3.14)

                obs, reward, terminated, truncated, info = self.step(action)

                current_pos = self.agent_pos_history[-1]
                t_index = info['t_index']
                u, v = self._get_uv_at(t_index, current_pos[0], current_pos[1])
                current_data = [current_pos[0], current_pos[1], u, v]

                if len(input_stack) >= 10:
                    input_stack.pop(0)  # 移除最早的时刻
                input_stack.append(current_data)  # 添加新的时刻

                if len(input_stack) == 10:
                    input_data = np.array(input_stack)
                    output_data = t_index
                    episode_data.append({'input': input_data, 'output': output_data})

            if len(input_stack) < 10:
                continue

            # 将episode_data转换为numpy数组
            episode_inputs = np.array([data['input'] for data in episode_data])
            episode_outputs = np.array([[data['output']] for data in episode_data])
            if inputs is not None:
                inputs = np.vstack((inputs, episode_inputs))
                outputs = np.vstack((outputs, episode_outputs))
            else:
                inputs = episode_inputs
                outputs = episode_outputs

            print(f"Episode {episode} complete.")

        savemat(file_path, {'inputs': inputs, 'outputs': outputs})
        print("Training data generation complete.")

    def render_frame_at_ts(self, time_step):
        if not isinstance(time_step, int):
            raise ValueError("time_step must be an integer.")

        t_index = time_step % self.env_cycle

        if time_step >= self.env_cycle:
            print(f"Time step {time_step} exceeds environment cycle. Rendering at time step {t_index} instead.")

        plt.clf()
        plt.imshow(np.rot90(self.Vel_grid[t_index]), origin='upper',
                   extent=(self.x_min, self.x_max, self.y_min, self.y_max),
                   cmap='coolwarm', aspect='equal', vmin=0, vmax=3)
        plt.colorbar()

        X, Y = np.meshgrid(np.linspace(self.x_min, self.x_max, self.render_quiver_x),
                           np.linspace(self.y_min, self.y_max, self.render_quiver_y))
        UU = np.zeros_like(X)
        VV = np.zeros_like(Y)

        for i in range(self.render_quiver_x):
            for j in range(self.render_quiver_y):
                x_index = int((X[j, i] - self.x_grid[0, 0]) / (self.x_grid[1, 0] - self.x_grid[0, 0]))
                y_index = int((Y[j, i] - self.y_grid[0, 0]) / (self.y_grid[0, 1] - self.y_grid[0, 0]))
                UU[j, i] = self.u_grid[t_index, x_index, y_index]
                VV[j, i] = self.v_grid[t_index, x_index, y_index]

        plt.quiver(X, Y, UU, VV)

        plt.xlim(self.x_min, self.x_max)
        plt.ylim(self.y_min, self.y_max)
        plt.title(f'Time Step = {time_step}', fontsize=14)
        plt.draw()
        plt.pause(0.01)

    # def render_frames_and_save(self, start_time_step, save_path='./double_gyre_rendered_frames'):
    #     if not os.path.exists(save_path):
    #         os.makedirs(save_path)
    #
    #     t_max = 150
    #     time_steps = int(t_max / self.dt)
    #
    #     for time_step in range(start_time_step, time_steps + 1):
    #         self.render_frame_at_ts(time_step)
    #         file_name = f'frame_{time_step}.png'
    #         file_path = os.path.join(save_path, file_name)
    #         plt.savefig(file_path)
    #         print(f'Saved frame {time_step} to {file_path}')
    #
    #     print("Rendering and saving frames complete.")


# 使用示例
if __name__ == '__main__':
    env = Cylinder2D_Re400(random_start_and_target=True,
                           render_mode='none',
                           swap_start_and_target=True,
                           random_t0=True,
                           swim_vel=0.9)
    env.apply(ObsNone())

    is_manual_control = False
    if is_manual_control:
        joy_stick = JoystickSimulator()
    env.generate_training_data(num_episodes=400)

    file_path = './training_data_CF_Re400.mat'
    data = loadmat(file_path)

    # 提取输入数据
    inputs = data['inputs']

    # 获取位置数据
    positions = inputs[:, 0, :2]

    # 创建绘图
    plt.figure(figsize=(10, 6))
    plt.scatter(positions[:, 0], positions[:, 1], s=1, c='blue', alpha=0.5)
    plt.title('Agent Positions')
    plt.xlabel('Position X')
    plt.ylabel('Position Y')
    plt.xlim(-3, 15)
    plt.ylim(-6, 6)
    plt.grid(True)
    plt.show()
