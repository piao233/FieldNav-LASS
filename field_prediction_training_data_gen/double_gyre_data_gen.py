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


class BaseEnvironment(gym.Env):

    def __init__(self, field_file_path, render_mode=None,
                 is_fixed_start_and_target=True, swap_start_and_target=False,
                 swim_vel=0.9, dxdy_norm_scale=8, seed=None):

        self.swim_vel = swim_vel
        self.render_mode = render_mode
        self.is_fixed_start_and_target = is_fixed_start_and_target
        self.swap_start_and_target = swap_start_and_target
        self.dxdy_norm_scale = dxdy_norm_scale
        self.seed(seed)

        # 仿真相关参数
        self.x_min, self.x_max, self.y_min, self.y_max = None, None, None, None  # need override
        self.dt = None  # need override
        self.t_step_max = 400  # may need override
        self.rand_t0_within = 0  # need override
        self.rand_start_target_r = 0  # need override
        self.target_r = None  # need override
        self.default_start, self.default_target = None, None  # need override
        self.env_cycle = np.inf  # may need override
        self.render_quiver_x = 21  # need override
        self.render_quiver_y = 11  # need override
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

        # 读取流场数据
        self.field_data = loadmat(field_file_path)
        self.x_grid = self.field_data['x_grid']
        self.y_grid = self.field_data['y_grid']
        self.u_grid = self.field_data['u_grid']
        self.v_grid = self.field_data['v_grid']
        self.o_grid = self.field_data['o_grid']
        self.Vel_grid = self.field_data['Vel_grid']

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
        self.observation_space = box

    def set_action_space(self, box):
        self.action_space = box

    def set_dxdy_norm_scale(self, k):
        self.dxdy_norm_scale = k

    def reward_shaping(self, sth):
        pass

    def set_ts_per_step(self, i):
        self.ts_per_step = i
        self.dt = self.dt * i

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t0 = np.random.randint(0, self.rand_t0_within) * self.dt
        self.t_step = 0
        if not self.is_fixed_start_and_target:
            self.start = self._rand_in_circle(self.default_start, self.rand_start_target_r)
            self.target = self._rand_in_circle(self.default_target, self.rand_start_target_r)
            if self.swap_start_and_target:
                if np.random.uniform() > 0.5:
                    self.start = self._rand_in_circle(self.default_target, self.rand_start_target_r)
                    self.target = self._rand_in_circle(self.default_start, self.rand_start_target_r)

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

    def _get_obs(self, input_info=None):
        if input_info is None:
            input_info = self._get_info()
        dx_norm = self._normalize(input_info["dx"], self.observation_space.low[0], self.observation_space.high[0],
                                  scale=self.dxdy_norm_scale)
        dy_norm = self._normalize(input_info["dy"], self.observation_space.low[1], self.observation_space.high[1],
                                  scale=self.dxdy_norm_scale)
        u_norm = self._normalize(input_info["u"], self.observation_space.low[2], self.observation_space.high[2])
        v_norm = self._normalize(input_info["v"], self.observation_space.low[3], self.observation_space.high[3])

        return {"dx": dx_norm, "dy": dy_norm, "u": u_norm, "v": v_norm}

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
        plt.clf()
        cur_time = self.t0 + self.t_step * self.dt

        plt.imshow(np.rot90(self.Vel_grid[self._env_cycle(int(cur_time / self.dt))]), origin='upper',
                   extent=(self.x_min, self.x_max, self.y_min, self.y_max),
                   cmap='coolwarm', aspect='equal', vmin=0, vmax=3)
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
        plt.legend()

        plt.xlim(self.x_min, self.x_max)
        plt.ylim(self.y_min, self.y_max)
        plt.title(f't={self.t0:.2f} + {self.t_step * self.dt:.2f}', fontsize=14)
        plt.draw()
        plt.pause(self.frame_pause)


class DoubleGyreEnvironmentGridFixSpeed(BaseEnvironment, ABC):

    def __init__(self, field_file_path='./DoubleGyre_grid_0.0to2.0of401_0.0to1.0of201.mat',
                 render_mode=None, is_fixed_start_and_target=True, swap_start_and_target=False,
                 swim_vel=0.9, dxdy_norm_scale=8, seed=None):

        super().__init__(field_file_path, render_mode, is_fixed_start_and_target, swap_start_and_target, swim_vel,
                         dxdy_norm_scale, seed)

        self.env_name = 'double_gyre'
        self.x_min, self.x_max = 0, 2
        self.y_min, self.y_max = 0, 1
        self.dt = 0.01
        self.t_step_max = 10086
        self.default_start = np.array([1.5, 0.5])
        self.default_target = np.array([0.5, 0.5])
        self.start = self.default_start
        self.target = self.default_target
        self.target_r = 1 / 50
        self.rand_t0_within = 33
        self.rand_start_target_r = 1 / 4
        self.env_cycle = 100
        self.render_quiver_x = 21
        self.render_quiver_y = 11

        self.action_space = gym.spaces.Box(low=0, high=2 * np.pi, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.array([-2, -1, -5, -5], dtype=np.float32),
            high=np.array([2, 1, 5, 5], dtype=np.float32)
        )

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
            if self.t_step >= self.t_step_max:
                _truncated = True
                _reward = -0
                print(colored("**** Episode Finished **** Reaches Env Time limit.", 'blue'))
            else:
                if d <= self.target_r * 2.5:
                    mid = 0.5 * (new_pos + pos_cur)
                    d_mid = np.linalg.norm(mid - self.target)
                    if d_mid <= self.target_r or d <= self.target_r:
                        _terminated = True
                        _reward = 200
                        print(colored("**** Episode Finished **** SUCCESS.", 'green'))
                else:
                    _reward = -self.dt - 10 * (d - self.d_2_target) / self.swim_vel

        self.d_2_target = d

        if self.render_mode == 'human':
            self._render_frame()

        return _observation, _reward, _terminated, _truncated, _info

    def generate_training_data(self, num_episodes=1000, save_path='./', file_name='training_data.mat'):
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        file_path = os.path.join(save_path, file_name)

        for episode in range(num_episodes):
            self.reset()
            episode_data = []
            terminated = False
            truncated = False
            input_stack = []

            while not (terminated or truncated):

                action = np.array([joy_stick.get_angle()])
                # action = [self.action + 0.1*np.pi*np.random.randn()]

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

            # 将episode_data转换为numpy数组
            episode_inputs = np.array([data['input'] for data in episode_data])
            episode_outputs = np.array([[data['output']] for data in episode_data])

            # 保存到MAT文件
            if os.path.exists(file_path):
                existing_data = loadmat(file_path)
                inputs = existing_data['inputs']
                outputs = existing_data['outputs']
                new_inputs = np.vstack((inputs, episode_inputs))
                new_outputs = np.vstack((outputs, episode_outputs))
            else:
                new_inputs = episode_inputs
                new_outputs = episode_outputs

            savemat(file_path, {'inputs': new_inputs, 'outputs': new_outputs})
            print(f"Episode {episode} data saved to {file_path}")

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
    env = DoubleGyreEnvironmentGridFixSpeed(is_fixed_start_and_target=False,
                                            render_mode='human',
                                            swap_start_and_target=True,
                                            swim_vel=0.9)
    is_manual_control = True
    if is_manual_control:
        joy_stick = JoystickSimulator()
    env.generate_training_data(num_episodes=1000)

# if __name__ == '__main__':
#     # 加载MAT文件
#     file_path = './training_data.mat'
#     data = loadmat(file_path)
#
#     # 提取输入数据
#     inputs = data['inputs']
#
#     # 获取位置数据
#     positions = inputs[:, 0, :2]
#
#     # 创建绘图
#     plt.figure(figsize=(10, 6))
#     plt.scatter(positions[:, 0], positions[:, 1], s=1, c='blue', alpha=0.5)
#     plt.title('Agent Positions')
#     plt.xlabel('Position X')
#     plt.ylabel('Position Y')
#     plt.grid(True)
#     plt.show()
