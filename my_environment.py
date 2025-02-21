from termcolor import colored
from abc import abstractmethod, ABC
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import math
import os
import gymnasium as gym
from gymnasium.utils import seeding
from env_obs_strategy import ObsAhead, ObsNone, ObsAheadPredict


##########################################################
# This is the pre-defined environment of DoubleGyre and CylinderFlow.
# RL observation strategies are defined in 'obs_strategy.py'.
##########################################################

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
        self.ts_per_step = 1  # how many timesteps are performed in step()
        self.frame_pause = 0.01

        # else
        self.agent_pos_history = []
        self.start, self.target = self.default_start, self.default_target
        self.action = None
        self.agent_pos = np.array([0, 0])
        self.prev_dist_to_start = 0
        self.t0 = 0
        self.t_step = 0
        self.d_2_target = 0
        self.strategy = None
        self.save_rendering = False

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
        self.strategy.apply(self)

    def _get_obs(self, input_info=None):
        if self.strategy is None:
            raise AttributeError("No strategy applied！ e.g. env.apply( *Obs strategy* )")
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
        normalize to [-k, k] rather than [-1, 1]
        """
        self.dxdy_norm_scale = k

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
                "x": self.agent_pos[0], "y": self.agent_pos[1]}

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
        # render agent path
        plt.plot(agent_positions[:, 0], agent_positions[:, 1], 'b-', linewidth=4)
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
            # change the file name if necessary
            file_name = f'frame_{self.t_step}.png'
            file_path = os.path.join(save_path, file_name)
            plt.savefig(file_path, transparent=True)

        # plt.pause(self.frame_pause)

    def render_with_compared_path(self, x_list, y_list):
        if not (x_list is None or y_list is None):
            plt.figure(self.env_name, figsize=(10, 6))
            plt.plot(x_list, y_list, 'm--', label='optimal traj')
            plt.legend()
            # plt.draw()
            plt.pause(self.frame_pause)

    def render_multiple_paths(self, successful_paths, failed_paths):
        """
        Render multiple paths including successful ones and failed ones.
        Successful ones are marked with green while the failed ones with red.
        Start points of the paths are marked with a dot and the ending points with a hollow circle.
        paras:
            successful_paths : Successful paths list with each element a dict containing {start, target, x_list, y_list}
            failed_paths : Filed paths list.
        """
        n = len(successful_paths) + len(failed_paths)
        plt.figure(self.env_name + f' {n} paths', figsize=(9, 6))
        plt.clf()

        # background
        quiver_density_multiplier = 1.5
        quiver_x_points = int(self.render_quiver_x * quiver_density_multiplier)
        quiver_y_points = int(self.render_quiver_y * quiver_density_multiplier)
        X, Y = np.meshgrid(np.linspace(self.x_min, self.x_max, quiver_x_points),
                           np.linspace(self.y_min, self.y_max, quiver_y_points))
        UU = np.zeros_like(X)
        VV = np.zeros_like(Y)
        for i in range(quiver_x_points):
            for j in range(quiver_y_points):
                x_index = int((X[j, i] - self.x_grid[0, 0]) / (self.x_grid[1, 0] - self.x_grid[0, 0]))
                y_index = int((Y[j, i] - self.y_grid[0, 0]) / (self.y_grid[0, 1] - self.y_grid[0, 0]))
                UU[j, i] = self.u_grid[0, x_index, y_index]
                VV[j, i] = self.v_grid[0, x_index, y_index]
        plt.quiver(X, Y, UU, VV, scale=45, width=0.002, headwidth=3, headlength=5)

        # render successful paths
        for path in successful_paths:
            start = path['start']
            target = path['target']
            x_list = path['x_list']
            y_list = path['y_list']

            # the path
            plt.plot(x_list, y_list, color='green', linestyle='-', linewidth=1.5,
                     label='Successful Path' if path == successful_paths[0] else "")
            # start and target
            plt.scatter(start[0], start[1], color='green', s=15)  # start
            plt.scatter(target[0], target[1], facecolors='none', edgecolors='green', s=50)  # target

        # render failed paths
        for path in failed_paths:
            start = path['start']
            target = path['target']
            x_list = path['x_list']
            y_list = path['y_list']

            # the path
            plt.plot(x_list, y_list, color='red', linestyle='-', linewidth=1.5,
                     label='Failed Path' if path == failed_paths[0] else "")
            # start and target
            plt.scatter(start[0], start[1], color='red', s=15)  # start
            plt.scatter(target[0], target[1], facecolors='none', edgecolors='red', s=50)  # target

        # other elements
        plt.tight_layout()
        plt.axis('equal')
        plt.xlim(self.x_min, self.x_max)
        plt.ylim(self.y_min, self.y_max)
        # plt.title('Comparison of Successful and Failed Paths')

        plt.show()
        plt.pause(self.frame_pause)

    def render_frame_at_ts(self, time_step):
        if not isinstance(time_step, int):
            raise ValueError("time_step must be an integer.")

        t_index = time_step % self.env_cycle

        if time_step >= self.env_cycle:
            print(f"Time step {time_step} exceeds environment cycle. Rendering at time step {t_index} instead.")

        plt.figure(self.env_name, figsize=(10, 6))
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

        plt.quiver(X, Y, UU, VV, scale=30)

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


class DoubleGyreEnvironmentGridFixSpeed(EnvironmentBase):
    def __init__(self, field_file_path='./flow_field/DoubleGyre_grid_0.0to2.0of401_0.0to1.0of201.mat',
                 render_mode=None, random_start_and_target=True, swap_start_and_target=False, random_t0=False,
                 swim_vel=0.9, dxdy_norm_scale=18, seed=None):

        super().__init__(field_file_path, render_mode, random_start_and_target, swap_start_and_target, random_t0,
                         swim_vel, dxdy_norm_scale, seed)

        self.env_name = 'double_gyre'
        self.x_min, self.x_max = 0, 2
        self.y_min, self.y_max = 0, 1
        self.dt = 0.01
        self.t_step_max = 400
        self.default_start = np.array([1.5, 0.5])
        self.default_target = np.array([0.5, 0.5])
        self.start = self.default_start
        self.target = self.default_target
        self.target_r = 1 / 50
        self.rand_t0_within = 33  # reset starting t0 within range
        self.rand_start_target_r = 0.25
        self.env_cycle = 100  # cycle for periodic field
        self.render_quiver_x = 21
        self.render_quiver_y = 11
        self.render_vel_max = 3

        self.action_space = gym.spaces.Box(low=0, high=2 * np.pi, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
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

        # renew position
        u = _info["u"]
        v = _info["v"]
        new_pos = self.agent_pos + (np.array([u, v]) + action_xy) * self.dt

        self.agent_pos = new_pos
        self.agent_pos_history.append(new_pos.copy())
        self.t_step += self.ts_per_step

        d = np.linalg.norm(new_pos - self.target)

        # check out-of-bound
        if not (self.x_min <= new_pos[0] <= self.x_max and self.y_min <= new_pos[1] <= self.y_max):
            _truncated = True
            _reward = -0
            print(colored("**** Episode Finished **** Hit Boundary.", 'red'))
        else:
            # check timeout
            if self.t_step >= self.t_step_max:
                _truncated = True
                _reward = -0
                print(colored("**** Episode Finished **** Reaches Env Time limit.", 'blue'))
            # normal case
            else:
                # check reaching target
                if d <= self.target_r * 2.5:
                    mid = 0.5 * (new_pos + pos_cur)
                    d_mid = np.linalg.norm(mid - self.target)
                    if d_mid <= self.target_r or d <= self.target_r:
                        _terminated = True
                        _reward = 200
                        print(colored("**** Episode Finished **** SUCCESS.", 'green'))
                # still running
                # DoubleGyre Reward Shaping
                else:
                    _reward = -self.dt - 10 * (d - self.d_2_target) / self.swim_vel

        self.d_2_target = d

        if self.render_mode == 'human':
            self._render_frame()

        return _observation, _reward, _terminated, _truncated, _info


class Cylinder2D_Re400(EnvironmentBase):

    def __init__(self, field_file_path='./flow_field/cylinder2D_Re400_ref_grid_-3.0to15.0of361_-6.0to6.0of241.mat',
                 render_mode=None, random_start_and_target=True, swap_start_and_target=False, random_t0=False,
                 swim_vel=0.9, dxdy_norm_scale=12, seed=None):

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
        self.rand_t0_within = 38
        self.rand_start_target_r = 2
        self.env_cycle = 66
        self.render_quiver_x = 20
        self.render_quiver_y = 24
        self.render_vel_max = 1.6

        self.action_space = gym.spaces.Box(low=0, high=2 * np.pi, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=np.array([-6, -6, -5, -5]),
                                                high=np.array([6, 6, 5, 5]),
                                                shape=(4,), dtype=np.float32)

        d_grid = self.x_grid[1, 0] - self.x_grid[0, 0]
        _, m, n = self.Vel_grid.shape
        x_coords = self.x_min + np.arange(m) * d_grid
        y_coords = self.y_min + np.arange(n) * d_grid
        X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
        distances = np.sqrt(X ** 2 + Y ** 2)
        self.Vel_grid[:, distances <= 0.5] = np.nan

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

        # renew position
        u = _info["u"]
        v = _info["v"]
        new_pos = self.agent_pos + (np.array([u, v]) + action_xy) * self.dt

        self.agent_pos = new_pos
        self.agent_pos_history.append(new_pos.copy())
        self.t_step += self.ts_per_step

        d = np.linalg.norm(new_pos - self.target)

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
                # Cylinder CF_Re400 Reward Shaping
                else:
                    _reward = -self.dt - 20 * (d - self.d_2_target) / self.swim_vel

        self.d_2_target = d

        if self.render_mode == 'human':
            self._render_frame()

        return _observation, _reward, _terminated, _truncated, _info


if __name__ == '__main__':
    # env = Cylinder2D_Re400(render_mode='human',
    #                        random_start_and_target=True, random_t0=True,
    #                        swap_start_and_target=True, swim_vel=0.9)

    env = DoubleGyreEnvironmentGridFixSpeed(render_mode='human',
                                            random_start_and_target=True, random_t0=True,
                                            swap_start_and_target=True, swim_vel=0.9)

    # ------------ Choose obs strategy ---------- #
    # ---- 1. BaseSS [dx, dy, u, v] ----
    # env.apply(ObsNone())
    # ---- 2. True-LookAhead ----
    # env.apply(ObsAhead(look_ahead_steps=10))
    # ---- 3.1 Predict-LookAhead for CF ----
    # env.apply(ObsAheadPredict(f'pretrained_models/field_predict_models/CF_pred_model.pth',
    #                           look_ahead_steps=10, plot_field_comparison=False))
    # ---- 3.2 Predict-LookAhead for DG ----
    env.apply(ObsAheadPredict(f'pretrained_models/field_predict_models/DG_pred_model.pth',
                              look_ahead_steps=10, plot_field_comparison=False))
    # ------------ ------------------- ---------- #

    env.frame_pause = 0.1
    env.save_rendering = False

    for episode in range(20):
        env.reset()
        terminated = False
        truncated = False
        while not (terminated or truncated):
            action = [np.pi]  # example action within [-pi, pi]
            obs, reward, terminated, truncated, _ = env.step(action)

        print(episode)
