import gymnasium as gym
import numpy as np
import math
import matplotlib.pyplot as plt
import torch
from field_predict_double_gyre import LSTMConvUpsamplePredictor as DG_Predictor
from field_predict_CF_Re400 import LSTMConvUpsamplePredictor as CF_Re400_Predictor


class ObsNone:
    """
    BaseSS: [dx, dy, u, v]
    """

    def __init__(self):
        self.name = "ObsNone"

    @staticmethod
    def apply(env):
        # modify env vars ones the strategy is set
        # default env state-space is already [dx dy u v], no need to add more.
        pass

    @staticmethod
    def _get_obs(env, input_info=None):
        if input_info is None:
            input_info = env._get_info()
        dx_norm = env._normalize(input_info["dx"], env.observation_space.low[0], env.observation_space.high[0],
                                 scale=env.dxdy_norm_scale)
        dy_norm = env._normalize(input_info["dy"], env.observation_space.low[1], env.observation_space.high[1],
                                 scale=env.dxdy_norm_scale)
        u_norm = env._normalize(input_info["u"], env.observation_space.low[2], env.observation_space.high[2])
        v_norm = env._normalize(input_info["v"], env.observation_space.low[3], env.observation_space.high[3])

        return {"dx": dx_norm, "dy": dy_norm, "u": u_norm, "v": v_norm}

    @staticmethod
    def render_more(env):
        pass


class ObsAhead:
    """
    True-LASS: apart from [dx dy u v], 8 Look-Ahead [dx dy u v]s are also included in state-space.
    36 dimensions in total.
    """

    def __init__(self, look_ahead_steps=10, field_predict_noise_sigma=0):
        self.look_ahead_steps = look_ahead_steps
        self.field_predict_noise_sigma = field_predict_noise_sigma
        self.name = "ObsAhead"

    def apply(self, env):
        env.look_forward_steps = self.look_ahead_steps
        state_space_low = np.concatenate((env.observation_space.low, np.zeros(32)))
        state_space_high = np.concatenate((env.observation_space.high, np.zeros(32)))
        env.set_state_space(gym.spaces.Box(low=state_space_low, high=state_space_high, shape=(36,), dtype=np.float32))

    def _get_obs(self, env, input_info=None):
        if input_info is None:
            input_info = env._get_info()
        dx_norm = env._normalize(input_info["dx"], env.observation_space.low[0], env.observation_space.high[0],
                                 scale=env.dxdy_norm_scale)
        dy_norm = env._normalize(input_info["dy"], env.observation_space.low[1], env.observation_space.high[1],
                                 scale=env.dxdy_norm_scale)
        u_norm = env._normalize(input_info["u"], env.observation_space.low[2], env.observation_space.high[2])
        v_norm = env._normalize(input_info["v"], env.observation_space.low[3], env.observation_space.high[3])

        env.sampled_paths = []  # save sampled paths in 8 directions

        # sample future paths
        num_steps = env.look_forward_steps
        actions = [0, 0.25 * np.pi, 0.5 * np.pi, 0.75 * np.pi, np.pi, 1.25 * np.pi, 1.5 * np.pi, 1.75 * np.pi]
        sample_list = []
        u, v = 0, 0
        for act in actions:
            new_pos = env.agent_pos.copy()
            action_xy = np.array([env.swim_vel * math.cos(act), env.swim_vel * math.sin(act)])
            path = [new_pos.copy()]
            for i in range(num_steps):
                x, y = new_pos[0], new_pos[1]
                u, v = env._get_uv_at(env.t_step + i, x, y)
                if not self.field_predict_noise_sigma:
                    new_pos += (np.array([u, v]) + action_xy) * env.dt
                else:
                    new_pos += (np.array([u, v]) + action_xy + np.random.normal(loc=0,
                                                                                scale=self.field_predict_noise_sigma,
                                                                                size=action_xy.shape)) * env.dt
                path.append(new_pos.copy())
                if not (env.x_min <= new_pos[0] <= env.x_max and env.y_min <= new_pos[1] <= env.y_max):
                    break
                # target check
                if np.linalg.norm(new_pos - env.target) < env.target_r:
                    new_pos = env.target
                    break
            sample_list.append([env.target[0] - new_pos[0], env.target[1] - new_pos[1], u, v])
            env.sampled_paths.append(path)

        rtn_obs = {"dx": dx_norm, "dy": dy_norm, "u": u_norm, "v": v_norm}

        for i in range(8):
            dx_norm = env._normalize(sample_list[i][0], env.observation_space.low[0], env.observation_space.high[0],
                                     scale=env.dxdy_norm_scale)
            dy_norm = env._normalize(sample_list[i][1], env.observation_space.low[1], env.observation_space.high[1],
                                     scale=env.dxdy_norm_scale)
            u_norm = env._normalize(sample_list[i][2], env.observation_space.low[2], env.observation_space.high[2])
            v_norm = env._normalize(sample_list[i][3], env.observation_space.low[3], env.observation_space.high[3])
            rtn_obs[f"dx{i}"] = dx_norm
            rtn_obs[f"dy{i}"] = dy_norm
            rtn_obs[f"u{i}"] = u_norm
            rtn_obs[f"v{i}"] = v_norm

        return rtn_obs

    @staticmethod
    def render_more(env):
        # render the LA paths
        for path in env.sampled_paths:
            path = np.array(path)
            plt.plot(path[:, 0], path[:, 1], '--', linewidth=0.5)
            plt.scatter(path[-1, 0], path[-1, 1], c='purple', marker='x')


class ObsAheadPredict:
    """
    PredLASS: apart from [dx dy u v], 8 Look-Ahead [dx dy u v]s are also included in state-space.
    36 dimensions in total.
    Look-Ahead is performed using predicted flow field by specific field prediction net.
    """

    def __init__(self, model_path, look_ahead_steps=5, input_size=4, hidden_size=256, num_layers=2, device='cpu',
                 plot_field_comparison=False, field_predict_noise_sigma=0):
        self.name = "ObsAheadPredict"
        self.device = device
        self.plot_field_comparison = plot_field_comparison
        self.field_predict_noise_sigma = field_predict_noise_sigma
        self.look_ahead_steps = look_ahead_steps
        self.model_path = model_path
        self.model = None
        self.history = []
        self.predicted_partial_field_lower_left_index = []
        self.x_index = 0
        self.y_index = 0
        self.env = None
        self.env_grid_d = None

        self.model_path, self.input_size, self.hidden_size, self.num_layers, self.look_ahead_steps = (
            model_path, input_size, hidden_size, num_layers, look_ahead_steps)

    def load_model(self, model_path, input_size, hidden_size, num_layers, output_steps):
        if self.env.env_name == 'double_gyre':
            model = DG_Predictor(input_size, hidden_size, num_layers, output_steps).to(self.device)
        elif self.env.env_name == 'cylinder2D_Re400':
            model = CF_Re400_Predictor(input_size, hidden_size, num_layers, output_steps).to(self.device)
        else:
            raise NotImplementedError("Field Prediction Model Loading ERROR.")
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        return model

    def apply(self, env):
        env.look_forward_steps = self.look_ahead_steps
        state_space_low = np.concatenate((env.observation_space.low, np.zeros(32)))
        state_space_high = np.concatenate((env.observation_space.high, np.zeros(32)))
        env.set_state_space(gym.spaces.Box(low=state_space_low, high=state_space_high, shape=(36,), dtype=np.float32))
        self.env = env
        self.env_grid_d = self.env.x_grid[1, 0] - self.env.x_grid[0, 0]
        self.model = self.load_model(self.model_path, self.input_size, self.hidden_size,
                                     self.num_layers, self.look_ahead_steps)

    @staticmethod
    def _normalize(value, low, high, scale=1.0):
        return (value - low) / (high - low) * scale

    def _get_obs(self, env, input_info=None):
        if input_info is None:
            input_info = env._get_info()

        dx_norm = env._normalize(input_info["dx"], env.observation_space.low[0], env.observation_space.high[0],
                                 scale=env.dxdy_norm_scale)
        dy_norm = env._normalize(input_info["dy"], env.observation_space.low[1], env.observation_space.high[1],
                                 scale=env.dxdy_norm_scale)
        u_norm = env._normalize(input_info["u"], env.observation_space.low[2], env.observation_space.high[2])
        v_norm = env._normalize(input_info["v"], env.observation_space.low[3], env.observation_space.high[3])

        # update history buffer
        self.history.append([input_info["x"], input_info["y"], input_info["u"], input_info["v"]])
        self.x_index = int((input_info["x"] - self.env.x_grid[0, 0]) // self.env_grid_d)
        self.y_index = int((input_info["y"] - self.env.y_grid[0, 0]) // self.env_grid_d)

        if len(self.history) > 10:
            self.history.pop(0)

        # reset history buffer when env is reset
        if env.t_step == 0:
            self.history = []

        if len(self.history) < 10:
            predicted_flow_fields = None
        else:
            predicted_flow_fields = self.predict_future_flow_fields(env)
            if self.plot_field_comparison:
                self.plot_comparison(predicted_flow_fields)  # 绘制比较图

        env.sampled_paths = []

        num_steps = env.look_forward_steps
        actions = [0, 0.25 * np.pi, 0.5 * np.pi, 0.75 * np.pi, np.pi, 1.25 * np.pi, 1.5 * np.pi, 1.75 * np.pi]
        sample_list = []
        u, v = 0, 0
        for act in actions:
            new_pos = env.agent_pos.copy()
            action_xy = np.array([env.swim_vel * math.cos(act), env.swim_vel * math.sin(act)])
            path = [new_pos.copy()]
            for i in range(num_steps):
                x, y = new_pos[0], new_pos[1]
                if not (env.x_min <= new_pos[0] <= env.x_max and env.y_min <= new_pos[1] <= env.y_max):
                    break
                u, v, flag_out_of_bound = self.get_predicted_uv(predicted_flow_fields, i, x, y)
                if flag_out_of_bound:
                    break
                new_pos += (np.array([u, v]) + action_xy) * env.dt
                path.append(new_pos.copy())
                if np.linalg.norm(new_pos - env.target) < env.target_r:
                    new_pos = env.target
                    break
            sample_list.append([env.target[0] - new_pos[0], env.target[1] - new_pos[1], u, v])
            env.sampled_paths.append(path)

        rtn_obs = {"dx": dx_norm, "dy": dy_norm, "u": u_norm, "v": v_norm}

        for i in range(8):
            dx_norm = env._normalize(sample_list[i][0], env.observation_space.low[0], env.observation_space.high[0],
                                     scale=env.dxdy_norm_scale)
            dy_norm = env._normalize(sample_list[i][1], env.observation_space.low[1], env.observation_space.high[1],
                                     scale=env.dxdy_norm_scale)
            u_norm = env._normalize(sample_list[i][2], env.observation_space.low[2], env.observation_space.high[2])
            v_norm = env._normalize(sample_list[i][3], env.observation_space.low[3], env.observation_space.high[3])
            rtn_obs[f"dx{i}"] = dx_norm
            rtn_obs[f"dy{i}"] = dy_norm
            rtn_obs[f"u{i}"] = u_norm
            rtn_obs[f"v{i}"] = v_norm

        return rtn_obs

    def predict_future_flow_fields(self, env):
        input_seq_normalized = np.array(self.history)
        if env.env_name == 'double_gyre':
            input_seq_normalized[:, 0] = 2 * (input_seq_normalized[:, 0] / 2.0) - 1  # x范围[0,2] -> [-1,1]
            input_seq_normalized[:, 1] = 2 * (input_seq_normalized[:, 1] / 1.0) - 1  # y范围[0,1] -> [-1,1]
            input_seq_normalized[:, 2] = input_seq_normalized[:, 2] / 2.0  # u范围[-2,2] -> [-1,1]
            input_seq_normalized[:, 3] = input_seq_normalized[:, 3] / 2.0  # v范围[-2,2] -> [-1,1]
        elif env.env_name == 'cylinder2D_Re400':
            input_seq_normalized[:, 0] = input_seq_normalized[:, 0] / 4.0 - 1  # x范围[0,8] -> [-1,1]
            input_seq_normalized[:, 1] = input_seq_normalized[:, 1] / 4.0 - 0  # y范围[-4,4] -> [-1,1]
            input_seq_normalized[:, 2] = input_seq_normalized[:, 2] / 0.4 - 2.5  # u范围[0.6,1.4] -> [-1,1]
            input_seq_normalized[:, 3] = input_seq_normalized[:, 3] / 0.4  # v范围[-0.4,0.4] -> [-1,1]
        else:
            raise NotImplementedError("Other jobs to do.")

        input_data = torch.tensor(input_seq_normalized, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            predicted_flow_fields = self.model(input_data).cpu().numpy()
        if not self.field_predict_noise_sigma:
            return predicted_flow_fields
        noise = np.random.normal(loc=0.0, scale=self.field_predict_noise_sigma, size=predicted_flow_fields.shape)
        t = predicted_flow_fields + noise
        return t

    def get_predicted_uv(self, predicted_flow_fields, t_step, x, y):
        flag_out_of_bound = False
        if predicted_flow_fields is None:
            return 0, 0, flag_out_of_bound

        d_x_index = int((x - self.env.x_grid[0, 0]) // self.env_grid_d) - self.x_index
        d_y_index = int((y - self.env.y_grid[0, 0]) // self.env_grid_d) - self.y_index

        if not (-63 <= d_x_index <= 64 and -63 <= d_y_index <= 64):
            flag_out_of_bound = True  # true if agent goes over boundary. result is not reliable.
        # agent LA end position forced set within the field prediction range
        x_index_in_predicted_field = np.clip(64 - 1 + d_x_index, 0, 127)
        y_index_in_predicted_field = np.clip(64 - 1 + d_y_index, 0, 127)

        u = predicted_flow_fields[0, t_step, 0, x_index_in_predicted_field, y_index_in_predicted_field]
        v = predicted_flow_fields[0, t_step, 1, x_index_in_predicted_field, y_index_in_predicted_field]
        return u, v, flag_out_of_bound

    def plot_comparison(self, predicted_flow_fields):
        fig = plt.figure('True vs Predicted', figsize=(9, 6))
        fig.clf()
        axes = fig.subplots(2, 3)

        cur_time = self.env.t0 + self.env.t_step * self.env.dt
        for (i, t) in enumerate([1, 5, 9]):
            u_pred = predicted_flow_fields[0, t, 0, :, :]
            v_pred = predicted_flow_fields[0, t, 1, :, :]

            ttt = self.env._env_cycle(int(cur_time / self.env.dt + t))
            u_true = self.env.u_grid[self.env._env_cycle(int(cur_time / self.env.dt + t))]
            v_true = self.env.v_grid[self.env._env_cycle(int(cur_time / self.env.dt + t))]
            # Pad the field with reflection mode
            padded_field = np.pad(u_true, ((64, 64), (64, 64)), mode='reflect')
            # Extract the patch from the padded field
            u_true = padded_field[self.x_index + 1:self.x_index + 129, self.y_index + 1:self.y_index + 129]
            # Pad the field with reflection mode
            padded_field = np.pad(v_true, ((64, 64), (64, 64)), mode='reflect')
            # Extract the patch from the padded field
            v_true = padded_field[self.x_index + 1:self.x_index + 129, self.y_index + 1:self.y_index + 129]

            # Downsample
            u_true_downsampled = u_true[::2, ::2]
            v_true_downsampled = v_true[::2, ::2]
            u_pred_downsampled = u_pred[::2, ::2]
            v_pred_downsampled = v_pred[::2, ::2]
            Vel_true_downsampled = np.sqrt(u_true_downsampled ** 2 + v_true_downsampled ** 2)
            Vel_pred_downsampled = np.sqrt(u_pred_downsampled ** 2 + v_pred_downsampled ** 2)

            im_true = axes[0, i].imshow(np.rot90(Vel_true_downsampled), cmap='coolwarm', vmin=0,
                                        vmax=self.env.render_vel_max, origin='upper', aspect='equal')
            axes[0, i].set_title(f'True, t+{t}')
            im_pred = axes[1, i].imshow(np.rot90(Vel_pred_downsampled), cmap='coolwarm', vmin=0,
                                        vmax=self.env.render_vel_max, origin='upper', aspect='equal')
            axes[1, i].set_title(f'Pred, t+{t}')

            # Create a grid for quiver
            X, Y = np.meshgrid(np.linspace(0, Vel_pred_downsampled.shape[0] - 1, 8),
                               np.linspace(0, Vel_pred_downsampled.shape[1] - 1, 8))

            U = np.rot90(u_true_downsampled[::8, ::8])
            V = np.rot90(v_true_downsampled[::8, ::8])
            axes[0, i].quiver(X, Y, U, V, scale=15, color='k')

            U = np.rot90(u_pred_downsampled[::8, ::8])
            V = np.rot90(v_pred_downsampled[::8, ::8])
            axes[1, i].quiver(X, Y, U, V, scale=15, color='k')

        plt.tight_layout()
        plt.draw()

    def render_more(self, env):
        d = self.env_grid_d
        lower_left = [(self.x_index - 63) * d + self.env.x_grid[0, 0], (self.y_index - 63) * d + self.env.y_grid[0, 0]]
        rect_x = [lower_left[0], lower_left[0], lower_left[0] + 128 * d, lower_left[0] + 128 * d, lower_left[0]]
        rect_y = [lower_left[1], lower_left[1] + 128 * d, lower_left[1] + 128 * d, lower_left[1], lower_left[1]]
        plt.plot(rect_x, rect_y, '--', c='red', linewidth=1.5)
        for path in env.sampled_paths:
            path = np.array(path)
            plt.plot(path[:, 0], path[:, 1], '--', c='red', linewidth=1)
            plt.scatter(path[-1, 0], path[-1, 1], c='red', marker='x')