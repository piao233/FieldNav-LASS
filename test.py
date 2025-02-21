from datetime import datetime
import numpy as np
import torch
from termcolor import colored
from colorama import init as colorama_init
import matplotlib.pyplot as plt
from my_PPO import PPO, force_cpu_as_device
from my_environment import Cylinder2D_Re400 as CF_Re400, DoubleGyreEnvironmentGridFixSpeed as DG
from env_obs_strategy import ObsAhead, ObsNone, ObsAheadPredict
colorama_init()


############################ Settings of Testing ###########################

# TODO: Set Environment to 'DoubleGyre' or 'CylinderFlow' ------- #
Env = 'DoubleGyre'
# --------------------------------------------------------------- #
# TODO: Set ObsStrategy to 'BaseSS', 'TrueLASS' or 'PredLASS' --- #
ObsStrategy = 'TrueLASS'
# --------------------------------------------------------------- #

test_episodes = 100  # Total test times
running_render = False  # Render during testing
example_path_num = 10  # Show some example paths once testing is finished. Set to 0 if not needed.

############################ -------------------- ###########################

def test(total_test_episodes=test_episodes, is_running_render=running_render, running_render_fail=False, frame_delay_s=0.001):
    print("============================================================================================")
    ################## hyperparameters #################
    random_seed = 0  # set this to load a particular checkpoint trained on random seed
    action_std_4_test = 0.002  # set std for action distribution when testing. irrelevent to the std of the model.
    has_continuous_action_space = True
    max_ep_len = 401  # max timesteps in one episode
    if Env == 'CylinderFlow':
        env = CF_Re400(random_start_and_target=True, random_t0=True,
                  swap_start_and_target=True, swim_vel=0.9)
        path_pred_net = './pretrained_models/field_predict_models/CF_pred_model.pth'
    elif Env == 'DoubleGyre':
        env = DG(random_start_and_target=True, random_t0=True,
                       swap_start_and_target=True, swim_vel=0.9)
        path_pred_net = './pretrained_models/field_predict_models/DG_pred_model.pth'
    else:
        raise NotImplementedError("Check Env name setting.")
    if ObsStrategy == 'BaseSS':
        env.apply(ObsNone())
    elif ObsStrategy == 'TrueLASS':
        env.apply(ObsAhead(look_ahead_steps=10))
    elif ObsStrategy == 'PredLASS':
        env.apply(ObsAheadPredict(path_pred_net, look_ahead_steps=10,
                                  device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                                  plot_field_comparison=False))
    else:
        raise NotImplementedError("Check ObsStrategy name setting.")

    #####################################################
    K_epochs = 1  # update policy for K epochs
    eps_clip = 0.2  # clip parameter for PPO
    gamma = 0.99  # discount factor

    lr_actor = 0.1  # learning rate for actor
    lr_critic = 0.1  # learning rate for critic
    #####################################################
    env_name = env.env_name
    # state space dimension
    state_dim = env.observation_space.shape[0]
    # action space dimension
    action_output_scale = np.array([])
    action_output_bias = np.array([])
    if has_continuous_action_space:
        action_dim = env.action_space.shape[0]
        action_output_scale = (env.action_space.high - env.action_space.low) / 2.0
        action_output_bias = (env.action_space.high + env.action_space.low) / 2.0
    else:
        action_dim = env.action_space.n

    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space,
                    action_std_4_test, continuous_action_output_scale=action_output_scale,
                    continuous_action_output_bias=action_output_bias)

    # preTrained weights directory
    if Env == 'DoubleGyre':
        if ObsStrategy == 'BaseSS':
            PPO_dir = './pretrained_models/PPO_models/DG_PPO_BaseSS.pth'
        elif ObsStrategy == 'TrueLASS' or ObsStrategy == 'PredLASS':
            PPO_dir = './pretrained_models/PPO_models/DG_PPO_LASS.pth'
        else: raise NotImplementedError()
    elif Env == 'CylinderFlow':
        if ObsStrategy == 'BaseSS':
            PPO_dir = './pretrained_models/PPO_models/CF_PPO_BaseSS.pth'
        elif ObsStrategy == 'TrueLASS' or ObsStrategy == 'PredLASS':
            PPO_dir = './pretrained_models/PPO_models/CF_PPO_LASS.pth'
        else: raise NotImplementedError()
    else: raise NotImplementedError()

    print("loading network from : " + PPO_dir)
    ppo_agent.load_full(PPO_dir)

    ########################################

    test_running_return = 0
    success_time_step = 0
    success_times = 0

    successful_paths = []
    failed_paths = []

    for ep in range(1, total_test_episodes + 1):
        ep_return = 0
        state, info = env.reset()
        global skip_episode
        skip_episode = False
        terminated = False
        truncated = False
        for t in range(1, max_ep_len + 1):
            action = ppo_agent.select_action(state)
            # print(action[0]/np.pi*180)
            state, reward, terminated, truncated, info = env.step(action)
            ep_return += reward
            if is_running_render or (running_render_fail and truncated):
                env.render()
                env.render_with_compared_path(the_path_x, the_path_y)
                plt.pause(frame_delay_s)
            if skip_episode:
                break
            if terminated:
                success_times += 1
                success_time_step += t
                # plt.pause(5)
                break
            if truncated:
                break
            if t == max_ep_len:
                print(colored("**** Episode Terminated **** Reaches Training Episode Time limit.", 'blue'))

        # save some paths for rendering
        if ep <= example_path_num:
            agent_positions = np.array(env.agent_pos_history)
            x_list = agent_positions[:, 0].tolist()
            y_list = agent_positions[:, 1].tolist()

            path_data = {
                'start': env.start.tolist(),
                'target': env.target.tolist(),
                'x_list': x_list,
                'y_list': y_list
            }
            if terminated:
                successful_paths.append(path_data)
            else:
                failed_paths.append(path_data)

        # clear buffer
        ppo_agent.buffer.clear()

        test_running_return += ep_return
        print('Episode: {} \t\t Reward: {}'.format(ep, round(ep_return, 2)))

    env.close()

    print("============================================================================================")
    avg_test_reward = test_running_return / total_test_episodes
    avg_test_reward = round(avg_test_reward, 2)
    print(f"Testing finished.")
    print(f"Success : {success_times}/{total_test_episodes} i.e. {success_times / total_test_episodes * 100:.2f}%")
    print("average test reward : " + str(avg_test_reward))
    print(f"average time of success : {success_time_step / success_times * env.dt:.2f}")
    print("============================================================================================")

    if example_path_num > 0:
        env.render_multiple_paths(successful_paths, failed_paths)


if __name__ == '__main__':
    force_cpu_as_device()
    # specific path (e.g. OC path) for comparison
    the_path_x = None
    the_path_y = None
    start_time = datetime.now().replace(microsecond=0)
    test()
    end_time = datetime.now().replace(microsecond=0)
    print(f'Elapsed: {end_time-start_time}')
