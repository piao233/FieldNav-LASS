import os
import argparse
import torch
import numpy as np
from datetime import datetime
from termcolor import colored
from colorama import init as colorama_init
import time
from my_environment import Cylinder2D_Re400 as CF, DoubleGyreEnvironmentGridFixSpeed as DG
from env_obs_strategy import ObsAhead, ObsNone
from my_PPO import PPO, force_cpu_as_device
colorama_init()

# ----- TODO: RL training settings -----
Env = CF  # CF or DG
Strategy = ObsNone  # ObsNone for BaseSS, ObsAhead for TrueLASS/PredLASS
lra = 0.0001  # lr_actor
lrc = 0.0002  # lr_critic
K = 80  # K_epoch
N = 20  # update network every N full episode lengths
# --------------------------------------


def print_train_info(model_id, lr_actor=None, lr_critic=None, K_epochs=None, update_after_N=None):
    print(f'ID {model_id}, lr_actor {lr_actor}, lr_critic {lr_critic}, K {K_epochs}, N {update_after_N}')


def train(model_id, render_mode=None, lr_actor=0.00005, lr_critic=0.0002, K_epochs=40, update_after_N=4, dxdy_scale=8):
    print("============================================================================================")

    env = Env(render_mode=render_mode, random_start_and_target=True, random_t0=True,
              swap_start_and_target=True, swim_vel=0.9, dxdy_norm_scale=dxdy_scale)
    env.apply(Strategy())

    env_name = env.env_name

    ####### initialize environment hyperparameters ######
    has_continuous_action_space = True  # continuous action space; else discrete

    # TODO: important settings
    max_training_timesteps = int(4e6)  # break training loop if timesteps > max_training_timesteps
    action_std = 0.4  # starting std for action distribution (Multivariate Normal)
    action_std_decay_rate = 0.012  # linearly decay action_std (action_std = action_std - action_std_decay_rate)
    min_action_std = 0.04  # minimum action_std (stop decay after action_std <= min_action_std)
    action_std_decay_freq = int(1e5)  # action_std decay frequency (in num timesteps)

    max_ep_len = 401  # max timesteps in one episode
    print_freq = max_ep_len * 15  # print avg reward in the interval (in num timesteps)
    log_freq = max_ep_len * 15  # log avg reward in the interval (in num timesteps)
    save_model_freq = int(1e5)  # save model frequency (in num timesteps)
    #####################################################
    ## Note : print/log frequencies should be > than max_ep_len
    ################ PPO hyperparameters ################
    update_timestep = max_ep_len * update_after_N  # update policy every n timesteps

    eps_clip = 0.2  # clip parameter for PPO
    gamma = 0.99  # discount factor

    random_seed = 0  # set random seed if required (0 = no random seed)

    pretrained_model_ID = model_id  # model ID, change this to prevent overwriting weights in same env_name folder
    #####################################################

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

    ###################### logging ######################

    #### log files for multiple runs are NOT overwritten
    log_dir = "PPO_logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_dir = log_dir + '/' + env_name + '/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    #### create new log file for each run
    log_f_name = log_dir + '/PPO_' + env_name + "_log_" + str(pretrained_model_ID) + ".csv"

    print("current logging model ID for " + env_name + " : ", pretrained_model_ID)
    print("logging at : " + log_f_name)
    if os.path.exists(log_f_name):
        raise ValueError(f"Check model ID. Current ID = {pretrained_model_ID} is already used.")
    #####################################################

    ################### checkpointing ###################

    directory = "PPO_preTrained"
    if not os.path.exists(directory):
        os.makedirs(directory)

    directory = directory + '/' + env_name + '/' + "PPO_{}_ID_{}_seed_{}".format(env_name, pretrained_model_ID,
                                                                                 random_seed) + '/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    print("save checkpoint directory : " + directory)
    #####################################################

    ############# print all hyperparameters #############
    print("--------------------------------------------------------------------------------------------")
    print("max training timesteps : ", max_training_timesteps)
    print("max timesteps per episode : ", max_ep_len)
    print("model saving frequency : " + str(save_model_freq) + " timesteps")
    print("log frequency : " + str(log_freq) + " timesteps")
    print("printing average reward over episodes in last : " + str(print_freq) + " timesteps")
    print("--------------------------------------------------------------------------------------------")
    print("state space dimension : ", state_dim)
    print("action space dimension : ", action_dim)
    print("--------------------------------------------------------------------------------------------")
    if has_continuous_action_space:
        print("Initializing a continuous action space policy")
        print("--------------------------------------------------------------------------------------------")
        print("starting std of action distribution : ", action_std)
        print("decay rate of std of action distribution : ", action_std_decay_rate)
        print("minimum std of action distribution : ", min_action_std)
        print("decay frequency of std of action distribution : " + str(action_std_decay_freq) + " timesteps")
    else:
        print("Initializing a discrete action space policy")
    print("--------------------------------------------------------------------------------------------")
    print("PPO update frequency : " + str(update_timestep) + " timesteps")
    print("PPO K epochs : ", K_epochs)
    print("PPO epsilon clip : ", eps_clip)
    print("discount factor (gamma) : ", gamma)
    print("--------------------------------------------------------------------------------------------")
    print("optimizer learning rate actor : ", lr_actor)
    print("optimizer learning rate critic : ", lr_critic)
    if random_seed:
        print("--------------------------------------------------------------------------------------------")
        print("setting random seed to ", random_seed)
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
    #####################################################

    print("============================================================================================")

    ################# training procedure ################

    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space,
                    action_std, continuous_action_output_scale=action_output_scale,
                    continuous_action_output_bias=action_output_bias)

    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    print("============================================================================================")

    # logging file
    log_f = open(log_f_name, "w+")
    log_f.write('episode,timestep,avg_return_in_period,action_std\n')

    # printing and logging variables
    print_running_return = 0
    print_running_episodes = 0
    log_running_return = 0
    log_running_episodes = 0

    cur_model_running_return = 0
    cur_model_running_episodes = 0
    max_model_avg_return = -np.inf

    time_step = 0
    i_episode = 0

    # training loop
    while time_step <= max_training_timesteps:
        if random_seed:
            state, info = env.reset(seed=random_seed)
        else:
            state, info = env.reset()

        current_ep_return = 0
        update_ready = False
        for t in range(1, max_ep_len + 1):

            # select action with policy
            action = ppo_agent.select_action(state)

            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # saving reward and is_terminals
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)

            time_step += 1
            current_ep_return += reward

            # update PPO agent
            if time_step % update_timestep == 0:
                update_ready = True
            if update_ready and done:
                update_ready = False
                # 计算上一个model的平均return per episode
                cur_model_avg_return = cur_model_running_return / cur_model_running_episodes
                if cur_model_avg_return > max_model_avg_return:
                    print("Better Model.")
                    checkpoint_path = directory + f"timestep_0_std_0.10086.pth"
                    print("saving model at : " + checkpoint_path)
                    ppo_agent.save_full(checkpoint_path)
                    max_model_avg_return = cur_model_avg_return
                cur_model_running_return = 0
                cur_model_running_episodes = 0

                print("Policy Updated. At Episode : {}  Timestep : {}".format(i_episode, time_step))
                ppo_agent.update()

            # if continuous action space; then decay action std of output action distribution
            if has_continuous_action_space and time_step % action_std_decay_freq == 0:
                action_std = ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

            # log in logging file
            if time_step % log_freq == 0:
                # log average return till last episode
                log_avg_return = log_running_return / log_running_episodes
                log_avg_return = round(log_avg_return, 4)

                log_f.write('{},{},{},{}\n'.format(i_episode, time_step, log_avg_return, action_std))
                log_f.flush()

                log_running_return = 0
                log_running_episodes = 0

            # printing average reward
            if time_step % print_freq == 0:
                # print average reward till last episode
                print_avg_return = print_running_return / print_running_episodes
                print_avg_return = round(print_avg_return, 2)

                print("Episode : {} \t\t Timestep : {} \t\t Average Return : {}".format(i_episode, time_step,
                                                                                        print_avg_return))

                print_running_return = 0
                print_running_episodes = 0

            # save model weights
            if time_step % save_model_freq == 0:
                print("--------------------------------------------------------------------------------------------")
                checkpoint_path = directory + f"timestep_{time_step}_std_{ppo_agent.action_std:.2f}.pth"
                print("saving model at : " + checkpoint_path)
                ppo_agent.save_full(checkpoint_path)
                print("model saved")
                print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                print("--------------------------------------------------------------------------------------------")

            # break; if the episode is over
            if done:
                break
            if t == max_ep_len:
                print(colored("**** Episode Terminated **** Reaches Training Episode Time limit.", 'blue'))

        print_running_return += current_ep_return
        print_running_episodes += 1

        log_running_return += current_ep_return
        log_running_episodes += 1

        cur_model_running_return += current_ep_return
        cur_model_running_episodes += 1

        i_episode += 1

    log_f.close()
    env.close()

    # print total training time
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")


if __name__ == '__main__':
    """
    (conda Env: RL_py3.9)python ./bloody_train.py start_id train_times
    """
    parser = argparse.ArgumentParser(description="Train PPO with optional arguments")
    parser.add_argument('start_id', type=int, help='Starting ID for training')
    parser.add_argument('train_times', type=int, help='Number of training times')
    args = parser.parse_args()
    start_id = args.start_id
    train_times = args.train_times
    force_cpu_as_device()

    total_delay = 5
    interval = min(1, total_delay)

    for i in range(train_times):
        print(f'======================== Training ID = {start_id + i} ========================')
        print_train_info(start_id + i,
                         lr_actor=lra,
                         lr_critic=lrc,
                         K_epochs=K,
                         update_after_N=N,
                         )

    for remaining in range(total_delay, 0, -interval):
        print(f"Starts in {remaining} seconds...")
        time.sleep(interval)

    print("Start training...")

    start_time = datetime.now().replace(microsecond=0)

    for i in range(train_times):
        print(f'======================== Training ID = {start_id + i} ========================')
        train(start_id + i, render_mode=None,
              lr_actor=lra,
              lr_critic=lrc,
              K_epochs=K,
              update_after_N=N)

    print(
        colored("============================================================================================", 'red'))
    print(colored(f"Training for {train_times} times Ended.", 'red'))
    end_time = datetime.now().replace(microsecond=0)
    print(colored(f"Started training at (GMT) : {start_time}", 'red'))
    print(colored(f"Finished training at (GMT) : {end_time}", 'red'))
    print(colored(f"Total training time  : {end_time - start_time}", 'red'))
    print(
        colored(f"============================================================================================", 'red'))
