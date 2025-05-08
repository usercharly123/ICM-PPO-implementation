import matplotlib.pyplot as plt
import numpy as np
import sys
import torch
from torch.utils.tensorboard import SummaryWriter

from utils import Memory
from ICMPPO import ICMPPO
import torch.nn as nn
from torch.distributions import Categorical

from mlagents_envs.environment import UnityEnvironment
from ml_agents.ml_agents_envs.mlagents_envs.envs.unity_gym_env import UnityToGymWrapper

import time

solved_reward = 1.7     # stop training if avg_reward > solved_reward
log_interval = 100     # print avg reward in the interval
max_episodes = 35 # WAS 350      # max training episodes
max_timesteps = 100    # WAS 1000 max timesteps in one episode
update_timestep = 18  # WAS 2048 Replay buffer size, update policy every n timesteps
log_dir= 'events/'           # Where to store tensorboard logs

# Initialize Unity env
multi_env_name = 'Pyramid1agent/UnityEnvironment.exe'
unity_env = UnityEnvironment(file_name=multi_env_name, worker_id=0, no_graphics=False)

multi_env = UnityToGymWrapper(unity_env, uint8_visual=False)

# Initialize log_writer, memory buffer, icmppo
writer = SummaryWriter(log_dir)
memory = Memory()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
agent = ICMPPO(writer=writer, device=device)

timestep = 0
T = np.zeros(1) # Was 16
state = multi_env.reset()
print("State: ", state.shape)

# Initialize the time to know how ong the script takes to run
start_time = time.time()
print("Start time: ", start_time)

# training loop
for i_episode in range(1, max_episodes + 1):
    print("Episode: ", i_episode)
    episode_rewards = np.zeros(16)
    episode_counter = np.zeros(16)
    for i in range(max_timesteps):
        if timestep % 10 == 0:
            print("Timestep: ", timestep)
        timestep += 1
        T += 1
        # Running policy_old:
        actions = np.atleast_1d(agent.policy_old.act(np.array(state), memory))  # was agent.policy_old.act(np.array(state), memory)
        state, rewards, dones, info = multi_env.step(list(actions))

        # Fix rewards
        rewards = np.atleast_1d(rewards)    # was np.array(rewards)
        dones = np.atleast_1d(dones)    # was np.array(dones)
        
        rewards += 2 * (rewards == 0) * (T < 1000)      # adds 2 to the reward of each agent that had zero reward but is still within the early timesteps
        episode_counter += dones
        T[dones] = 0
        # Saving reward and is_terminal:
        memory.rewards.append(rewards)
        memory.is_terminals.append(dones)
        
        # If the episode is done, reset the environment
        if dones.any():
            print(f"Episode {i_episode} done at timestep {i}")
            state = multi_env.reset()  # Reset the environment when done = True

        # update if its time
        if timestep % update_timestep == 0:
            agent.update(memory, timestep)
            memory.clear_memory()

        episode_rewards += rewards

    if episode_counter.sum() == 0:
        episode_counter = np.ones(16)

    # stop training if avg_reward > solved_reward
    if episode_rewards.sum() / episode_counter.sum() > solved_reward:
        print("########## Solved! ##########")
        writer.add_scalar('Mean_extr_reward_per_1000_steps',
                          episode_rewards.sum() / episode_counter.sum(),
                          timestep
        )
        torch.save(agent.policy.state_dict(), './ppo.pt')
        torch.save(agent.icm.state_dict(), './icm.pt')
        break

    # logging
    if timestep % log_interval == 0:
        print('Episode {} \t episode reward: {} \t'.format(i_episode, episode_rewards.sum() / episode_counter.sum()))
        writer.add_scalar('Mean_extr_reward_per_1000_steps',
                          episode_rewards.sum() / episode_counter.sum(),
                          timestep
        )
        
# Print the time it took to run the script
end_time = time.time()
print("End time: ", end_time)
print("Time taken: ", end_time - start_time)

multi_env.close()   # Closes the UnityToGymWrapper
writer.close()      # Closes TensorBoard logging