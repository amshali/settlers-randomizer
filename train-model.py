import numpy as np
from catan_board import CatanBoardEnv
from stable_baselines3 import PPO
from reward_funcs import reward_mean_var4

# Choose the reward function
reward_func = reward_mean_var4

env = CatanBoardEnv(reward_function=reward_func)


def linear_schedule(initial_value):
    def schedule(progress_remaining):
        return progress_remaining * initial_value

    return schedule


# linear_schedule(0.0008): 98
# linear_schedule(0.0005): 102
# linear_schedule(0.0001): 79
# linear_schedule(0.001): 82
# 0.0005: 74

## linear_schedule(0.0005) -> 724
# 0.0003 -> 758

# Train the agent
model = PPO("MlpPolicy", env, learning_rate=0.0002, verbose=1)
model.learn(total_timesteps=5000000)
model.save("catan_ppo_" + reward_func.__name__)
env.save_good_assignments("good_assignments_" + reward_func.__name__)
