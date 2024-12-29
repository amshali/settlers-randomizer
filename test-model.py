from stable_baselines3 import PPO
from catan_board import CatanBoardEnv

from reward_funcs import reward_mean_var4

# Choose the reward function
reward_func = reward_mean_var4

env = CatanBoardEnv(reward_function=reward_func)
# Load the model and environment
model = PPO.load("catan_ppo_" + reward_func.__name__)

# Test the trained agent
all_steps = []
all_rewards = []
for g in range(1, 2):
    obs = env.reset()
    print(f"Game {g}")
    steps = 0
    info = {}
    for i in range(10000):
        steps += 1
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        if info.get("mean") > 9.5 and info.get("variance") < 2:
            break
    # store the steps in an array and then average them
    print(f"Steps: {steps}")
    print(f"Var: {info.get("variance")}")
    print(f"Mean: {info.get("mean")}")
    all_steps.append(steps)
    all_rewards.append(info.get("variance"))

env.render_board()
print(f"Average steps: {sum(all_steps) / len(all_steps)}")
print(f"Average var: {sum(all_rewards) / len(all_rewards)}")
