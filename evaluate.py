import numpy as np
import os, sys
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from env import SumoEnv
from stable_baselines3.common.monitor import Monitor

from stable_baselines3.common.evaluation import evaluate_policy
import pathlib

def main():
    gui = True
    eval_timesteps = 3000

    # defining model path
    result_name = ""
    save_path = os.path.join(pathlib.Path(__file__).parent.resolve(), "Models", result_name, result_name, "Best Model", "best_model")

    model = DQN.load(save_path)
    env = DummyVecEnv([lambda: SumoEnv(gui=gui)])
    reward = 0
    obs = env.reset()
    time = 0

    # run evaluation
    for _ in range(eval_timesteps):
        action, _ = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        time += 0.1
        reward += rewards
    
    # print the total reward
    print('the final reward is {}'.format(reward))
    
# run code on script run
if __name__ =="__main__":
    main()