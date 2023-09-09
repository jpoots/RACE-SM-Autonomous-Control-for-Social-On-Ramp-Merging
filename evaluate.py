import numpy as np
import os, sys
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from eval_env import SumoEnv
from stable_baselines3.common.monitor import Monitor

from stable_baselines3.common.evaluation import evaluate_policy
import pathlib

def main():
    gui = True
    # defining model path
    result_name = ""
    timestep_mil = 14
    save_path = os.path.join(pathlib.Path(__file__).parent.resolve(), "Models", result_name, f"rl_model_{int(timestep_mil * 10 ** 6)}_steps.zip")

    model = PPO.load(save_path)
    env = SubprocVecEnv([lambda: SumoEnv(gui=gui)])
    reward = 0
    obs = env.reset()
    time = 0
    for _ in range(10000000):
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, done, info = env.step(action)
    
    print('the final reward is {}'.format(reward))
    
# run code on script run
if __name__ =="__main__":
    main()