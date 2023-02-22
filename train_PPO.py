import os
import pathlib
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from env import SumoEnv
from eval_env import SumoEnv as EvalEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
from stable_baselines3.common.monitor import Monitor

def main():

    # input variables
    num_envs = 20
    timesteps = 40000000
    eval_freq = 100000
    iteration = 66
    algo = "PPO_UNCOOP_ENV_RERUN"
    gui = False

    # paths and eval adjusting for multiple envs
    result_name = f"{iteration}_{algo}_{timesteps}"
    current_path = pathlib.Path(__file__).parent.resolve()
    log_path = os.path.join(current_path,"Logs", result_name)
    checkpoint_save_path = os.path.join(current_path, "Models", result_name)
    final_save_path = os.path.join(checkpoint_save_path, result_name)
    best_save_path = os.path.join(final_save_path, "Best Model")
    eval_freq_adjusted = eval_freq/num_envs

    # setup env
    env = SubprocVecEnv([lambda: Monitor(SumoEnv(gui=gui), info_keywords=("is_success",))  for i in range(num_envs)], start_method="spawn")
    eval_env = SubprocVecEnv([lambda: Monitor(EvalEnv(gui=gui), info_keywords=("is_success",))])

    # Callback to evaluate model every 5000 steps and save best
    checkpoint_callback = CheckpointCallback(
        save_path=checkpoint_save_path,
        name_prefix= "rl_model",
        save_freq=eval_freq_adjusted, 
        verbose = 1
    )

    # Callback to evaluate model every 5000 steps and save best
    eval_callback = EvalCallback(
        eval_env,
        n_eval_episodes= 50,
        best_model_save_path=best_save_path,
        eval_freq=eval_freq_adjusted, 
        render=False,
        deterministic=True,
        verbose = 1
    )

    # Create the callback list
    callback = CallbackList([checkpoint_callback, eval_callback])

    # create, train, save model
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_path, device="cuda")
    model.learn(total_timesteps=timesteps, callback=callback)

    print("Training complete. Saving model.")
    model.save(final_save_path)
    print("Model Saved")

# run code on script run
if __name__ =="__main__":
    main()