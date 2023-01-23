import os
import pathlib
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import SubprocVecEnv
from env import SumoEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
from stable_baselines3.common.monitor import Monitor

def main():

    # input variables
    num_envs = 8
    timesteps = 20000000
    eval_freq = 50000
    iteration = 20
    algo = "DQN"
    gui = False

    # paths and eval adjusting for multiple envs
    result_name = f"{iteration}_{algo}_{timesteps}"
    current_path = pathlib.Path(__file__).parent.resolve()
    log_path = os.path.join(current_path,"Logs", result_name)
    checkpoint_save_path = os.path.join(current_path, "Models", result_name)
    final_save_path = os.path.join(checkpoint_save_path, result_name)
    best_save_path = os.path.join(final_save_path, "Best Model")
    eval_freq_adjusted = eval_freq/num_envs

    # setup envs
    env = SubprocVecEnv([lambda: Monitor(SumoEnv(gui=gui))  for i in range(num_envs)], start_method="spawn")
    eval_env = SubprocVecEnv([lambda: Monitor(SumoEnv(gui=gui))])

    # Callback to save checkpoints
    checkpoint_callback = CheckpointCallback(
        save_path=checkpoint_save_path,
        name_prefix= "rl_model",
        save_freq=eval_freq_adjusted, 
        verbose = 1
    )

    # Callback to evaluate model and save best
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=best_save_path,
        eval_freq=eval_freq_adjusted, 
        render=False,
        verbose = 1
    )

    # Create the callback list
    callback = CallbackList([checkpoint_callback, eval_callback])

    # create, train, save model
    model = DQN("MlpPolicy", env, verbose=1, tensorboard_log=log_path, device="cuda")
    model.learn(total_timesteps=timesteps, callback=callback)

    print("Training complete. Saving model.")
    model.save(final_save_path)
    print("Model Saved")

# run code on script run
if __name__ =="__main__":
    main()
