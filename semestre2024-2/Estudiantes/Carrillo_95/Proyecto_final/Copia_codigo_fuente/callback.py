# Standard
import os

# Third party
import numpy as np
import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback

class SaveTrainingLogCallback(BaseCallback):
    def __init__(self, log_dir="./logs/", verbose=1):
        super(SaveTrainingLogCallback, self).__init__(verbose)
        self.log_dir = log_dir
        self.log_file = os.path.join(log_dir, "training_log_20.csv")
        self.training_data = []

        # Ensure log directory exists
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

    def _on_step(self) -> bool:
        # Get info from training logs
        logs = self.locals["infos"]  # Env information
        if len(logs) > 0 and "episode" in logs[0]:  # Ensure episode data exists
            episode_reward = logs[0]["episode"]["r"]
            self.training_data.append([
                self.num_timesteps,  # Number of timesteps
                episode_reward,  # Episode reward
                self.model.logger.name_to_value["train/policy_loss"],  # Policy loss
                self.model.logger.name_to_value["train/value_loss"],  # Value loss
                self.model.logger.name_to_value["train/entropy_loss"],  # Entropy loss
                self.model.logger.name_to_value["train/approx_kl"]  # KL divergence
            ])

        return True  # Continue training

    def _on_training_end(self) -> None:
        # Convert data to DataFrame and save as CSV
        df = pd.DataFrame(self.training_data,
                          columns=["Timesteps", "EpisodeReward", "PolicyLoss", "ValueLoss", "EntropyLoss", "KL_Divergence"])
        df.to_csv(self.log_file, index=False)
        if self.verbose:
            print(f"Training log saved to {self.log_file}")
