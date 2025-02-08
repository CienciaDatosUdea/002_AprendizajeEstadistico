# Standard 
from typing import List, Tuple

# Third-party 
from stable_baselines3 import PPO
import numpy as np

# Local application
from environment import FirePropagationEnv
from utils import squareAnimationPlot
from utils import parameters

# Testing environment with animation render 
class TestForestFireEnv(FirePropagationEnv):
    """
    A test environment for forest fire propagation simulations that extends the FirePropagationEnv class.
    This environment integrates an agent trained with PPO to make decisions and saves the historical steps
    of the forest grid for rendering an animation of the simulation.
    
    Attributes:
        historicalSteps (list): Stores the state of the forest at each time step for animation.
    """
    def __init__(self,
                 neighbours:List[Tuple[int,int]],
                 neighboursBoolTensor:np.ndarray,
                 grid_size:int, threshold:float,
                 initial_iters:int,
                 pre_defined_actions:bool) -> None:
        """
        Initializes the test environment by inheriting attributes from the parent class and 
        setting up storage for historical data.
        
        Args:
            neighbours (list): List of neighbor coordinates for each cell in the grid.
            neighboursBoolTensor (np.ndarray): Boolean tensor indicating valid neighbor relationships.
            grid_size (tuple): The size of the forest grid (rows, columns).
            threshold (float): Probability threshold for fire propagation.
            max_steps (int): Maximum number of steps in the simulation.
            extinguish_limit (int): Maximum number of extinguished cells allowed.
            pre_defined_actions (bool): Whether the model will use predefined actions or not.
        """
        # Initialize parent class
        super().__init__(neighbours,neighboursBoolTensor, grid_size, threshold, initial_iters, pre_defined_actions)
        self.historicalSteps = []
    
    # Save historical animation data to render
    def episode(self) -> None:
        """
        Runs a single episode of the environment using a pre-trained PPO agent, saving historical
        grid states for animation and rendering the results.
        
        Steps:
        - Resets the environment to its initial state.
        - Loads a pre-trained PPO model for decision-making.
        - Simulates the environment until the episode ends, saving forest states at each step.
        - Renders the animation of the simulation.
        """
        # Load the PPO model
        model = PPO.load("./models/cnn_ppo_fire_agent_20")
        observation, info = self.reset()
        
        # Save the initial state
        self.historicalSteps.append(np.copy(info['initial_forest']))
        done = False
        
        while not done:
            # The agent evaluates the environment and chooses an action
            actions,_ = model.predict(observation, deterministic = True)
            
            # Perform the action and advance one time step in the environment
            observation, reward, done, truncated, info = self.step(actions=actions)
            
            # Save the forest state after the step
            self.historicalSteps.append(np.copy(info['forest']))
            
        # Render the animation after the episode ends
        self.render()
        
    def render(self, fileName:str='animation_expo_9', interval:int=100) -> None:
        """
        Renders an animation of the forest fire simulation.

        Args:
            fileName (str): Name of the output file (default is 'animation').
            interval (int): Time interval (in milliseconds) between animation frames (default is 100ms).
        """
        print('Starting simulation, wait a sec...')
        squareAnimationPlot(fileName,
                            self.historicalSteps,
                            interval,
                            p_bond=self.threshold,
                            p_site=1)
        print('Done.')
            
if __name__ == "__main__":
    TestFire = TestForestFireEnv(**parameters)
    TestFire.episode()