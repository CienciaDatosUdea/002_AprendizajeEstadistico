# Standard
from typing import final, List, Tuple

# Third-party 
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Local application
from utils import parameters
from feature_extraction import CustomCNNFeaturExtractor
from callback import SaveTrainingLogCallback


class FirePropagationEnv(gym.Env):
    """
    Fire Propagation Environment for simulating forest fires. The environment
    simulates a grid-based forest where cells can have different states:
    - 1: Healthy
    - 2: Burning
    - 3: Burned
    - 4: Empty
    
    The environment tracks the spread of fire and allows actions like extinguishing cells.
    """
    def __init__(self,
                 neighbours:List[Tuple[int,int]],
                 neighboursBoolTensor:np.ndarray,
                 grid_size:int,
                 threshold:float,
                 initial_iters:int,
                 pre_defined_actions:bool):
        """
        Initialize the fire propagation environment.
        
        Args:
            neighbours (list of tuples): List of tuples representing the relative positions of neighboring cells.
            neighboursBoolTensor (np.ndarray): A tensor representing the neighborhood connectivity for the grid.
            grid_size (int): The size of the grid (forest). It's a square grid of shape (grid_size, grid_size).
            threshold (float): The burning threshold, which dictates how susceptible cells are to catching fire.
            initial_iters (int): The initial number of iterations before the agent can take action.
            pre_defined_actions (bool): Whether the model will use predefined actions or not.
            

        Returns:
            None
        """
        super(FirePropagationEnv, self).__init__()
        
        # Neighbours list
        self.neighbours = neighbours
        # Boolean neighbour tensor for regular tesellation
        self.neighboursBoolTensor = neighboursBoolTensor
        # Size of the forest
        self.grid_size = grid_size
        # Burning threshold
        self.threshold = threshold
        # inital iterations
        self.initial_iters = initial_iters
        
        self.pre_defined_actions = pre_defined_actions
        # Action space
        if self.pre_defined_actions:
            # Extinguish cells on predefined structures
            self.action_space = spaces.MultiDiscrete(np.array([4,self.grid_size-2,self.grid_size-2]), dtype=np.int32)
        else: 
            # Extinguish cells (limited per step)
            self.action_space = spaces.Discrete(self.grid_size*self.grid_size)
            #self.action_space = spaces.MultiDiscrete(np.array([self.grid_size,self.grid_size]), dtype=np.int32)
            
        # Historical actions record
        self.historical_actions = []
        # Observation space: forest of cell states (0, 1=healthy, 2=burning, 3=burned, 4=empty)
        self.observation_space = spaces.Box(low=0, high=4, shape=(4,grid_size, grid_size), dtype=np.int32)
        # Initialize environment
        self.reset()

    def reset(self, seed:int = None, options:dict = None):
        """
        Resets the environment to its initial state.

        Returns:
            observation (np.ndarray): The initial state of the forest, with cell values indicating their states.
            info (dict): Additional information, such as the initial state of the forest.
        """
        super().reset(seed=seed)
        # Create initial forest: majority healthy, one burning
        self.forest = np.ones((self.grid_size, self.grid_size), dtype=np.int32)
        # Start initial fire in the center of the lattice
        self.forest[self.grid_size//2,  self.grid_size//2] = 2
        self.forest[self.grid_size//2,  self.grid_size//2 + 1] = 2
        
        # Propagate few initial steps if desired
        self.steps = 0
        self._propagate_fire(iters=self.initial_iters)

        self.historical_actions = []
        
        observation = self._to_image_like(matrix=self.forest)
        
        info = {
            'initial_forest': self.forest
        }
        
        return observation, info

    def step(self, actions):
        """
        Take a step in the environment by applying the given action.
        
        Args:
            action (int): The index of the cell to extinguish (flattened 2D grid to 1D).
        
        Returns:
            observation (np.ndarray): The updated state of the forest after the action.
            reward (float): The reward for taking the action.
            done (bool): Whether the episode is done (if the fire has been extinguished or max steps reached).
            truncated (bool): Whether the episode was truncated due to exceeding the max steps.
            info (dict): Additional information, such as the current fire state.
        """
        if self.pre_defined_actions:
            # The action is mapped to a option on desired position
            shape, x, y = actions  # Extract action components
    
            x += 1      # Prevent out of bound shapes
            y += 1

            # Apply the action (turn off cells)
            self.apply_shape(shape, x, y)

        else: 
            # The action represents an index of the grid, calculate its 2D coordinates
            #x,y = actions
            x,y = divmod(actions,self.grid_size)
            # Apply extinguish action (mark the selected cell as empty)
            self.forest[x,y] = 4

        # Run one time step after agent's action (Update environment)
        self._propagate_fire()
                
        # Stablish reward criteria
        # Check if the episode is terminated
        terminated = (np.sum(self.forest == 2) == 0)
        
        if terminated:
            burned_cells = np.sum(self.forest == 3)/(self.grid_size*self.grid_size)
        else:
            burned_cells = 0
            
        # Set truncated in false for no use
        truncated = False
        
        reward = - burned_cells     # Penalize bruned cells after the complete episode
        observation = self._to_image_like(matrix=self.forest)   # Reshape observation to image like format
        
        info = {
            'forest': self.forest       # Save forest for trazability
        }
    
        return observation, reward, terminated, truncated, info

    # Method for executing pre defined action
    def apply_shape(self,shape:int, x:int, y:int):
        
        # Vertical line
        if shape == 0:
            self.forest[x,y] = 4
            self.forest[x+1,y] = 4
            self.forest[x-1,y] = 4
        # Horizontal
        elif shape == 1:
            self.forest[x,y] = 4
            self.forest[x,y+1] = 4
            self.forest[x,y-1] = 4
        #Diagonal right
        elif shape == 2:
            self.forest[x,y] = 4
            self.forest[x-1,y+1] = 4
            self.forest[x+1,y-1] = 4
        # Diagonal left
        elif shape == 3:
            self.forest[x,y] = 4
            self.forest[x-1,y-1] = 4
            self.forest[x+1,y+1] = 4
    
    # Method for fire spread
    @final
    def _propagate_fire(self,iters:int=1) -> None:
        """
        Propagate fire given some time steps
        
        Args:
            inters: number of times steps the fire propagates
        
        Returns:
            None
        """
        i = 1
        
        while (0<i<=iters):
            neighboursTensor = self.createNeighbourTensor()
            # Find those current burning trees
            couldPropagate = (neighboursTensor == 2)
            # Number of burning neighbours
            amountOfBurningNeighbours = np.sum(couldPropagate, axis=0)
            # Filter those cells wiht burning neighbours
            cellsToEvaluate = amountOfBurningNeighbours > 0
            # Boleand mask of burning trees
            burningTrees = (self.forest == 2)
            # Generate probality propagation matrix
            probabilityMatrixForest = np.random.rand(*self.forest.shape)
            probabilityMatrixForest[cellsToEvaluate] = 1. - (1. - probabilityMatrixForest[cellsToEvaluate]) ** (1/amountOfBurningNeighbours[cellsToEvaluate])
            # Find cells that could burn
            couldBurn = (probabilityMatrixForest <= self.threshold)
            # Find new burning trees
            newBurningTrees = (self.forest == 1) & couldBurn & np.logical_or.reduce(couldPropagate,axis=0)

            # Update forest matrix for the next step
            self.forest[burningTrees] = 3
            self.forest[newBurningTrees] = 2

            # Update steps count
            self.steps += 1
            i += 1
        
        
    # Method necessary for step method propagating time
    @final
    def createNeighbourTensor(self) -> np.ndarray:
        """
        Creates a tensor representing the shifted views of the forest grid for each neighbor direction.

        This function generates a 3D tensor where each slice corresponds to the forest grid shifted
        according to the directions specified in `self.neighbours`. It also accounts for boundary conditions,
        setting cells outside the grid boundaries to zero.

        The tensor is further filtered by `self.neighboursBoolTensor` to account for valid neighbor relationships.

        Returns:
            np.ndarray: A 3D tensor of shape `(len(self.neighbours), *self.forest.shape)`, where:
                - `len(self.neighbours)` is the number of neighbor directions.
                - `self.forest.shape` is the shape of the 2D forest grid (rows, columns).
        """
        # Get the number of neighbors
        neighborhoodSize = len(self.neighbours)
         # Initialize the tensor with zeros; shape is (number of neighbors, forest rows, forest columns)
        tensor = np.zeros((neighborhoodSize, *self.forest.shape))
        
        # Iterate over each neighbor direction
        for i, neigh in enumerate(self.neighbours):
            x,y = neigh   # Neighbor shift in (row, column) format
            # Shift the forest grid using np.roll; axes are flipped due to the roll behavior
            tensor[i] = np.roll(self.forest, (-x,y), axis=(1,0))
            
            # Handle boundary conditions for rows
            if x:
                if x == 1.:   # Bottom row boundary
                    tensor[i, : , -1 ] = 0
                elif x == -1.:  # Top row boundary
                    tensor[i, : , 0 ] = 0
                else:
                    continue 
                
            # Handle boundary conditions for columns
            if y:
                if y == 1.:  # Right column boundary
                    tensor[i, 0 , : ] = 0
                elif y == -1.:   # Left column boundary
                    tensor[i, -1 , : ] = 0
                else:
                    continue 
            else:
                continue
            
        # Apply a mask to account for valid neighbor relationships
        tensor = tensor * self.neighboursBoolTensor
        return tensor
    
    # Reshape the forest into image-like format
    @final
    def _to_image_like(self, matrix:np.ndarray) -> np.ndarray:
        """
        Convert a 2D matrix (grid) into a one-hot encoded 3D tensor.

        This function takes a 2D grid (matrix) where each value represents a state (e.g., healthy, burning, burned, vacancy),
        and converts it into a 3D tensor (one-hot encoding). The output tensor will have one channel for each state.

        For example, if the grid has 4 possible states (healthy, burning, burned, and vacancy), the output will be a tensor 
        with 4 channels, where each channel is a binary matrix that indicates which cells belong to each state.

        Args:
            matrix (np.ndarray): A 2D matrix (grid) of shape (grid_size, grid_size), where each element represents a state.

        Returns:
            np.ndarray: A 3D tensor (one-hot encoded), with shape (num_channels, grid_size, grid_size), where each channel
                        represents one of the states in the input matrix.
        """
        # Define the number of channels (states). Here, 4 states are considered:
        # 1 - healthy, 2 - burning, 3 - burned, 4 - vacancy.
        num_channels = 4  
        
        # Initialize the one-hot encoded tensor with zeros.
        # Shape will be (num_channels, grid_size, grid_size)
        one_hot = np.zeros((num_channels, self.grid_size, self.grid_size), dtype=np.int32)
        
         # Loop through each channel (state)
        for channel in range(num_channels):
        # For each channel, mark the positions in the matrix that match the channel's state.
        # For example, (matrix == (channel + 1)) will create a boolean array where:
        # - True for cells with the state (channel + 1), and False for others.
        # Convert this boolean array into a int32 array (0 or 1) to represent the one-hot encoding.
            one_hot[channel] = (matrix == (channel+1)).astype(np.int32)
        return one_hot


# -------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    # Set Up the PPO Agent
    
    # Create the environment
    # Define the Policy with custom feature extractor
    policy_kwargs = dict(
        features_extractor_class = CustomCNNFeaturExtractor,
        features_extractor_kwargs = dict(features_dim=100)
        )

    # Wrap the environment for vectorized training
    vec_env = make_vec_env(lambda: FirePropagationEnv(**parameters), n_envs =4)

    # Create the PPO agent
    model = PPO("CnnPolicy",
                vec_env,
                policy_kwargs=policy_kwargs,
                verbose=1,
                learning_rate=3e-4,     # Increase to avoid local minumums
                clip_range=0.1,     # Reduce for larger updates
                ent_coef=0.02)      # Increase for encouraging exploration - reducing exploitation

    # Ceate the personalized callback
    callback = SaveTrainingLogCallback(log_dir="./logs/")
    
    # Train the agent
    model.learn(total_timesteps=4000000,log_interval=1, callback=callback)
    
    # Save the trained model
    model.save("./models/cnn_ppo_fire_agent_20_2")