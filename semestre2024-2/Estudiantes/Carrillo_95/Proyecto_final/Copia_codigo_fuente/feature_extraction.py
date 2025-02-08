# Third-party 
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym


# Define the custor CC Feature Extractor
class CustomCNNFeaturExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor using a Convolutional Neural Network (CNN) followed by a fully connected (FC) layer.
    
    This extractor is used to transform the input observations (e.g., images) from the environment into a feature
    vector, typically used by reinforcement learning algorithms like PPO or A2C.
    
    Args:
        observation_space (gym.spaces.Box): The observation space of the environment, which provides information about
                                             the shape of the input (e.g., image dimensions).
        features_dim (int): The dimension of the feature vector that will be output by the extractor. Default is 512.
        normalized_image (bool): Whether or not to normalize the input image. Default is False.
    """
    def __init__(self,
                 observation_space:gym.spaces.Box,
                 features_dim:int = 512,
                 normalized_image:bool = False) -> None:
        
        # Call the parent constructor to initialize the feature extractor base class
        super(CustomCNNFeaturExtractor,self).__init__(observation_space, features_dim)
        
        # Extract the dimensions of the input (assumed to be an image in the environment)
        input_channels = observation_space.shape[0] # Number of input channels (e.g., 3 for RGB images)
        grid_height = observation_space.shape[1]    # Height of the image
        grid_width = observation_space.shape[2]    # Width of the image
        
        # Define the CNN layers (Convolutional + Activation layers)
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels,32, kernel_size=2,stride=1, padding=0),    # First convolutional layer
            nn.ReLU(),                                                          # ReLU activation
            nn.Conv2d(32,64,kernel_size=2, stride=1,padding=0),      # Second convolutional layer
            nn.ReLU(),                                                # ReLU activation
            nn.Conv2d(64,64,kernel_size=2, stride=1,padding=0),     # Third convolutional layer
            nn.ReLU(),                                              # ReLU activation
            nn.Flatten()                                            # Flatten the output for FC layers
        )
        
         # Calculate the output dimension of the CNN (after convolutions and flattening)
        with torch.no_grad():    # No need to track gradients while calculating the output size
            dummy_input = torch.zeros(1, input_channels, grid_height, grid_width)   # Create a dummy input tensor
            cnn_output_dim = self.cnn(dummy_input).shape[1]          # Get the flattened output dimension
              
        # Define a fully connected (FC) layer to map CNN output to the desired feature dimension
        self.fc = nn.Sequential(
            nn.Linear(cnn_output_dim, features_dim),   # Fully connected layer
            nn.ReLU()                           # ReLU activation
        )
        
        # Update feature dimension
    
    def forward(self, observations:torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the CNN and fully connected (FC) layers.
        
        Args:
            observations (torch.Tensor): The input observations, typically of shape (batch_size, input_channels, height, width)
        
        Returns:
            torch.Tensor: The output feature vector of shape (batch_size, features_dim)
        """
        # Pass the input observations through the CNN
        cnn_output = self.cnn(observations)
        # Pass the CNN output through the fully connected layers
        return self.fc(cnn_output)
    
