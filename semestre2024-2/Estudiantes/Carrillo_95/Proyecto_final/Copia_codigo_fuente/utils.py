# Standard 
from typing import List

# Third-party 
from matplotlib.colors import ListedColormap
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


# Important setting variables for the animation aesthetics
colors = ['green', 'red', 'black', 'white']
ticksLabels = ['Healthy tree', 'Burning tree', 'Burned tree', 'Vacant Agent']
customCmap = ListedColormap(colors)
ticksLocation = [0.375, 1.125, 1.875, 2.625]

# Functions for animations
def squareAnimationPlot(filename:str, historical:List[List[int]], interval:int, p_bond, p_site) -> None:
    """
    Creates and saves an animated GIF of a square tessellation simulation for fire propagation.

    Args:
        filename (str): The name of the file (without extension) where the animation will be saved.
        historical (List[List[int]]): A list of 2D arrays (as lists of lists) representing the states of the simulation at each time step.
        interval (int): Time interval between frames in milliseconds.
        p_bond (float): Probability of bond formation in the simulation. (Probability of propagation)
        p_site (float): Probability of site occupation in the simulation. (Vaccancies not implemented)

    Returns:
        None: The function saves the animation as a GIF file.
    """
    
    # Create a figure and axis for the animation
    fig, ax = plt.subplots(figsize=(10, 10))
    cax = ax.matshow(historical[0], cmap=customCmap, vmin=1, vmax=4)
    ax.set_title('Square tessellation simulation', size=20)
    ax.set_xlabel(r'$P_{bond}=$' + str(round(p_bond,2)) + r'  $P_{site}=$' + str(round(p_site,2)), size=15)
    
    # Add a color bar with labels and ticks
    cbar = plt.colorbar(cax, ticks=ticksLocation)
    cbar.set_ticklabels(ticksLabels)
    cbar.set_label('Tree status')

    # Define the update function for the animation
    def update(i):
        """
        Updates the matrix displayed in the animation at each frame.

        Args:
            i (int): The current frame index.

        Returns:
            List: The updated image object to be re-rendered.
        """
        cax.set_array(historical[i])  # Update the displayed matrix
        return [cax]

    # Set up the animation
    squareAni = animation.FuncAnimation(
        fig, update, frames=len(historical), interval=interval, blit=True)
    
    # Save the animation as a GIF file
    squareAni.save('gifs/' + filename + ".gif", writer="pillow")
    
# Parameters for training and testing PPO agent
parameters = {
'neighbours': [(-1,0),(1,0),(0,1),(0,-1)],
'neighboursBoolTensor': np.ones((4,20,20), dtype=bool),
'grid_size': 20,
'threshold':0.55,
'initial_iters': 1,
'pre_defined_actions': True
}
