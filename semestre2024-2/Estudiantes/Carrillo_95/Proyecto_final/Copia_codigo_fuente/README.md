# 🌲🔥 FireForest_RL 🌲🔥


FireForest_RL is a reinforcement learning project aimed at training an agent using the PPO algorithm (Proximal Policy Optimization) from **Stable-Baselines3** to control and extinguish a forest fire propagating on a square lattice. The fire spreads using principles from percolation theory, and the agent is tasked with taking optimal actions to minimize damage.

---

## ℹ️ Purpose

The primary goal of this project is to:

- Train a reinforcement learning (RL) agent to mitigate forest fire spread by extinguishing individual cells of a grid-based simulation.
- Utilize **PPO** for its clipping property, which helps maintain stability during training, making it a suitable algorithm for handling the stochastic nature of the problem.
- Implement a scalable observation space with a tensor-based representation of the environment; the structure of the code allows topography and wind model implementation.

The project demonstrates how deep learning and RL can be applied to dynamic, complex problems with high-dimensional observation spaces under stochastic time development.

---

## 🧠 Features

- **Proximal Policy Optimization (PPO):** The PPO algorithm ensures that policy updates remain stable by clipping excessive updates, avoiding divergence during training.
- **Scalable Observation Space:**
  - The environment is initially constructed as an array of shape (grid_size,grid_size) where it's entries can be numbers from 1 to 4:
    - 1: Healthy tree
    - 2: Burning tree
    - 3: Burned tree.
    - 4: Vacant cell.
    The challenge arises when the agent needs to extract features from the environment. Given the structure we described, it is more convenient to represent the forest as a simple image and use a convolutional neural network (CNN) to identify significant patterns in the environment's state at a given time step. Implementing a neural network for this purpose is crucial to preserving the spatial relationships inherent in fire propagation, the layers of the image_like environment are defined as follows.
  - The environment is represented as a tensor of shape `(4, size, size)` where:
    - Layer 1: Healthy trees.
    - Layer 2: Burning cells.
    - Layer 3: Burned cells.
    - Layer 4: Vacant cells or cells where the agent has acted.
  - Each cell state type is separated in a separate tensor layer, so the entries of the tensor are 0 or 1.
- **Convolutional Neural Network (CNN):**
  - A custom CNN is used for feature extraction, enabling the agent to interpret the environment as image-like data.
  - Each update step transforms the fire propagation into this interpretable tensor format, i.e, the fire propagation method is implemented for a 2D array that must then be transformed to a image_like format on each discrete time step.
- **Avoidance of Tabular Methods:**
  - Traditional methods like Q-learning are impractical due to the exponential growth in state-action combinations. PPO instead constructs a probability density function over this space, ensuring scalability and a proper memory usage.

---

## 🌐 Technologies

- **Stable-Baselines3:** Provides the PPO implementation for reinforcement learning.
- **PyTorch:** Used for building and training the custom CNN for feature extraction.

---

## 🔧 Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/S3b4sc/ForestFire_RL.git
2. Navigate to the project directory:
   ```bash
   cd FireForest_RL
   ```
3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚧 Usage

*environment.py* This code should be ran to start the agent training and specify the total time  steps for constructing the policy, the parameters for its training can be modified on *utils.py* whre you can modify the grid size, p_bond, and if to use defined structues as agent's actions or simply use one vacancy per time step.

*test_environment.py* this screip is used for runing a test using the trained model, ir generated a gif animation that can be found on gifs diretory,  you can modify the name of the model to load for this purpose.

*test_plots.py* after training a agent, some trainnig information is saved on the log directory, this training parameters can be plotted running this script.


## 🤝 Contributors

- **Sebastian Carrillo Mejia** ([@S3b4sc](https://github.com/S3b4sc))
- **Santiago Ramirez Gaviria** ([@santiago-Ramirez3](https://github.com/santiagoRamirez3))

---

## ⚖️ License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.

---

## 🌟 Acknowledgments

- The authors of Stable-Baselines3 for providing a robust RL framework.
- The PyTorch community for its extensive documentation and tools for building neural networks.
- Researchers in percolation theory whose work inspired the fire propagation model.

---