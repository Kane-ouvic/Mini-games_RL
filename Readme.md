# **Mini-Games_RL**

A reinforcement learning (RL) project focused on training AI agents to play classic games. The project uses **Deep Q-Networks (DQN)** to train an agent for playing Mario and implements **both DQN and Policy Gradient methods** to train an agent for Tetris. The goal is to explore different RL approaches and compare their effectiveness in learning complex game strategies.

---

## **Table of Contents**

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [File Structure](#file-structure)

---

## **Introduction**

This project aims to train reinforcement learning models to play two popular games: **Super Mario Bros.** and **Tetris**. Using RL techniques, the models learn to maximize their scores through trial and error.

* **Mario AI**: Uses Deep Q-Networks (DQN) to train an agent to navigate the game world efficiently.
* **Tetris AI**: Implements both **DQN** and **Policy Gradient** methods to compare performance and strategy formation.

The project helps in understanding the effectiveness of different RL techniques for decision-making in sequential game environments.

![Example Screenshot](https://github.com/Kane-ouvic/Mini-games_RL/blob/main/Results/mario.png)
![Example Screenshot](https://github.com/Kane-ouvic/Mini-games_RL/blob/main/Results/tetris.png)

---

## **Features**

* **Mario AI with DQN**: Uses a deep Q-learning algorithm to train an AI agent to play Mario.
* **Tetris AI with DQN and Policy Gradient**: Trains two different models for Tetris using DQN and Policy Gradient, enabling performance comparisons.
* **Custom Training Pipelines**: Implements training loops, reward functions, and environment preprocessing for effective learning.
* **Inference & Evaluation**: Allows running trained models to test their performance on unseen game scenarios.

---

## **Installation**

Step-by-step guide on how to set up the project.

1. Clone the repository:
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## **Usage**

### **Training**

Train the Mario model using DQN:

```bash
python mario/run.py
```

Train the Tetris model using DQN:

```bash
python tetris/run.py
```

Train the Tetris model using Policy Gradient:

```bash
python tetris/run_pg.py
```

### **Inference**

Run the trained Mario model:

```bash
python mario/eval.py
```

Run the trained Tetris model (DQN):

```bash
python tetris/eval.py
```

Run the trained Tetris model (Policy Gradient):

```bash
python tetris/eval_pg.py
```

---

## **File Structure**

Describe the project's directory and file layout.

```plaintext
mini-games-rl/
├── results/              # Inference results (saved models, logs, evaluation results)
├── mario/                # Training and inference code for Mario
├── tetris/               # Training and inference code for Tetris
├── requirements.txt      # Dependencies
├── README.md             # Project documentation
```

---

## **Future Work**

* Fine-tuning hyperparameters for better performance.
* Implementing additional RL techniques such as PPO or A2C.
* Expanding the project to other classic games.

This project is a great way to learn about reinforcement learning in gaming applications while experimenting with different algorithms. 🚀

### Example Image

![Example Screenshot](https://github.com/Kane-ouvic/CARLA_Segmentation/blob/main/result/imgs/1.png)

