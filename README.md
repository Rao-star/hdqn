# Hierarchical DQN for MiniGrid DoorKey
Using H-DQN, train agent to find the key, open the door, and finally reach the end point in a MiniGrid environment

## Introduction
1. Creat virtual enviroment (Using conda)
```
conda create -n hdqn-env python=3.8 <br>
conda activate hdqn-env <br>
```
2. Install 'MiniGrid' enviroments and relevant libraries
```
pip install -r requirements.txt
```

## How to use
1. Set the parameters in file 'run_hqdn' (Examples are given here for references ) <br>
**MiniGrid 5x5 DoorKey:**
NUM_EPISODES = 3000<br>
BATCH_SIZE = 128<br>
GAMMA = 0.99<br>
REPLAY_MEMORY_SIZE = 20000<br>
LEARNING_RATE = 0.0005<br>
ALPHA = 0.99<br>
EPS = 0.01<br>
