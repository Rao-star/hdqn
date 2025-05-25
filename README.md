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
1. Set the parameters in file 'run_hqdn' <br>
**MiniGrid 5x5 DoorKey:** (Examples are given here for references )
NUM_EPISODES = 3000 
BATCH_SIZE = 128 
GAMMA = 0.99 
REPLAY_MEMORY_SIZE = 20000 
LEARNING_RATE = 0.0005 
ALPHA = 0.99 
EPS = 0.01 
