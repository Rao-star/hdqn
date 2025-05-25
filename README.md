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
**MiniGrid 5x5 DoorKey:** Using RMSprop as optimizer<br>
**MiniGrid 8x8 DoorKey:** Using AdamW as optimizer<br>
