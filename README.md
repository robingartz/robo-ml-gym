# Robo ML Gym
Reinforcement Learning for Robot Bin-Picking with the ABB 120 Robot  
using PyBullet Physics Engine and Gymnasium Environment

![image](https://github.com/robingartz/robo-ml-gym/assets/76932159/34465559-9470-41b0-9ad7-18facbd522cd)

## Install Gymnasium on Windows:
```
$ conda create -n gymenv  
$ conda activate gymenv  
$ conda install python=3.9  
$ conda install -c conda-forge pip   
$ conda install swig  
```
Install C++ build tools: https://visualstudio.microsoft.com/visual-cpp-build-tools/  
```
$ pip install gymnasium pybullet==3.2.5 stable-baselines3==2.3.0 pyyaml wandb
```

Other libraries:  
``` 
$ pip install rl_zoo3  
```

## Handy References:
Gymnasium: https://gymnasium.farama.org/  
PyBullet Docs: https://pybullet.org/wordpress/index.php/forum-2/  
PyBullet Quickstart: https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA  
panda-gym: https://github.com/qgallouedec/panda-gym/  
robo-gym: https://github.com/jr-robotics/robo-gym  

## Custom Environment Setup:
https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/#sphx-glr-tutorials-gymnasium-basics-environment-creation-py  
```  
$ cd robo-ml-gym/  
$ conda activate gymenv  
$ python3 train.py
$ python3 visualise.py  
```

## PyCharm Setup:
Set default Python interpreter to gymenv as outlined in:  
https://stackoverflow.com/questions/42746732/use-conda-environment-in-pycharm/46133678#46133678
Set Configuration script to: validation.py  
Working directory: robo-ml-gym  

## Also Checkout:
### MuJoCo:
DeepMind's physics engine for development in robotics + etc fast & accurate simulation with emphasis on contacts

### SB3:
Set of implementations of RL algorithms in PyTorch

### Weights & Biases: experiment tracking
### Hugging Face: storing/sharing models
