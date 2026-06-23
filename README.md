Repository for Praktikum 2 for the MA Computer Science Course for the university of Vienna.

# Application of Autoencoders on Grid-Like Reinforcement Learning Environments for State Space Transformations

The repository lists the code for a series of custom environments used for the completion of the report for the "Praktikum 2" course. 

## Environments

IDE: PyCharm 2024.1.4 (Community Edition)

- `MovementPrototype_Single_Agent` : .
- `Construction_Collision_Avoidance`: .
- `Construction_Support` : . 

## Installation 

- Dependencies: Python (3.9), Farama Foundation Gymnasium (1.1.1), pygame (2.6.1), numpy (2.0.2) , stable-baselines3 (2.7.0), pytorch (2.8.0)

After creating a new project that uses python 3.9, install the dependencies and move the environment files under the project root, all environments have a `runEnv.py` file that can be used to run the corresponding environment. 

## Common Files Names Among Environments
- `runEnv.py` : Script that runs and renders the environment meant for quick inspection. 
- `trainEnv.py` : Script that uses stable baselines3 to train a model for the environment. 
