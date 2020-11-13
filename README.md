# Reinforcement Learning Experiment

This is the implementation for SJTU EI339 teamwork project of reinforcement learning.



## File Structure

```
└───Easy21
        ├─easy21_pi.py          
        ├─easy21_TD.py
        ├─easy21_MC.py
        └─storage/           # results stored

└───pytorch-trpo
    ├─ conjugate_gradients.py
    ├─ main.py            # entrance for training
    ├─ models.py
    ├─ replay_memory.py
    ├─ running_state.py
    ├─ test.py            # entrance for testing
    ├─ trpo.py
    ├─ utils.py
    └───__pycache__/
    └─── save_model/        # model parameters saved
		└─── storage/           # figures of loss and reward

└───MPC
    └─── MPC-Qube
        ├─ README.md
        ├─ RandomShooting.py
        ├─ SimAnneal.py
        ├─ __init__.py
        ├─ config.yml
        ├─ controller.py
        ├─ dynamics.py
        ├─ run.py
        ├─ test.py
        ├─ utils.py
        └─── Hive/
        └─── storage/
      
    └─── MPC-CartPoleSwing/		(similar)
    └─── MPC-BallBalancerSim/  	(similar)
    └─── README.md
    └─── __init__.py
    
└───vedio                    # vedio attached for testing in MPC and TRPO

```



## Usage

#### Easy21

It can be directly run in pycharm or in terminal using 

```
python easy21_xx.py
```

#### TRPO

Take the environment "Qube-100-v0" as an example

- Training phase

  ```
  python main.py --env-name Qube-100-v0
  ```

  The learned model parameters will be stored in direction "save_model" with ".pt" ending.

  The loss and award during training is located in "storage".

- Testing phase

  ```
  python test.py --model-path save_model/... --env-name Qube-100-v0
  ```

#### MPC

- Training phase

  ```
  python run.py -p storage/config-1.yml
  ```

- Testing phase

  ```
  python test.py -p storage/config-1.yml
  ```



## Parameters Setting

#### Easy21

Take "Easy21_MC" as an example, we enter the training phase and testing phase by calling "Q_learning.train()" and "Q_learning.test()" separately. The parameters, namely, learning rate and epsilon, are passed into the model in line 213. 

#### TRPO

The parameters are set in the "argparse" part in "pytorch-trpo/main.py" with "help" elaborating different meanings of the hyper-parameters.

#### MPC

Use configure file ```config.yml```, to set all the parameters.

