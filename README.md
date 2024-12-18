# Learning Deep Reinforcement Learning (DRL) and Multi-Agent DRL


## RL Algorithms 

### DQN Algorithm

Gradient:

<img src="pics/dqn2.png" width="700">

Pseudo-code:

<img src="pics/dqn1.png" width="700">

### Double DQN

### REINFORCE

### Actor-Critic

Pseudo-code:

<img src="pics/valina_policy_gradient.png" width="700">


### DDPG

Pseudo-code:

<img src="pics/ddpg_v2.png" width="700">

### SAC

Pseudo-code:

<img src="pics/ppo.png" width="700">

### TD3

Pseudo-code:

<img src="pics/td3_v2.png" width="700">


### SAC

Pseudo-code:

<img src="pics/sac_v2.png" width="700">


### RL Algorithms Interconnections

<img src="pics/rl_history.png" width="700">

## MARL Algorithms

### IQL

The idea is simple. Just run independent RL agents in some environments. 
The main problem: nonstationarity of the world that is dependent on actions of other agents. Surprisingly, in some cases IQL works great (e.g. ping-pong env). 
In [IQL (2015)](https://arxiv.org/abs/1511.08779) paper the authors show how by playing with the reward definitions the behaviour of agents change from cooperative to competitive.


### VDN

Implicitly, the value decomposition network aims to learn an optimal linear value decomposition from the team reward signal, 
by back-propagating the total Q gradient through deep neural networks representing the individual component value functions.

This additive value decomposition is speciﬁcally motivated by avoiding the spurious reward signals that emerge in purely independent learners.
The implicit value function learned by each agent depends only on local observations, and so is more easily learned.

<img src="pics/vdn_q_function.png" width="700">

Basic architecture:

<img src="pics/vdn_basic_arc.png" width="700">


### QMix

QMIX is a value-based method that can train decentralised policies in a centralised end-to-end fashion. 
QMIX employs a network that estimates joint action-values as a complex non-linear combination of per-agent values that condition only on local observations. 
THe authors structurally enforce that the joint-action value is monotonic in the per-agent values, which allows tractable maximisation of the joint action-value in off-policy learning, and guarantees consistency between the centralised and decentralised policies.

Basically, QMix is like VDN but is not constrained to linear dependencies between agents' Q values.
QMix also makes use of external state of the environment.

The graphical representation of QMix NNs:

<img src="pics/qmix.png" width="700">


### COMA

Counterfactual Multi-Agent (COMA) is an actor-critic algorithm with a centralised critic.
Three main ideas underly COMA:
1) centralisation of the critic, 
2) use of a counterfactual baseline, and 
3) use of a critic representation that allows efﬁcient evaluation of the baseline.

COMA also uses external state of the environment in the learning stage.

The architecture:

<img src="pics/coma.png" width="700">

### DGN

The overview of DGN:

<img src="pics/dgn.png" width="700">

<img src="pics/dgn2.png" width="700">

Code: [https://github.com/PKU-RL/DGN](https://github.com/PKU-RL/DGN)

### QTRAN
### IPPO
### ROMA
### MADDPG
### QPlex
### MAPPO
### Belief-PPO
### IDDPG
### SHAQ

## MARL for MAPF

### PRIMAL


## Environments

### MAgent2

The code for parallel execution:

```python
# from magent2.environments.adversarial_pursuit import parallel_env
# from magent2.environments.battle import parallel_env
# from magent2.environments.battlefield import parallel_env
# from magent2.environments.combined_arms import parallel_env
# from magent2.environments.gather import parallel_env, raw_env
from magent2.environments.tiger_deer import parallel_env

render_mode='human'
# render_mode=None

env = parallel_env(render_mode=render_mode, max_cycles=200)
observations, infos = env.reset()

i_step = 0
while env.agents:
    # this is where you would insert your policy
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}

    observations, rewards, terminations, truncations, infos = env.step(actions)

    i_step += 1
    print(f'{i_step}')
    # time.sleep(0.1)
env.close()
```

Envs: [https://github.com/Farama-Foundation/MAgent2/tree/main/magent2/environments](https://github.com/Farama-Foundation/MAgent2/tree/main/magent2/environments)



## Credits

Stand on the shoulders of giants.

### General

- [OpenAI | Spinning Up in Deep RL](https://spinningup.openai.com/en/latest/index.html)
- [graphviz in python](https://graphviz.readthedocs.io/en/stable/index.html)
- [OpenAI | Key Papers in Deep RL](https://spinningup.openai.com/en/latest/spinningup/keypapers.html)

### Algorithms RL

- [PyTorch | DQN](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
- [PyTorch | REINFORCE, Actor-Critic Examples](https://github.com/pytorch/examples/tree/main/reinforcement_learning)
- [Medium | DDPG](https://medium.com/geekculture/a-deep-dive-into-the-ddpg-algorithm-for-continuous-control-2718222c333e)
- [colab | PPO](https://colab.research.google.com/github/nikhilbarhate99/PPO-PyTorch/blob/master/PPO_colab.ipynb)
- [Medium | TD3](https://medium.com/geekculture/a-deep-dive-into-the-ddpg-algorithm-for-continuous-control-2718222c333e)
- [colab | SAC](https://colab.research.google.com/github/nikhilbarhate99/PPO-PyTorch/blob/master/PPO_colab.ipynb#scrollTo=Z4VJcUT2GlJz) ([github](https://github.com/nikhilbarhate99/PPO-PyTorch/blob/master/PPO.py))
- [cleanrl | SAC](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_continuous_action.py) ([github](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/sac_continuous_action.py))
- [Medium | SAC](https://towardsdatascience.com/soft-actor-critic-demystified-b8427df61665) ([github](https://github.com/vaishak2future/sac/blob/master/sac.ipynb))

### Algorithms MARL

- [github | DGN](https://github.com/PKU-RL/DGN)







[//]: # (<img src="" width="700">)