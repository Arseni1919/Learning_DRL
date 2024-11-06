# Learning Deep Reinforcement Learning (DRL) and Multi-Agent DRL

### DQN Algorithm

Gradient:
![](pics/dqn2.png)
Pseudo-code:
![](pics/dqn1.png)

### Double DQN

### REINFORCE

### Actor-Critic

Pseudo-code:
![valina_policy_gradient.png](pics/valina_policy_gradient.png)

### DDPG

Pseudo-code:
![](pics/ddpg_v2.png)

### PPO

Pseudo-code:
![](pics/ppo.png)

### TD3

Pseudo-code:
![](pics/td3_v2.png)


### SAC

Pseudo-code:
![](pics/sac_v2.png)

[//]: # (![]&#40;pics/sac1.png&#41;)

[//]: # (![]&#40;pics/sac2.png&#41;)

[//]: # (![]&#40;pics/sac3.png&#41;)

<img src='https://g.gravizo.com/svg?
  digraph G {
    size ="4,4";
    REINFORCE -> ActorCritic;
    ActorCritic -> TRPO;
    TRPO -> PPO;
    PPO -> SAC;
    ActorCritic -> DDPG;
    DDPG -> TD3;
    TD3 -> SAC;
    DQN -> ActorCritic;
    DQN -> DoubleDQN;
    DoubleDQN -> TD3;
    DoubleDQN -> SAC;
  }
'/>
![Alt text](https://g.gravizo.com/svg?
  digraph G {
    size ="4,4";
    REINFORCE -> ActorCritic;
    ActorCritic -> TRPO;
    TRPO -> PPO;
    PPO -> SAC;
    ActorCritic -> DDPG;
    DDPG -> TD3;
    TD3 -> SAC;
    DQN -> ActorCritic;
    DQN -> DoubleDQN;
    DoubleDQN -> TD3;
    DoubleDQN -> SAC;
  }
)


## Credits

- [OpenAI | Spinning Up in Deep RL](https://spinningup.openai.com/en/latest/index.html)
- [PyTorch | DQN](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
- [PyTorch | REINFORCE, Actor-Critic Examples](https://github.com/pytorch/examples/tree/main/reinforcement_learning)
- [Medium | TD3](https://medium.com/geekculture/a-deep-dive-into-the-ddpg-algorithm-for-continuous-control-2718222c333e)
- [colab | PPO](https://colab.research.google.com/github/nikhilbarhate99/PPO-PyTorch/blob/master/PPO_colab.ipynb#scrollTo=Z4VJcUT2GlJz) ([github](https://github.com/nikhilbarhate99/PPO-PyTorch/blob/master/PPO.py))
- [cleanrl | PPO](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_continuous_action.py)
- [Medium | SAC](https://towardsdatascience.com/soft-actor-critic-demystified-b8427df61665) ([github](https://github.com/vaishak2future/sac/blob/master/sac.ipynb))
- [cleanrl (github) | SAC](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/sac_continuous_action.py)
