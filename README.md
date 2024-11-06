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
   main -> parse -> execute;
   main -> init;
   main -> cleanup;
   execute -> make_string;
   execute -> printf
   init -> make_string;
   main -> printf;
   execute -> compare;
 }
'/>
![Alt text](https://g.gravizo.com/svg?%20digraph%20G%20{%20size%20=%224,4%22;%20maaaaain%20[shape=box];%20maaaain%20-%3E%20parse%20[weight=8];%20parse%20-%3E%20execute;%20main%20-%3E%20init%20[style=dotted];%20main%20-%3E%20cleanup;%20execute%20-%3E%20{%20make_string;%20printf}%20init%20-%3E%20make_string;%20edge%20[color=red];%20main%20-%3E%20printf%20[style=bold,label=%22100%20times%22];%20make_string%20[label=%22make%20a%20string%22];%20node%20[shape=box,style=filled,color=%22.7%20.3%201.0%22];%20execute%20-%3E%20compare;%20})
<img src='https://g.gravizo.com/svg?
  digraph G {
    REINFORCE -> Actor-Critic
    Actor-Critic -> TRPO
    TRPO -> PPO
    PPO -> SAC
    Actor-Critic -> DDPG
    DDPG -> TD3
    TD3 -> SAC
    DQN -> Actor-Critic
    DQN -> Double-DQN
    Double-DQN -> TD3
    Double-DQN -> SAC
  }
'/>

## Credits

- [OpenAI | Spinning Up in Deep RL](https://spinningup.openai.com/en/latest/index.html)
- [PyTorch | DQN](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
- [PyTorch | REINFORCE, Actor-Critic Examples](https://github.com/pytorch/examples/tree/main/reinforcement_learning)
- [Medium | TD3](https://medium.com/geekculture/a-deep-dive-into-the-ddpg-algorithm-for-continuous-control-2718222c333e)
- [colab | PPO](https://colab.research.google.com/github/nikhilbarhate99/PPO-PyTorch/blob/master/PPO_colab.ipynb#scrollTo=Z4VJcUT2GlJz) ([github](https://github.com/nikhilbarhate99/PPO-PyTorch/blob/master/PPO.py))
- [cleanrl | PPO](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_continuous_action.py)
- [Medium | SAC](https://towardsdatascience.com/soft-actor-critic-demystified-b8427df61665) ([github](https://github.com/vaishak2future/sac/blob/master/sac.ipynb))
- [cleanrl (github) | SAC](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/sac_continuous_action.py)
