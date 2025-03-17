# Learning Deep Reinforcement Learning (DRL) and Multi-Agent DRL

[//]: # (##########################################################)
[//]: # (##########################################################)
[//]: # (##########################################################)
## RL Algorithms 

### Terms 

- `credit assignment problem` - This problem is a key source of difficulty in RL that is the long time delay between actions and their positive or negative effect on rewards

### DQN Algorithm

- Paper: [https://arxiv.org/pdf/1312.05602](https://arxiv.org/pdf/1312.05602)
- Code: [PyTorch | DQN](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)

Gradient:

<img src="pics/dqn2.png" width="700">

Pseudo-code:

<img src="pics/dqn1.png" width="700">

### Double DQN (2016)

- Paper: [Deep Reinforcement Learning with Double Q-Learning](https://ojs.aaai.org/index.php/AAAI/article/view/10295)
- Code: ~

### Dueling DQN (2016)

- Paper: [Dueling Network Architectures for Deep Reinforcement Learning](https://proceedings.mlr.press/v48/wangf16.pdf)
- Env: [Farama | Atari Games](https://ale.farama.org/environments/)
- Code: ~

In this paper, the authors showed a nice trick to change the structure of the NNs inside already existing DRL algorithms, that provides better results.
They separated the $Q$-network from a single stream that outputs a value per each action to two separate streams for the $V$ part (state value) and $A$ parts (advantage values for every action). The illustration of the idea is presented here:

<img src="pics/dueling_dqn_1.png" width="500">

The question is how to combine the two outputs to get the required $Q$ value at the end. The naive approach to sum the $V$ and $A$ parts does not work. The authors say that this way the effect of separation cancels out.
So the alternative is to subtract from the advantage values the maximum $A$ value or.. the average of $A$ values as here:

<img src="pics/dueling_dqn_2.png" width="500">

Note that $Q(s, a*) = V(s,a*)$ and $A(s, a*) = 0$, where $a*$ is the best action. That's why the formula makes sense.

The great part of this new trick is that it naturally provides two following advantages:

1. **The $V$ part is updated with every $Q$ update in the dueling schema, unlike previous approaches:** "The dueling architecture has an ability to learn the state-value function efﬁciently. With every update of the Q values in the dueling architecture, the value stream $V$ is updated – this contrasts with the updates in a single-stream architecture where only the value of one actions is updated, the values for all other actions remain untouched. This more frequent updating of the value stream allocates more resources to $V$, and thus allows for better approximation of the state values, which need to be accurate for temporal-difference-based methods like $Q$-learning to work (Sutton & Barto, 1998). This phenomenon is reﬂected in the experiments, where the advantage of the dueling architecture over single-stream $Q$ networks grows when the number of actions is large."
2. **Dueling separation helps to distinguish between small differences in action values:** "The differences between $Q$-values for a given state are often very small relative to the magnitude of $Q$. For example, after training with DDQN on the game of Seaquest, the average action gap (the gap between the Q values of the best and the second best action in a given state) across visited states is roughly 0.04, whereas the average state value across those states is about 15. This difference in scale means that small amount of noise in the updates could reorder the actions, and thus making the nearly greedy policy switch abruptly. The dueling architecture with its separate advantage stream is robust to such effects."

### REINFORCE

- Explanation: [PyTorch | REINFORCE, Actor-Critic Examples](https://github.com/pytorch/examples/tree/main/reinforcement_learning)

### Actor-Critic

- Paper: [https://proceedings.mlr.press/v48/mniha16.pdf](https://proceedings.mlr.press/v48/mniha16.pdf)
- Code: [PyTorch | REINFORCE, Actor-Critic Examples](https://github.com/pytorch/examples/tree/main/reinforcement_learning)

Pseudo-code:

<img src="pics/valina_policy_gradient.png" width="700">


### DDPG

- Paper: [https://arxiv.org/pdf/1509.02971](https://arxiv.org/pdf/1509.02971)
- Code: [Medium | DDPG](https://medium.com/geekculture/a-deep-dive-into-the-ddpg-algorithm-for-continuous-control-2718222c333e)

Pseudo-code:

<img src="pics/ddpg_v2.png" width="700">

### GAE (2016)

- Paper: [HIGH-DIMENSIONAL CONTINUOUS CONTROL USING GENERALIZED ADVANTAGE ESTIMATION](https://arxiv.org/pdf/1506.02438)
- Code: ~

The idea of GAE is to implement $TD(\lambda)$ ideas on the advantage values instead of state-values.
As in $TD(\lambda)$, GAE helps to balance between variance and bias of advantage values. 
Here is the formulation of GAE:

<img src="pics/gae_1.png" width="700">

<img src="pics/gae_2.png" width="700">

<img src="pics/gae_3.png" width="700">

The usage of GAE is primarily in policy optimization algorithms (PPO, TRPO, etc.) to reduce variance without adding to much bias.

### PPO (2017)

- Paper: [https://arxiv.org/pdf/1707.06347](https://arxiv.org/pdf/1707.06347)
- Code: [colab | PPO](https://colab.research.google.com/github/nikhilbarhate99/PPO-PyTorch/blob/master/PPO_colab.ipynb)

Pseudo-code:

<img src="pics/ppo.png" width="700">

### TD3

- Paper: [https://proceedings.mlr.press/v80/fujimoto18a/fujimoto18a.pdf](https://proceedings.mlr.press/v80/fujimoto18a/fujimoto18a.pdf)
- Code: 
  - [Medium | TD3](https://medium.com/geekculture/a-deep-dive-into-the-ddpg-algorithm-for-continuous-control-2718222c333e)

Pseudo-code:

<img src="pics/td3_v2.png" width="700">


### SAC

- Paper: [https://proceedings.mlr.press/v80/haarnoja18b/haarnoja18b.pdf](https://proceedings.mlr.press/v80/haarnoja18b/haarnoja18b.pdf)
- Code:
  - [Medium | TD3](https://medium.com/geekculture/a-deep-dive-into-the-ddpg-algorithm-for-continuous-control-2718222c333e)
  - [colab | SAC](https://colab.research.google.com/github/nikhilbarhate99/PPO-PyTorch/blob/master/PPO_colab.ipynb#scrollTo=Z4VJcUT2GlJz) ([github](https://github.com/nikhilbarhate99/PPO-PyTorch/blob/master/PPO.py))
  - [cleanrl | SAC](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_continuous_action.py) ([github](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/sac_continuous_action.py))
  - [Medium | SAC](https://towardsdatascience.com/soft-actor-critic-demystified-b8427df61665) ([github](https://github.com/vaishak2future/sac/blob/master/sac.ipynb))

Pseudo-code:

<img src="pics/sac_v2.png" width="700">


### RL Algorithms Interconnections

<img src="pics/rl_history.png" width="700">

--- 


[//]: # (##########################################################)
[//]: # (##########################################################)
[//]: # (##########################################################)
## MARL Algorithms

### Terms

- `non-stationary` - each agent’s policy is changing as training progresses, and the environment becomes non-stationary from the perspective of any individual agent (in a way that is not explainable by changes in the agent’s own policy)
- `CTDE` - centralised training decentralised execution
- `IGM (Individual-Global-Max)` - To enable effective CTDE for multi-agent Q-learning, it is critical that the joint greedy action should be equivalent to the collection of individual greedy actions of agents, which is called the IGM (Individual-Global-Max) principle (Son et al., 2019).

### IQL (1993, 2015)

- Paper (1993): [https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=b3fc56876ad1cdf35ad4af13b991bbb24d219bd9](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=b3fc56876ad1cdf35ad4af13b991bbb24d219bd9)
- Paper (2015): [https://proceedings.mlr.press/v70/foerster17b/foerster17b.pdf](https://proceedings.mlr.press/v70/foerster17b/foerster17b.pdf)

The idea is simple. Just run independent RL agents in some environments. 
The main problem: nonstationarity of the world that is dependent on actions of other agents. Surprisingly, in some cases IQL works great (e.g. ping-pong env). 
In [IQL (2015)](https://arxiv.org/abs/1511.08779) paper the authors show how by playing with the reward definitions the behaviour of agents change from cooperative to competitive.


### VDN (2017)

- Paper: [https://arxiv.org/pdf/1706.05296](https://arxiv.org/pdf/1706.05296)

Implicitly, the value decomposition network aims to learn an optimal linear value decomposition from the team reward signal, 
by back-propagating the total Q gradient through deep neural networks representing the individual component value functions.

This additive value decomposition is speciﬁcally motivated by avoiding the spurious reward signals that emerge in purely independent learners.
The implicit value function learned by each agent depends only on local observations, and so is more easily learned.

<img src="pics/vdn_q_function.png" width="700">

Basic architecture:

<img src="pics/vdn_basic_arc.png" width="700">


### QMix (2018)

- Paper: [https://www.jmlr.org/papers/volume21/20-081/20-081.pdf](https://www.jmlr.org/papers/volume21/20-081/20-081.pdf)

QMIX is a value-based method that can train decentralised policies in a centralised end-to-end fashion. 
QMIX employs a network that estimates joint action-values as a complex non-linear combination of per-agent values that condition only on local observations. 
THe authors structurally enforce that the joint-action value is monotonic in the per-agent values, which allows tractable maximisation of the joint action-value in off-policy learning, and guarantees consistency between the centralised and decentralised policies.

Basically, QMix is like VDN but is not constrained to linear dependencies between agents' Q values.
QMix also makes use of external state of the environment.

The graphical representation of QMix NNs:

<img src="pics/qmix.png" width="700">


### COMA (2018)

- Paper: [https://ojs.aaai.org/index.php/AAAI/article/view/11794](https://ojs.aaai.org/index.php/AAAI/article/view/11794)

Counterfactual Multi-Agent (COMA) is an actor-critic algorithm with a centralised critic.
Three main ideas underly COMA:
1) centralisation of the critic, 
2) use of a counterfactual baseline, and 
3) use of a critic representation that allows efﬁcient evaluation of the baseline.

COMA also uses external state of the environment in the learning stage.

The architecture:

<img src="pics/coma.png" width="700">

### QTRAN (2019)

- Paper: [https://proceedings.mlr.press/v97/son19a/son19a.pdf](https://proceedings.mlr.press/v97/son19a/son19a.pdf)
- Code: [https://github.com/himelbrand/marl-qtran](https://github.com/himelbrand/marl-qtran)

Also value-based algorithm like VDN and QMix.
VDN uses sum for the values, QMix uses monotonicity assumption.
VDN and QMIX address only a fraction of factorizable MARL tasks due to their structural constraint in factorization such as additivity and monotonicity. 
QTRAN is a new factorization method for MARL, which is free from such structural constraints and takes on a new approach to transforming the original joint action-value function into an easily factorizable one, with the same optimal actions. 
QTRAN guarantees more general factorization than VDN or QMIX, thus covering a much wider class of MARL tasks than does previous methods.

One of the main big ideas here in QTRAN is that rather than directly factorizing Q function (VDN did it by sum and QMix did it by non-linear NN), the authors consider an alternative joint action-value
function that is factorized by additive decomposition, but the Q function (and V function) itself is learned separately from the agents.

- very complex
- have good theoretical properties

A toy example where QTRAN is better than VDN and QMix:

<img src="pics/qtran_toy_example.png" width="400">

The architecture of QTRAN is as follows: 

<img src="pics/qtran.png" width="700">

Pseudocode: 

<img src="pics/qtran_pseudocode.png" width="700">


### DGN (2020)

- Paper: [https://arxiv.org/pdf/1810.09202](https://arxiv.org/pdf/1810.09202)
- Code: [https://github.com/PKU-RL/DGN](https://github.com/PKU-RL/DGN)

The main idea is to use the underlying connection graph of agents.
No additional information about the state is not provided.

DGN consists of three types of modules: (1) observation encoder, (2) convolutional layer and (3) Q network, as illustrated in Figure 1 below. 
The local observation $o^h_i$ is encoded into a feeture vector $h^{'t}_i$ by MLP for low-dimensional input or CNN for visual input. 
The convolutional layer integrates the feature vectors in the local region (including node $i$ and its neighbors $B_i$) and generates the
latent feature vector $h^{'t}_i$. 
By stacking more convolutional layers, the receptive ﬁeld of an agent gradually grows, where more information is gathered, and thus the scope of cooperation can also increase.

In the paper the experiments are done with the `MAgent` environments. 

The overview of three modules of DGN:

<img src="pics/dgn.png" width="700">

Computational layer (relation kernel):

<img src="pics/dgn2.png" width="700">

Temporal relation regularization:

<img src="pics/dgn3.png" width="700">

### IPPO (2022)

- Paper: [https://arxiv.org/pdf/2011.09533](https://arxiv.org/pdf/2011.09533)
- Code:

Each agent `a` learns a local observation based critic $V_φ(z^a_t )$ parameterised by $φ$ using Generalized Advantage Estimation (GAE).
The network parameters $φ, θ$ are shared across critics, and actors, respectively.
The authors also add an entropy regularization term to the ﬁnal policy loss.
For each agent, the overall learning loss becomes:

<img src="pics/ippo_1.png" width="700">

The architecture is three `conv1d` layers and two MLP layers.

The algorithms outperforms QMIX and others on the SMAC env.
It seems that IPPO's (approximate) surrogate objective might mitigate certain forms of environment non-stationarity that other independent learning algorithms are prone to, e.g., by suppressing updates catastrophic to performance.

In general, the idea is that IL is somehow surprisingly can be ok. The authors are unsure exactly why is that a case, probably because of the magic of the PPO's surrogate objective.

In the paper, the authors used a very complicated language.


### ROMA (2020)

- Paper: [ROMA: Multi-Agent Reinforcement Learning with Emergent Roles](https://arxiv.org/pdf/2003.08039)
- Code: ~

The paper introduces a concept of _role_ into MARL. 
The idea is that this concept will allow to agents to specialize in their specific tasks and hence improve performance. The trick is that many agents have similar roles, so they can enjoy from mutual learning. But how to distinguish between different roles, so that non-related agent will not disturb others?
The authors define three properties that are important for every role: (1) it needs to be dynamic to the changes of an env; (2) it needs to identiﬁable, i.e. to be temporary stable, so that the behaviour is consistent with time; (3) it needs to be specialized so that different robots can identify each other's roles and to be able to learn from agents with similar roles.
So the ROMA works as follows (plus-minus): it encodes the trajectory into role, composes the loss for _identiﬁable_ property. Then, it computes the loss for _specialized_ property. Then, it uses the QMIX loss for the $Q_{tot}$. After the reword the gradients propagate back.

The method achieved SOTA results on the SMAC benchmark. That is good. But the disadvantage is that the method is relatively complicated. There are 5 learnable different NNs, the loss function is complex, and there are many parameters. 
Maybe that is why ROMA is not used as a benchmark in the papers that came afterword.


<img src="pics/roma_1.png" width="700">

### MADDPG (2020)

- Paper: [Multi-Agent Actor-Critic for Mixed
Cooperative-Competitive Environments](https://proceedings.neurips.cc/paper/2017/file/68a9750337a418a86fe06c1991a1d64c-Paper.pdf)
- The env they used: [PettingZoo - MPE](https://pettingzoo.farama.org/environments/mpe/)
- Code: ~

In this paper, the authors adapt DDPG to the multi-agent case.
They describe the same problems of non-stationary and that it is tricky to use memory buffer in the multi-agent case.
Their trick is that there is a separate $Q_i$ function for every agent $i$. And each such function sees the actions, observations and, potentially, policies of other agents. They also describe how to approximate policies of other agents using only their observations.
Another trick for competitive environments is to preserve several policies for every agent and to switch between them randomly during both the training and the execution.
The schema of the algorithms is in the following pic:

<img src="pics/maddpg_1.png" width="400">

The pseudo-code of MADDPG is as follows:

<img src="pics/maddpg_2.png" width="700">

The paper presented no theoretical guarantees.
A really well-written paper, I should say.

### Qatten (2020)

- Paper: [Qatten: A General Framework for Cooperative Multiagent Reinforcement Learning](https://arxiv.org/pdf/2002.03939)
- Env: [SMAC](https://github.com/oxwhirl/smac/tree/master)
- Code: ~

This paper generalizes the VDN, QMIX, and QTRAN findings. They all use some combination of $Q_i$ values to compose the $Q_{tot}$ function.
The Qatten paper generalizes the idea in the following Theorem:

<img src="pics/Qatten_1.png" width="700">

Basically, it says that, given every state $s$, there are some linear combinations of $Q$ values that properly describe $Q_{tot}$.
Then they presented the approximation for this ideal$Q_{tot}$ in order to construct an algorithm: 

<img src="pics/Qatten_2.png" width="700">

An the overall algorithm schema is presented here:

<img src="pics/Qatten_3.png" width="700">

The achieved SOTA results at the time. They used multi-head attention as the ultimate approximator for the $\lambda$ values, referring to this paper: ["ARE TRANSFORMERS UNIVERSAL APPROXIMATORS OF SEQUENCE-TO-SEQUENCE FUNCTIONS?" (ICLR, 2020)](https://arxiv.org/pdf/1912.10077).
The details of the formulas are hard to track, unfortunately.


### QPlex (2021)

- Paper: [QPLEX: DUPLEX DUELING MULTI-AGENT Q-LEARNING](https://arxiv.org/pdf/2008.01062)
- Env: [SMAC](https://github.com/oxwhirl/smac/tree/master)
- Code: ~

This paper defines advantage-based IGM (_individual global max_) principle and bridges it with the original IGM principle. This principle was used by previous algorithms such as VDN, QMIX, and QTRAN. There, they said that the combination of individual $Q$ values compose the total $Q$ value. So, in this paper, the authors declare that it is more beneficial to look at the advantage values $A$ instead of $Q$ values. The inspiration is taken from the "Dueling DQN" paper (explained earlier).
Not only that, the paper also took inspiration from "Qatten" paper to combine the $A$ values in a cleaver manner.
The overall flow of the algorithm is here: 

<img src="pics/qplex_1.png" width="700">

The bad news, once again as in "Qatten" paper, the approach is complex and it is hard to fully understand the theoretical strength of the paper.

They took the smart combination of $Q$ values from "Qatten" paper, attention mechanism, duelling trick from the "Dueling DQN" paper and smashed it all together on top of the MARL problem and it worked.

### MAPPO (2022)

- Paper: [The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games](https://proceedings.neurips.cc/paper_files/paper/2022/file/9c1535a02f0ce079433344e14d910597-Paper-Datasets_and_Benchmarks.pdf)
- Env: [SMAC](https://github.com/oxwhirl/smac/tree/master), [Google Research Football](https://github.com/google-research/football), [Hanabi](https://pettingzoo.farama.org/environments/classic/hanabi/), [MPE](https://pettingzoo.farama.org/environments/mpe/)
- Code: [https://github.com/marlbenchmark/on-policy](https://github.com/marlbenchmark/on-policy)

Basicalle, MAPPO works exactly like PPO.
There is a common value function. The policy function may be common, if agents are similar, or different for every agent. The value function may use some global state info.
When the authors implemented MAPPO, they wanted to point out the following 5 main advises for the implementation:

1. **value normalisation**: Utilize value normalization to stabilize value learning. Keep track of averages and variances.
2. **input representation to $V$ function**: use global data if applicable. Do not duplicate data.
3. **training data usage**: do not overtrain on the batch.
4. **PPO clipping**: very important - do the clipping. 0.2 worked well for them.
5. **batch size**: use big batches for better performance.


The pseudo-code from the paper:

<img src="pics/mappo_1.png" width="700">



### Belief-PPO (2023)

- Paper: [Dynamic Belief for Decentralized Multi-Agent Cooperative Learning](https://www.ijcai.org/proceedings/2023/0039.pdf)
- Env: [SMAC](https://github.com/oxwhirl/smac/tree/master)
- Code: ~

Ok, here the setting is full decentralised. As far as I understand, they use beliefs (embedded policy approximations) of other agents to cover for the non-stationary problem. The authors use history of observations of other agents to infer their behaviour.
Once again, the problem, imo, is that the approach, though achieves SOTA results, is very complex for simple peoples like me to quickly implement. Not trivial how to adjust all those NNs and attention mechanism. No code from authors neither. The paper uses a bit high language which complicated understanding. The formulas lack intuitive explanations. But the idea is nice, that is why the paper here.

Here is the general structure of Belief-PPO:

<img src="pics/belief_ppo_1.png" width="700">


### IDDPG

- Paper: 
- Code:

### SHAQ (2023)

- Paper: [https://proceedings.neurips.cc/paper_files/paper/2022/file/27985d21f0b751b933d675930aa25022-Paper-Conference.pdf](https://proceedings.neurips.cc/paper_files/paper/2022/file/27985d21f0b751b933d675930aa25022-Paper-Conference.pdf)
- Code: [https://github.com/hsvgbkhgbv/shapley-q-learning](https://github.com/hsvgbkhgbv/shapley-q-learning)


A simple example that describes Shapley Value principle from ChatGPT (hope it is correct):

```txt
Simple Example with Numbers
Let's take a small example with two players: Alice and Bob.

- Alice alone can make $100.
- Bob alone can make $50.
- Together, they make $200.

The question is: how much did each contribute to the extra value created when they work together?

- If Alice joins first, she contributes $100, then Bob joins and increases the total from $100 to $200, so his contribution is $100.
- If Bob joins first, he contributes $50, then Alice joins and increases the total from $50 to $200, so her contribution is $150.

Averaging these two cases:

- Alice's Shapley value = (100+150)/2=125.
- Bob's Shapley value = (50+100)/2=75.

So, a fair split of the $200 profit is Alice = $125, Bob = $75.
```

The SHAQ paper the Shapley value principle to the fair division of reward in MARL case and achieves SOTA results, of course.
The additional nice advantage is that with SHAQ we have some explainability of the reward distribution between agents.

Pseudo-code: 

<img src="pics/shaq_1.png" width="700">

<img src="pics/shaq_2.png" width="700">

As you can see the implementation is also not so simple. At list, they have code to play with in GitHub.

---

[//]: # (##########################################################)
[//]: # (##########################################################)
[//]: # (##########################################################)
## MARL for MAPF

### PRIMAL

[//]: # (##########################################################)
[//]: # (##########################################################)
[//]: # (##########################################################)
## Environments

### MAgent2

- Env: [https://magent2.farama.org/](https://magent2.farama.org/)
- Paper: [https://arxiv.org/pdf/1712.00600](https://arxiv.org/pdf/1712.00600)
- Envs list: [https://github.com/Farama-Foundation/MAgent2/tree/main/magent2/environments](https://github.com/Farama-Foundation/MAgent2/tree/main/magent2/environments)

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

### VMAS (now is a part of BenchMARL)

- Env: [https://github.com/proroklab/VectorizedMultiAgentSimulator](https://github.com/proroklab/VectorizedMultiAgentSimulator)
- Paper: [https://arxiv.org/pdf/2207.03530](https://arxiv.org/pdf/2207.03530)

Parallel execution:

```python
import vmas

# Create the environment
env = vmas.make_env(
    # scenario="waterfall", # can be scenario name or BaseScenario class
    # scenario="dropout",
    # scenario="transport",
    # scenario="wheel",
    # scenario="drone",
    # scenario="kinematic_bicycle",
    # scenario="road_traffic",
    # scenario="multi_give_way",
    # scenario="football",
    # scenario="give_way",
    # scenario="simple",
    scenario="simple_adversary",
    num_envs=32,
    device="cpu", # Or "cuda" for GPU
    continuous_actions=True,
    max_steps=None, # Defines the horizon. None is infinite horizon.
    seed=None, # Seed of the environment
    n_agents=3  # Additional arguments you want to pass to the scenario
)
# Reset itr
obs = env.reset()

# Step it with deterministic actions (all agents take their maximum range action)
for i in range(1000):
    obs, rews, dones, info = env.step(env.get_random_actions())
    print(i)
    env.render(
        # mode="rgb_array",  # "rgb_array" returns image, "human" renders in display
        mode="human",  # "rgb_array" returns image, "human" renders in display
        # agent_index_focus=4, # If None keep all agents in camera, else focus camera on specific agent
        # index=0, # Index of batched environment to render
        # visualize_when_rgb=True,  # Also run human visualization when mode=="rgb_array"
    )
```


### SMAC - StarCraft Multi-Agent Challenge

- Env: [https://github.com/oxwhirl/smac](https://github.com/oxwhirl/smac)
- Paper: [https://arxiv.org/pdf/1902.04043](https://arxiv.org/pdf/1902.04043)

Parallel execution:

```python
from smac.env import StarCraft2Env
import numpy as np


def main():
    # env = StarCraft2Env(map_name="8m")
    # env = StarCraft2Env(map_name="2s_vs_1sc", replay_dir='/Users/perchik/PycharmProjects/Learning_DRL/saved_replays')
    env = StarCraft2Env(map_name="2s_vs_1sc")
    env_info = env.get_env_info()

    n_actions = env_info["n_actions"]
    n_agents = env_info["n_agents"]

    n_episodes = 10

    for e in range(n_episodes):
        env.reset()
        terminated = False
        episode_reward = 0

        while not terminated:
            obs = env.get_obs()
            state = env.get_state()
            # env.render()  # Uncomment for rendering

            actions = []
            for agent_id in range(n_agents):
                avail_actions = env.get_avail_agent_actions(agent_id)
                avail_actions_ind = np.nonzero(avail_actions)[0]
                action = np.random.choice(avail_actions_ind)
                actions.append(action)

            reward, terminated, _ = env.step(actions)
            episode_reward += reward

        print(f"Total reward in episode {e} = {episode_reward}")

    # env.save_replay()
    env.close()

if __name__ == '__main__':
    main()
```


[//]: # (##########################################################)
[//]: # (##########################################################)
[//]: # (##########################################################)
## Credits

Stand on the shoulders of giants.

### General

- [OpenAI | Spinning Up in Deep RL](https://spinningup.openai.com/en/latest/index.html)
- [OpenAI | Key Papers in Deep RL](https://spinningup.openai.com/en/latest/spinningup/keypapers.html)
- [graphviz in python](https://graphviz.readthedocs.io/en/stable/index.html)

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