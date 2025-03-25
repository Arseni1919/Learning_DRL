import graphviz

g = graphviz.Digraph('G', filename='pics/rl_history', engine='sfdp', format='png')
g.attr(rankdir='BT', size='10,10')

g.edge('TD-Learning', 'DQN')
g.edge('TD-Learning', 'REINFORCE')
g.edge('DQN', 'DoubleDQN')
g.edge('DQN', 'DuelingDQN')
g.edge('REINFORCE', 'ActorCritic')
g.edge('ActorCritic', 'TRPO', label='on-policy')
g.edge('TRPO', 'PPO')
g.edge('ActorCritic', 'DPG', label='off-policy')
g.edge('DPG', 'DDPG')
g.edge('DDPG', 'TD3')
g.edge('TD3', 'SAC')

u = g.unflatten(stagger=7)
u.view()

# <img src='https://g.gravizo.com/svg?
#   digraph G {
#     size ="4,4";
#     REINFORCE -> ActorCritic;
#     ActorCritic -> TRPO;
#     TRPO -> SAC;
#     SAC -> SAC;
#     ActorCritic -> DDPG;
#     DDPG -> TD3;
#     TD3 -> SAC;
#     DQN -> ActorCritic;
#     DQN -> DoubleDQN;
#     DoubleDQN -> TD3;
#     DoubleDQN -> SAC;
#   }
# '/>