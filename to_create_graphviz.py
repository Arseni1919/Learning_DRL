import graphviz

g = graphviz.Digraph('G', filename='pics/rl_history', engine='sfdp', format='png')
g.attr(rankdir='TB', size='10,10')

g.edge('DQN', 'DoubleDQN')
g.edge('REINFORCE', 'ActorCritic')
g.edge('DPG', 'DDPG')
g.edge('DoubleDQN', 'TD3')
g.edge('DoubleDQN', 'SAC')
g.edge('ActorCritic', 'TRPO')
g.edge('TRPO', 'PPO')
g.edge('PPO', 'SAC')
g.edge('ActorCritic', 'DDPG')
g.edge('DDPG', 'TD3')
g.edge('TD3', 'SAC')

u = g.unflatten(stagger=2)
u.view()

# <img src='https://g.gravizo.com/svg?
#   digraph G {
#     size ="4,4";
#     REINFORCE -> ActorCritic;
#     ActorCritic -> TRPO;
#     TRPO -> PPO;
#     PPO -> SAC;
#     ActorCritic -> DDPG;
#     DDPG -> TD3;
#     TD3 -> SAC;
#     DQN -> ActorCritic;
#     DQN -> DoubleDQN;
#     DoubleDQN -> TD3;
#     DoubleDQN -> SAC;
#   }
# '/>