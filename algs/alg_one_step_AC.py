from algs.alg_one_step_AC_sup import *
from plot_functions import *


def main():
    # parameters
    GAMMA = 0.99
    LR = 1e-3
    BATCH_SIZE = 100
    EPS = np.finfo(np.float32).eps.item()
    running_reward = 10
    reward_threshold = 400
    steps = 0
    render_mode = "human"  # "human" / "rgb_array" / None
    # render_mode = None
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )
    # for plots
    fig, ax = plt.subplots(1, 2, figsize=(14, 7))
    plot_rate = 0.001
    episode_durations: List[int] = []
    running_rewards: List[float] = []

    # seeds
    torch.manual_seed(123)

    # create env
    env = gym.make("CartPole-v1", render_mode=render_mode)

    # create NNs
    ac_net: ActorCriticNet = ActorCriticNet(4, 2).to(device)
    optimizer = optim.AdamW(ac_net.parameters(), lr=LR)

    # main loop
    for i_episode in count():
        state, _ = env.reset()
        ep_reward = 0

        for t in count():
            action, log_prob, state_value = select_action(state, ac_net, device)
            next_state, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            done: bool = terminated or truncated

            # save to batch
            pass

            state = next_state
            steps += 1
            if done:
                break

        if steps > 0 and steps % BATCH_SIZE == 0:
            # TODO: transform to vectors
            # learning stage
            with torch.no_grad():
                t_next_state: Tensor = torch.from_numpy(next_state).float().unsqueeze(0).to(device=device)
                _, v_next_state = ac_net(t_next_state)
                target = reward + GAMMA * v_next_state.item() * (1 - done)
            target = torch.tensor(target, device=device, dtype=torch.float32)
            advantage = target - state_value.item()
            actor_loss = - log_prob * advantage
            critic_loss = F.smooth_l1_loss(state_value.squeeze(), target).unsqueeze(0)
            optimizer.zero_grad()
            loss = actor_loss + critic_loss
            loss.backward()
            optimizer.step()



        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward

        # print & plot
        episode_durations.append(ep_reward)
        running_rewards.append(running_reward)
        print(f'\rEpisode: {i_episode}, Last reward: {ep_reward}, Avr. reward: {running_reward}', end='')
        plot_episode_durations(ax[0], info={
            'episode_durations': episode_durations,
            'running_rewards': running_rewards,
        })
        plt.pause(plot_rate)

        # stop condition
        if running_reward > reward_threshold:
            print(f'Solved. Avr. reward is above {reward_threshold}.')
            break


if __name__ == '__main__':
    main()




# def finish_episode(ac_net: ActorCriticNet, optimizer: optim.Optimizer, device, GAMMA: float, EPS: float):
#     R = 0
#     actor_losses = []
#     critic_losses = []
#     returns = deque()
#     for r in ac_net.rewards[::-1]:
#         R = r + GAMMA * R
#         returns.appendleft(R)
#     returns = torch.tensor(returns, device=device, dtype=torch.float32)
#     returns = (returns - returns.mean()) / (returns.std() + EPS)
#     for (log_prob, value), R in zip(ac_net.saved_actions, returns):
#         advantage = R - value.item()
#         # Note that we use a negative because optimizers use gradient descent,
#         # whilst the REINFORCE rule assumes gradient ascent.
#         actor_losses.append(-log_prob * advantage)
#         critic_losses.append(F.smooth_l1_loss(value.squeeze(), torch.tensor(R).to(device=device).clone().detach()).unsqueeze(0))
#     optimizer.zero_grad()
#     loss = torch.cat(actor_losses).sum() + torch.cat(critic_losses).sum()
#     loss.backward()
#     optimizer.step()
#     ac_net.rewards = []
#     ac_net.saved_actions = []
