from algs_RL.alg_PPO_TorchRL_sup import *


def main():
    # parameters
    is_fork = multiprocessing.get_start_method() == 'fork'
    num_cells = 256 # num of cells in each layer
    lr = 3e-4
    max_grad_norm = 1.0
    frames_per_batch = 1000
    total_frames = 50_000
    sub_batch_size = 64
    num_epochs = 10
    clip_epsilon = (
        0.2
    )
    gamma = 0.99
    lmbda = 0.95
    entropy_eps = 1e-4


    # render_mode="rgb_array"
    render_mode = "human"
    # render_mode = None
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )

    # create env
    env = gym.make('InvertedPendulum-v5', reset_noise_scale=0.1, render_mode="human")

    


if __name__ == '__main__':
    main()