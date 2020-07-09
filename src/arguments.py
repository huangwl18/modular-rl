import argparse

def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument("--seed", default=0, type=int,
        help="sets Gym, PyTorch and Numpy seeds")
    parser.add_argument("--morphologies", nargs="*", type=str, default=['walker'],
        help="which morphology env to run (walker, hopper, etc)")
    parser.add_argument("--custom_xml", type=str, default=None,
        help="path to MuJoCo xml files (can be either one file or a directory containing multiple files)")
    parser.add_argument("--start_timesteps", default=1e4, type=int,
        help="How many time steps purely random policy is run for?")
    parser.add_argument('--max_timesteps', type=int, default=20e6,
        help='number of timesteps to train')
    parser.add_argument("--expl_noise", default=0.126, type=float,
        help="std of Gaussian exploration noise")
    parser.add_argument("--batch_size", default=100, type=int,
        help="batch size for both actor and critic")
    parser.add_argument("--discount", default=0.99, type=float,
        help="discount factor")
    parser.add_argument("--tau", default=0.046, type=float,
        help="target network update rate")
    parser.add_argument("--policy_noise", default=0.2, type=float,
        help="noise added to target policy during critic update")
    parser.add_argument("--noise_clip", default=0.5, type=float,
        help="range to clip target policy noise")
    parser.add_argument("--policy_freq", default=2, type=int,
        help="frequency of delayed policy updates")
    parser.add_argument("--expID", default=0, type=int)
    parser.add_argument('--video_length', default=10, type=int,
        help='length of video to generate (in seconds)')
    parser.add_argument('--msg_dim', default=32,
        help='message dimension when trained modularly with message passing')
    parser.add_argument('--disable_fold', action="store_true",
        help='disable the use of pytorch fold (used for accelerating training)')
    parser.add_argument('--lr', default=0.0005, type=float,
        help='learning rate for Adam')
    parser.add_argument("--max_episode_steps", type=int, default=1000,
        help="maximum number of timesteps allowed in one episode")
    parser.add_argument("--save_freq", default=5e5, type=int,
        help="How often (time steps) we save the model and the replay buffer?")
    parser.add_argument("--td", action="store_true",
        help="enable top down message passing")
    parser.add_argument("--bu", action="store_true",
        help="enable bottom up message passing")
    parser.add_argument("--rb_max", type=int, default=10e6,
        help="maximum replay buffer size across all morphologies")
    parser.add_argument("--max_children", type=int, default=None,
        help="maximum number of children allowed at each node (optional; facilitate model loading if max_children is different at training time)")
    args = parser.parse_args()
    return args
