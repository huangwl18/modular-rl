import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os
import argparse


def extract(args):
    os.makedirs(args.output_dir, exist_ok=True)
    for exp in args.expID:
        # identify the exp folder in the data dir
        exp_folder = [folder for folder in os.listdir(args.data_dir) if exp in folder]
        assert len(exp_folder) == 1, 'there must exist only one folder containing the experiment ID {}, but found the following: {}'.format(exp, exp_folder)
        # extract data from event files
        full_exp_path = os.path.join(args.data_dir, exp_folder[0])
        print('=' * 30, '\nstart extracting experiment {} from {}'.format(exp, full_exp_path))
        event_acc = EventAccumulator(full_exp_path)
        event_acc.Reload()
        # Show all tags in the log file
        tags = event_acc.Tags()['scalars']
        data = []
        for t in tags:
            if args.tag in t:
                w_times, steps, vals = zip(*event_acc.Scalars(t))
                data.append(np.vstack([steps, vals]))
        # data shape: [agents #, (steps, vals), steps #]
        # take average reward across all training agents
        data = np.mean(np.array(data), axis=0)
        # save extracted data
        if args.tag == 'episode_reward':
            output_path = os.path.join(args.output_dir, '{}.npy'.format(exp))
        else:
            output_path = os.path.join(args.output_dir, '{}_{}.npy'.format(exp, args.tag))
        np.save(output_path, np.array(data))
        print('experiment {} extraction saved to {} \n'.format(exp, output_path), '=' * 30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--expID', nargs="*", type=str, help="experiments to extract data from", required=True)
    parser.add_argument('--data_dir', type=str, help='data directory that contains all the experiment folders', required=True)
    parser.add_argument('--output_dir', type=str, help='output directory', required=True)
    parser.add_argument('--tag', type=str, default='episode_reward', help='tag to look for in event files (e.g. episode_reward)')
    args = parser.parse_args()
    extract(args)
