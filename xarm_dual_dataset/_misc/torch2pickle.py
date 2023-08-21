import torch
import numpy as np
import pickle
import glob
import os
from tqdm import tqdm


if __name__ == '__main__':
    # unfolding a towel: 32 trajectory --> {'train': 29, 'val': 3}
    # reaching a towel: 38 trajectory --> {'train': 35, 'val': 3}
    dataset_split = {
        'unfold': {
            'train': {
                'length': 29,
                'dir': '/root/robot-qnap-1/pickle_dataset/xarm_dual/train',
            },
            'val': {
                'length': 3,
                'dir': '/root/robot-qnap-1/pickle_dataset/xarm_dual/val',
            }
        },
        'reach': {
            'train': {
                'length': 35,
                'dir': '/root/robot-qnap-1/pickle_dataset/xarm_dual/train',
            },
            'val': {
                'length': 3,
                'dir': '/root/robot-qnap-1/pickle_dataset/xarm_dual/val',
            }
        }
    }
    os.makedirs(dataset_split['unfold']['train']['dir'], exist_ok=True)
    os.makedirs(dataset_split['reach']['val']['dir'], exist_ok=True)

    episode_counter = 0

    # unfolding a towel
    # dataset_path = '/root/robot-qnap-1/xarm_dual_torch/dual_reach.pkl'
    dataset_path = '/root/robot-qnap-1/xarm_dual_torch/dual_spread.pkl'

    with open(dataset_path, 'rb') as f:
        data = pickle.load(f)
    assert len(data) == 32

    for episode in tqdm(data):
        if episode_counter < dataset_split['unfold']['train']['length']:
            filename = os.path.join(dataset_split['unfold']['train']['dir'], f'epi_{episode_counter}.pkl')
        else:
            filename = os.path.join(dataset_split['unfold']['val']['dir'], f'epi_{episode_counter}.pkl')
        episode_dict = {}

        episode_dict['language_instruction'] = 'Unfold a wrinkled towel.'

        episode_dict['image'] = episode['images'].numpy().astype(np.uint8)
        episode_dict['pose_l'] = episode['pose_l'].numpy()
        episode_dict['pose_r'] = episode['pose_r'].numpy()
        episode_dict['action_l'] = episode['actions_l'].numpy()
        episode_dict['action_r'] = episode['actions_r'].numpy()
        episode_dict['action'] = np.concatenate(
            [episode_dict['action_l'], episode_dict['action_r']], axis=1
        )
        with open(filename, 'wb') as f:
            pickle.dump(episode_dict, f)
        episode_counter += 1
    assert episode_counter == 32

    # reaching a towel
    dataset_path = '/root/robot-qnap-1/xarm_dual_torch/dual_reach.pkl'
    # dataset_path = '/root/robot-qnap-1/xarm_dual_torch/dual_spread.pkl'

    with open(dataset_path, 'rb') as f:
        data = pickle.load(f)
    assert len(data) == 38

    for episode in tqdm(data):
        if episode_counter - 32 < dataset_split['reach']['train']['length']:
            filename = os.path.join(dataset_split['reach']['train']['dir'], f'epi_{episode_counter}.pkl')
        else:
            filename = os.path.join(dataset_split['reach']['val']['dir'], f'epi_{episode_counter}.pkl')
        episode_dict = {}

        episode_dict['language_instruction'] = 'Reach a towel.'

        episode_dict['image'] = episode['images'].numpy().astype(np.uint8)
        episode_dict['pose_l'] = episode['pose_l'].numpy()
        episode_dict['pose_r'] = episode['pose_r'].numpy()
        episode_dict['action_l'] = episode['actions_l'].numpy()
        episode_dict['action_r'] = episode['actions_r'].numpy()
        episode_dict['action'] = np.concatenate(
            [episode_dict['action_l'], episode_dict['action_r']], axis=1
        )
        with open(filename, 'wb') as f:
            pickle.dump(episode_dict, f)
        episode_counter += 1
    assert episode_counter == (32 + 38)
