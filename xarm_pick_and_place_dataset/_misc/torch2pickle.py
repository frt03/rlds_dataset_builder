import torch
import numpy as np
import pickle
import glob
import os
from tqdm import tqdm


if __name__ == '__main__':
    # 102 trajectory --> {'train': 92, 'val': 10}
    dataset_split = {
        'train': {
            'length': 92,
            'dir': '/root/robot-qnap-1/pickle_dataset/xarm_pick_and_place/train',
        },
        'val': {
            'length': 10,
            'dir': '/root/robot-qnap-1/pickle_dataset/xarm_pick_and_place/val',
        }
    }
    os.makedirs(dataset_split['train']['dir'], exist_ok=True)
    os.makedirs(dataset_split['val']['dir'], exist_ok=True)

    episode_list = glob.glob('/root/robot-qnap-1/xarm_1201_plate_torch/*_10hz')

    assert len(episode_list) == 102

    for i, episode_file in tqdm(enumerate(episode_list)):
        if i < dataset_split['train']['length']:
            filename = os.path.join(dataset_split['train']['dir'], f'epi_{i}.pkl')
        else:
            filename = os.path.join(dataset_split['val']['dir'], f'epi_{i}.pkl')
        episode_dict = {}

        episode_dict['language_instruction'] = 'Pick up a white plate, and then place it on the red plate.'

        # controller_pose.pt
        controller_pose = torch.load(
            os.path.join(episode_file, 'controller_pose.pt'),
            map_location=torch.device('cpu')
        ).numpy()
        # controller_trigger.pt
        controller_trigger = torch.load(
            os.path.join(episode_file, 'controller_trigger.pt'),
            map_location=torch.device('cpu')
        ).numpy()
        episode_dict['action'] = np.concatenate(
            [controller_pose, controller_trigger.reshape([-1, 1])], axis=1)
        # joint_trajectory.pt
        joint_trajectory = torch.load(
            os.path.join(episode_file, 'joint_trajectory.pt'),
            map_location=torch.device('cpu')
        ).numpy()
        episode_dict['joint_trajectory'] = joint_trajectory
        # joint_state.pt
        joint_state = torch.load(
            os.path.join(episode_file, 'joint_state.pt'),
            map_location=torch.device('cpu')
        ).numpy()
        episode_dict['joint_state'] = joint_state
        # end_effector_pose.pt
        end_effector_pose = torch.load(
            os.path.join(episode_file, 'end_effector_pose.pt'),
            map_location=torch.device('cpu')
        ).numpy()
        episode_dict['end_effector_pose'] = end_effector_pose
        # camera_color.pt
        camera_color = torch.load(
            os.path.join(episode_file, 'camera_color.pt'),
            map_location=torch.device('cpu')
        ).numpy().astype(np.uint8)
        episode_dict['image'] = camera_color
        # camera2_color.pt
        camera2_color = torch.load(
            os.path.join(episode_file, 'camera2_color.pt'),
            map_location=torch.device('cpu')
        ).numpy().astype(np.uint8)
        episode_dict['image2'] = camera2_color
        # handcamera_color.pt
        handcamera_color = torch.load(
            os.path.join(episode_file, 'handcamera_color.pt'),
            map_location=torch.device('cpu')
        ).numpy().astype(np.uint8)
        episode_dict['hand_image'] = handcamera_color

        with open(filename, 'wb') as f:
            pickle.dump(episode_dict, f)
