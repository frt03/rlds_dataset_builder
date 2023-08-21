from typing import Iterator, Tuple, Any

import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import pickle


class XArmPickAndPlaceDataset(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for xarm_pick_and_place dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image': tfds.features.Image(
                            shape=(224, 224, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Main camera RGB observation.',
                        ),
                        'image2': tfds.features.Image(
                            shape=(224, 224, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Another camera RGB observation from different view point.',
                        ),
                        'hand_image': tfds.features.Image(
                            shape=(224, 224, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Hand camera RGB observation.',
                        ),
                        'joint_trajectory': tfds.features.Tensor(
                            shape=(21,),
                            dtype=np.float32,
                            doc='Robot joint trajectory, consists of [7x robot joint angles, '
                                '7x robot joint velocity, 7x robot joint acceralation].',
                        ),
                        'joint_state': tfds.features.Tensor(
                            shape=(14,),
                            dtype=np.float32,
                            doc='Robot joint state, consists of [7x robot joint angles, '
                                '7x robot joint velocity].',
                        ),
                        'end_effector_pose': tfds.features.Tensor(
                            shape=(6,),
                            dtype=np.float32,
                            doc='Robot end effector pose, consists of [3x EEF position, '
                                '3x EEF orientation yaw/pitch/roll].',
                        ),
                    }),
                    'action': tfds.features.Tensor(
                        shape=(7,),
                        dtype=np.float32,
                        doc='Robot action, consists of [3x EEF position, '
                            '3x EEF orientation yaw/pitch/roll, '
                            '1x gripper open/close position].',
                    ),
                    'discount': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Discount if provided, default to 1.'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Reward if provided, 1 on final step for demos.'
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode if it is a terminal step, True for demos.'
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Language Instruction.'
                    ),
                    'language_embedding': tfds.features.Tensor(
                        shape=(512,),
                        dtype=np.float32,
                        doc='Kona language embedding. '
                            'See https://tfhub.dev/google/universal-sentence-encoder-large/5'
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                }),
            }))

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        return {
            'train': self._generate_examples(path='/root/robot-qnap-1/pickle_dataset/xarm_pick_and_place/train/*.pkl'),
            'val': self._generate_examples(path='/root/robot-qnap-1/pickle_dataset/xarm_pick_and_place/val/*.pkl'),
        }

    def _generate_examples(self, path) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        def _parse_example(episode_path):
            # load raw data --> this should change for your dataset
            with open(episode_path, 'rb') as f:
                data = pickle.load(f)

            # assemble episode --> here we're assuming demos so we set reward to 1 at the end
            episode = []
            lang = data['language_instruction']
            for i, (action, jt, js, eep, image, image2, hand) in enumerate(zip(data['action'], data['joint_trajectory'], data['joint_state'], data['end_effector_pose'], data['image'], data['image2'], data['hand_image'])):
                # compute Kona language embedding
                language_embedding = self._embed([lang])[0].numpy()

                episode.append({
                    'observation': {
                        'image': image,
                        'image2': image2,
                        'hand_image': hand,
                        'joint_trajectory': jt,
                        'joint_state': js,
                        'end_effector_pose': eep,
                    },
                    'action': action,
                    'discount': 1.0,
                    'reward': float(i == (len(data) - 1)),
                    'is_first': i == 0,
                    'is_last': i == (len(data) - 1),
                    'is_terminal': i == (len(data) - 1),
                    'language_instruction': lang,
                    'language_embedding': language_embedding,
                })

            # create output data sample
            sample = {
                'steps': episode,
                'episode_metadata': {
                    'file_path': episode_path
                }
            }

            # if you want to skip an example for whatever reason, simply return None
            return episode_path, sample

        # create list of all examples
        episode_paths = glob.glob(path)

        # for smallish datasets, use single-thread parsing
        for sample in episode_paths:
            yield _parse_example(sample)

        # for large datasets use beam to parallelize data parsing (this will have initialization overhead)
        # beam = tfds.core.lazy_imports.apache_beam
        # return (
        #         beam.Create(episode_paths)
        #         | beam.Map(_parse_example)
        # )
