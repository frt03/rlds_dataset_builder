# Weblab xArm Dataset (xarm_pick_and_place)

We here provide the dataset where a single xArm manipulates two plates.
We converted the ROS topic in rosbag files into Python format.
Because the interaction between publisher and subscriber is asynchronous for each ROS topic, similar sensory values in the same timestamp may differ (e.g. `joint_trajectory` and `joint_state`).
