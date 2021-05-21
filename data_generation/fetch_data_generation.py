import gym
import time
import random
import numpy as np

import os
import argparse
import torch

# import rospy
# import roslaunch

# from random import randint
# from std_srvs.srv import Empty
# from sensor_msgs.msg import JointState
# from geometry_msgs.msg import PoseStamped
# from geometry_msgs.msg import Pose
# from std_msgs.msg import Float64
# from controller_manager_msgs.srv import SwitchController
# from gym.utils import seeding


"""Data generation for the case of a single block with Fetch Arm pick and place"""

ep_returns = []
observations = []
actions = []
rewards = []
lens = []
infos = []

def main():
    parser = argparse.ArgumentParser(
        'Get expert trajectories on FetchPickAndPlace-v1 with pt format.')
    parser.add_argument(
        '--pt-file',
        default='trajs_fetchpickandplace_heuristics.pt',
        help='output pt file to save demonstrations',
        type=str)
    parser.add_argument(
        '--render',
        action='store_true',
        default=False,
        help='render simulator')
    parser.add_argument(
        '--save-episode',
        type=int,
        default=3000,
        help='the number of episodes to save demonstrations (default: 100)')

    args = parser.parse_args()

    # if args.pt_file is None:
    #     args.pt_file = os.path.splitext(args.h5_file)[0] + '.pt'

    env = gym.make('FetchPickAndPlace-v1')
    numItr = 100
    initStateSpace = "random"

    env.reset()
    print("Reset!")
    time.sleep(1)

    while len(actions) < numItr:
        obs = env.reset()

        if args.render:
            env.render()

        print("Reset!")
        print("ITERATION NUMBER ", len(actions))
        goToGoal(env, obs)
        

    # fileName = "data_fetch"
    # fileName += "_" + initStateSpace
    # fileName += "_" + str(numItr)
    # fileName += ".npz"
    # np.savez_compressed(fileName, acs=actions, obs=observations, info=infos)

    data = {
        'states': observations,
        'actions': actions,
        'rewards': rewards,
        'lengths': lens
    }

    torch.save(data, args.pt_file)


def goToGoal(env, lastObs):

    #goal = self.sampleGoal()
    goal = lastObs['desired_goal']

    #objectPosition
    objectPos = lastObs['observation'][3:6]
    gripperPos = lastObs['observation'][:3]
    gripperState = lastObs['observation'][9:11]
    object_rel_pos = lastObs['observation'][6:9]
    goal_rel_pose = lastObs['desired_goal'] - lastObs['achieved_goal']

    #print("relative position ", object_rel_pos)
    #print("Goal position ", goal)
    #print("gripper Position ", gripperPos)
    #print("Object Position ", objectPos)
    #print("Gripper state  ", gripperState)

    episodeAcs = []
    episodeObs = []
    episodeRes = []
    episodeInfo = []

    object_oriented_goal = object_rel_pos.copy()
    object_oriented_goal[2] += 0.03
    
    print("Max episode steps ", env._max_episode_steps)

    timeStep = 0

    episodeObs.append(np.concatenate((lastObs, goal_rel_pose), axis=-1))

    

    while np.linalg.norm(object_oriented_goal) >= 0.005 and timeStep <= env._max_episode_steps:
        env.render()
        action = [0, 0, 0, 0]

        object_oriented_goal = object_rel_pos.copy()
        object_oriented_goal[2] += 0.03

        for i in range(len(object_oriented_goal)):
            action[i] = object_oriented_goal[i]*6

        action[len(action)-1] = 0.05

        obsDataNew, reward, done, info = env.step(action)
        goal_rel_pose = obsDataNew['desired_goal'] - obsDataNew['achieved_goal']
        timeStep += 1

        episodeAcs.append(action)
        episodeInfo.append(info)
        episodeRes.append(reward)
        # episodeObs.append(obsDataNew)
        episodeObs.append(np.concatenate((obsDataNew, goal_rel_pose), axis=-1))

        objectPos = obsDataNew['observation'][3:6]
        gripperPos = obsDataNew['observation'][:3]
        gripperState = obsDataNew['observation'][9:11]
        object_rel_pos = obsDataNew['observation'][6:9]

    while np.linalg.norm(object_rel_pos) >= 0.005 and timeStep <= env._max_episode_steps :
        env.render()
        action = [0, 0, 0, 0]

        

        for i in range(len(object_rel_pos)):
            action[i] = object_rel_pos[i]*6

        action[len(action)-1] = -0.005

        obsDataNew, reward, done, info = env.step(action)
        goal_rel_pose = obsDataNew['desired_goal'] - obsDataNew['achieved_goal']
        timeStep += 1

        episodeAcs.append(action)
        episodeInfo.append(info)
        episodeRes.append(reward)
        # episodeObs.append(obsDataNew)
        episodeObs.append(np.concatenate((obsDataNew, goal_rel_pose), axis=-1))

        objectPos = obsDataNew['observation'][3:6]
        gripperPos = obsDataNew['observation'][:3]
        gripperState = obsDataNew['observation'][9:11]
        object_rel_pos = obsDataNew['observation'][6:9]


    while np.linalg.norm(goal - objectPos) >= 0.01 and timeStep <= env._max_episode_steps :
        env.render()
        action = [0, 0, 0, 0]

        

        for i in range(len(goal - objectPos)):
            action[i] = (goal - objectPos)[i]*6

        action[len(action)-1] = -0.005

        obsDataNew, reward, done, info = env.step(action)
        goal_rel_pose = obsDataNew['desired_goal'] - obsDataNew['achieved_goal']
        timeStep += 1

        episodeAcs.append(action)
        episodeInfo.append(info)
        episodeRes.append(reward)
        # episodeObs.append(obsDataNew)
        episodeObs.append(np.concatenate((obsDataNew, goal_rel_pose), axis=-1))

        objectPos = obsDataNew['observation'][3:6]
        gripperPos = obsDataNew['observation'][:3]
        gripperState = obsDataNew['observation'][9:11]
        object_rel_pos = obsDataNew['observation'][6:9]


    while True:
        env.render()
        action = [0, 0, 0, 0]

        action[len(action)-1] = -0.005

        obsDataNew, reward, done, info = env.step(action)
        goal_rel_pose = obsDataNew['desired_goal'] - obsDataNew['achieved_goal']
        timeStep += 1

        episodeAcs.append(action)
        episodeInfo.append(info)
        episodeRes.append(reward)
        # episodeObs.append(obsDataNew)
        episodeObs.append(np.concatenate((obsDataNew, goal_rel_pose), axis=-1))

        objectPos = obsDataNew['observation'][3:6]
        gripperPos = obsDataNew['observation'][:3]
        gripperState = obsDataNew['observation'][9:11]
        object_rel_pos = obsDataNew['observation'][6:9]

        if timeStep >= env._max_episode_steps: break

    print("Toatal timesteps taken ", timeStep)
    print("len(episodeObs): ", len(episodeObs))
    print("len(episodeAcs): ", len(episodeAcs))
    print("len(episodeRes): ", len(episodeRes))

    actions.append(episodeAcs)
    observations.append(episodeObs)
    infos.append(episodeInfo)
    lens.append(env._max_episode_steps)
    print("Update Result: len(lens) is ", len(lens))


if __name__ == "__main__":
    main()
