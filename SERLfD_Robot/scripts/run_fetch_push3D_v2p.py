# -*- coding: utf-8 -*-
"""Train or test baselines on LunarLanderContinuous-v2.

- Author: Curt Park
- Contact: curt.park@medipixel.io
"""

import argparse
import random
import importlib
import os
import gym
from gym import wrappers
import algorithms.common.helper_functions as common_utils
# from envs.fetch.pushing_env_acc_fetch import fetchGraspingProceduralEnv
from envs.fetch.pushing3D_env_acc_fetch_1 import fetchPush3DEnv as fetchGraspingProceduralEnv
# from envs.fetch.pushing_env_2_fetch import fetchGraspingProceduralEnv

# configurations
parser = argparse.ArgumentParser(description="Pytorch RL baselines")
parser.add_argument(
    "--seed", type=int, default=777, help="random seed for reproducibility"
)
parser.add_argument("--algo", type=str, default="sac", help="choose an algorithm")
parser.add_argument("--mode", type=str, default="GUI", help="choose a simulator mode")

parser.add_argument(
    "--load-from",
    type=str,
    default=None,
    help="load the saved model and optimizer at the beginning",
)
parser.add_argument(
    "--robot-env-config",
    type=str,
    default=os.getcwd() + "/../config/fetch_serl_push3d_env_acc_1.yaml",
    help="load the robot environment configurations",
)
parser.add_argument("--episode-num", type=int, default=1500, help="total episode num")
parser.add_argument(
    "--max-episode-steps", type=int, default=60, help="max episode step"
)
parser.add_argument(
    "--max-total-steps", type=int, default=1e6, help="max total steps"
)
parser.add_argument(
    "--avg-score-window", type=int, default=100, help="window for computing average episode scores"
)
parser.add_argument(
    "--off-render", dest="render", action="store_false", help="turn off rendering"
)
parser.add_argument(
    "--render-after",
    type=int,
    default=0,
    help="start rendering after the input number of episode",
)
# parser.add_argument(
#     "--use-se", dest="render", action="store_false", help="use self-explainer to help training"
# )

parser.add_argument("--save-period", type=int, default=100, help="save model period")
parser.add_argument("--log", action="store_true", help="turn on logging")
parser.add_argument("--test", action="store_true", help="test mode (no training)")
parser.add_argument("--replay", action="store_true", help="replay the demonstrations")
parser.add_argument("--record", action="store_true", help="record the demonstrations")

parser.add_argument(
    "--demo-path",
    type=str,
    default="data/lunarlander_continuous_demo.pkl",
    help="demonstration path",
)
parser.set_defaults(render=True)

args = parser.parse_args()


def main():
    """Main."""
    # ros env initialization
    # os.environ["ROS_MASTER_URI"] = "10.153.71.29:11311"
    # os.environ["ROS_IP"] = "yzha3@10.218.108.88"

    # openai_ros env initialization
    env = fetchGraspingProceduralEnv(render_mode=args.mode, render_width=640, render_height=480, downsample_width=120, downsample_height=90, camera_random=1.0)
    env._max_episode_steps = args.max_episode_steps

    # env = gym.wrappers.Monitor(
    #     env, '../data/fetch-1', force=True)    # set a random seed
    new_seed = random.randint(0, 2**32 - 1)
    #args.seed = new_seed
    common_utils.set_random_seed(args.seed, env)

    # run
    module_path = "config.agent.fetch_serl_push_v0." + args.algo
    agent = importlib.import_module(module_path)
    agent = agent.get(env, args)

    if args.record:
        agent.record_mouse()
    elif args.replay:
        agent.replay()
    else:
        # run
        if args.test:
            agent.test()
        else:
            agent.train()


if __name__ == "__main__":
    main()
