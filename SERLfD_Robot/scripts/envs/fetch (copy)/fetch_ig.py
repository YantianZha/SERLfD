from gibson2.envs.igibson_env import iGibsonEnv
from gibson2.utils.motion_planning_wrapper import MotionPlanningWrapper
import gibson2
import argparse
import numpy as np
import os

def test_base_planning():
    print("Test env")
    # download_assets()
    # download_demo_data()
    config_filename = os.path.join(gibson2.root_path, 'test', 'test_house_occupancy_grid.yaml')

    nav_env = iGibsonEnv(config_file=config_filename,
                                  mode=args.mode,
                                  action_timestep=1.0 / 120.0,
                                  physics_timestep=1.0 / 120.0)
    motion_planner = MotionPlanningWrapper(nav_env)
    state = nav_env.reset()
    nav_env.robots[0].set_position_orientation([0,0,0],[0,0,0,1])
    nav_env.simulator.step()
    plan = None
    itr = 0
    while plan is None and itr < 10:
        plan = motion_planner.plan_base_motion([0.5,0.9,0])
        print(plan)
        itr += 1
    motion_planner.dry_run_base_plan(plan)

    assert len(plan) > 0
    nav_env.clean()

def run_base_planning_once(args):

    nav_env = iGibsonEnv(config_file=args.config,
                                  mode=args.mode,
                                  action_timestep=1.0 / 120.0,
                                  physics_timestep=1.0 / 120.0)
    motion_planner = MotionPlanningWrapper(nav_env)
    state = nav_env.reset()
    nav_env.robots[0].set_position_orientation([0,0,0],[0,0,0,1])
    nav_env.simulator.step()
    plan = None
    itr = 0
    while plan is None and itr < 10:
        joint_goal = motion_planner.get_arm_joint_positions(arm_ik_goal=)
        plan = motion_planner.plan_arm_motion(arm_joint_positions=joint_goal)
        print(plan)
        itr += 1
    motion_planner.dry_run_arm_plan(plan)

    assert len(plan) > 0
    # nav_env.clean()

def run_example(args):
    nav_env = iGibsonEnv(config_file=args.config,
                                  mode=args.mode,
                                  action_timestep=1.0 / 120.0,
                                  physics_timestep=1.0 / 120.0)

    motion_planner = MotionPlanningWrapper(nav_env)
    state = nav_env.reset()

    while True:
        action = np.zeros(nav_env.action_space.shape)
        state, reward, done, _ = nav_env.step(action)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        '-c',
        default=os.path.join(gibson2.example_config_path, 'fetch_motion_planning.yaml'),
        help='which config file to use [default: use yaml files in examples/configs]')
    parser.add_argument('--mode',
        '-m',
        choices=['headless', 'gui', 'iggui'],
        default='iggui',
        help='which mode for simulation (default: iggui)')

    args = parser.parse_args()
    # run_example(args)
    # test_base_planning()
    run_base_planning_once(args)