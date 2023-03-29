#!/bin/bash

ROS_DISTRO=kinetic
CATKIN_WS=/root/catkin_ws
KAIR=$CATKIN_WS/src/kair_algorithms_draft

source /opt/ros/$ROS_DISTRO/setup.bash
source $CATKIN_WS/devel/setup.bash

if [ "$1" == "lunarlander" ]; then
	cd $KAIR/scripts; \
		python run_lunarlander_continuous.py --algo $2 --off-render
elif [ "$1" == "reacher" ]; then
	cd $KAIR/scripts; \
		python run_reacher_v1.py --algo $2 --off-render
elif [ "$1" == "openmanipulator" ]; then
	cd $CATKIN_WS; \
		roslaunch kair_algorithms open_manipulator_env.launch gui:=false &
	cd $KAIR/scripts; \
		rosrun kair_algorithms run_open_manipulator_reacher_v0.py --algo $2 --off-render
else
	echo "Unknown parameter"
fi
