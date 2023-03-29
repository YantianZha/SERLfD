# SERLfD (Self-Explanation-guided Reinforcement Learning from Demonstrations)

This repository contains the implementation for the paper [Learning from Ambiguous Demonstrations with Self-Explanation Guided Reinforcement Learning](https://arxiv.org/pdf/2110.05286.pdf).



In this paper, we present an algorithm that iteratively learning to self-explain potentially ambiguous demonstrations thorough a shared vocabulary with humans, and a policy to accomplish tasks specified by demonstrations. We provide a comprehensive evaluation over three continuous robot control domains and one discrete Pacman domain. 

<p align="center">
    <img src="figures/env.png" alt="envs" width="800" />
</p>

## Installation

- `git clone https://github.com/YantianZha/SERLfD.git`

- The code has been tested on 
  - Operating System: Ubuntu 18.04, CentOS 7
  - Python Version: 3.7, 3.8
  - GPU: RTX 1080, RTX 3080

#### Prerequisites

- In the project folder, create a virtual environment in Anaconda:

  ```
  conda env create -f SERLfD.yml
  conda activate SERLfD
  ```

- SERLfD

  ```
  cd SERLfD
  pip install -e .
  ```


## Training

Running the following commands in `examples` folder allows to train Ant with SHAC.
```
python train_shac.py --cfg ./cfg/shac/ant.yaml --logdir ./logs/Ant/shac
```

We also provide a one-line script in the `examples/train_script.sh` folder to replicate the results reported in the paper for both our method and for baseline method. The results might slightly differ from the paper due to the randomness of the cuda and different Operating System/GPU/Python versions. The plot reported in paper is produced with TITAN X on Ubuntu 16.04.

#### SHAC (Our Method)

For example, running the following commands in `examples` folder allows to train Ant and SNU Humanoid (Humanoid MTU in the paper) environments with SHAC respectively for 5 individual seeds.

```
python train_script.py --env Ant --algo shac --num-seeds 5
```

```
python train_script.py --env SNUHumanoid --algo shac --num-seeds 5
```

#### Baseline Algorithms

For example, running the following commands in `examples` folder allows to train Ant environment with PPO implemented in RL_games for 5 individual seeds,

```
python train_script.py --env Ant --algo ppo --num-seeds 5
```

## Testing

To test the trained policy, you can input the policy checkpoint into the training script and use a `--play` flag to indicate it is for testing. For example, the following command allows to test a trained policy (assume the policy is located in `logs/Ant/shac/policy.pt`)

```
python train_shac.py --cfg ./cfg/shac/ant.yaml --checkpoint ./logs/Ant/shac/policy.pt --play [--render]
```

The `--render` flag indicates whether to export the video of the task execution. If does, the exported video is encoded in `.usd` format, and stored in the `examples/output` folder. To visualize the exported `.usd` file, refer to [USD at NVIDIA](https://developer.nvidia.com/usd).

## Citation

If you find our paper or code is useful, please consider citing:
```kvk
@article{yantian-self-expl,
  title={Learning from Ambiguous Demonstrations with Self-Explanation Guided Reinforcement Learning},
  author={Zha, Yantian and Guan, Lin and Kambhampati, Subbarao},
  journal={AAAI-22 Workshop on Reinforcement Learning in Games and arXiv preprint arXiv:2110.05286},
  year={2021}
}
```
