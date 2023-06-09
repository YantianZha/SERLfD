# SERLfD (Self-Explanation-guided Reinforcement Learning from Demonstrations)

This repository contains the implementation for the paper [Learning from Ambiguous Demonstrations with Self-Explanation Guided Reinforcement Learning](https://arxiv.org/pdf/2110.05286.pdf).



In this paper, we present an algorithm that iteratively learning to self-explain potentially ambiguous demonstrations thorough a shared vocabulary with humans, and a policy to accomplish tasks specified by demonstrations. We provide a comprehensive evaluation over three continuous robot control domains and one discrete Pacman domain. 

<p align="center">
    <img src="figures/env.png" alt="envs" width="800" />
</p>
<p align="center">
    <img src="figures/SERL_gif_1.gif" alt="envs" width="400" />
    <img src="figures/SERL_gif_2.gif" alt="envs" width="400" />
</p>
<p align="center">
    <img src="figures/SERL_gif_3.gif" alt="envs" width="400" />
    <img src="figures/SERL_gif_4.gif" alt="envs" width="400" />
</p>
<p align="center">
    <img src="figures/SERL_gif_9.gif" alt="envs" width="400" />
    <img src="figures/SERL_gif_10.gif" alt="envs" width="400" />
</p>
<p align="center">
    <img src="figures/SERL_gif_Pacman.gif" alt="envs" width="600" />
</p>

## Installation

- `git clone https://github.com/YantianZha/SERLfD.git`

- Download the demonstrations from [here](https://drive.google.com/drive/folders/18_PAerU15nanE3PMuu-eInZCKIosEN4F?usp=share_link).

- The code has been tested on 
  - Operating System: Ubuntu 18.04, CentOS 7
  - Python Version: 3.7, 3.8
  - gym version <= 0.25.2
  - numpy version == 1.20.1
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
  ```


## Training

#### TD3fD+SE and SACfD+SE (Our Method)

For example, running the following commands allows to train TD3fD and SACfD augmented with using self-explanations.


#### Baseline Algorithms

For example, running the following commands allows to train baseline RLfD agents,

```
CUDA_VISIBLE_DEVICES=1 nohup python3 run_fetch_push_v0p.py --algo sacfd --episode-num 3000 --off-render --max-episode-steps 50 --demo-path /data/Yantian/datasets/SERL/Fetch/Push-v0 --log > /data/Yantian/nohup_sacfd.out &
```

#### RLfD+SE Algorithms
```
CUDA_VISIBLE_DEVICES=1 nohup python3 run_fetch_push_v0p.py --algo sesacfd_v2_s --episode-num 3000 --off-render --max-episode-steps 50 --demo-path /data/Yantian/datasets/SERL/Fetch/Push-v0 --log > /data/Yantian/nohup_sesacfd_v2_s.out &
```

## Results

<p align="center">
    <img src="figures/result_robot.PNG" alt="envs" width="1000" />
</p>

<div class="col-sm-6 col-xs-6">
  <p><font size="+1"> Self-Explanation Guided Robot Learning</font></p>
  <br />
  <iframe width="560" height="315" src="https://www.youtube.com/embed/w5nGYOdVMiA" frameborder="0" gesture="media" allow="encrypted-media" allowfullscreen></iframe>
  </iframe>
</div>
        
## Citation

If you find our paper or code is useful, please consider citing:
```kvk
@article{yantian-self-expl,
  title={Learning from Ambiguous Demonstrations with Self-Explanation Guided Reinforcement Learning},
  author={Zha, Yantian and Guan, Lin and Kambhampati, Subbarao},
  journal={AAAI-22 Workshop on Reinforcement Learning in Games},
  year={2021}
}
```
