# Self-Explanation-Guided-Reinforcement-Learning

The directory contains the source code for running SEGL **Pacman** experiments.

- To run the SQLfD+SE, use the following command:

```
python -m one_ghost_run_segl_single_sql_fd_discrete_predicate
```

- To run the SQLfD+SE+nu, use the following command: 

```
python -m one_ghost_run_segl_single_sql_fd_discrete
```

- To run the SQLfD+SE+nrs, use the following command:

```
python -m run_baseline_no_shaping
```


### Reference
Our implementation (especially RL baseline) takes the KAIR open-source RL repository as the codebase: https://github.com/kairproject/kair_algorithms_draft
