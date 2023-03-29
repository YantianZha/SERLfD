# Pacman (Eat-Ghost Task)

## Directory Structure
- agents: the original agents used by the game system
- eat\_ghost\_env: the eat-ghost game environment, including the utility map graphic display
- layouts: the pacman game layouts
- pacman\_src: the source codes of the original Berkeley project (most of the codes are unmodified)
- learning\_agents: implementation of different RL algorithms
- experiment\_results: save important experiment results, e.g., human demonstrations
- scripts: save experiment launcher/runner scripts that work well


## To launch an experiment

### SEGL-SQLfD (Eat One Ghost)
- file: **one\_ghost\_run\_segl\_single\_sql\_fd\_discrete.py**
- line 22: **experiment\_log\_dir**, specify the directory to save experiment-related files, currently it's set to \'tmp/\'
- line 24: **wandb\_project\_name**, specify the project name on wandb
- **line 26 and 27**: **demo\_dir** and **demo_expr_id**, specify the dir saving demonstration and the id of demonstration
- line 57: **--save-period**, specify how frequently (every n episode) we save the model during training
- line 73: **--no\_shaping**, for debugging, specify whether to return 0 shaping reward
- line 75: **--print\_shaping**, for debugging, specify whether print out the shaping reward information during training
- line 77: **--manual\_shaping**, for debugging, specify whether to use manually-set shaping reward
- **line 93**: **config\_file**, specify the config file of eat ghost domain, use this to set the number of ghosts in the env
- line 94 and 95: make a copy of the env config file and the launcher script
- **line 138**: specify the decay factor in learning from demonstration (this can significantly affect the performance of DQfD and our algo)
- **line 166 and 167**: this can affect the performance a lot
- line 178: **negative\_reward\_only**, specify whether to return negative shaping reward
- line 199: **lr_explainer**, the learning rate, might affect numerical stability sometimes 
- **line 200**: can use this to save additional experiment arguments on wandb
- Run existing/trained model: python one\_ghost\_run\_segl\_single\_sql\_fd\_discrete.py --test --explainer-load-from \[path to explainer model\] --policy-load-from \[path to policy model\] 



