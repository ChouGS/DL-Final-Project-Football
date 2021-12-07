# CGT-Final-Project-Football

Run `objects/gameyard.py` to see several sample visualizations

## Newest: instructions for running `main.py`

`-ns` indicates the number of simulations.

`-np` or `--num_players` indicates the number of players on each team.

`-op` or `--offender_pattern` indicates the strategy for offenders. Must be 'H'/'L'.

`-pp` or `--passing_pattern` indicates the strategy for quaterback when they make `pass_or_not` decisions. Must be 'H'/'L'.

`-cp` or `--control_pattern` indicates the configuration of `u_penalty`. Must be 'H'/'L'.

For example, if you wish to run an experiment for 100 games with `offender_pattern=H, passing_pattern=L, control_pattern=H`, you should key in the command

```bash
./main.py -ns 100 -op H -pp L -cp H
```
