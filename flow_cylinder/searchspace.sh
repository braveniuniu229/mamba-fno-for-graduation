#!/bin/bash

# Set the PYTHONPATH
export PYTHONPATH=/mnt/d/codespace/mambaFieldReconstruction/flow_cylinder:/mnt/d/codespace/mambaFieldReconstruction:/home/braveniuniu/.pycharm_helpers/pycharm_display:/home/braveniuniu/anaconda3/envs/mamba/lib/python310.zip:/home/braveniuniu/anaconda3/envs/mamba/lib/python3.10:/home/braveniuniu/anaconda3/envs/mamba/lib/python3.10/lib-dynload:/home/braveniuniu/anaconda3/envs/mamba/lib/python3.10/site-packages:/home/braveniuniu/.pycharm_helpers/pycharm_matplotlib_backend

# Arrays of parameters
d_states=(4 8 16 32 64 80 128 151)
num_blocks=(2 5 8 10 12 15 20 24)

# Loop through all combinations of d_state and num_blocks
for d_state in "${d_states[@]}"
do
  for num_block in "${num_blocks[@]}"
  do
    echo "Running script with d_state=${d_state} and num_blocks=${num_block}"
    python /mnt/d/codespace/mambaFieldReconstruction/flow_cylinder/trainimambawithpod_space.py --d_state ${d_state} --num_blocks ${num_block}

    # Check if the script finished successfully
    if [ $? -eq 0 ]; then
      echo "Finished running script with d_state=${d_state} and num_blocks=${num_block}"
    else
      echo "Error running script with d_state=${d_state} and num_blocks=${num_block}" >&2
      exit 1
    fi
  done
done

echo "All combinations have been run."
