#!/bin/bash

# Set the PYTHONPATH
export PYTHONPATH=/mnt/d/codespace/mambaFieldReconstruction/cylinder:/mnt/d/codespace/mambaFieldReconstruction:/home/braveniuniu/.pycharm_helpers/pycharm_display:/home/braveniuniu/anaconda3/envs/mamba/lib/python310.zip:/home/braveniuniu/anaconda3/envs/mamba/lib/python3.10:/home/braveniuniu/anaconda3/envs/mamba/lib/python3.10/lib-dynload:/home/braveniuniu/anaconda3/envs/mamba/lib/python3.10/site-packages:/home/braveniuniu/.pycharm_helpers/pycharm_matplotlib_backend

# Arrays of parameters
d_states=(4 8 16 32 64 151)
num_blocks=(2 5 8 10 24)
modes1_modes2=(90 50 10)
widths=(1 5 12)

# Loop through all combinations of d_state, num_blocks, modes1, modes2, modes, and width
for d_state in "${d_states[@]}"
do
  for num_block in "${num_blocks[@]}"
  do
    for modes1 in "${modes1_modes2[@]}"
    do
      modes2=$modes1
      for width in "${widths[@]}"
      do
        modes_half=$((d_state / 2))
        modes_quarter=$((d_state / 4))

        for modes in $modes_half $modes_quarter
        do
          echo "Running script with d_state=${d_state}, num_blocks=${num_block}, modes1=${modes1}, modes2=${modes2}, modes=${modes}, width=${width}"

          # Check if the Python script exists
          if [ ! -f /mnt/d/codespace/mambaFieldReconstruction/cylinder/trainmambapod_timewithFNO.py ]; then
            echo "Python script not found: /mnt/d/codespace/mambaFieldReconstruction/cylinder/trainmambapod_timewithFNO.py" >&2
            exit 1
          fi

          # Execute the Python script and capture its output
          python /mnt/d/codespace/mambaFieldReconstruction/cylinder/trainmambapod_timewithFNO.py --d_state ${d_state} --num_blocks ${num_block} --modes1 ${modes1} --modes2 ${modes2} --modes ${modes} --width ${width} 2>&1 | tee /tmp/script_output.log

          # Check if the script finished successfully
          if [ $? -eq 0 ]; then
            echo "Finished running script with d_state=${d_state}, num_blocks=${num_block}, modes1=${modes1}, modes2=${modes2}, modes=${modes}, width=${width}"
          else
            echo "Error running script with d_state=${d_state}, num_blocks=${num_block}, modes1=${modes1}, modes2=${modes2}, modes=${modes}, width=${width}" >&2
            echo "Script output:"
            cat /tmp/script_output.log
            exit 1
          fi
        done
      done
    done
  done
done

echo "All combinations have been run."
