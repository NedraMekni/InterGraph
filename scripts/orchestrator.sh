#!/bin/bash

SNAPSHOT_DIR="/data/shared/projects/NLRP3/IntegraphSnapshots" 
INCREMENTAL_DIR="/data/shared/projects/NLRP3/incremental_data"

conda activate graph_env
python3 get_structures.py
if [ $? -eq 0 ]; then
  echo OK STEP 1
  conda deactivate
  conda activate pym
  python3 test_pymol.py
  if [ $? -eq 0 ]; then
    echo OK STEP 2
    conda deactivate
    conda activate pyt-cuda
    python3 build_graphs.py
    if [ $? -eq 0 ]; then
      echo OK STEP 3
      conda deactivate
      echo "incremental_directory valid"
      SNAPSHOT_N=$(ls $SNAPSHOT_DIR| wc -l)
      echo "creating $SNAPSHOT_DIR/snapshot_$SNAPSHOT_N.tar.gz"
      tar -zcvf $SNAPSHOT_DIR/snapshot_$SNAPSHOT_N.tar.gz $INCREMENTAL_DIR


    else
      conda deactivate
      echo FAIL STEP 3
    fi    
  else
    conda deactivate
    echo FAIL STEP 2
  fi
else
  conda deactivate
  echo FAIL STEP 1
fi


