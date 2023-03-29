#!/bin/bash

SNAPSHOT_DIR="/data/shared/projects/NLRP3/IntegraphSnapshots" 
INCREMENTAL_DIR="/data/shared/projects/NLRP3/incremental_data"
SNAPSHOT_N=9

rm -rf $INCREMENTAL_DIR
mkdir $INCREMENTAL_DIR
tar -zxvf $SNAPSHOT_DIR/snapshot_$SNAPSHOT_N.tar.gz -C $INCREMENTAL_DIR
cp -r $INCREMENTAL_DIR$INCREMENTAL_DIR $INCREMENTAL_DIR'1'
rm -rf $INCREMENTAL_DIR
mv $INCREMENTAL_DIR'1' $INCREMENTAL_DIR