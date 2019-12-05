#!/bin/bash

if [[ $@ -lt 1 ]]; then
  exit 0
fi

pid=$1
while kill -0 $pid 2>&1 1>/dev/null; do
  sleep 10m
done

source ./server_data
rsync -avrzP nohup.out $SRVR_HORACIO_ENV:$remote_base_dir/train.log
sudo poweroff
