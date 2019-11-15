#!/bin/bash

beacon=0
if [[ -f "./server_data" ]]; then
  source "./server_data"
  beacon=1
fi

function check_nof_connection(){
  echo $(ss | grep -i ssh | wc -l)
}

function restart_sshd(){
  echo "Restarting ssh..."
  systemctl restart sshd.service
}

function send_beacon(){
  if [[ "${beacon}" -eq 1 ]]; then
    echo "Last beacon $(date)" > beacon.log
    rsync -avrzP ./beacon.log $SRVR_HORACIO_ENV:$remote_base_dir/beacon.log 1>/dev/null 2>/dev/null
    rsync -avrzP ./nohup.out $SRVR_HORACIO_ENV:$remote_base_dir/beacon.nohup.out 1>/dev/null 2>/dev/null
  else
    echo "No server data"
  fi
}

while /bin/true; do
  nof_connections=$(check_nof_connection)
  if [[ "${nof_connections}" -eq 0 ]]; then
    restart_sshd
    send_beacon
  else
    send_beacon
  fi
  sleep 5m
done
