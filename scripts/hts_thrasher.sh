#! /usr/bin/env bash

log_file=~/hts_multistream.log
function log {
    echo "[$(date)] ${1}" >> $log_file
}

log "Starting hts_multistream thrasher" 

loop=0
max_loops=9999
while [[ $loop -lt $max_loops ]]; do

    timeout 300 ./HAPI/hts_multistream.py --spad=1,8,0 --check_spad=1 --auto=1 --recycle=0 --sig_gen=sg0761 acq2206_010 acq2206_009 acq2206_008 acq2206_007 acq2206_006 acq2206_005
    rtnval=$?
    if [[ $rtnval != 0 ]]; then
        log "[RUN $loop] (ERROR) hts_multistream.py $rtnval"
    else
        log "[RUN $loop] (SUCCESS) hts_multistream.py succeeded $rtnval"
    fi

    grep "Errors 0" /mnt/afhba.*/acq2206_0??/checker.log
    rtnval=$?
    if [[ $rtnval != 0 ]]; then
        log "[RUN $loop] (ERROR) Spad Checker Failed $rtnval"
    else
        log "[RUN $loop] (SUCCESS) Spad Checker succeeded $rtnval"
    fi

    ((loop++))
    sleep 1
done
