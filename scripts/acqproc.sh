#!/bin/bash

UUT1=acq2106_085
UUT2=acq2106_085
UUT3=acq2106_085
UUTS=$UUT1,$UUT2,$UUT3
echo $UUTS

POST=400000 # 20kHz for 20s

HAPI_DIR=/home/dt100/PROJECTS/acq400_hapi/
AFHBA404_DIR=/home/dt100/PROJECTS/AFHBA404/
export PYTHONPATH=/home/dt100/PROJECTS/acq400_hapi

control_program() {
    # Run the control program here
    cd $AFHBA404_DIR
    # Which channel to duplicate. 1 is CH02.
    export DUP1=0
    # Number of AI channels in the system.
    export NCHAN=128
    # Number of AO channels in the system.
    export AOCHAN=32
    # Number of DO Longwords in the system.
    export DO32=1
    # Number of longwords in the scratch pad.
    export SPADLONGS=16
    # Which port to use.
    export DEVNUM=0
    # Whether or not to print samples.
    export VERBOSE=1 
    export TASKET="taskset --cpu-list 1"
    [ "x$TASKSET" != "x" ] && echo TASKSET $TASKSET
    $TASKSET ./LLCONTROL/afhba-llcontrol-cpucopy $POST &
    wait
}


control_script() {

    cd $HAPI_DIR
    python3.6 user_apps/acq400/acq400_capture.py --transient='POST=400000' $UUT1
}


configure_script() {

    # Setup is done here.
    cd $AFHBA404_DIR
    python3.6 scripts/llc-config-utility.py --include_dio_in_aggregator=0 $UUT1

    cd $HAPI_DIR
    # The sync_role command can be changed to 'fpmaster' for external clk and trg.
    python3.6 user_apps/acq400/sync_role.py --toprole="master" --fclk=20000 $UUT1
    
    cd $AFHBA404_DIR
}


case "x$1" in
xconfigure_uut)
    configure_uut;;
xcontrol_program)
    control_program;;
xcontrol_script)
    control_script;;
*)
    # Execution starts here.
    configure_script
    sudo $0 control_program &
    control_script
    ;;
esac
