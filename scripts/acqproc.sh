#!/bin/bash


# This script is intended to be a "catch-all" script for Low Latency Control
# systems. What this script WILL do:
# - Configure the clocks
# - Configure the system for LLC capture.
# - Start the control program. At present this is always a cpucopy.
# - Take a transient capture.
# - Analyse the T_LATCH from the data acquired.
# - Configure the latency measurement FPGA registers.
# - Analyse the FPGA register data encoded in the SPAD.
#
# What this script will NOT do:
# - Set up isolated cpus for use with taskset.
# - Set up the python environment for the user.
# - Set ANY OUTPUT_DELAY values for the AO.

# - Analyse the full latency of the system. This is best done by hand and done
#   according to the LLC-system-latency-measurement-guide
#   (https://github.com/seanalsop/LLC-system-latency-measurement-guide)
#
# Please note:
# - This script should be run as root if the user wishes to use tasket.
# - The script does not have to be run as root if taskset is not to be used.

UUT1=acq2106_085

POST=400000 # 20kHz for 20s
CLK=20000 # Set clock speed here

HAPI_DIR=/home/dt100/PROJECTS/acq400_hapi/
AFHBA404_DIR=/home/dt100/PROJECTS/AFHBA404/
MDS_DIR=/home/dt100/PROJECTS/ACQ400_MDSplus/

ANALYSIS=true # Whether or not to run the analysis scripts.
TRANSIENT=true # Take a transient capture if true, else stream.

PYTHON="python3.6"
# comment out if NOT using MDSplus
USE_MDSPLUS=1

export PYTHONPATH=/home/dt100/PROJECTS/acq400_hapi

if [ "$USE_MDSPLUS" = "1" ]; then
# Below is the UUT_path for MDSplus. The server is set to andros as this
# is the internal D-TACQ MDSplus server. Please change this to the name of
# your MDSplus server if you wish to use MDSplus. Ignore if not using MDSplus
export $UUT1'_path=andros:://home/dt100/TREES/'$UUT1


mdsplus_upload() {
    # An optional function that uploads the scratchpad data to MDSplus.
    export NCOLS=16
    export STORE_COLS="0:15"
    $MDS_DIR/mds_put_slice.py --ncols $NCOLS --dtype np.uint32 --store_cols $STORE_COLS \
        --tlatch_report=1 --node_name "CH%02d" --default_node ST \
        $UUT1 afhba.0.log

    cd $AFHBA404_DIR

}
fi



check_uut() {
    ping -c 1 $UUT1 &>/dev/null
    if ! [ $? -eq 0 ]; then
        echo "Cannot ping $UUT1, please check UUT is available."
        exit 1
    fi
}


analysis() {
    echo ""
    echo "Running analysis now."
    echo "--------------------"
    # Takes the LLC data and runs various analysis scripts.
    cd $HAPI_DIR
    DEVNUM=$(echo $cmd | cut -d" " -f6 | cut -d"=" -f2)
    NCHAN=$(echo $cmd | cut -d" " -f2 | cut -d"=" -f2)
    SPADLONGS=$(echo $cmd | cut -d" " -f5 | cut -d"=" -f2)
    # echo $DEVNUM
    $PYTHON test_apps/t_latch_histogram.py --src=$AFHBA404_DIR/afhba.$DEVNUM.log --ones=1 --nchan=$NCHAN --spad_len=$SPADLONGS

    cd $AFHBA404_DIR
    $PYTHON ./scripts/latency_on_diff_histo.py --file="./afhba.0.log" $UUT1
}


control_program() {
    # Run the control program here
    cd $AFHBA404_DIR

    # Export all of the environment variables the system needs to run the
    # correct cpucopy.
    for ii in 1 2 3 4 5 6
    do
        var=$(echo $cmd | cut -d" " -f$ii)
        export $var
    done

    # Cut off all of the environment variables as they have been exported.
    runcmd=$(echo $cmd | cut -d" " -f7)
    runcmd="${runcmd} ${POST}"
    echo "Command: $cmd"

    export TASKET="taskset --cpu-list 1"
    [ "x$TASKSET" != "x" ] && echo TASKSET $TASKSET

    eval "${TASKSET} ${runcmd}"
    wait
    
    [ "$USE_MDSPLUS" = "1" ] && mdsplus_upload
}


control_script() {

    cd $HAPI_DIR
    if $TRANSIENT; then
        $PYTHON user_apps/acq400/acq400_capture.py --transient="POST=${POST}" $UUT1
    else
        $PYTHON user_apps/acq400/acq400_streamtonowhere.py --samples=$POST $UUT1
    fi

}


configure_uut() {
    # Setup is done here.

    cd $HAPI_DIR
    # The sync_role command can be changed to 'fpmaster' for external clk and trg.
    $PYTHON user_apps/acq400/sync_role.py --toprole="master" --fclk=$CLK $UUT1

    cd $AFHBA404_DIR
    cmd="$($PYTHON scripts/llc-config-utility.py --include_dio_in_aggregator=0 $UUT1)"
    success=$?
    cmd="$(echo "$cmd" | tail -n1)"
    if ! [ $success -eq 0 ]; then
        echo "Host did not find $UUT1 connected the AFHBA404 card. Please check connections."
        exit 1
    fi

    # cd $AFHBA404_DIR
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
    check_uut
    configure_uut
    control_program &
    control_script

    if $ANALYSIS; then
        analysis
    fi
    ;;
esac
