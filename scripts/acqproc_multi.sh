#!/bin/bash

# This script is intended to be a "catch-all" script for Low Latency Control using ACQPROC
#
# What this script WILL do:
# - Configure the clocks on all DEVMAX systems (eg one master and two HDMI slaves)
# - Configure the system for LLC capture.
# - Start the control program - acqproc
# - Take a transient capture (or stream if specified).
# - Analyse the T_LATCH from the data acquired.
# - Configure the latency measurement FPGA registers.
# - Analyse the FPGA register data encoded in the SPAD.
#
# What this script will NOT do:
# - Set up isolated cpus for use with cpu affinity
# - Set up the python environment for the user.
# - Set ANY OUTPUT_DELAY values for the AO.
# - Analyse the full latency of the system. This is best done by hand and done
#   according to the LLC-system-latency-measurement-guide
#   (https://github.com/seanalsop/LLC-system-latency-measurement-guide)
#
# Please note:
# - This script should be run as root if the user wishes to use RPPRIO and AFFINITY
# - The script does not have to be run as root if taskset is not to be used.
# - run ./ACQPROC/acqproc $ACQPROC_CONFIG directly to review configuration

ACQPROC_CONFIG=$1 #{ACQPROC_CONFIG:-./ACQPROC/configs/pcs1.json}
if [ -z "$ACQPROC_CONFIG" ]; then
    echo "Script argument should be a configuration file (json)."
    echo "This argument is required. Example files are included in:"
    echo "ACQPROC/configs/"
    echo "ls ACQPROC/configs/*.json"
    exit
fi

cat - >acqproc_multi.env <<EOF
$(./scripts/acqproc_getconfig.py $ACQPROC_CONFIG)
EOF
source acqproc_multi.env
echo UUT1 $UUT1
echo UUT2 $UUT2
echo UUTS $UUTS
echo DEVMAX $DEVMAX

#CAPTURE_UUTS=${CAPTURE_UUTS:-$UUTS}
CAPTURE_UUTS=$(python3 scripts/list_capture_uuts.py --json_file=${ACQPROC_CONFIG})
POST=${POST:-40000} 	# Number of samples to capture
CLK=${CLK:-100000} 		# Set desired clock speed here.
VERBOSE=${VERBOSE:-0}
SYNC_ROLE_MODE=${SYNC_ROLE_MODE:-parallel} # serial: default, parallel, none
AFFINITY=${AFFINITY:-2}        # cpu affinity. 0=none, 1=use cpu0 2=use cpu1, for example

# UUT1 is the master in clock/trigger terms.
# The sync_role command can be changed to 'fpmaster' for external clk and trg.
TOPROLE=${TOPROLE:-master}		# alt: fpmaster for front panel clk/trg.

TOP=${TOP:-/home/dt100/PROJECTS/}
HAPI_DIR=$TOP/acq400_hapi/
AFHBA404_DIR=$TOP/AFHBA404/
MDS_DIR=$TOP/ACQ400_MDSplus/

ANALYSIS=false # Whether or not to run the analysis scripts.
TRANSIENT=true # Take a transient capture if true, else stream

PYTHON="python3"
# comment out if NOT using MDSplus
USE_MDSPLUS=0

export PYTHONPATH=/home/$USER/PROJECTS/acq400_hapi

if [ "$USE_MDSPLUS" = "1" ]; then
# Below is the UUT_path for MDSplus. The server is set to andros as this
# is the internal D-TACQ MDSplus server. Please change this to the name of
# your MDSplus server if you wish to use MDSplus. Ignore if not using MDSplus
for uut in $UUTS;do
    export $uut'_path=andros:://home/dt100/TREES/'$uut
done

mdsplus_upload() {
    # An optional function that uploads the scratchpad data to MDSplus.
    export NCOLS=15
    export STORE_COLS="0:14"
    DEVNUM=0
    for uut in $UUTS; do
        filename=uut${DEVNUM}_data.dat
        $MDS_DIR/mds_put_slice.py --ncols $NCOLS --dtype np.uint32 --store_cols $STORE_COLS \
            --tlatch_report=1 --node_name "CH%02d" --default_node ST $uut $filename
        DEVNUM=$((DEVNUM+1))
        echo ""
    done
    cd $AFHBA404_DIR

}
fi



check_uut() {
    for UUT in $UUTS; do

        ping -c 1 $UUT &>/dev/null
        if ! [ $? -eq 0 ]; then
            echo "Cannot ping $UUT1, please check UUT is available."
            exit 1
        fi
    done
}


analysis() {
    echo ""
    echo "Running analysis now."
    echo "--------------------"
    cd $AFHBA404_DIR
    # Change the json_src path here if the json file is not located in 
    # ~/PROJECTS/AFHBA404/runtime.json
    $PYTHON scripts/acqproc_analysis.py --ones=1 --json=1 --json_src="./runtime.json" --src="$AFHBA404_DIR"
}


control_program() {
    # Run the control program here
    cd $AFHBA404_DIR
    cat - > control_program.env <<EOF
    export DEVMAX=$DEVMAX
    export VERBOSE=$VERBOSE
    export HW=1
    export RTPRIO=10
    export AFFINITY=$AFFINITY
    export SINGLE_THREAD_CONTROL=control_dup1
EOF
    sudo bash -c 'source control_program.env; ./ACQPROC/acqproc '${ACQPROC_CONFIG}' '$POST''
    echo "Finished"
    [ "$USE_MDSPLUS" = "1" ] && mdsplus_upload
}


control_script() {

    cd $HAPI_DIR
    if $TRANSIENT; then
        $PYTHON user_apps/acq400/acq400_capture.py --transient="POST=${POST}" $CAPTURE_UUTS
    else
        $PYTHON user_apps/acq400/acq400_streamtonowhere.py --samples=$POST $CAPTURE_UUTS
    fi

}


configure_uut() {
    # Setup is done here.

    cd $HAPI_DIR
    case $SYNC_ROLE_MODE in
    n*)
	    echo "WARNING: omit sync_role";;
    p*)
        INDEX=0
        uuts=($uuts)
        SYNC_ROLES=($SYNC_ROLES)
        for uut in $UUTS; do
            TOPROLE=${SYNC_ROLES[$INDEX]}
            if [ "$TOPROLE" = "notouch" ] ; then continue ; fi
            echo "$TOPROLE"
            $PYTHON user_apps/acq400/sync_role.py --toprole="$TOPROLE" --fclk=$CLK $uut &
            TOPROLE=slave
            ((INDEX++))
        done
        for uut in $UUTS; do
            wait
        done;;
    *)
        $PYTHON user_apps/acq400/sync_role.py --toprole="$TOPROLE" --fclk=$CLK $UUTS;;
    esac 

    cd $AFHBA404_DIR
    $PYTHON scripts/llc-config-utility.py --include_dio_in_aggregator=1 --json_file=$ACQPROC_CONFIG $UUTS

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
