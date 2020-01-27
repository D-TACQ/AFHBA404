#!/bin/bash


# This script is intended to be a "catch-all" script for Low Latency Control
# systems, with use for three systems at once. What this script WILL do:
# - Configure the clocks on all three systems (one master and two HDMI slaves)
# - Configure the system for LLC capture.
# - Start the control program. This is always a multiuut-4AI1AO1DX.
# - Take a transient capture (or stream if specified).
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
UUT2=acq2106_130
UUT3=acq2106_176
UUT_BOLO=acq2106_123

# If you are using more than one UUT fill in the UUTs:
UUTS="${UUTS:-$UUT1 $UUT2 $UUT3}"

DEVMAX=0
for u in $UUTS; do
	DEVMAX=$(($DEVMAX+1))
done


if [ -n "$UUT_BOLO" ]; then
	# If we are using a BOLO system then increment DEVNUM.
	DEVMAX=$(($DEVMAX+1))
fi


POST=${POST:-200000} 	# Number of samples to capture
CLK=${CLK:-10000} 		# Set desired clock speed here.
VERBOSE=${VERBOSE:-1}
SYNC_ROLE_MODE=${SYNC_ROLE_MODE:-parallel} # serial: default, parallel, none

# UUT1 is the master in clock/trigger terms.
# The sync_role command can be changed to 'fpmaster' for external clk and trg.
TOPROLE=${TOPROLE:-fpmaster}		# alt: fpmaster for front panel clk/trg.

TOP=${TOP:-/home/dt100/PROJECTS/}
HAPI_DIR=$TOP/acq400_hapi/
AFHBA404_DIR=$TOP/AFHBA404/
MDS_DIR=$TOP/ACQ400_MDSplus/

ANALYSIS=true # Whether or not to run the analysis scripts.
TRANSIENT=false # Take a transient capture if true, else stream.

PYTHON="python3"
# comment out if NOT using MDSplus
USE_MDSPLUS=0

export PYTHONPATH=/home/dt100/PROJECTS/acq400_hapi

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
    # Takes the LLC data and runs various analysis scripts.
    cd $HAPI_DIR
    
    NCHAN=130
    SPADLONGS=15
    # echo $DEVNUM

    DEVNUM=0
    for uut in $UUTS; do
        filename=uut${DEVNUM}_data.dat
        cd $HAPI_DIR
        $PYTHON test_apps/t_latch_histogram.py --src=$AFHBA404_DIR/$filename --ones=1 --nchan=$NCHAN --spad_len=$SPADLONGS

        cd $AFHBA404_DIR
        $PYTHON ./scripts/latency_on_diff_histo.py --file="./$filename" $UUT1
        DEVNUM=$((DEVNUM+1))
    done
}


control_program() {
    # Run the control program here
    cd $AFHBA404_DIR

    export TASKET="taskset --cpu-list 1"
    [ "x$TASKSET" != "x" ] && echo TASKSET $TASKSET
    export DEVMAX=$DEVMAX
    export VERBOSE=$VERBOSE

    export BOLODEV=3

    $TASKSET ./LLCONTROL/afhba-llcontrol-multiuut-4AI1AO1DX $POST
    wait
    echo "Splitting data now."
    ./scripts/split_multi_uut_data.py --nuuts=$DEVMAX
    echo "Starting MDSplus put now."
    [ "$USE_MDSPLUS" = "1" ] && mdsplus_upload
}


control_script() {

    cd $HAPI_DIR
    if $TRANSIENT; then
        $PYTHON user_apps/acq400/acq400_capture.py --transient="POST=${POST}" $UUTS
    else
        $PYTHON user_apps/acq400/acq400_streamtonowhere.py --samples=$POST $UUTS $UUT_BOLO
    fi

}


configure_uut() {
    # Setup is done here.

    cd $HAPI_DIR
    case $SYNC_ROLE_MODE in
    n*)
	    echo "WARNING: omit sync_role";;
    p*)
        for uut in $UUTS; do
            $PYTHON user_apps/acq400/sync_role.py --toprole="$TOPROLE" --fclk=$CLK $uut &
            TOPROLE=slave
        done
        for uut in $UUTS; do
            wait
        done;;
    *)
        $PYTHON user_apps/acq400/sync_role.py --toprole="$TOPROLE" --fclk=$CLK $UUTS;;
    esac 

    cd $AFHBA404_DIR
    cmd="$($PYTHON scripts/llc-config-utility.py --include_dio_in_aggregator=1 $UUTS)"
    success=$?
    info=$(echo "$cmd" | tail -n6 | sed '$d')
    printf "$info"
    echo -e "\n"
    cmd="$(echo "$cmd" | tail -n1)"
    if ! [ $success -eq 0 ]; then
        echo "Host did not find $UUT1 connected the AFHBA404 card. Please check connections."
        exit 1
    fi

    # cd $AFHBA404_DIR
    if [ -n $UUT_BOLO ]; then
        echo "Configuring BOLO now."
        SPAD_LEN=16 AISITES=1,2 ./HAPI/llc-bolo-harness.py $UUT_BOLO
    fi
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
    time configure_uut
    control_program &
    control_script

    if $ANALYSIS; then
        analysis
    fi
    ;;
esac
