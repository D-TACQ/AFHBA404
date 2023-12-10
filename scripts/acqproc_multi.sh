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
if [ -z "$ACQPROC_CONFIG" ] || [ ! -e $ACQPROC_CONFIG ]; then
	echo "Script argument should be a configuration file (json)."
	echo "This argument is required. Example files are included in:"
	echo "ACQPROC/configs/"
	echo "ls ACQPROC/configs/*.json"
	exit 1
fi
shift

NOCONFIGURE=${NOCONFIGURE:-0}


./scripts/acqproc_getconfig.py $ACQPROC_CONFIG
source acqproc_multi.env
cat acqproc_multi.env

LLC_CALLBACKS=${LLC_CALLBACKS:-0}
CAPTURE_UUTS=$(python3 scripts/list_capture_uuts.py --json_file=${ACQPROC_CONFIG})
POST=${POST:-1000000} 	# Number of samples to capture
CLK=${CLK:-50000} 		# Set desired clock speed here.
VERBOSE=${VERBOSE:-0}
SYNC_ROLE_MODE=${SYNC_ROLE_MODE:-parallel} # serial: default, parallel, none
AFFINITY=${AFFINITY:-0}        # cpu affinity. 0=none, 2=use cpu0, for example
LOOP_FOREVER=${LOOP_FOREVER:-0} # set to one to loop forever
THE_ACQPROC=${THE_ACQPROC:-./ACQPROC/acqproc}  # selects ACQPROC variant.
SINGLE_THREAD_CONTROL=${SINGLE_THREAD_CONTROL:-control_dup1}

# UUT1 is the master in clock/trigger terms.
# The sync_role command can be changed to 'fpmaster' for external clk and trg.
TOPROLE=${TOPROLE:-master}		# alt: fpmaster for front panel clk/trg.

TOP=${TOP:-/home/$USER/PROJECTS}
HAPI_DIR=$TOP/acq400_hapi
AFHBA404_DIR=$TOP/AFHBA404/
MDS_DIR=$TOP/ACQ400_MDSplus/

ANALYSIS=${ANALYSIS:-true} # Whether or not to run the analysis scripts.
if [ $LLC_CALLBACKS -eq 0 ]; then
    TRANSIENT=${TRANSIENT:-false} # Take a transient capture if true, else stream
fi

PYTHON="python3"

PYTHON() {
	echo python3 $*
	python3 $*
	echo python3 $1 done $?
}

# comment out if NOT using MDSplus
USE_MDSPLUS=${USE_MDSPLUS:-0}

if [ $USE_MDSPLUS -ne 0 ]; then
	MDSPLUS_SERVER=${MDSPLUS_SERVER:-andros}
fi

export PYTHONPATH=/home/$USER/PROJECTS/acq400_hapi

if [ "$USE_MDSPLUS" = "1" ]; then
	for uut in $UUTS;do
        	export $uut'_path=${MDSPLUS_SERVER}:://home/dt100/TREES/'$uut
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
	PYTHON scripts/acqproc_analysis.py --ones=1 --json=1 --json_src="./runtime.json" --src="$AFHBA404_DIR"
}


control_program() {
	# Run the control program here. We have to pass environment through the sudo barrier.
	cd $AFHBA404_DIR
	cat - > control_program.env <<EOF
	export DEVMAX=$DEVMAX
	export VERBOSE=$VERBOSE
	export HW=1
	export RTPRIO=10
	export AFFINITY=$AFFINITY
	export SINGLE_THREAD_CONTROL=$SINGLE_THREAD_CONTROL
	export LOOP_FOREVER=$LOOP_FOREVER
	export THE_ACQPROC=$THE_ACQPROC
EOF
	sudo bash -c 'source control_program.env; rm -f *.dat; ${THE_ACQPROC} '${ACQPROC_CONFIG}' '$POST''
	echo "Control Program Finished"
	[ "$USE_MDSPLUS" = "1" ] && mdsplus_upload
}


control_script() {
	if $TRANSIENT; then
		PYTHON $HAPI_DIR/user_apps/acq400/acq400_capture.py --transient="PRE=0 POST=${POST}" $CAPTURE_UUTS
	else
		PYTHON $HAPI_DIR/user_apps/acq400/acq400_streamtonowhere.py --samples=$POST $CAPTURE_UUTS
	fi
}


configure_uut() {
	PYTHON scripts/llc-config-utility.py $ACQPROC_CONFIG
	case $SYNC_ROLE_MODE in
	n*)
		echo "WARNING: omit sync_role";;
	*)
        	PYTHON $HAPI_DIR/user_apps/acq400/sync_role.py --toprole="$TOPROLE" --fclk=$CLK $UUTS;;
	esac 
}

PID_CP=0
trap ctrl_c INT

ctrl_c() {
	echo "Trapped CTRL-C"
	if [ $PID_CP -ne 0 ]; then
		echo kill $PID_CP
		kill -9 $PID_CP
		sudo pkill acqproc
	fi
	exit
}

control_program_with_analysis() {
        control_program & PID_CP=$!
	wait 
        if $ANALYSIS; then
            analysis
        else
            sleep 2
	fi
}
control_program_loop() {
	shot=0
	while [ $shot -lt $1 ]; do
		echo control_program_loop SHOT $shot / $1
		control_program_with_analysis
		((shot++))
	done
}

case "x$1" in
xconfigure_uut)
	configure_uut;;
xcontrol_program)
	control_program_with_analysis;;
xcontrol_program_loop*)
	maxshot=${1#*=}
	control_program_loop ${maxshot:-10};;
xlooptest*)
	maxshot=${1#*=}
	control_program_loop ${maxshot:-10};;
xcontrol_script)
	control_script;;
xanalysis)
	analysis;;
xhelp)
	echo "USAGE: acqproc_multi.sh CONFIG [ops]"
	echo "[ops] : configure_uut control_program control_script analysis default:all";;
xall|*)
	# Execution starts here.
	check_uut
	[ $NOCONFIGURE -eq 0 ] && configure_uut
	if [ $LLC_CALLBACKS -ne 0 ]; then
		echo control_program_with_analysis
		control_program_with_analysis
	else
		echo control_program vanilla
		control_program & PID_CP=$!
        	control_script 
		wait $PID_CP
	fi

	if $ANALYSIS; then
		analysis
	fi;;
esac


