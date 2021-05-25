#################################################
#  launch_feedback.py
#    This contains the script that launches the gpu_control_common program
#    Things that are performed:
#     a) Read setting files from the HIT-SI mdsplus tree.
#     b) Initialize the D-TACQ hardware (acq2106_122). - Sub-launches python2 program
#     c) Pre-process gpu_control_common to launch.
#     d) Launch program.
#     e) Post-process gpu_control_common output.
################################################

#import modules
import os
import subprocess
import time
import acq400_hapi

#declare some global variables:
sourcepath = ''
dtname = 'acq2106_122'

def run_hapi():
    # Runs a python2 script, which runs the HAPI configuration script to
    #  initialize the D-TACQ acq2106_122 device.
    scriptname = 'llc-acq480-dio.py'
    run_str = 'python2 ' + sourcepath + '/' + scriptname + ' ' + dtname
    print(run_str)
    subprocess.call(run_str.split(), env=dict(os.environ,DOSITES="6",SITECLIENT_TRACE="1"))
    uut = acq400_hapi.Acq2106(dtname)
#    print('Waiting for armed status')
#    uut.statmon.wait_armed()
    print(dtname + ' has been armed')
    return 0

def stop_datastream():
    # Runs a python2 script, to use HAPI to send a terminate signal to
    # the acq2106_122
    scriptname = 'stop_datastream.py'
    run_str = 'python2 ' + sourcepath + '/' + scriptname + ' ' + dtname
    print(run_str)
    subprocess.call(run_str.split())
    return 0

#module_loaded from stackoverflow
def load_afhba():
    print("Looking for afhba kernel module:")
    lsmod_proc = subprocess.Popen(['lsmod'],stdout = subprocess.PIPE)
    grep_proc = subprocess.Popen(['grep','afhba'],stdin=lsmod_proc.stdout)
    grep_proc.communicate()
    if grep_proc.returncode == 0:
        print("afhba kernel module already loaded.")
        return
    else:
        print("afhba kernel module not loaded, loading now")
        subprocess.call(['sudo','/home/ctfusion/hitsi_gpucontrol/afhba404_gpu/scripts/loadNIRQ_gpu'])
        return

def launch_gpu():
    run_str = './afhba-llcontrol-gpu'
    print(run_str)
    subprocess.call(run_str.split(),shell=True)
    return 0;

if __name__ == "__main__":
    print("Launcher started")
    sourcepath = os.path.dirname(os.path.abspath(__file__))
    load_afhba()

# Initialize acq2106_122
    run_hapi()

# Launch gpu_control_common:
    launch_gpu()

