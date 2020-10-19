import numpy as np
import argparse
import json


#uuts = 7
#data = []
#total_data = np.array([]).reshape((-1, 192))
#uuts = ["acq2106_238", "acq2106_261", "acq2106_262", "acq2106_263", "acq2106_264", "acq2106_265", "acq2106_266"]
#nchan= 192


# for uut in uuts:
#	dat = np.fromfile("{}_VI.dat".format(uut), dtype=np.int32).reshape((-1, 112))
#	data.append(dat)
#	print(dat.shape)
#

#total_data = np.concatenate(data, axis=1).flatten()


# total_data.tofile("LLCONTROL/afhba.0.log")


def get_args():
    parser = argparse.ArgumentParser(description='THOMSON data mux tool.')
    parser.add_argument('--json_file', type=str,
                        default="ACQPROC/configs/7_uut_thom.json")
    args = parser.parse_args()
    return args


def load_json(json_file):
    with open(json_file) as _json_file:
        json_data = json.load(_json_file)
    return json_data


def get_uut_info(uut_json):
    uuts = []
    longwords = 0
    for uut in uut_json["AFHBA"]["UUT"]:
        uuts.append(uut["name"])
        longwords = int(uut["VI"]["SP32"]) + int((uut["VI"]["AI16"]) / 2)
    return longwords, uuts


def main():
    args = get_args()
    uut_json = load_json(args.json_file)
    longwords, uuts = get_uut_info(uut_json)

    data = []
    for uut in uuts:
        uut_data = np.fromfile("{}_VI.dat".format(
            uut), dtype=np.int32).reshape((-1, 112))
        data.append(uut_data)

    total_data = np.concatenate(data, axis=1).flatten()
    # hardcoded for thomson_analysis.py
    total_data.tofile("LLCONTROL/afhba.0.log")


if __name__ == '__main__':
    main()
