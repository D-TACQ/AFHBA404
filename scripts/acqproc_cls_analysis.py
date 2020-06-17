#!/usr/bin/python


import argparse
import json
import numpy as np


def get_args():
    parser = argparse.ArgumentParser(description='acqproc analysis')

    parser.add_argument('--json_file', default="./configs/ns32.json",
    help="json file to load.")

    parser.add_argument('--composite_file', default="./ns32.log",
    help="composite file to load.")

    args = parser.parse_args()
    return args


def load_json(json_file):
    with open(json_file) as _json_file:
        json_data = json.load(_json_file)
    return json_data


def verify_data(json, composite_file):
    data = []
    for uut in json['AFHBA']['UUT']:
        nchan_total = uut['VI']['AI32'] + uut['VI']['SP32']
        data.append(np.fromfile("{}_VI.dat".format(uut['name']), dtype=np.int32).reshape((-1, nchan_total)))

    composite = np.fromfile(composite_file, dtype=np.int32)
    composite = composite.reshape((-1, 64))

    # Use col stack to zip together the VI.dat files without the SPAD.
    test_composite = [ array[:,0:uut['VI']['AI32']] for array in data ]
    test_composite = np.column_stack(test_composite)

    if not np.array_equal(composite, test_composite):
        print("Error found. Composite array does not contain the same data as VI files.")

    test_spad = [0,22222222,33333333,44444444,55555555,66666666,77777777,0,0,0,0,0,0,0,0]
    test_spad = np.array([ int(str(item),16) for item in test_spad ])
    for uut_data in data:
        for spad in uut_data[:,uut['VI']['AI32']+1:]:
            if not np.array_equal(spad, test_spad):
                print(spad)
                print(test_spad)
                print("SPAD error detected.")


    return data


def main():
    args = get_args()
    json = load_json(args.json_file)
    verify_data(json, args.composite_file)

    return None


if __name__ == '__main__':
    main()
