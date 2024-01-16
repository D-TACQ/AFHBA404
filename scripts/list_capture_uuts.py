#!/usr/bin/env python3


import acq400_hapi
import json
import argparse


def get_args():
    parser = argparse.ArgumentParser(description='LLC capture UUTs tool.')

    parser.add_argument('--json_file', type=str, default='./test.json',
        help='Where to load the json file from.')

    return parser.parse_args()



def load_json(json_file):
    with open(json_file) as _json_file:
        json_data = json.load(_json_file)
    return json_data


def main():
    uuts = []
    args = get_args()
    uut_config = load_json(args.json_file)
    for uut in uut_config['AFHBA']['UUT']:
        def add():
            if 'master' in uut['sync_role']:
                uuts.insert(0, uut['name'])
            else:
                uuts.append(uut['name'])

        if "AI16" in uut['VI'].keys():
            if uut['VI']['AI16'] != 0:
                add()
        if "AI32" in uut['VI'].keys():
            if uut['VI']['AI32'] != 0:
                add()
    print(" ".join(uuts))


if __name__ == '__main__':
    main()

