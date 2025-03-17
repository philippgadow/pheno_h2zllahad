#!/bin/env python
from argparse import ArgumentParser


def main():
    parser = ArgumentParser(description='Hack LHE files')
    parser.add_argument('input', help='Input LHE file')
    args = parser.parse_args()

    hack_file(args.input)


def hack_file(infile):
    with open(infile, 'r') as f1, \
         open(infile.replace('.lhe', '_bsm.lhe'), 'w') as f2:
        for line in f1:
            if line.startswith('      25     1'):
                f2.write(line.replace('      25     1', '      35     1'))
            else:
                f2.write(line)


if __name__ == '__main__':
    main()
