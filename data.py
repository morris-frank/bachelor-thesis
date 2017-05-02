#!/usr/bin/env python3
from ba.data import Generator
import sys


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('No arguments given')
        sys.exit()
    gen = Generator(sys.argv[1:])
    gen.run()
