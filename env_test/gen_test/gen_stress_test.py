#!/usr/bin/env python
import random
import sys


n_addresses = 10000
address_length = 8
with open('stress_trace', 'w') as f:
    for i in range(n_addresses):
        address = ''
        for j in range(address_length):
            address += random.choice('0 1 2 3 4 5 6 7 8 9 a b c d e f'.split())
        if len(sys.argv) == 2:
            cache_size = int(sys.argv[1])
        # the output address space is twice the size of cache_size
        # one half for the sender and one half for the receiver	
            address = hex(int(address, 16) % (2*cache_size))[2:]	
        address = '0' * ( address_length - len(address) ) + address

        f.write(address + ' ' + random.choice(['R', 'R']) + '\n')
        
