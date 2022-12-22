CacheSimulator
==============



A cache simulator in Python for CS 530

Documentation on cache configuration and trace files are located in their respective folders

### Requirements

CacheSimulator needs two extra Python modules: pyyaml and terminaltables

These can both be installed using pip:

    sudo pip install pyyaml/terminaltables

### Running

To run a quick test simulation, enter the src folder and run this command:

    ./cache_simulator.py -pdc ../configs/config_simple_multilevel -t ../traces/trace2.txt

For more details, run:

    ./cache_simulator.py --help

### Goals

This simulator will create a memory heirarchy from a YAML configuration file
and calculate the AMAT for a given tracefile.

The memory heirarchy is configurable with the following features:
- Word size, block size
  - Address size does not need to be defined
- L1 cache with user-defined parameters
  - Associativity
  - Hit time
  - Write time
- Optional L2 and L3 caches
- Simulate write back and write through
- Pretty print the cache layouts

