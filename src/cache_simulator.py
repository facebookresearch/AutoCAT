# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed according to the terms of the
# GNU General Public License version 2.

#!/usr/bin/env python

import yaml, cache, argparse, logging, pprint
from terminaltables.other_tables import UnixTable
from replacement_policy import *
def main():
    #Set up our arguments
    parser = argparse.ArgumentParser(description='Simulate a cache')
    parser.add_argument('-c','--config-file', help='Configuration file for the memory heirarchy', required=True)
    parser.add_argument('-t', '--trace-file', help='Tracefile containing instructions', required=True)
    parser.add_argument('-l', '--log-file', help='Log file name', required=False)
    parser.add_argument('-p', '--pretty', help='Use pretty colors', required=False, action='store_true')
    parser.add_argument('-d', '--draw-cache', help='Draw cache layouts', required=False, action='store_true')
    parser.add_argument('-f', '--result-file', help='Result trace', required=False)
    arguments = vars(parser.parse_args())
    
    if arguments['pretty']:
        import colorer

    log_filename = 'cache_simulator.log'
    if arguments['log_file']:
        log_filename = arguments['log_file']

    result_file = 'result.txt'
    if arguments['result_file']:
        result_file = arguments['result_file']

    with open(result_file, 'w'):
        pass

    #Clear the log file if it exists
    with open(log_filename, 'w'):
        pass

    logger = logging.getLogger()
    fh = logging.FileHandler(log_filename)
    sh = logging.StreamHandler()
    logger.addHandler(fh)
    logger.addHandler(sh)

    fh_format = logging.Formatter('%(message)s')
    fh.setFormatter(fh_format)
    sh.setFormatter(fh_format)
    logger.setLevel(logging.INFO)
    
    logger.info('Loading config...')
    config_file = open(arguments['config_file'])
    configs = yaml.full_load(config_file)
    hierarchy = build_hierarchy(configs, logger)
    logger.info('Memory hierarchy built.')

    logger.info('Loading tracefile...')
    trace_file = open(arguments['trace_file'])
    trace = trace_file.read().splitlines()
    trace = [item for item in trace if not item.startswith('#')]
    logger.info('Loaded tracefile ' + arguments['trace_file'])
    logger.info('Begin simulation!')
    simulate(hierarchy, trace, logger, result_file = result_file)
    if arguments['draw_cache']:
        for cache in hierarchy:
            if hierarchy[cache].next_level:
                print_cache(hierarchy[cache])
    
    

#Print the contents of a cache as a table
#If the table is too long, it will print the first few sets,
#break, and then print the last set
def print_cache(cache):
    table_size = 5
    ways = [""]
    sets = []
    set_indexes = sorted(cache.data.keys())
    if len(cache.data.keys()) > 0:
        first_key = list(cache.data.keys())[0]
        way_no = 0
        
        #Label the columns
        for way in range(cache.associativity):
            ways.append("Way " + str(way_no))
            way_no += 1
        
        #Print either all the sets if the cache is small, or just a few
        #sets and then the last set
        sets.append(ways)
        if len(set_indexes) > table_size + 4 - 1:
            for s in range(min(table_size, len(set_indexes) - 4)):
                temp_way = ["Set " + str(s)]
                for w in range(0, cache.associativity):
                    temp_way.append(cache.data[set_indexes[s]][w][1].address)
                sets.append(temp_way)
            
            for i in range(3):
                temp_way = ['.']
                for w in range(cache.associativity):
                    temp_way.append('')
                sets.append(temp_way)
            
            ##set_ways = cache.data[set_indexes[len(set_indexes) - 1]].keys()
            temp_way = ['Set ' + str(len(set_indexes) - 1)]
            for w in range(0, cache.associativity):
                temp_way.append(cache.data[set_indexes[len(set_indexes) - 1]][w][1].address)
            sets.append(temp_way)
        else: 
            for s in range(len(set_indexes)):
                temp_way = ["Set " + str(s)]
                for w in range(0, cache.associativity):
                    temp_way.append(cache.data[set_indexes[s]][w][1].address)
                sets.append(temp_way)

        table = UnixTable(sets)
        table.title = cache.name
        table.inner_row_border = True
        print(table.table)

#Loop through the instructions in the tracefile and use
#the given memory hierarchy to find AMAT
def simulate(hierarchy, trace, logger, result_file=''):
    responses = []
    if result_file != '':
        f = open(result_file, 'w')
    #We only interface directly with L1. Reads and writes will automatically
    #interact with lower levels of the hierarchy
    l1 = hierarchy['cache_1']
    if 'cache_1_core_2' in hierarchy:
        l1_c2 = hierarchy['cache_1_core_2']
    for current_step in range(len(trace)):
        instruction = trace[current_step]
        address, op = instruction.split()
        #Call read for this address on our memory hierarchy
        if op == 'R' or op == 'R2':
            logger.info(str(current_step) + ':\tReading ' + address + ' ' + op)
            if op == 'R2':
                l = l1_c2
            else:
                l = l1
            r, _, _, _ = l.read(address, current_step)
            logger.warning('\thit_list: ' + pprint.pformat(r.hit_list) + '\ttime: ' + str(r.time) + '\n')
            responses.append(r)
        elif op == 'RL' or op == 'RL2':      # pl cache lock cacheline
            assert(l1.rep_policy == plru_pl_policy) # must be pl cache 
            # multilcore not implemented
            assert(op == 'RL')
            logger.info(str(current_step) + ':\tReading ' + address + ' ' + op)
            r, _, _, _ = l1.read(address, current_step, pl_opt = PL_LOCK )
            logger.warning('\thit_list: ' + pprint.pformat(r.hit_list) + '\ttime: ' + str(r.time) + '\n')
            responses.append(r)
        elif op == 'RU' or op == 'RU2':      # pl cache unlock cacheline
            assert(l1.rep_policy == plru_pl_policy)
            # multilcore not implemented
            assert(op == 'RU')
            logger.info(str(current_step) + ':\tReading ' + address + ' ' + op)
            r, _, _, _ = l1.read(address, current_step, pl_opt = PL_UNLOCK )
            logger.warning('\thit_list: ' + pprint.pformat(r.hit_list) + '\ttime: ' + str(r.time) + '\n')
            responses.append(r)
        #Call write
        elif op == 'W' or op == 'W2':
            # multilcore not implemented
            #assert(op == 'W')
            logger.info(str(current_step) + ':\tWriting ' + address + ' ' + op)
            r, _, _= l1.write(address, True, current_step)
            logger.warning('\thit_list: ' + pprint.pformat(r.hit_list) + '\ttime: ' + str(r.time) + '\n')
            responses.append(r)
        #Call cflush
        elif op == 'F' or op == 'F2':
            ## multilcore not implemented
            #assert(op == 'F')
            logger.info(str(current_step) + ':\tFlushing ' + address + ' ' + op)
            r, _, _ = l1.cflush(address, current_step)
            #logger.warning('\thit_list: ' + pprint.pformat(r.hit_list) + '\ttime: ' + str(r.time) + '\n')            
        else:
            raise InvalidOpError
        
        #if result_file != '':
        # print the trace 
        print(address + ' ' + str(r.time), file = f )
        for cache in hierarchy:
            if hierarchy[cache].next_level:
                print_cache(hierarchy[cache])
    
    logger.info('Simulation complete')
    analyze_results(hierarchy, responses, logger)

def analyze_results(hierarchy, responses, logger):
    #Parse all the responses from the simulation
    n_instructions = len(responses)

    total_time = 0
    for r in responses:
        total_time += r.time
    logger.info('\nNumber of instructions: ' + str(n_instructions))
    logger.info('\nTotal cycles taken: ' + str(total_time) + '\n')

    amat = compute_amat(hierarchy['cache_1'], responses, logger)
    logger.info('\nAMATs:\n'+pprint.pformat(amat))

def compute_amat(level, responses, logger, results={}):
    #Check if this is main memory
    #Main memory has a non-variable hit time
    if not level.next_level:
        results[level.name] = level.hit_time
    else:
        #Find out how many times this level of cache was accessed
        #And how many of those accesses were misses
        n_miss = 0
        n_access = 0
        for r in responses:
            if level.name in r.hit_list.keys():
                n_access += 1
                if r.hit_list[level.name] == False:
                    n_miss += 1

        if n_access > 0:
            miss_rate = float(n_miss)/n_access
            #Recursively compute the AMAT of this level of cache by computing
            #the AMAT of lower levels
            results[level.name] = level.hit_time + miss_rate * compute_amat(level.next_level, responses, logger)[level.next_level.name] #wat
        else:
            results[level.name] = 0 * compute_amat(level.next_level, responses, logger)[level.next_level.name] #trust me, this is good

        logger.info(level.name)
        logger.info('\tNumber of accesses: ' + str(n_access))
        logger.info('\tNumber of hits: ' + str(n_access - n_miss))
        logger.info('\tNumber of misses: ' + str(n_miss))
    return results


def build_hierarchy(configs, logger):
    #Build the cache hierarchy with the given configuration
    hierarchy = {}
    #Main memory is required
    main_memory = build_cache(configs, 'mem', None, logger)
    prev_level = main_memory
    hierarchy['mem'] = main_memory
    if 'cache_3' in configs.keys():
        cache_3 = build_cache(configs, 'cache_3', prev_level, logger)
        prev_level = cache_3
        hierarchy['cache_3'] = cache_3
    if 'cache_2' in configs.keys():
        cache_2 = build_cache(configs, 'cache_2', prev_level, logger)
        prev_level = cache_2
        hierarchy['cache_2'] = cache_2
    if 'cache_1_core_2' in configs.keys():
        cache_1_core_2 = build_cache(configs, 'cache_1_core_2', prev_level, logger)
        prev_level = cache_2
        hierarchy['cache_1_core_2'] = cache_1_core_2        
    #Cache_1 is required
    cache_1 = build_cache(configs, 'cache_1', prev_level, logger)
    if 'cache_1_core_2' in configs.keys(): 
        cache_1.add_same_level_cache(cache_1_core_2)
        cache_1_core_2.add_same_level_cache(cache_1)
    #print(len(cache_1.same_level_caches))
    hierarchy['cache_1'] = cache_1
    return hierarchy

def build_cache(configs, name, next_level_cache, logger):
    return cache.Cache(name,
                configs['architecture']['word_size'],
                configs['architecture']['block_size'],
                configs[name]['blocks'] if (name != 'mem') else -1,
                configs[name]['associativity'] if (name != 'mem') else -1,
                configs[name]['hit_time'],
                configs[name]['hit_time'],
                configs['architecture']['write_back'],
                logger,
                next_level_cache,
                rep_policy = configs[name]['rep_policy'] if 'rep_policy' in configs[name] else '',
                prefetcher = configs[name]['prefetcher'] if 'prefetcher' in configs[name] else "none",
                verbose = configs['verbose'] if 'verbose' in configs else 'False' )


if __name__ == '__main__':
    main()
