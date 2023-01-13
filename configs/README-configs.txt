CacheSimulator uses YAML for formatting the cache configuration file

To prevent any errors, use python2 as the default
otherwise you may encounter TypeError: 'dict_keys' object is not subscriptable

There are three required sections for a config file:
- architecture
- cache_1
- mem

They must be named exactly as above. Additionally, you can specify L2 and L3
caches, which must have sections titled cache_2 and cache_3, respectively.

Elements within these sections must be indented with 2 spaces.

The architecture section requires word_size, block_size, and write_back.
Word_size and block_size are in bytes, and write_back can be either 'true' or
'false'.

Cache_1 (and subsequent levels) requires blocks, associativity, and hit_time.
Blocks defines the total number of blocks in the cache. This number multiplied
with the block_size in architecture would give the total size of cache data in
bytes. Associativity defines n-way set associativity. Direct mapped is 1 and
fully associative would be equal to the number of blocks. Finally, hit_time is
in cycles.

Mem is essentially a specific case of cache. For memory, you only need
hit_time, which again is in cycles.
