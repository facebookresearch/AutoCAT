#!/bin/bash
cd spectre_stream
make
cd ..
cd covert_channel_LRU_1thread_8way
make 
cd ..
cd covert_channel_stream_1thread_2bits_8way
make
cd ..
cd process_error_rate_1thread
make
cd ..