# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed according to the terms of the
# GNU General Public License version 2.

# run unit test
../../src/cache_simulator.py -pdc $1/config -t $1/test.txt -f $1/result.txt

if diff -u "$1/result.txt" "$1/expected_result.txt" ; then
    echo "$1 test passed! "
else
    echo "$1 test failed!" 
    : 
fi