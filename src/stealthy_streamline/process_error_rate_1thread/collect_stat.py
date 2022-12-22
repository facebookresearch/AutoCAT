#!/usr/bin/env python


Error_all = [[ 0 for i in range(5)] for j in range(5)]
for test_idx in range(5):
    for bandwidth_idx in range (1,6):
        filename = "Error_rate_{}_{}.txt".format(bandwidth_idx,test_idx)
        f = open(filename, "r")

        for line in f:
            pass
        last_line = line
        error = float(line)
        Error_all[bandwidth_idx-1][test_idx] = error
    

for i in range(5):
    for j in range (5):
        print(Error_all[i][j], end=" ")
    print()
