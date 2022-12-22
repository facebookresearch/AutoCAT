from cProfile import label
from tkinter import font
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import numpy as np
import matplotlib.pyplot as plt


fontaxes = {
    'family': 'Arial',
        'color':  'black',
        'weight': 'bold',
        'size': 8,
}
fontaxes_title = {
    'family': 'Arial',
        'color':  'black',
        'weight': 'bold',
        'size': 9,
}


lsmarkersize = 2.5
lslinewidth = 0.6
plt.figure(num=None, figsize=(3.5, 1.5), dpi=200, facecolor='w')
plt.subplots_adjust(right = 0.99, top =0.90, bottom=0.24, left=0.15, wspace=0.2, hspace=0.2)  

# Without CCHunter, generated with  python sample_cchunter.py checkpoint=/media/research/yl3469/RLSCA/CacheSimulator/data/guess97_cchunter100/ppo_agent-53.pth 
# trace4 = [0, 1, 0, 2, 1, 0, 0, 0, 0, 0, 1, 2, 1, 2]
# trace3 = [0, 0, 2, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 1, 2, 0, 0, 0, 1, 2, 2, 0, 0, 0, 0, 0, 2]
# trace2 = [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 2]
# trace1 = [1, 2, 1, 2, 0, 0, 0, 1, 2, 0, 1, 2]


# trace4= [1, 1, 2, 0, 1, 2, 0, 2, 1, 2, 0, 2, 0, 0, 1, 2, 1, 2, 0, 2, 0, 1, 2, 0, 0, 2, 1, 2, 0, 0, 1, 2, 1, 2, 0, 2, 0, 1, 2, 1, 2, 0, 2, 1, 2, 1, 2, 0, 0, 0, 0, 2, 0, 2, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 1, 2, 1, 2, 1, 2, 0, 2, 0, 1, 2, 0, 0, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2]
# trace3= [1, 0, 2, 0, 0, 0, 2, 0, 2, 0, 1, 2, 1, 2, 0, 2, 1, 2, 1, 2, 0, 0, 1, 2, 0, 2, 0, 2, 0, 1, 2, 0, 0, 2, 1, 2, 0, 1, 2, 1, 2, 0, 2, 0, 0, 0, 2, 1, 0, 2, 0, 0, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 0, 2, 0, 0, 1, 2, 0, 2, 1, 2, 0, 1, 2, 2, 0, 0, 2]
# trace2= [1, 0, 2, 1, 2, 0, 2, 1, 2, 0, 2, 0, 1, 2, 0, 1, 2, 1, 2, 1, 2, 1, 2, 0, 2, 0, 1, 2, 1, 2, 1, 2, 0, 0, 0, 2, 0, 2, 0, 0, 0, 1, 2, 1, 2, 0, 2, 0, 0, 0, 1, 2, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 2, 0, 0, 2, 0, 0, 1, 2, 1, 2, 1, 2, 0, 1, 2, 2, 0, 1, 2]
# trace1= [1, 0, 2, 0, 0, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 1, 2, 0, 0, 1, 2, 1, 2, 0, 2, 1, 2, 0, 2, 0, 1, 2, 0, 0, 2, 1, 2, 0, 0, 0, 1, 2, 1, 2, 1, 2, 0, 0, 0, 1, 2, 0, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 2, 0, 0, 2, 1, 2, 0, 2, 0, 0, 0, 1, 2, 0, 1, 2, 1, 2]

# Without CCHUnter, generated with python sample_cchunter.py c
# heckpoint=/media/research/yl3469/RLSCA/CacheSimulator/data/guess99_cchunter100/ppo_agent-338.pth

trace4 = [1, 0, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 1, 2, 0, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 2, 0, 0, 1, 2, 1, 2, 0, 2, 0, 1, 2, 2, 1, 2, 0, 2, 0, 1, 2, 2, 1, 2, 1, 2, 1, 2, 0, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 2, 1, 2, 0, 2, 0, 1, 2, 0, 1, 2, 2, 0, 2]
trace3 = [1, 0, 2, 0, 0, 0, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 2, 0, 1, 2, 0, 1, 2, 1, 2, 0, 2, 1, 2, 0, 2, 0, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 2, 1, 2, 1, 0, 2, 1, 2, 0, 2, 0, 1, 2, 0, 1, 2, 0, 2, 0, 2, 0, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 1, 2]
trace2 = [1, 1, 2, 1, 2, 0, 2, 0, 1, 2, 1, 2, 0, 2, 0, 1, 2, 0, 1, 2, 0, 2, 0, 2, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 1, 2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 2, 0, 0, 1, 2, 1, 2, 0, 2, 1, 2, 0, 2, 0, 1, 2, 2, 0, 1, 2, 1, 2, 1, 0, 2, 1, 2, 0, 2, 1, 2]
trace1 = [0, 1, 2, 0, 0, 1, 2, 1, 2, 0, 2, 0, 0, 1, 2, 1, 2, 0, 2, 0, 0, 1, 2, 1, 2, 0, 2, 1, 2, 0, 2, 1, 2, 1, 2, 1, 2, 0, 2, 1, 2, 0, 2, 0, 1, 2, 1, 2, 0, 2, 0, 0, 1, 2, 1, 2, 0, 2, 1, 2, 0, 2, 0, 0, 1, 2, 0, 2, 1, 2, 1, 2, 0, 2, 0, 0, 1, 2, 2, 0, 2]

# With CCHUnter, generated with python sample_cchunter.py checkpoint=/media/research/yl3469/RLSCA/CacheSimulator/data/guess95_cchunter0/ppo_agent-699.pth
# trace4= [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2]
# trace3= [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2]
# trace2= [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2]
# trace1= [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2]

# With CCHUnter, nondeterministic, generated with python sample_cchunter.py checkpoint=/media/research/yl3469/RLSCA/CacheSimulator/data/guess95_cchunter0/ppo_agent-699.pth
# trace4 = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2]
# trace3 = [1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2]
# trace2 = [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2]
# trace1 = [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2]
# trace = trace1 + trace2 + trace3 + trace4
trace = trace1


# With CCHunter
# ctrace4 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2]
# ctrace3 = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 2]
# ctrace2 = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 2]
# ctrace1 = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 2]

# With CCHunter
# ctrace4 = [1, 2, 0, 1, 2, 0, 1, 2, 1, 2, 0, 2, 0, 0, 1, 2, 0, 2, 1, 2, 0, 1, 2, 1, 2, 1, 2, 1, 2, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 2, 0, 1, 2, 1, 2, 1, 2, 0, 0, 1, 2, 1, 2, 2, 0, 1, 2, 1, 2, 2, 0, 0, 1, 2, 1, 2, 2, 1, 2, 2, 0, 1, 2, 2, 1, 0, 1, 2, 2, 0, 0, 0, 2, 2, 2, 2, 1, 0, 0, 2]
# ctrace3 = [1, 2, 1, 2, 0, 2, 0, 1, 2, 1, 2, 1, 2, 1, 2, 0, 2, 0, 0, 1, 2, 0, 2, 0, 2, 0, 1, 2, 1, 2, 2, 0, 0, 0, 1, 2, 1, 2, 0, 2, 0, 0, 1, 2, 0, 2, 2, 1, 2, 1, 2, 2, 1, 2, 1, 2, 2, 0, 0, 1, 2, 0, 0, 2, 2, 0, 0, 1, 2, 2, 1, 0, 2, 2, 1, 0, 0, 2, 2, 1, 0, 0, 2, 2, 2, 2, 1, 0, 2, 2]
# ctrace2 = [1, 2, 0, 0, 0, 2, 0, 2, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 2, 0, 2, 0, 0, 1, 2, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 2, 1, 2, 0, 1, 2, 0, 0, 2, 2, 1, 2, 1, 2, 0, 2, 1, 2, 2, 1, 2, 0, 1, 2, 1, 2, 2, 0, 1, 2, 1, 2, 2, 0, 1, 2, 2, 0, 0, 1, 2, 2, 1, 0, 0, 2, 2, 2, 2, 0, 1, 2, 2]
# ctrace1 = [1, 2, 0, 1, 2, 1, 2, 0, 2, 0, 0, 0, 2, 0, 2, 0, 0, 1, 2, 0, 2, 1, 2, 0, 0, 1, 2, 0, 2, 2, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 1, 2, 0, 2, 0, 0, 1, 2, 1, 2, 2, 0, 0, 1, 2, 2, 1, 2, 2, 1, 2, 1, 2, 2, 0, 1, 2, 1, 0, 2, 2, 0, 0, 1, 2, 2, 0, 0, 0, 2, 2, 2, 2, 0, 0, 0, 2, 2]


# [0, 0, 2, 0, 0, 1, 2, 1, 2, 0, 2, 0, 0, 1, 2, 0, 2, 1, 2, 1, 2, 1, 2, 0, 0, 1, 2, 1, 2, 1, 2, 0, 0, 0, 2, 0, 2, 0, 1, 2, 2, 0, 1, 2, 1, 2, 1, 2, 1, 2, 0, 0, 2, 0, 0, 0, 1, 2, 1, 2, 0, 2, 0, 1, 2, 2, 0, 1, 2, 1, 2, 0, 0, 2, 0, 0, 1, 2, 2, 0, 2]

# ctrace = ctrace1 + ctrace2 + ctrace3 + ctrace4

# With CCHunter, deterministic, 0.13333333333333333: python sample_cchunter.py checkpoint=/media/research/yl3469/RLSCA/CacheSimulator/data/guess99_cchunter11/ppo_agent-458.pth
ctrace4 = [1, 2, 0, 1, 2, 0, 1, 2, 1, 2, 0, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 2, 0, 1, 2, 0, 1, 2, 0, 2, 0, 2, 0, 1, 2, 1, 2, 1, 2, 1, 2, 2, 0, 1, 2, 1, 2, 2, 0, 1, 2, 2, 1, 2, 2, 1, 2, 1, 2, 2, 0, 1, 2, 2, 1, 2, 1, 0, 1, 2, 2, 1, 0, 0, 2, 2, 2, 2, 1, 0, 0, 2]
ctrace3 = [0, 2, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 1, 2, 0, 0, 0, 0, 2, 2, 1, 2, 1, 2, 0, 2, 1, 2, 1, 2, 0, 0, 0, 2, 2, 1, 2, 0, 1, 2, 1, 2, 2, 1, 2, 0, 1, 2, 1, 2, 2, 1, 2, 1, 2, 2, 2, 1, 1, 2, 2, 1, 0, 0, 2, 2, 1, 0, 0, 2, 2, 2, 2, 1, 0, 2, 2]
ctrace2 = [0, 2, 0, 0, 1, 2, 0, 2, 0, 2, 0, 1, 2, 1, 2, 0, 2, 0, 1, 2, 1, 2, 0, 2, 0, 1, 2, 1, 2, 2, 1, 2, 1, 2, 1, 2, 0, 0, 0, 2, 1, 2, 0, 1, 2, 0, 0, 2, 2, 1, 2, 0, 1, 2, 0, 1, 2, 1, 2, 0, 2, 1, 2, 2, 0, 1, 2, 0, 1, 2, 1, 0, 2, 2, 1, 0, 0, 2, 2, 0, 0, 0, 2, 2, 2, 2, 1, 0, 0, 2]
ctrace1 = [0, 2, 0, 1, 2, 1, 2, 0, 2, 0, 1, 2, 0, 1, 2, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 2, 0, 2, 2, 0, 0, 1, 2, 1, 2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 0, 0, 2, 2, 1, 2, 1, 2, 2, 0, 1, 2, 0, 0, 2, 2, 0, 0, 1, 2, 2, 1, 0, 2, 2, 1, 0, 1, 2, 2, 1, 0, 0, 2, 2, 2, 2, 0, 0, 1, 2]

ctrace = ctrace1

mask = [i != 2 for i in trace] 
trace_lean = [i for i, v in zip(trace, mask) if v]

mask = [i != 2 for i in ctrace] 
ctrace_lean = [i for i, v in zip(ctrace, mask) if v]

def calculate_autocorrelation_coefficients(x, lags):
    """
    Calculate the autocorrelation coefficients for the given data and lags.
    """
    # n = len(x)
    series = pd.Series([i[0] for i in x])
    # print("Series is:\n", series)
    # print("series correlation:\n",series.autocorr())
    # data = np.asarray(x)
    # print(data)
    # x_mean = np.mean(data)
    # y_mean = np.mean(data)
    # rho = np.zeros(lags)
    # for lag in range(0, lags):
    #     x_m = data[:-lag]
    #     y_m = data[lag:]
    #     x_m -= x_mean
    #     y_m -= y_mean
    #     rho[lag] = np.sum(x_m * y_m) / (n - lag)
    return series.autocorr(lags)

def autocorrelation_plot_forked(series, ax=None, n_lags=None, change_deno=False, change_core=False, **kwds):
    """
    Autocorrelation plot for time series.
    Parameters:
    -----------
    series: Time series
    ax: Matplotlib axis object, optional
    n_lags: maximum number of lags to show. Default is len(series)
    kwds : keywords
        Options to pass to matplotlib plotting method
    Returns:
    -----------
    class:`matplotlib.axis.Axes`
    """
    import matplotlib.pyplot as plt
    
    n_full = len(series)
    if n_full <= 2:
      raise ValueError("""len(series) = %i but should be > 2
      to maintain at least 2 points of intersection when autocorrelating
      with lags"""%n_full)
      
    # Calculate the maximum number of lags permissible
    # Subtract 2 to keep at least 2 points of intersection,
    # otherwise pandas.Series.autocorr will throw a warning about insufficient
    # degrees of freedom
    n_maxlags = n_full - 2
    
    
    # calculate the actual number of lags
    if n_lags is None:
      # Choosing a reasonable number of lags varies between datasets,
      # but if the data longer than 200 points, limit this to 100 lags as a
      # reasonable default for plotting when n_lags is not specified
      n_lags = min(n_maxlags, 100)
    else:
      if n_lags > n_maxlags:
        raise ValueError("n_lags should be < %i (i.e. len(series)-2)"%n_maxlags)
    
    if ax is None:
        ax = plt.gca(xlim=(1, n_lags), ylim=(-1.0, 1.0), label=label)

    if not change_core:
      data = np.asarray(series)
      mean = np.mean(data)
      c0 = np.sum((data - mean) ** 2) / float(n_full)
      def r(h):
          deno = n_full if not change_deno else (n_full - h)
          return ((data[:n_full - h] - mean) *
                  (data[h:] - mean)).sum() / float(deno) / c0
    else:
      def r(h):
        return series.autocorr(lag=h)
      
    x = np.arange(n_lags) + 1
    # y = lmap(r, x)
    y = np.array([r(xi) for xi in x])
    z95 = 1.959963984540054
    z99 = 2.5758293035489004
    ax.axhline(y=0.8, linestyle='--', color='grey')
    # ax.axhline(y=z95 / np.sqrt(n_full), color='grey')
    ax.axhline(y=0.0, color='black')
    # ax.axhline(y=-z95 / np.sqrt(n_full), color='grey')
    # ax.axhline(y=-z99 / np.sqrt(n_full), linestyle='--', color='grey')

    ax.plot(x, y, **kwds)
    if 'label' in kwds:
        ax.legend()
    ax.grid()
    
    return ax



data = pd.Series(trace)
cdata = pd.Series(ctrace)
# data = pd.Series(trace_lean)
# cdata = pd.Series(ctrace_lean)

ax = autocorrelation_plot_forked(data, n_lags=len(data)-2, change_deno=True, label='Baseline')
autocorrelation_plot_forked(cdata, ax = ax, n_lags=len(cdata)-2, change_deno=True, label='With Autocorrelation Based Detection Penalty')

plt.tick_params(labelsize=6)
ax.set_xlabel("Lag (p)",fontdict = fontaxes)
ax.set_ylabel("Autocorrelation",fontdict = fontaxes)
ax.legend(prop={'size': 6})

plt.savefig('cchunter_hit_trace_{}_acf.pdf'.format(0))