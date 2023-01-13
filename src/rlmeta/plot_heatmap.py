# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed according to the terms of the
# GNU General Public License version 2.

import seaborn as sns

data=[[2, 6, 1, 1, 1, 2, 1, 1, 9, 5, 1, 4, 1, 8, 0, 2, 2, 0, 6, 1, 2, 2, 0, 0, 1, 2, 1, 1, 2, 4, 3, 3, 1, 0, 1, 2, 0, 3, 2, 1], [2, 2, 1, 2, 2, 1, 1, 1, 0, 1, 3, 2, 1, 0, 5, 1, 1, 0, 1, 1, 0, 3, 1, 5, 2, 5, 0, 3, 1, 0, 1, 1, 2, 4, 4, 1, 3, 0, 1, 2], [1, 0, 4, 1, 2, 0, 6, 4, 2, 1, 4, 1, 3, 1, 7, 3, 1, 7, 2, 4, 5, 1, 3, 2, 1, 3, 4, 1, 1, 1, 6, 5, 3, 1, 4, 2, 2, 2, 1, 1], [1, 1, 1, 4, 2, 4, 1, 2, 0, 1, 1, 0, 1, 1, 0, 1, 2, 2, 0, 3, 2, 0, 6, 1, 3, 0, 3, 2, 2, 2, 0, 1, 1, 3, 0, 3, 3, 6, 3, 4]]

p=sns.heatmap(self.cyclone_heatmap, vmin=0, vmax=20)
p.set_xlabel('Time intervals (40 cycles)')
p.set_ylabel('Set index')
fig= p.get_figure()
fig.set_size_inches(3, 3)
