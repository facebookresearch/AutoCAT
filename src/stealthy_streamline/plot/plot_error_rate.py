#!/usr/bin/env python
import matplotlib.pyplot as plt


fontaxes = {
    'family': 'Arial',
        'color':  'black',
        'weight': 'bold',
        'size': 11,
}
fontaxes_title = {
    'family': 'Arial',
        'color':  'black',
#        'weight': 'bold',
        'size': 10,
}


lsmarkersize = 2.5
lslinewidth = 0.6


Error_all = [[[[ 0 for i in range(5)] for j in range(5)] for k in range(2)] for l in range(4)]


Error_stat_all = [[[[ 0 for i in range(5)] for j in range(3)] for k in range(2)] for l in range(4)]

# machine order
# 0 fukushima
# 1 cornell
# 2 potato
# 3 cat
# channel order
# 0 LRU
# 1 SS
path_all = [
    ["../covert_channel_LRU_1thread_8way/test", #measurement_fukushima",
    "../covert_channel_stream_1thread_2bits_8way/test"], #measurement_8way_fukushima"],
##    ["../covert_channel_LRU_1thread_8way/measurement_core",
##    "../covert_channel_stream_1thread_2bits_8way/measurement_8way_core"],
##    ["../covert_channel_LRU_1thread/measurement_202206",
##    "../covert_channel_stream_1thread_2bits/measurement_202206"],
##    ["../covert_channel_LRU_1thread_ubuntu/measurement_Xeon", #TO DO
##    "../covert_channel_stream_1thread_2bits_ubuntu/measurement"]
]

bit_rate_ch = [[6.20606,7.6704]] #,[7.314, 8.904],[6.8267,11.378],[4.26666,7.31428]]
bit_rate_all = [[[0 for j in range(5)] for k in range(2)] for l in range(4)]

for machine_idx in range(1):
    for channel_idx in range(2):
        # read from file
        path = path_all[machine_idx][channel_idx]
        for test_idx in range(5):
            for bandwidth_idx in range (1,6):
                filename = "{}/Error_rate_{}_{}.txt".format(path,bandwidth_idx,test_idx)
                f = open(filename, "r")

                for line in f:
                    pass
                last_line = line
                error = float(line)
                Error_all[machine_idx][channel_idx][bandwidth_idx-1][test_idx] = error

        # process each bandwitch
        for i in range(5):
            max_tmp = 0
            min_tmp = 1
            avg_tmp = 0
            for j in range (5):
                print(Error_all[machine_idx][channel_idx][i][j], end=" ")
                if Error_all[machine_idx][channel_idx][i][j] > max_tmp:
                    max_tmp =  Error_all[machine_idx][channel_idx][i][j]
                if Error_all[machine_idx][channel_idx][i][j] < min_tmp:
                    min_tmp =  Error_all[machine_idx][channel_idx][i][j]
                avg_tmp = avg_tmp + Error_all[machine_idx][channel_idx][i][j]
            avg_tmp = avg_tmp / 5

            print(avg_tmp,min_tmp,max_tmp)
            Error_stat_all[machine_idx][channel_idx][0][i] = avg_tmp
            Error_stat_all[machine_idx][channel_idx][1][i] = max_tmp
            Error_stat_all[machine_idx][channel_idx][2][i] = min_tmp

            bit_rate_all[machine_idx][channel_idx][i] = bit_rate_ch[machine_idx][channel_idx]/(i+1)

for machine_idx in range(1):
    for channel_idx in range(2):
        print(bit_rate_all[machine_idx][channel_idx])
#Error_rate_stram=[[0.2177733333, 0.04370133333, 0.01709, 0.007975, 0.005696666667], [0.227539, 0.046631,0.022217,0.009277,0.006592],[0.210693, 0.041016,0.013916,0.007324,0.00415]]       
#Error_rate_LRU=[[0.1423338333,0.02587883333,0.003662,0.004801666667,0.0013835],[0.583496, 0.054199, 0.006836, 0.008789, 0.005371],[0.01416, 0.009766, 0.001465, 0.001465,0]]

for machine_idx in range(1):
    for channel_idx in range(2):
        for i in range(3):
            for j in range(5):
                Error_stat_all[machine_idx][channel_idx][i][j] = Error_stat_all[machine_idx][channel_idx][i][j]*100


#bit_rate_stream=[113.78, 56.89, 37.92666667, 28.445, 22.756]
#bit_rate_LRU=[68.267,34.1335,22.75566667,17.06675,13.6534]

plt.figure(num=None, figsize=(3.5, 2.5), dpi=300, facecolor='w')
fig,axs = plt.subplots(1, 1)

plt.subplots_adjust(right = 0.98, top =0.88, bottom=0.1,left=0.1,wspace=0.3, hspace=0.5)  
    
#fig,axs = plt.subplots(2, 2)


labels=["LRU addr_based","Stealthy Streamline"]
titles=["Xeon E5-2687W v2"] #,"Core i7-6700", "Core i5-11600K", "Xeon W-1350P"]
colors = ['b.-', 'go-']
colors_error_bar = ['b-', 'g-']
for machine_idx in range(1):
    ax=axs#[int(machine_idx/2), machine_idx%2]
    for channel_idx in range(2):
        ax.plot(Error_stat_all[machine_idx][channel_idx][0], bit_rate_all[machine_idx][channel_idx],colors[channel_idx], linewidth=1, markersize=lsmarkersize, markeredgewidth=0, label=labels[channel_idx])
        #error bar
        bar_len_y=0.2
        for i in range(5):
            ax.plot([Error_stat_all[machine_idx][channel_idx][2][i],Error_stat_all[machine_idx][channel_idx][1][i]],[bit_rate_all[machine_idx][channel_idx][i], bit_rate_all[machine_idx][channel_idx][i]], colors_error_bar[channel_idx], linewidth=0.5)
            ax.plot([Error_stat_all[machine_idx][channel_idx][2][i],Error_stat_all[machine_idx][channel_idx][2][i]],[bit_rate_all[machine_idx][channel_idx][i]-bar_len_y, bit_rate_all[machine_idx][channel_idx][i]+bar_len_y], colors_error_bar[channel_idx], linewidth=0.5)
            ax.plot([Error_stat_all[machine_idx][channel_idx][1][i],Error_stat_all[machine_idx][channel_idx][1][i]],[bit_rate_all[machine_idx][channel_idx][i]-bar_len_y, bit_rate_all[machine_idx][channel_idx][i]+bar_len_y], colors_error_bar[channel_idx], linewidth=0.5)
              
    ax.set_title(titles[machine_idx],fontdict = fontaxes_title) #plt.title('Hor. symmetric')
    ax.set_xlim([0,25])
    ax.set_ylim([0,12])
    ax.set_xlabel("Error rate (%)",fontdict = fontaxes)
    ax.set_ylabel('Bit Rate (Mbps)',fontdict = fontaxes)

    #plt.tick_params(labelsize=6)

    #plt.tight_layout()
    if machine_idx ==0:
        ax.legend(ncol=2, bbox_to_anchor=(2,1.4),prop={'size': 12})
#plt.show()
#plt.savefig('stealthy_streamline_error.pdf')  
plt.savefig('stealthy_streamline_error.png')  
