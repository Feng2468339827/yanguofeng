import pandas as pd
import matplotlib.pyplot as plt
import xlrd
import numpy as np
import xlwt

data3 = pd.read_excel('data\DTOMDRL-20-a-0.001-c-0.001-da-0.0001-dc-0.001.xls', sheet_name='sac奖励值')
data4 = pd.read_excel('data\DTOMDRL-20-a-0.0001-c-0.001-da-0.0001-dc-0.001.xls', sheet_name='sac奖励值')
data5 = pd.read_excel('data\DTOMDRL-20-a-1e-05-c-0.001-da-0.0001-dc-0.001.xls', sheet_name='sac奖励值')

# 写入表格
# book = xlwt.Workbook(encoding='utf-8', style_compression=0)
# sheet_sac = book.add_sheet('奖励值', cell_overwrite_ok=True)
# col = ('episode', '0.01', '0.001', '0.0001')
# for i in range(0, 4):
#     sheet_sac.write(0, i, col[i])
# savepath = 'E:\python\pytorch\VEC-distributed-11.2\VEC-astringency\data' \
#                '\\reward-a-0.0001-test.xls'

sac_reward_3 = []
sac_reward_4 = []
sac_reward_5 = []
episodes = []
sac_reward_3.append(data3['ave_reward'][0])
sac_reward_4.append(data4['ave_reward'][0])
sac_reward_5.append(data5['ave_reward'][0])
episodes.append(1)
j = 1
for i in range(1, len(data5['episode']) + 1):
    if i % 5 == 0:
        sac_reward_3.append(data3['ave_reward'][i-1])
        sac_reward_4.append(data4['ave_reward'][i-1])
        sac_reward_5.append(data5['ave_reward'][i-1])
        episodes.append(i)
        # sheet_sac.write(j, 0, i)
        # sheet_sac.write(j, 1, data3['ave_reward'][i-1])
        # sheet_sac.write(j, 2, data4['ave_reward'][i-1])
        # sheet_sac.write(j, 3, data5['ave_reward'][i-1])
        # j = j + 1
        # book.save(savepath)


plt.plot(episodes, sac_reward_3, label='lr-A=0.0001 lr-c=0.01', color="red")
plt.plot(episodes, sac_reward_4, label='lr-A=0.0001 lr-c=0.001')
plt.plot(episodes, sac_reward_5, label='lr-A=0.0001 lr-c=0.0001', color="green")


plt.legend(loc="best")
plt.xticks(np.arange(0, 1501, 300))
plt.yticks(np.arange(0.1, 1.0, 0.1))
plt.xlim(0, 1500)
plt.xlabel('Episodes')
plt.ylabel('Average Reward per Episode')
plt.show()
