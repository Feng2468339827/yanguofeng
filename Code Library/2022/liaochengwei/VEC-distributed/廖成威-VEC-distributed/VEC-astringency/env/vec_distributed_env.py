
import random
import numpy as np
from matplotlib import pyplot as plt
from .transmission_rate import calculate_data_rate
from .task_generate_time import Task_arrivals

rsu_coverage = 300  # RSU覆盖范围
d_max = 200  # 车辆最大通信范围
channel_bandwidth = 30  # 20
power = 600  # mw 600

vehicle_num_TV = 30  # 任务车辆数 20 10-30
vehicle_num_SV = 10  # 服务车辆数 10
sub_task_num = 10  # 子任务个数  10

date_rate_sum = calculate_data_rate(rsu_coverage, power, channel_bandwidth)
data_rate_V2I = date_rate_sum / vehicle_num_TV  # V2I传输速率
data_rate_I2V = date_rate_sum / vehicle_num_SV  # V2V传输速率
vehicle_pow = 600  # 车辆传输功率  600mW

time_cs = 0.05  # 单位大小数据从RSU传输到云服务器的时延 0.05

F_mec = 15 * 1e9  # mec服务器总资源
F_SV = 2 * 1e9  # SV总资源
F_TV = 1 * 1e9  # TV总资源
F_cs = 25 * 1e9  # CS总资源

F_TVs = []

task_queue_mec = []  # mec上的队列，[[任务到达时间,任务计算时间]]
task_queue_cs = []   # cs上的队列，[[任务到达时间,任务计算时间]]
task_queue_svs = []  # svs上的队列，[[[任务到达时间,任务计算时间],[任务到达时间,任务计算时间]],[],[]]
svs_queue_time = []  # SVs的队列时间


subtask_messages = {}  # 使用字典存储子任务信息

state_space = []
state_space1 = []
"""
状态空间
    [  [c1,c2,...],[d1,d2,...],[pred,succ],[车辆前一时隙卸载决策]]  
    [   0,              1,         2,                3        ]
"""
"""
动作空间
[LOC,SV1,SV2,...,SV10,MEC,CS]
[0,1,2,...,10,11,12][0-12]
"""
action_space = []
for i in range(vehicle_num_SV + 3):
    action_space.append(i)


class Vec_env():
    def __init__(self, T):
        super(Vec_env, self).__init__()

        self.T = T
        self.vehicle_num_TV = vehicle_num_TV  # 任务车辆数
        self.vehicle_num_SV = vehicle_num_SV  # 服务车辆数
        self.sub_task_num = sub_task_num  # 子任务个数

        self.data_rate_V2I = data_rate_V2I  # V2I传输速率
        self.data_rate_V2V = data_rate_I2V  # I2V传输速率
        self.vehicle_pow = vehicle_pow  # 车辆传输功率
        self.F_mec = F_mec  # mec服务器总资源
        self.F_SV = F_SV  # SV可用资源
        self.F_TV = F_TV  # TV可用资源
        self.F_TVs = F_TVs  # 所有TV可用资源集合
        self.F_cs = F_cs  # CS总资源

        self.cal = Calculate_proioritiy()  # 初始化子任务优先级计算算法
        self.subtask_messages = subtask_messages  # 存储子任务信息，包括任务序号，任务大小、所需CPU转数、最大可容忍时延
        self.subtask_queue = []  # 存储任务执行顺序
        self.decides = []  # 存储前一时隙TV的卸载决策
        self.sub_decides = []  # 存储前一时隙某个TV子任务的卸载决策

        self.task_queue_mec = task_queue_mec  # 构建mec上计算的任务的队列，[任务到达时间，任务计算时间]
        self.task_queue_cs = task_queue_cs    # 构建cs上计算的任务的队列，[任务到达时间，任务计算时间]
        self.task_queue_svs = task_queue_svs  # svs上的队列,[[[任务到达时间,任务计算时间],[任务到达时间,任务计算时间]],[],[]]
        self.SV_index = 0  # 记录要卸载的SV编号

        # 生成任务的到达时间
        self.Task_arrivals = Task_arrivals()
        self.arrival_times = []
        # 队列时间
        self.mec_queue_time = 0  # mec上的队列时间
        self.cs_queue_time = 0  # cs上的队列时间
        self.svs_queue_time = svs_queue_time  # 所有sv上的队列时间[0,0,0,0...]
        # 状态空间
        self.state_space = state_space  # VEC环境中使用
        self.state_space1 = state_space1  # DRL算法中使用
        # 动作空间
        self.action_space = action_space

        self.subtask_count = 1  # 子任务个数计数
        self.TV_count = 1  # TV个数计数

        self.cost_time_loc = 0
        self.cost_time = 0
        self.task_max_time = 0
        self.task_count_success = 0

    # 随机生成任务,所需的cpu转数、任务数据大小、任务最大可容忍时延[c,d,t]
    def task_generate(self, ):
        subtask_cycle = round(random.uniform(0.05, 0.1), 2) * 1e9  # 子任务所需的cpu转数
        subtask_size = round(random.uniform(0.1, 0.4), 2)  # 子任务数据大小 0.1-0.5M
        subtask_max_time = random.choice([0.05, 0.1, 0.15])  # 子任务最大可容忍时延
        return subtask_cycle, subtask_size, subtask_max_time

    def cal_time(self, action_decide, F_rt, subtask_cycle, subtask_size):
        """
        计算任务完成时间
        :param action_decide: 卸载决策
        :param F_rt: 服务器计算能力
        :param subtask_cycle: 任务所需cpu周期数
        :param subtask_size: 任务大小
        :return: 完成任务的时延 传输+队列+执行
        """
        time_cost = 0
        exe_time = subtask_cycle / F_rt
        transmission_time = subtask_size / data_rate_V2I  # 传输到RSU的时间
        if action_decide == 'LOC':
            time_cost = exe_time

        elif action_decide == 'MEC':
            arrival_time = self.arrival_times[0] + transmission_time  # 任务到达时间
            self.task_queue_mec.append([arrival_time, exe_time])  # 将执行的任务加入MEC队列
            time_cost = transmission_time + exe_time + self.mec_queue_time

        elif action_decide == 'CS':
            arrival_time = self.arrival_times[0] + transmission_time + subtask_size * time_cs
            self.task_queue_cs.append([arrival_time, exe_time])  # 将执行的任务加入CS队列
            time_cost = transmission_time + exe_time + subtask_size * time_cs + self.cs_queue_time

        elif action_decide == 'V2V':
            arrival_time = self.arrival_times[0] + transmission_time + subtask_size / data_rate_I2V
            self.task_queue_svs[self.SV_index].append([arrival_time, exe_time])  # 将执行的任务加入SV队列
            time_cost = transmission_time + (subtask_size / data_rate_I2V) + exe_time + self.svs_queue_time[self.SV_index]

        return time_cost

    def cal_queue_time(self, task_queue):
        """
        计算队列时间
        :param task_queue: 任务队列[[任务到达时间,执行时间]]
        :return: 队列时延
        """
        exe_sum_time = 0
        length = len(task_queue)
        if length == 0:
            return 0, 0, 0
        for row in range(length - 1):
            exe_sum_time += task_queue[row][1]
        queue_time = exe_sum_time - task_queue[length - 1][0]
        if queue_time < 0:
            queue_time = 0
        return queue_time, exe_sum_time, task_queue[length - 1][0]

    # 生成任务，构建任务依赖图字典和任务计算时延字典
    def task_structure_graph(self, ):
        # 任务依赖图
        task_dependencie1 = {
            '1': ['2', '3'],
            '2': ['4', '5'],
            '3': ['6', '7'],
            '4': ['8'],
            '5': ['9'],
            '6': ['9'],
            '7': ['9'],
            '8': ['10'],
            '9': ['10'],
            '10': [],
        }
        task_dependencie2 = {
            '1': ['2', '3', '4'],
            '2': ['5'],
            '3': ['5'],
            '4': ['6'],
            '5': ['7'],
            '6': ['7'],
            '7': ['9'],
            '8': ['10'],
            '9': ['10'],
            '10': [],
        }
        task_dependencie3 = {
            '1': ['2', '3', '4'],
            '2': ['5'],
            '3': ['6'],
            '4': ['6'],
            '5': ['7'],
            '6': ['7', '8'],
            '7': ['9'],
            '8': ['10'],
            '9': [],
            '10': [],
        }
        task_dependencies = [task_dependencie1, task_dependencie2, task_dependencie3]
        task_dependencie = random.choice(task_dependencies)
        task_delays = {}
        max_time = []
        for i in range(1, 11):
            # 生成子任务
            subtask_cycle, subtask_size, subtask_max_time = self.task_generate()
            self.subtask_messages[f'{i}'] = [subtask_cycle, subtask_size, subtask_max_time]
            max_time.append(self.subtask_messages[f'{i}'][2])
            # 计算任务的最小时延
            delay_loc = self.cal_time('LOC', self.F_TV, subtask_cycle, subtask_size)
            delay_mec = self.cal_time('MEC', self.F_mec, subtask_cycle, subtask_size)
            delay_cs = self.cal_time('CS', self.F_cs, subtask_cycle, subtask_size)
            delay_v2v = self.cal_time('V2V', self.F_SV, subtask_cycle, subtask_size)
            delay = min(delay_loc, delay_mec, delay_cs, delay_v2v)
            task_delays[f'{i}'] = delay
        self.task_max_time = sum(max_time)
        return task_dependencie, task_delays, subtask_messages

    """
    任务优先级： ['1', '3', '4', '2', '5', '6', '8', '7', '10', '9']
    任务依赖关系： {'1': ['2', '3', '4'], '2': ['5'], '3': ['6'], '4': ['6'], '5': ['7'], 
                '6': ['7', '8'], '7': ['9'], '8': ['10'], '9': [], '10': []}
    任务计算时延： {'1': 0.0022093718843469593, '2': 0.001759222333000997, '3': 0.002448654037886341, 
                '4': 0.0021495513459621135, '5': 0.0021994017946161517, '6': 0.001978564307078764, 
                '7': 0.0021096709870388835, '8': 0.001928713858424726, '9': 0.00228913260219342, 
                '10': 0.0024985044865403786}
    子任务信息： {'1': [90000000.0, 0.33, 0.2], '2': [50000000.0, 0.39, 0.2], '3': [70000000.0, 0.34, 0.15],
                 '4': [100000000.0, 0.21, 0.1], '5': [80000000.0, 0.18, 0.05], '6': [80000000.0, 0.18, 0.1], 
                 '7': [70000000.0, 0.32, 0.15],'8': [70000000.0, 0.13, 0.2], '9': [70000000.0, 0.29, 0.1],
                  '10': [100000000.0, 0.24, 0.15]}
    0.21
    """

    # 初始化环境,输出当前state,mec总资源，cs总资源
    def reset(self):
        self.state_space.clear()  # 重置状态空间
        self.arrival_times = self.Task_arrivals.task_arrivals(self.vehicle_num_TV * self.T + 1, self.T).tolist()
        self.task_queue_mec.clear()  # 重置在mec队列
        self.task_queue_cs.clear()  # 重置cs队列
        self.task_queue_svs.clear()
        self.mec_queue_time = 0
        self.cs_queue_time = 0
        self.svs_queue_time.clear()  # 清空SVs队列时间
        self.subtask_messages.clear()  # 重置存储的子任务信
        self.subtask_queue.clear()  # 重置存储任务执行顺序
        self.task_count_success = 0  # 重置任务成功个数

        for j in range(vehicle_num_SV):
            self.task_queue_svs.append([])
            svs_queue_time.append(0)

        self.F_TVs = []
        for m in range(vehicle_num_TV):
            self.F_TVs.append(F_TV)
        # 生成任务信息
        task_dependencies, task_delays, subtask_messages = self.task_structure_graph()

        # 计算任务优先级
        priorities = {}
        priorities = self.cal.priority_result(task_dependencies, task_delays, priorities)
        # 得到任务序列
        self.subtask_queue = sorted(priorities, key=priorities.get, reverse=True)
        subtask_cycle = []
        subtask_data = []
        for index in range(10):
            subtask_cycle.append(subtask_messages[f'{self.subtask_queue[index]}'][0])
            subtask_data.append(subtask_messages[f'{self.subtask_queue[index]}'][1])
        self.state_space.append(subtask_cycle)  # 任务所需cpu周期数c  0
        self.state_space.append(subtask_data)  # 任务大小d  1

        pred_succ = []
        for j in range(1, self.sub_task_num):
            pred_succ.append(self.subtask_queue[j])
        pred_succ = [int(x) for x in pred_succ]
        self.state_space.append(pred_succ)  # 子任务前驱+后继  2

        time_queue_m_c = [0, 0]
        self.state_space.append(time_queue_m_c)  # mec和cs的队列时延 3

        time_queue_svs = [0 for index in range(self.vehicle_num_SV)]
        self.state_space.append(time_queue_svs)   # SVs的队列时延  # 4

        # 初始化TV前一时隙的卸载决策,10*vehicle_num_TV,默认为0，本地计算
        decide = []
        for i in range(self.sub_task_num * vehicle_num_TV):
            decide.append(0)
        self.state_space.append(decide)  # 前一时隙卸载决策 5
        self.state_space1 = sum(self.state_space, [])
        return self.state_space1

    """
    状态空间
    [  [c1,c2,...],[d1,d2,...],[pred,succ],[mec_q,cs_q],[svs_q,...],[车辆前一时隙卸载决策]]  
    [   0,              1,         2,              3,           4,          5 ]
    """
    """
    动作空间
    [LOC,SV1,SV2,...,SV10,MEC,CS]
    [0,1,2,...,10,11,12][0-12]
    """

    # 执行决策，输入action,输出reward，next_state
    def step(self, action):
        cost_time = 0
        done = False
        s = self.state_space
        s_ = s  # 存储下一个状态
        self.sub_decides.append(action)
        index = self.subtask_count - 1
        cost_time_loc = self.cal_time('LOC', self.F_TV, s[0][index], s[1][index])
        self.cost_time_loc += cost_time_loc
        if action == 0:  # 本地计算 action_decide
            cost_time = self.cal_time('LOC', self.F_TV, s[0][index], s[1][index])

        elif action == 11:  # mec计算
            cost_time = self.cal_time('MEC', self.F_mec, s[0][index], s[1][index])

        elif action == 12:  # cs计算
            cost_time = self.cal_time('CS', self.F_cs, s[0][index], s[1][index])

        elif action in range(1, 11):  # SV计算
            self.SV_index = action - 1
            cost_time = self.cal_time('V2V', self.F_SV, s[0][index], s[1][index])

        self.cost_time += cost_time
        reward = (self.cost_time_loc - self.cost_time) / self.cost_time_loc
        if reward < -1:
            reward = -1
        # 生成任务图
        if self.subtask_count == self.sub_task_num:  # 10个子任务已决策完毕
            if self.cost_time <= self.task_max_time:  # 统计任务成功的个数
                self.task_count_success += 1

            self.subtask_count = 0

            # 重置任务时间
            self.cost_time = 0
            self.cost_time_loc = 0
            self.task_max_time = 0
            # 更新任务到达时间
            self.arrival_times.remove(self.arrival_times[0])
            # 更新MEC、CS服务器上的队列时间
            self.mec_queue_time, exe_time, arr_time = self.cal_queue_time(self.task_queue_mec)
            self.cs_queue_time, exe_time, arr_time = self.cal_queue_time(self.task_queue_cs)
            s_[3] = [self.mec_queue_time, self.cs_queue_time]
            # 更新SV上的队列时间
            for SV_index in range(self.vehicle_num_SV):
                self.svs_queue_time[SV_index], exe_time, arr_time = self.cal_queue_time(self.task_queue_svs[SV_index])
                s_[4][SV_index] = self.svs_queue_time[SV_index]

            sub_decides_temp = self.sub_decides
            self.decides.append(sub_decides_temp)  # 将子任务决策加入到车辆决策列表（二维列表）
            self.TV_count += 1
            self.sub_decides = []  # 将存储的子任务决策清空
            # 生成任务信息
            task_dependencies, task_delays, subtask_messages = self.task_structure_graph()
            # 计算任务优先级
            priorities = {}
            priorities = self.cal.priority_result(task_dependencies, task_delays, priorities)
            # 得到任务序列
            self.subtask_queue = sorted(priorities, key=priorities.get, reverse=True)
            # 修改所有子任务所需的cpu周期数和大小 c,d
            subtask_cycle = []
            subtask_data = []
            for index in range(10):
                subtask_cycle.append(subtask_messages[f'{self.subtask_queue[index]}'][0])
                subtask_data.append(subtask_messages[f'{self.subtask_queue[index]}'][1])
            s_[0] = subtask_cycle  # 修改c
            s_[1] = subtask_data   # 修改d

        # 修改前驱+后继
        pred_succ = []
        for j in range(10):
            pred_succ.append(self.subtask_queue[j])
        pred_succ.remove(pred_succ[self.subtask_count])
        pred_succ = [int(x) for x in pred_succ]
        s_[2] = pred_succ

        self.subtask_count += 1

        # 修改前一时隙的卸载决策
        if self.TV_count > self.vehicle_num_TV:
            # print(self.decides)
            decide_temp = sum(self.decides, [])  # 转换成一维列表
            s_[5] = decide_temp
            self.TV_count = 1
            self.decides.clear()
            done = True

        self.state_space = s_  # 更新状态
        s_1 = sum(s_, [])  # 将下一个转换为一维列表返回到DRL算法

        return s_1, reward, done, cost_time


if __name__ == '__main__':
    env = Vec_env(T=3)
    # print(env.reset())
    cal = Calculate_proioritiy()
    priorities = {}
    env.arrival_times = env.Task_arrivals.task_arrivals(env.vehicle_num_TV * env.T + 1, env.T).tolist()
    task_dependencies, task_delays, subtask_messages = env.task_structure_graph()
    properties = cal.priority_result(task_dependencies, task_delays, priorities)
    # print(priorities)
    priorities = sorted(priorities, key=priorities.get, reverse=True)
    print("任务优先级：", priorities)
    print("任务依赖关系：", task_dependencies)
    print("任务计算时延：", task_delays)
    print("子任务信息：", subtask_messages)
    print(subtask_messages[f'{priorities[5]}'])
    print(env.arrival_times)
    print(len(env.arrival_times))
