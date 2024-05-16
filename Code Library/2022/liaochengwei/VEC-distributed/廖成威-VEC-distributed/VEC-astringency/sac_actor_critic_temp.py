import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import env.env_main as env_main

from env.env_main import VecEnv

TV_num = env_main.vehicle_num_TV * env_main.num_lanes
input_size = 128 + (TV_num - 10) * 16


class PolicyNet(nn.Module):
    def __init__(self, state_dim_fc, state_dim_conv, hidden_dim, action_dim, device):
        super(PolicyNet, self).__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        # 卷积层处理的状态
        self.conv1 = nn.Conv2d(in_channels=state_dim_conv, out_channels=16, kernel_size=(3, 3))
        # 池化层
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # LSTM 层
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        # 全连接层处理的状态
        self.fc2 = nn.Linear(state_dim_fc, hidden_dim)
        # 共享的隐藏层,全连接层
        self.shared_fc = nn.Linear(hidden_dim, hidden_dim)
        # 输出层
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    # def _initialize_fc_layer(self, x_conv_size, hidden_dim):
    #     self.fc1 = nn.Linear(x_conv_size, hidden_dim)  # 创建并初始化全连接层
    #     self.fc1 = self.fc1.to(self.device)  # 将全连接层移动到 GPU
    #
    # def _initialize_lstm_layer(self, input_size, hidden_size):
    #     #  创建并初始化lstm层
    #     self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
    #     self.lstm = self.lstm.to(self.device)  # 将lstm层移动到 GPU

    def forward(self, x_fc, x_conv, hidden=None):
        # 卷积层处理
        x_conv = F.relu(self.conv1(x_conv))
        x_conv = self.pool1(x_conv)  # 添加池化层
        x_conv = x_conv.view(x_conv.size(0), -1)  # 展平卷积输出
        # LSTM 层处理
        x_conv = x_conv.unsqueeze(1)  # 增加时间步维度
        # 如果lstm层还没有初始化，进行初始化
        # if self.lstm is None:
        #     self._initialize_lstm_layer(x_conv.size(2), self.hidden_dim)

        lstm_out, hidden = self.lstm(x_conv, hidden)
        x_conv = lstm_out[:, -1, :]  # 取最后一个时间步的输出
        # 如果全连接层还没有初始化，进行初始化
        # if self.fc1 is None:
        #     self._initialize_fc_layer(x_conv.size(1), self.hidden_dim)

        x_conv = F.relu(self.fc1(x_conv))
        # 全连接层处理
        x_fc = F.relu(self.fc2(x_fc))
        # 合并卷积和全连接层的输出
        x = x_conv + x_fc  # 或者可以使用其他方式来整合
        # 共享隐藏层
        x = F.relu(self.shared_fc(x))
        # 输出动作的概率分布
        action_pro = F.softmax(self.fc3(x), dim=1)
        return action_pro


class QValueNet(nn.Module):
    def __init__(self, state_dim_fc, state_dim_conv, hidden_dim, action_dim, device):
        super(QValueNet, self).__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        # 卷积层处理的状态
        self.conv1 = nn.Conv2d(in_channels=state_dim_conv, out_channels=16, kernel_size=(3, 3))
        # 池化层
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # LSTM 层
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        # 全连接层处理的状态
        self.fc2 = nn.Linear(state_dim_fc, hidden_dim)
        # 共享的隐藏层,全连接层
        self.shared_fc = nn.Linear(hidden_dim, hidden_dim)
        # 输出层
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    # def _initialize_fc_layer(self, x_conv_size, hidden_dim):
    #     self.fc1 = nn.Linear(x_conv_size, hidden_dim)  # 创建并初始化全连接层
    #     self.fc1 = self.fc1.to(self.device)  # 将全连接层移动到 GPU
    #
    # def _initialize_lstm_layer(self, input_size, hidden_size):
    #     #  创建并初始化lstm层
    #     self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
    #     self.lstm = self.lstm.to(self.device)   # 将lstm层移动到 GPU

    def forward(self, x_fc, x_conv, hidden=None):
        # 卷积层处理
        x_conv = F.relu(self.conv1(x_conv))
        x_conv = self.pool1(x_conv)  # 添加池化层
        x_conv = x_conv.view(x_conv.size(0), -1)  # 展平卷积输出

        # LSTM 层处理
        x_conv = x_conv.unsqueeze(1)  # 增加时间步维度
        # 如果lstm层还没有初始化，进行初始化
        # if self.lstm is None:
        #     self._initialize_lstm_layer(x_conv.size(2), self.hidden_dim)
        lstm_out, hidden = self.lstm(x_conv, hidden)
        x_conv = lstm_out[:, -1, :]  # 取最后一个时间步的输出

        # 如果全连接层还没有初始化，进行初始化
        # if self.fc1 is None:
        #     self._initialize_fc_layer(x_conv.size(1), self.hidden_dim)
        x_conv = F.relu(self.fc1(x_conv))
        # 全连接层处理
        x_fc = F.relu(self.fc2(x_fc))
        # 合并卷积和全连接层的输出
        x = x_conv + x_fc  # 或者可以使用其他方式来整合
        # 共享隐藏层
        x = F.relu(self.shared_fc(x))
        return self.fc3(x)


class SAC:
    """处理离散动作的SAC算法"""

    def __init__(self, state_dim_fc, state_dim_conv, hidden_dim, action_dim, actor_lr, critic_lr,
                 alpha_lr, target_entropy, tau, gamma, device):
        # 策略网络
        self.actor = PolicyNet(state_dim_fc, state_dim_conv, hidden_dim, action_dim, device).to(device)
        # 加载策略网络的参数
        # self.actor.load_state_dict(torch.load('sac_actor_state_dict.pth'))
        # 第一个Q网络
        self.critic_1 = QValueNet(state_dim_fc, state_dim_conv, hidden_dim, action_dim, device).to(device)
        # 加载第一个Q网络参数
        # self.critic_1.load_state_dict(torch.load('sac_critic1_state_dict.pth'))
        # 第二个Q网络
        self.critic_2 = QValueNet(state_dim_fc, state_dim_conv, hidden_dim, action_dim, device).to(device)
        # 加载第二个Q网络参数
        # self.critic_2.load_state_dict(torch.load('sac_critic2_state_dict.pth'))
        # 第一个目标Q网络
        self.target_critic_1 = QValueNet(state_dim_fc, state_dim_conv, hidden_dim,
                                         action_dim, device).to(device)
        # 第二个目标Q网络
        self.target_critic_2 = QValueNet(state_dim_fc, state_dim_conv, hidden_dim,
                                         action_dim, device).to(device)
        # 令目标Q网络的初始参数和Q网络一样
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        # 使用Adam优化器 (输入为评估网络(策略网络)的参数和学习率)，更新模型参数
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(),
                                                   lr=critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(),
                                                   lr=critic_lr)
        # 使用alpha的log值,可以使训练结果比较稳定
        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float)
        self.log_alpha.requires_grad = True  # 可以对alpha求梯度
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr=alpha_lr)

        self.target_entropy = target_entropy  # 目标熵的大小
        self.gamma = gamma  # 折扣因子
        self.tau = tau  # 软更新参数，更新目标网络
        self.device = device

    def take_action(self, state_fc, state_conv):
        # 将状态转为tensor类型
        state_fc = torch.tensor([state_fc], dtype=torch.float).to(self.device)  # 全连接层
        state_conv = torch.tensor(state_conv, dtype=torch.float).unsqueeze(0).unsqueeze(0).to(self.device)  # 卷积层

        # 获取动作概率分布
        action_probs = self.actor(state_fc, state_conv)
        # 使用概率分布获取动作
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()  # 采样一个动作
        return action.item()

    # 计算目标Q值,直接用策略网络的输出概率进行期望计算
    # (连续动作空间熵为策略网络输出，目标q值得计算直接输入下一状态和动作)
    # q1_value = self.target_critic_1(next_states, next_actions)
    def calc_target(self, rewards, next_states_fc, next_states_conv, dones):
        # 得到下一状态的概率分布
        next_probs = self.actor(next_states_fc, next_states_conv)
        # 计算下一状态的熵
        next_log_probs = torch.log(next_probs + 1e-8)
        entropy = -torch.sum(next_probs * next_log_probs, dim=1, keepdim=True)
        # 计算下一状态的目标Q值
        q1_value = self.target_critic_1(next_states_fc, next_states_conv)
        q2_value = self.target_critic_2(next_states_fc, next_states_conv)
        # 用min操作选择两个Q值中较小的一个，并与动作概率相乘
        min_qvalue = torch.sum(next_probs * torch.min(q1_value, q2_value),
                               dim=1,
                               keepdim=True)
        # 计算目标Q值
        next_value = min_qvalue + self.log_alpha.exp() * entropy
        td_target = rewards + self.gamma * next_value * (1 - dones)
        return td_target

    # 更新目标Q网络的参数
    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(),
                                       net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) +
                                    param.data * self.tau)

    def update(self, transition_dict):
        states_fc = torch.tensor(transition_dict['states_fc'],
                                 dtype=torch.float).to(self.device)

        states_conv = torch.tensor(transition_dict['states_conv'], dtype=torch.float) \
            .unsqueeze(0).mean(dim=1, keepdim=True).to(self.device)  # 卷积层

        actions = torch.tensor(transition_dict['actions']).unsqueeze(0).to(
            self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states_fc = torch.tensor(transition_dict['next_states_fc'],
                                      dtype=torch.float).to(self.device)
        next_states_conv = torch.tensor(transition_dict['next_states_conv'], dtype=torch.float) \
            .unsqueeze(0).mean(dim=1, keepdim=True).to(self.device)

        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)

        # 更新两个Q网络
        td_target = self.calc_target(rewards, next_states_fc, next_states_conv, dones)

        # 计算每个动作的 Q 值
        critic_1_q_values = self.critic_1(states_fc, states_conv).gather(1, actions)
        # 对 Q 值进行整合，例如取平均值
        average_1_q_value = torch.mean(critic_1_q_values, dim=1)
        # 调整average_1_q_value维度
        average_1_q_value = average_1_q_value.expand_as(td_target)
        critic_1_loss = torch.mean(
            F.mse_loss(average_1_q_value, td_target.detach()))

        critic_2_q_values = self.critic_2(states_fc, states_conv).gather(1, actions)
        # 对 Q 值进行整合，例如取平均值
        average_2_q_value = torch.mean(critic_2_q_values, dim=1)
        # 调整average_2_q_value维度
        average_2_q_value = average_2_q_value.expand_as(td_target)
        critic_2_loss = torch.mean(
            F.mse_loss(average_2_q_value, td_target.detach()))

        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()

        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        # 更新策略网络
        probs = self.actor(states_fc, states_conv)
        log_probs = torch.log(probs + 1e-8)
        # 直接根据概率计算熵，求期望(和连续动作空间不同点,连续动作空间直接从策略网络输出，取负值)
        entropy = -torch.sum(probs * log_probs, dim=1, keepdim=True)
        q1_value = self.critic_1(states_fc, states_conv)
        q2_value = self.critic_2(states_fc, states_conv)
        min_qvalue = torch.sum(probs * torch.min(q1_value, q2_value),
                               dim=1,
                               keepdim=True)  # 直接根据概率计算期望
        actor_loss = torch.mean(-self.log_alpha.exp() * entropy - min_qvalue)
        # 更新策略网络
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 更新alpha（温度参数，和熵有关）值
        alpha_loss = torch.mean(
            (entropy - self.target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        # 更新目标Q网络的参数(使用两个V critic, 提高算法稳定性)
        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)

# device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
#     "cpu")
# num_T = 10
# T = 1
# env = VecEnv(num_T, T)
# state1, state2 = env.reset()
# state1_dim = len(state1)
# action_dim = len(env.action_space)
# actor = PolicyNet(state1_dim, 1, 128, action_dim, device).to(device)
# sac = SAC(state1_dim, 1, 128, action_dim, 0.0001, 0.001, 0.05, -1, 0.005, 0.9, device)
# action = sac.take_action(state1, state2)
# print(action)


# q_net = QValueNet(state1_dim, 1, 128, action_dim).to(device)
# state1_input = torch.tensor([state1, state1], dtype=torch.float).to(device)  # 全连接层
#
# # torch.Size([1, 1, 20, 6])
# state2_input = torch.tensor((state2, state2), dtype=torch.float).unsqueeze(0).mean(dim=1, keepdim=True).to(device)  # 卷积层
# # print(state2_input.shape)
# # print(state2_input)
# a = [3, 5, 7, 4, 5, 6]
# actions = torch.tensor(a).unsqueeze(0).to(device)
# critic_1_q_values = q_net(state1_input, state2_input).gather(1, actions)
# # print(critic_1_q_values)
# average_1_q_value = torch.mean(critic_1_q_values, dim=1)
# print(average_1_q_value)
