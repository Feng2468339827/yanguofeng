from env.vehicles import Vehicles
from env.task import Task
from env.availability_SV import AvailabilitySV

# 生成车辆所用参数
road_length = 600  # 道路长度（单位：米）
num_lanes = 2  # 车道数
lane_width = 3  # 车道宽度（单位：米）
vehicle_num_TV = 10  # 单个车道的任务车辆数
vehicle_num_SV = 5
min_vehicle_speed = 10  # 最大车速（单位：米/秒）
max_vehicle_speed = 20  # 最大车速（单位：米/秒）

# 构建csv所用参数
availability_threshold = 0.9
max_communication = 300
sigma = 0.2


class Env:
    def __init__(self, ):
        super(Env, self).__init__()
        # 生成任务车辆 服务车辆
        self.vehicle_TVs = Vehicles(vehicle_num_TV, road_length, num_lanes, lane_width, min_vehicle_speed
                                    , max_vehicle_speed).initialization_vehicle()
        self.vehicle_SVs = Vehicles(vehicle_num_SV, road_length, num_lanes, lane_width, min_vehicle_speed
                                    , max_vehicle_speed).initialization_vehicle()
        self.task = Task()
        self.task_generate_times = self.task.task_generate_time(vehicle_num_TV * 2, 3, 1).tolist()

        self.availabilitySV = AvailabilitySV(availability_threshold, max_communication, sigma)


env = Env()
SVs_effective = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
print(env.vehicle_SVs)
task_messages = env.task.task_generate()
csv = env.availabilitySV.csv(env.vehicle_TVs[0], env.vehicle_SVs, SVs_effective, 2, 1000, env.task_generate_times[0],
                             task_messages)
print(csv)
