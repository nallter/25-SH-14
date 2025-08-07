import numpy as np
from typing import List, Dict, Tuple, Set
import math
import random


class Action:
    """动作表示类 - 完整实现"""
    
    def __init__(self, 
                 start_position: Tuple[float, float], 
                 end_position: Tuple[float, float], 
                 direction: float, 
                 duration: float):
        """
        初始化动作
        
        参数:
        start_position: 起始位置 (x, y) 坐标
        end_position: 结束位置 (x, y) 坐标
        direction: 运动方向 (角度，0-360度)
        duration: 动作持续时间 (秒)
        """
        self.start_position = start_position
        self.end_position = end_position
        self.direction = direction
        self.duration = duration
        
        # 计算移动距离
        dx = end_position[0] - start_position[0]
        dy = end_position[1] - start_position[1]
        self.distance = math.sqrt(dx**2 + dy**2)
        
        # 计算速度
        self.speed = self.distance / duration if duration > 0 else 0
    
    def __repr__(self):
        return (f"Action(start={self.start_position}, end={self.end_position}, "
                f"dir={self.direction}°, dist={self.distance:.1f}m, "
                f"speed={self.speed:.1f}m/s)")

class DistributedMPC:
    """分布式模型预测控制器 - 功能完善版"""
    
    def __init__(self, agents: List['Agent'], grid_map: 'SearchGrid', 
                 prediction_horizon: int = 5, control_horizon: int = 3):
        """初始化分布式MPC控制器
        
        参数:
        agents: 智能体列表
        grid_map: 搜索网格地图
        prediction_horizon: 预测时域
        control_horizon: 控制时域
        """
        self.agents = agents
        self.grid_map = grid_map
        self.prediction_horizon = prediction_horizon
        self.control_horizon = control_horizon
        self.time_step = 0
        
        # 权重参数
        self.w1 = 0.6  # 搜索效能权重 (提高)
        self.w2 = 0.2  # 目标发现权重 (降低)
        self.w3 = 0.2  # 协同权重 (降低)
        
        # 预测路径和控制序列
        self.predicted_paths = {}
        self.control_sequences = {}
        
        # 协同参数
        self.communication_range = float('inf')  # 无通信范围限制
        self.boundary_weight = 0.3  # 初始边界权重 (降低)
        self.boundary_decay_rate = 0.01  # 边界权重衰减率 (提高)
        
        # 目标跟踪器
        self.target_tracker = TargetTracker()
        
        # 协同信息缓存
        self.shared_info = {}

    def sense_update(self):
        """传感器探测更新 - 完善版"""
        for agent in self.agents:
            # 获取传感器探测数据
            detections = agent.sensor.scan(self.grid_map, agent.position, agent.heading)
            
            # 更新目标跟踪器
            self.target_tracker.update_detections(detections, agent.id, self.time_step)
            
            # 更新环境信息
            self.grid_map.update_from_detections(detections, agent.id)
            
            # 更新本地状态
            agent.update_local_state(self.grid_map)
            
            # 共享关键信息（分布式协同）
            self.share_critical_info(agent)

    def share_critical_info(self, agent: 'Agent'):
        """共享关键信息（分布式协同）"""
        # 1. 共享高概率目标位置
        high_prob_targets = self.grid_map.get_high_probability_targets(threshold=0.6)
        self.shared_info[agent.id] = {
            'position': agent.position,
            'heading': agent.heading,
            'high_prob_targets': high_prob_targets,
            'planned_path': agent.planned_path[:2] if agent.planned_path else None,
            'timestamp': self.time_step
        }
        
        # 2. 广播给所有其他智能体（无通信范围限制）
        for other_agent in self.agents:
            if other_agent.id != agent.id:
                other_agent.receive_shared_info(agent.id, self.shared_info[agent.id])

    def decide_actions(self):
        """决策过程 - 完善版"""
        # 更新边界权重（随时间衰减）
        self.update_boundary_weight()
        
        for agent in self.agents:
            # 生成未来多步可能路径
            possible_paths = agent.generate_possible_paths(
                self.prediction_horizon, 
                self.grid_map
            )
            
            # 评估各路径性能指标
            path_rewards = self.evaluate_paths(agent, possible_paths)
            
            # 分布式优化控制序列
            optimal_sequence = self.distributed_optimize(
                agent, 
                possible_paths, 
                path_rewards
            )
            
            # 存储优化结果
            self.predicted_paths[agent.id] = possible_paths
            self.control_sequences[agent.id] = optimal_sequence
            agent.planned_path = optimal_sequence

    def evaluate_paths(self, agent: 'Agent', paths: List[List['Action']]) -> Dict[int, float]:
        """评估路径性能指标 - 完善版"""
        rewards = {}
        for idx, path in enumerate(paths):
            # 1. 搜索效能回报（环境探测回报）
            search_efficiency = self.calculate_search_efficiency(agent, path)
            
            # 2. 目标发现回报
            target_discovery = self.calculate_target_discovery(agent, path)
            
            # 3. 协同回报（避撞和协同）
            collaboration = self.calculate_collaboration(agent, path)
            
            # 4. 边界优先回报
            boundary_priority = self.calculate_boundary_priority(agent, path)
            
            # 综合性能指标
            total_reward = (
                self.w1 * search_efficiency +
                self.w2 * target_discovery +
                self.w3 * collaboration +
                self.boundary_weight * boundary_priority
            )
            
            rewards[idx] = total_reward
        
        return rewards

    def calculate_search_efficiency(self, agent: 'Agent', path: List['Action']) -> float:
        """计算搜索效能回报（环境探测回报）"""
        efficiency = 0.0
        for i, action in enumerate(path):
            # 预测执行动作后的位置
            next_pos = agent.predict_position(action)
            
            # 获取该位置的环境不确定度（熵）
            uncertainty = self.grid_map.get_uncertainty(next_pos)
            
            # 考虑探测能力
            if agent.can_detect_position(next_pos):
                # 预计能探测到的网格
                detectable_grids = agent.sensor.get_detectable_grids(
                    next_pos, agent.heading, self.grid_map
                )
                
                # 计算预计减少的不确定度
                uncertainty_reduction = sum(
                    self.grid_map.get_uncertainty(grid_pos) 
                    for grid_pos in detectable_grids
                )
                
                efficiency += uncertainty_reduction * math.exp(-0.2 * i)  # 时间衰减
        
        return efficiency

    def calculate_target_discovery(self, agent: 'Agent', path: List['Action']) -> float:
        """计算目标发现回报"""
        discovery = 0.0
        for i, action in enumerate(path):
            next_pos = agent.predict_position(action)
            
            if agent.can_detect_position(next_pos):
                # 获取可探测网格
                detectable_grids = agent.sensor.get_detectable_grids(
                    next_pos, agent.heading, self.grid_map
                )
                
                # 计算目标发现潜力
                for grid_pos in detectable_grids:
                    target_prob = self.grid_map.get_target_probability(grid_pos)
                    
                    # 考虑目标跟踪状态
                    tracking_status = self.target_tracker.get_tracking_status(grid_pos)
                    if tracking_status == 'confirmed':
                        # 已确认目标价值较低
                        discovery += 0.1 * target_prob
                    elif tracking_status == 'partial':
                        # 部分探测目标价值高
                        discovery += 0.8 * target_prob
                    else:
                        # 未探测目标价值中等
                        discovery += 0.5 * target_prob
        
        return discovery

    def calculate_collision_risk(self, agent: 'Agent', path: List['Action']) -> float:
        """计算碰撞风险"""
        collision_risk = 0.0
        
        for i, action in enumerate(path):
            next_pos = agent.predict_position(action)
            
            # 检查与其他智能体的碰撞风险
            for other_agent in self.agents:
                if other_agent.id != agent.id:
                    if other_agent.planned_path and len(other_agent.planned_path) > i:
                        other_pos = other_agent.predict_position(other_agent.planned_path[i])
                        distance = np.linalg.norm(np.array(next_pos) - np.array(other_pos))
                        min_safe_distance = agent.safe_distance + other_agent.safe_distance
                        if distance < min_safe_distance:
                            collision_risk += 1.0 / (distance + 0.1)  # 距离越近风险越大
        
        return collision_risk

    def calculate_collaboration(self, agent: 'Agent', path: List['Action']) -> float:
        """计算协同回报（避撞和协同）"""
        collaboration = 0.0
        
        # 1. 避撞惩罚
        collision_risk = self.calculate_collision_risk(agent, path)
        collaboration -= collision_risk
        
        # 2. 协同奖励（避免重复探测）
        duplicate_penalty = 0.0
        for i, action in enumerate(path):
            next_pos = agent.predict_position(action)
            
            if agent.can_detect_position(next_pos):
                detectable_grids = agent.sensor.get_detectable_grids(
                    next_pos, agent.heading, self.grid_map
                )
                
                for grid_pos in detectable_grids:
                    # 检查是否有其他智能体计划探测此网格
                    for other_agent in self.agents:
                        if other_agent.id != agent.id:
                            if other_agent.planned_path:
                                for other_action in other_agent.planned_path:
                                    other_pos = other_agent.predict_position(other_action)
                                    if other_agent.can_detect_position(other_pos):
                                        other_detectable = other_agent.sensor.get_detectable_grids(
                                            other_pos, other_agent.heading, self.grid_map
                                        )
                                        if grid_pos in other_detectable:
                                            # 重复探测惩罚
                                            duplicate_penalty += 0.3
        
        collaboration -= duplicate_penalty
        
        return collaboration

    def calculate_boundary_priority(self, agent: 'Agent', path: List['Action']) -> float:
        """计算边界优先回报"""
        boundary_reward = 0.0
        
        for i, action in enumerate(path):
            next_pos = agent.predict_position(action)
            
            if self.grid_map.is_boundary_position(next_pos):
                # 边界位置奖励
                boundary_reward += 1.0 * math.exp(-0.2 * i)  # 时间衰减
                
                # 边界探测能力奖励
                if agent.can_detect_position(next_pos):
                    detectable_grids = agent.sensor.get_detectable_grids(
                        next_pos, agent.heading, self.grid_map
                    )
                    
                    boundary_grids = [
                        pos for pos in detectable_grids 
                        if self.grid_map.is_boundary_position(pos)
                    ]
                    
                    boundary_reward += 0.5 * len(boundary_grids)
        
        return boundary_reward

    def update_boundary_weight(self):
        """更新边界权重（随时间衰减）"""
        self.boundary_weight = max(0.1, self.boundary_weight * (1 - self.boundary_decay_rate))

    def distributed_optimize(self, agent: 'Agent', 
                            paths: List[List['Action']], 
                            rewards: Dict[int, float]) -> List['Action']:
        """分布式优化控制序列 - 完善版"""
        # 选择最优路径索引
        best_path_idx = max(rewards, key=rewards.get)
        best_path = paths[best_path_idx]
        
        # 考虑协同约束（避让其他智能体）
        optimized_path = self.adjust_path_for_collaboration(agent, best_path)
        
        return optimized_path[:self.control_horizon]

    def run_step(self):
        """执行单步MPC控制流程"""
        # 1. 传感器探测更新
        self.sense_update()
        
        # 2. 决策过程
        self.decide_actions()
        
        # 3. 执行动作
        for agent in self.agents:
            if agent.planned_path and len(agent.planned_path) > 0:
                action = agent.planned_path[0]
                agent.position = action.end_position
                # 确保heading与运动方向一致
                agent.heading = action.direction
        
        # 4. 更新时间步
        self.time_step += 1

    def adjust_path_for_collaboration(self, agent: 'Agent', path: List['Action']) -> List['Action']:
        """调整路径以考虑协同约束"""
        adjusted_path = []
        
        for i, action in enumerate(path):
            # 预测执行动作后的位置
            next_pos = agent.predict_position(action)
            
            # 检查是否有碰撞风险
            collision_risk = False
            for other_agent in self.agents:
                if other_agent.id != agent.id:
                    # 获取其他智能体的计划位置
                    if other_agent.planned_path and len(other_agent.planned_path) > i:
                        other_pos = other_agent.predict_position(other_agent.planned_path[i])
                        
                        # 计算距离
                        distance = np.linalg.norm(np.array(next_pos) - np.array(other_pos))
                        
                        # 检查是否低于安全距离
                        min_safe_distance = agent.safe_distance + other_agent.safe_distance
                        if distance < min_safe_distance:
                            collision_risk = True
                            break
            
            if collision_risk:
                # 生成避让动作
                avoidance_action = self.generate_avoidance_action(agent, next_pos, i)
                if avoidance_action:
                    adjusted_path.append(avoidance_action)
                else:
                    # 无法避让则悬停
                    adjusted_path.append(Action(
                        start_position=agent.position,
                        end_position=agent.position,
                        direction=agent.heading,
                        duration=1
                    ))
            else:
                adjusted_path.append(action)
        
        return adjusted_path

    def generate_avoidance_action(self, agent: 'Agent', risky_position: Tuple[float, float], step: int) -> 'Action':
        """生成避让动作"""
        # 获取当前可行动作
        possible_actions = agent.motion_model.generate_possible_actions(agent.position)
        
        # 评估每个动作的避让效果
        best_action = None
        best_score = -float('inf')
        
        for action in possible_actions:
            next_pos = agent.predict_position(action)
            
            # 计算避让分数
            score = 0.0
            
            # 1. 距离风险位置越远越好
            distance_to_risk = np.linalg.norm(np.array(next_pos) - np.array(risky_position))
            score += distance_to_risk
            
            # 2. 保持探测能力
            if agent.can_detect_position(next_pos):
                score += 0.5
            
            # 3. 边界优先
            if self.grid_map.is_boundary_position(next_pos):
                score += 1.0
            
            # 4. 路径一致性（避免频繁转向）
            angle_diff = abs(action.direction - agent.heading)
            if angle_diff < 30:  # 小于30度转向
                score += 0.3
            
            if score > best_score:
                best_score = score
                best_action = action
        
        return best_action

class Agent:
    """智能体基类 - 完善版"""
    
    def __init__(self, agent_id: str, position: Tuple[float, float], 
                 heading: float, agent_type: str, safe_distance: float):
        """初始化智能体
        
        参数:
        agent_id: 智能体唯一标识
        position: 初始位置 (x,y)
        heading: 初始航向角度
        agent_type: 智能体类型 ('drone'或'usv')
        safe_distance: 安全距离
        """
        self.id = agent_id
        self.position = position
        self.heading = heading
        self.agent_type = agent_type
        self.safe_distance = safe_distance
        self.sensor = Sensor(agent_type)
        self.motion_model = MotionModel(agent_type)
        
        # 其他属性
        self.planned_path = None  # 计划路径
        self.shared_info = {}  # 接收到的共享信息

    def can_detect_position(self, position: Tuple[float, float]) -> bool:
        """检查是否能探测到指定位置"""
        # 计算距离
        distance = np.linalg.norm(np.array(self.position) - np.array(position))
        
        if self.agent_type == 'drone':
            # 无人机需要检查角度
            dx = position[0] - self.position[0]
            dy = position[1] - self.position[1]
            angle = math.degrees(math.atan2(dy, dx))
            angle_diff = abs((angle - self.heading + 180) % 360 - 180)
            
            return distance <= self.sensor.range and angle_diff <= self.sensor.fov/2
        else:
            # 无人艇只需检查距离
            return distance <= self.sensor.range

    def receive_shared_info(self, sender_id: str, info: Dict):
        """接收其他智能体共享的信息"""
        self.shared_info[sender_id] = info

    def generate_possible_actions(self) -> List['Action']:
        """生成可能的动作集合 - 考虑转弯半径约束"""
        actions = []
        
        # 无人机只能直线移动（4方向）
        if self.agent_type == 'drone':
            directions = [0, 90, 180, 270]
            for direction in directions:
                # 检查转弯半径约束
                if self.is_turn_feasible(direction):
                    end_pos = self.motion_model.calculate_end_position(self.position, direction)
                    actions.append(Action(
                        start_position=self.position,
                        end_position=end_pos,
                        direction=direction,
                        duration=1
                    ))
        
        # 无人艇可以8方向移动
        else:
            directions = [0, 45, 90, 135, 180, 225, 270, 315]
            for direction in directions:
                # 检查转弯半径约束
                if self.is_turn_feasible(direction):
                    end_pos = self.motion_model.calculate_end_position(self.position, direction)
                    actions.append(Action(
                        start_position=self.position,
                        end_position=end_pos,
                        direction=direction,
                        duration=1
                    ))
        
        return actions

    def is_turn_feasible(self, new_direction: float) -> bool:
        """检查转弯是否满足最小转弯半径约束"""
        # 计算转向角度差
        angle_diff = abs((new_direction - self.heading + 180) % 360 - 180)
        
        if angle_diff < 1:  # 基本直行
            return True
        
        # 计算最小转弯半径要求
        if self.agent_type == 'drone':
            min_turn_radius = 100  # 无人机最小转弯半径100m
        else:
            min_turn_radius = 20   # 无人艇最小转弯半径20m
        
        # 计算实际转弯半径 (v^2 / (g * tan(θ)))
        # 简化计算：假设速度恒定，转弯半径与角度差成反比
        required_radius = self.motion_model.max_speed**2 / (9.8 * math.tan(math.radians(angle_diff)))
        
        return required_radius >= min_turn_radius

    def generate_possible_paths(self, horizon: int, grid_map: 'SearchGrid') -> List[List['Action']]:
        """生成可能的路径集合"""
        paths = []
        
        # 生成初始动作集合
        initial_actions = self.generate_possible_actions()
        
        # 递归生成路径
        def _generate_path(current_path: List['Action'], remaining_horizon: int):
            if remaining_horizon == 0:
                paths.append(current_path)
                return
            
            last_action = current_path[-1] if current_path else None
            current_position = last_action.end_position if last_action else self.position
            current_heading = last_action.direction if last_action else self.heading
            
            # 生成下一步可能的动作
            next_actions = []
            for direction in [0, 45, 90, 135, 180, 225, 270, 315]:
                if self.is_turn_feasible(direction):
                    action = Action(
                        start_position=current_position,
                        end_position=self.motion_model.calculate_end_position(current_position, direction),
                        direction=direction,
                        duration=1
                    )
                    next_actions.append(action)
            
            # 递归生成后续路径
            for action in next_actions:
                _generate_path(current_path + [action], remaining_horizon - 1)
        
        _generate_path([], horizon)
        return paths

    def update_local_state(self, grid_map: 'SearchGrid'):
        """更新智能体本地状态信息"""
        # 获取当前位置的网格信息
        i, j = grid_map.position_to_index(self.position)
        
        # 更新本地目标概率信息
        self.local_target_prob = grid_map.target_probability[i, j]
        
        # 更新本地不确定度信息
        self.local_uncertainty = grid_map.uncertainty[i, j]
        
        # 更新边界状态
        self.on_boundary = grid_map.is_boundary_position(self.position)

    def predict_position(self, action: 'Action') -> Tuple[float, float]:
        """预测执行动作后的位置"""
        # 直接返回动作的结束位置
        return action.end_position

class Sensor:
    """传感器模型 - 完善版"""
    
    def __init__(self, agent_type: str):
        self.agent_type = agent_type
        if agent_type == 'drone':
            self.range = 3000  # 3km
            self.fov = 60      # 60度扇形视野
        else:  # 'usv'
            self.range = 800   # 800m
            self.fov = 360     # 360度全向视野

    def scan(self, grid_map: 'SearchGrid', position: Tuple[float, float], 
            heading: float) -> Dict[Tuple[int, int], float]:
        """
        执行传感器扫描 - 基于MovingTarget的探测
        
        返回:
        探测到的网格位置和累计探测时间
        """
        detected_grids = {}
        
        # 检查所有MovingTarget是否在探测范围内
        for target in grid_map.targets:
            if isinstance(target, MovingTarget):
                target.update_grid_position(grid_map)
                if self.is_target_detectable(target, position, heading):
                    grid_pos = target.grid_position
                    # 累计探测时间
                    current_duration = grid_map.get_detection_duration(grid_pos)
                    new_duration = current_duration + 1
                    detected_grids[grid_pos] = new_duration
                    
                    # 检查是否达到确认阈值
                    if new_duration >= 10:
                        grid_map.confirm_target(grid_pos)
        
        return detected_grids

    def is_target_detectable(self, target: 'MovingTarget', 
                           sensor_position: Tuple[float, float], 
                           sensor_heading: float) -> bool:
        """检查目标是否在传感器探测范围内"""
        if self.agent_type == 'drone':
            # 无人机需要检查角度和距离
            dx = target.position[0] - sensor_position[0]
            dy = target.position[1] - sensor_position[1]
            distance = math.sqrt(dx**2 + dy**2)
            if distance > self.range:
                return False
                
            angle = math.degrees(math.atan2(dy, dx)) % 360
            angle_diff = abs((angle - sensor_heading + 180) % 360 - 180)
            return angle_diff <= self.fov/2
        else:
            # 无人艇只需检查距离
            dx = target.position[0] - sensor_position[0]
            dy = target.position[1] - sensor_position[1]
            distance = math.sqrt(dx**2 + dy**2)
            return distance <= self.range

    def get_detectable_grids(self, position: Tuple[float, float], 
                            heading: float, 
                            grid_map: 'SearchGrid') -> Set[Tuple[int, int]]:
        """
        获取传感器可探测的网格位置
        """
        # 转换位置为网格索引
        center_i, center_j = grid_map.position_to_index(position)
        
        # 计算覆盖半径 (网格单位)
        grid_range = int(self.range / (grid_map.resolution * 1852))
        
        detectable_grids = set()
        
        if self.agent_type == 'usv' or self.fov == 360:
            # 无人艇或全向探测 - 圆形区域
            for i in range(max(0, center_i-grid_range), min(grid_map.grid_height, center_i+grid_range+1)):
                for j in range(max(0, center_j-grid_range), min(grid_map.grid_width, center_j+grid_range+1)):
                    if np.hypot(i-center_i, j-center_j) <= grid_range:
                        detectable_grids.add((i, j))
        else:
            # 无人机 - 扇形区域
            # 计算扇形角度范围
            angle_min = heading - self.fov/2
            angle_max = heading + self.fov/2
            
            for i in range(max(0, center_i-grid_range), min(grid_map.grid_height, center_i+grid_range+1)):
                for j in range(max(0, center_j-grid_range), min(grid_map.grid_width, center_j+grid_range+1)):
                    # 计算相对角度
                    dx = j - center_j
                    dy = i - center_i
                    distance = np.hypot(dx, dy)
                    
                    if distance <= grid_range:
                        angle = math.degrees(math.atan2(dy, dx)) % 360
                        
                        # 检查角度是否在扇形范围内
                        if angle_min <= angle <= angle_max or angle_min <= angle+360 <= angle_max:
                            detectable_grids.add((i, j))
        
        return detectable_grids


class MotionModel:
    """运动模型 - 完善版"""
    
    def __init__(self, agent_type: str):
        """初始化运动模型
        
        参数:
        agent_type: 智能体类型 ('drone'或'usv')
        """
        self.agent_type = agent_type
        if agent_type == 'drone':
            self.max_speed = 30  # 无人机最大速度30m/s
        else:  # 'usv'
            self.max_speed = 10  # 无人艇最大速度10m/s
    
    def calculate_end_position(self, start: Tuple[float, float], direction: float) -> Tuple[float, float]:
        """计算动作结束位置 - 考虑速度约束"""
        # 将角度转换为弧度
        angle_rad = math.radians(direction)
        
        # 计算位移 (速度 * 时间步长)
        # 注意坐标系: x向右为正，y向上为正
        dx = self.max_speed * math.sin(angle_rad)  # 东向应为x增加
        dy = self.max_speed * math.cos(angle_rad)  # 北向应为y增加
        
        # 计算结束位置
        end_x = start[0] + dx
        end_y = start[1] + dy
        
        return (end_x, end_y)

    def generate_possible_actions(self, position: Tuple[float, float]) -> List['Action']:
        """生成可能的动作集合 - 考虑转弯半径约束"""
        actions = []
        
        # 无人机只能直线移动（4方向）
        if self.agent_type == 'drone':
            directions = [0, 90, 180, 270]
            for direction in directions:
                end_pos = self.calculate_end_position(position, direction)
                actions.append(Action(
                    start_position=position,
                    end_position=end_pos,
                    direction=direction,
                    duration=1
                ))
        
        # 无人艇可以8方向移动
        else:
            directions = [0, 45, 90, 135, 180, 225, 270, 315]
            for direction in directions:
                end_pos = self.calculate_end_position(position, direction)
                actions.append(Action(
                    start_position=position,
                    end_position=end_pos,
                    direction=direction,
                    duration=1
                ))
        
        return actions


class SearchGrid:
    """环境网格地图类 - 完善版"""
    
    def __init__(self, width: float, height: float, resolution: float):
        """
        初始化网格地图
        
        参数:
        width: 区域宽度 (海里)
        height: 区域高度 (海里) 
        resolution: 网格分辨率 (海里/格)
        """
        self.width = width
        self.height = height
        self.resolution = resolution
        self.time_step = 0  # 初始化时间步
        
        # 计算网格尺寸
        self.grid_width = int(width / resolution)
        self.grid_height = int(height / resolution)
        
        # 初始化状态变量
        self.target_probability = np.zeros((self.grid_height, self.grid_width))
        self.uncertainty = np.ones((self.grid_height, self.grid_width))  # 初始不确定性最高
        self.detection_duration = np.zeros((self.grid_height, self.grid_width))  # 累计探测时间
        self.confirmed_targets = np.zeros((self.grid_height, self.grid_width), dtype=bool)  # 已确认目标
        self.last_update_time = np.zeros((self.grid_height, self.grid_width))  # 最后更新时间
        self.targets = []  # 存储目标对象列表

    def update_from_detections(self, detections: Dict[Tuple[int, int], float], agent_id: str):
        """
        改进的目标概率更新逻辑：
        1. 所有网格基础概率保持10%-30%背景值
        2. 探测到目标：概率显著增加 (70%-100%)
        3. 探测但未发现目标：概率降低 (5%-15%)
        """
        # 基础背景概率 (10%-30%)
        background_min = 0.1
        background_max = 0.3
        
        # 首先设置所有网格的基础背景概率
        for i in range(self.grid_height):
            for j in range(self.grid_width):
                if self.last_update_time[i, j] < self.time_step:
                    # 未探测区域保持随机背景概率
                    if self.target_probability[i, j] < background_min:
                        self.target_probability[i, j] = random.uniform(background_min, background_max)
        
        # 处理当前探测结果
        for grid_idx, duration in detections.items():
            i, j = grid_idx
            self.detection_duration[i, j] = duration
            
            if duration > 0:  # 探测到目标
                # 概率显著增加 (70%-100%)
                self.target_probability[i, j] = min(1.0, 
                    max(0.7, self.target_probability[i, j] * 1.5 + 0.2))
            else:  # 探测但未发现目标
                # 概率降低 (5%-15%)
                self.target_probability[i, j] = max(0.05,
                    min(0.15, self.target_probability[i, j] * 0.3))
            
            # 更新不确定性（熵）
            p = self.target_probability[i, j]
            self.uncertainty[i, j] = 0 if p in (0, 1) else -p*math.log2(p) - (1-p)*math.log2(1-p)
            self.last_update_time[i, j] = self.time_step

    def add_target_object(self, target: 'Target'):
        """添加目标对象"""
        self.targets.append(target)
        # 更新网格概率
        i, j = self.position_to_index(target.position)
        self.target_probability[i, j] = 0.8  # 初始概率
        self.uncertainty[i, j] = 0.5
        self.confirmed_targets[i, j] = False
        self.detection_duration[i, j] = 0.0

    def confirm_target(self, grid_idx: Tuple[int, int]):
        """确认目标位置"""
        i, j = grid_idx
        self.target_probability[i, j] = 1.0
        self.uncertainty[i, j] = 0.0
        self.confirmed_targets[i, j] = True
        self.detection_duration[i, j] = 10.0  # 设置为最大值

    def get_target_probability(self, position: Tuple[float, float]) -> float:
        """获取目标存在概率"""
        i, j = self.position_to_index(position)
        return self.target_probability[i, j]

    def get_uncertainty(self, position: Tuple[float, float]) -> float:
        """获取环境不确定度（熵）"""
        i, j = self.position_to_index(position)
        return self.uncertainty[i, j]

    def get_detection_duration(self, position: Tuple[float, float]) -> float:
        """获取累计探测时间"""
        i, j = self.position_to_index(position)
        return self.detection_duration[i, j]

    def is_boundary_position(self, position: Tuple[float, float]) -> bool:
        """检查位置是否在边界区域"""
        i, j = self.position_to_index(position)
        # 定义边界区域（最外两层网格）
        return (i < 2 or i >= self.grid_height-2 or 
                j < 2 or j >= self.grid_width-2)

    def get_high_probability_targets(self, threshold: float = 0.6) -> List[Tuple[float, float]]:
        """获取高概率目标位置"""
        high_prob_targets = []
        for i in range(self.grid_height):
            for j in range(self.grid_width):
                if self.target_probability[i, j] >= threshold:
                    pos = self.index_to_position((i, j))
                    high_prob_targets.append(pos)
        return high_prob_targets
    def position_to_index(self, position: Tuple[float, float]) -> Tuple[int, int]:
        """
        将实际位置转换为网格索引
        
        参数:
        position: (x, y) 坐标，单位米
        
        返回:
        (i, j) 网格索引
        """
        # 1海里 = 1852米
        # 网格分辨率 (米) = resolution * 1852
        grid_resolution_m = self.resolution * 1852
        
        # 计算索引
        j = int(position[0] / grid_resolution_m)  # x方向
        i = int(position[1] / grid_resolution_m)  # y方向
        
        # 确保索引在有效范围内
        i = max(0, min(self.grid_height - 1, i))
        j = max(0, min(self.grid_width - 1, j))
        
        return i, j
    
    def index_to_position(self, index: Tuple[int, int]) -> Tuple[float, float]:
        """
        将网格索引转换为实际位置
        
        参数:
        index: (i, j) 网格索引
        
        返回:
        (x, y) 坐标，单位米
        """
        # 解包索引元组
        i, j = index
        
        grid_resolution_m = self.resolution * 1852
        x = j * grid_resolution_m + grid_resolution_m / 2  # 网格中心
        y = i * grid_resolution_m + grid_resolution_m / 2
        return x, y

class TargetTracker:
    """目标跟踪器"""
    
    def __init__(self):
        self.targets = {}  # 目标ID: {position, last_seen, duration, status}
        self.next_target_id = 1

    def update_detections(self, detections: Dict[Tuple[int, int], float], agent_id: str, time_step: int):
        """更新目标跟踪状态"""
        for grid_idx, duration in detections.items():
            if duration > 0:  # 有探测到目标
                # 检查是否已有目标在此位置
                existing_target = None
                for target_id, target_data in self.targets.items():
                    if target_data['position'] == grid_idx:
                        existing_target = target_id
                        break
                
                if existing_target:
                    # 更新现有目标
                    self.targets[existing_target]['last_seen'] = time_step
                    self.targets[existing_target]['duration'] = duration
                    
                    # 更新状态
                    if duration >= 10:
                        self.targets[existing_target]['status'] = 'confirmed'
                    elif duration > 0:
                        self.targets[existing_target]['status'] = 'partial'
                else:
                    # 创建新目标
                    target_id = f"target_{self.next_target_id}"
                    self.next_target_id += 1
                    self.targets[target_id] = {
                        'position': grid_idx,
                        'last_seen': time_step,
                        'duration': duration,
                        'status': 'partial' if duration > 0 else 'new'
                    }
            else:
                # 重置未探测目标的持续时间
                for target_id, target_data in self.targets.items():
                    if target_data['position'] == grid_idx:
                        self.targets[target_id]['duration'] = 0
                        self.targets[target_id]['status'] = 'lost'
                        break

    def get_tracking_status(self, grid_idx: Tuple[int, int]) -> str:
        """获取目标的跟踪状态"""
        for target_data in self.targets.values():
            if target_data['position'] == grid_idx:
                return target_data['status']
        return 'unknown'
    
class Target:
    """目标基类"""
    def __init__(self, position: Tuple[float, float]):
        self.position = position
        self.detected = False
        self.detection_count = 0
        self.grid_position = None
    
    def update_grid_position(self, grid_map: 'SearchGrid'):
        """更新网格位置"""
        self.grid_position = grid_map.position_to_index(self.position)
    
    def is_in_detection_range(self, agent: 'Agent') -> bool:
        """检查是否在智能体探测范围内"""
        return agent.can_detect_position(self.position)

class MovingTarget(Target):
    """移动目标类 - 增强版"""
    def __init__(self, start_position: Tuple[float, float]):
        super().__init__(start_position)
        self.speed = 7.72  # 15节 = 7.72米/秒
        self.direction = random.uniform(0, 360)  # 随机初始方向
    
    def move(self, time_step: float):
        """移动目标"""
        # 随机改变方向 (10%概率)
        if random.random() < 0.1:
            self.direction = (self.direction + random.uniform(-30, 30)) % 360
        
        angle_rad = math.radians(self.direction)
        dx = self.speed * time_step * math.cos(angle_rad)
        dy = self.speed * time_step * math.sin(angle_rad)
        self.position = (self.position[0] + dx, self.position[1] + dy)

class StaticTarget(Target):
    """静态目标类"""
    def __init__(self, position: Tuple[float, float]):
        """
        初始化静态目标
        
        参数:
        position: 目标位置 (x, y) 单位米
        """
        super().__init__(position)
        self.position = position
        self.detected = False
        self.detection_count = 0  # 被探测次数
        self.grid_position = None  # 所在网格位置
    
    def update_grid_position(self, grid_map: 'SearchGrid'):
        """更新网格位置"""
        self.grid_position = grid_map.position_to_index(self.position)
    
    def is_in_detection_range(self, agent: 'Agent') -> bool:
        """检查是否在智能体探测范围内"""
        return agent.can_detect_position(self.position)
