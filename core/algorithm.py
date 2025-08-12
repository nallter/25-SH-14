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
                 prediction_horizon: int = 3, control_horizon: int = 2):
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
        
        # 权重参数（强调搜索与发现，兼顾协同与边界）
        self.w1 = 0.35  # 搜索效能权重
        self.w2 = 0.5   # 目标发现/拦截权重（进一步提高以缩短发现/处置时间）
        self.w3 = 0.15  # 协同权重
        
        # 预测路径和控制序列
        self.predicted_paths = {}
        self.control_sequences = {}
        
        # 协同参数
        self.communication_range = float('inf')  # 无通信范围限制
        self.boundary_weight = 0.6  # 初期更重视边界巡航以提升P
        self.boundary_decay_rate = 0.01
        
        # 目标跟踪器
        self.target_tracker = TargetTracker()
        
        # 协同信息缓存
        self.shared_info = {}
        
        # USV任务分配（agent_id -> intercept_point）
        self.usv_assignments: Dict[str, Tuple[float, float]] = {}

    def sense_update(self):
        """传感器探测更新 - 完善版"""
        for agent in self.agents:
            # 获取传感器探测数据（包含正例与负例）
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

    def get_moving_targets_world(self) -> List[Dict]:
        """汇总移动目标的世界坐标与速度向量，用于拦截计算"""
        moving = []
        for t in self.grid_map.targets:
            if isinstance(t, MovingTarget):
                # 速度向量（m/s）
                angle_rad = math.radians(t.direction)
                vt = np.array([t.speed * math.cos(angle_rad), t.speed * math.sin(angle_rad)], dtype=float)
                # 网格索引位置以匹配跟踪器
                t.update_grid_position(self.grid_map)
                moving.append({'pos': np.array(t.position, dtype=float), 'vel': vt, 'grid_idx': t.grid_position})
        return moving

    def compute_intercept_point(self, agent: 'Agent', tgt: Dict) -> Tuple[float, float]:
        """给定agent与目标，枚举时间窗口计算可达拦截点；返回最早可达点或None"""
        pa = np.array(agent.position, dtype=float)
        va = agent.motion_model.max_speed
        pt = tgt['pos']
        vt = tgt['vel']
        # 候选到达时间（秒）
        for t_sec in [30, 60, 90, 120, 180]:
            p_int = pt + vt * t_sec
            # 边界裁剪
            width_m = self.grid_map.width * 1852
            height_m = self.grid_map.height * 1852
            p_int[0] = np.clip(p_int[0], 0, width_m)
            p_int[1] = np.clip(p_int[1], 0, height_m)
            dist = np.linalg.norm(p_int - pa)
            # agent在t_sec内是否可达（留出安全距离裕度）
            if dist <= va * t_sec:
                return (float(p_int[0]), float(p_int[1]))
        return None

    def update_usv_assignments(self):
        """为USV分配追击目标并计算拦截点，避免多USV重复追同一目标"""
        moving_targets = self.get_moving_targets_world()
        # 优先选择被跟踪器标注为partial/confirmed的位置
        candidate_idxs = set()
        for tid, tdata in self.target_tracker.targets.items():
            if tdata['status'] in ('partial', 'confirmed'):
                candidate_idxs.add(tdata['position'])
        # 过滤移动目标（存在于移动目标列表）
        candidates = [t for t in moving_targets if t['grid_idx'] in candidate_idxs]
        if not candidates:
            self.usv_assignments.clear()
            for ag in self.agents:
                if ag.agent_type == 'usv':
                    ag.assigned_waypoint = None
            return
        # 贪心匹配：按USV到目标当前距离排序逐个分配
        usvs = [ag for ag in self.agents if ag.agent_type == 'usv']
        remaining = candidates.copy()
        assignments: Dict[str, Tuple[float, float]] = {}
        for ag in sorted(usvs, key=lambda a: min(np.linalg.norm(np.array(a.position)-c['pos']) for c in remaining) if remaining else 1e9):
            best = None
            best_dist = 1e12
            best_point = None
            for c in remaining:
                inter_pt = self.compute_intercept_point(ag, c)
                if inter_pt is None:
                    # 不可达则用当前目标点距离
                    d = np.linalg.norm(np.array(ag.position)-c['pos'])
                else:
                    d = np.linalg.norm(np.array(ag.position)-np.array(inter_pt))
                if d < best_dist:
                    best_dist = d
                    best = c
                    best_point = inter_pt if inter_pt is not None else tuple(c['pos'])
            if best is not None:
                assignments[ag.id] = best_point
                remaining.remove(best)
        self.usv_assignments = assignments
        # 写入到agent，便于评估使用
        for ag in self.agents:
            if ag.agent_type == 'usv':
                ag.assigned_waypoint = self.usv_assignments.get(ag.id, None)

    def decide_actions(self):
        """决策过程 - 完善版"""
        # 更新边界权重（随时间衰减）
        self.update_boundary_weight()
        
        # USV任务分配（追击/拦截）
        self.update_usv_assignments()
        
        for agent in self.agents:
            # 生成未来多步可能路径（Beam Search 限界）
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
            
            # 2. 目标发现/拦截回报
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
            next_pos = agent.predict_position(action)
            # 使用未来姿态计算可探测网格
            detectable_grids = agent.sensor.get_detectable_grids(
                next_pos, action.direction, self.grid_map
            )
            if not detectable_grids:
                continue
            # 预计减少的不确定度（直接索引网格数组，避免坐标转换）
            uncertainty_reduction = 0.0
            for (gi, gj) in detectable_grids:
                uncertainty_reduction += float(self.grid_map.uncertainty[gi, gj])
            efficiency += uncertainty_reduction * math.exp(-0.2 * i)
        return efficiency

    def calculate_target_discovery(self, agent: 'Agent', path: List['Action']) -> float:
        """计算目标发现/拦截回报"""
        discovery = 0.0
        for i, action in enumerate(path):
            next_pos = agent.predict_position(action)
            # 未来姿态探测扇区
            detectable_grids = agent.sensor.get_detectable_grids(
                next_pos, action.direction, self.grid_map
            )
            if not detectable_grids:
                continue
            # 目标发现潜力（直接索引概率数组）
            for grid_idx in detectable_grids:
                gi, gj = grid_idx
                target_prob = float(self.grid_map.target_probability[gi, gj])
                tracking_status = self.target_tracker.get_tracking_status((gi, gj))
                if tracking_status == 'confirmed':
                    discovery += 0.2 * target_prob
                elif tracking_status == 'partial':
                    discovery += 1.0 * target_prob
                else:
                    discovery += 0.5 * target_prob
            # USV拦截奖励：靠近已跟踪目标（缩短处置时间）
            if agent.agent_type == 'usv' and self.target_tracker.targets:
                # 已分配的拦截点更优先
                if getattr(agent, 'assigned_waypoint', None) is not None:
                    dist_wp = np.linalg.norm(np.array(next_pos) - np.array(agent.assigned_waypoint))
                    discovery += 600.0 / (dist_wp + 50.0)
                else:
                    for t_data in self.target_tracker.targets.values():
                        t_pos_m = self.grid_map.index_to_position(t_data['position'])
                        dist = np.linalg.norm(np.array(next_pos) - np.array(t_pos_m))
                        discovery += 300.0 / (dist + 50.0)
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
            # 本机未来可探测网格
            my_detectable = agent.sensor.get_detectable_grids(
                next_pos, action.direction, self.grid_map
            )
            if not my_detectable:
                continue
            for other_agent in self.agents:
                if other_agent.id == agent.id:
                    continue
                if other_agent.planned_path and len(other_agent.planned_path) > i:
                    other_action = other_agent.planned_path[i]
                    other_pos = other_agent.predict_position(other_action)
                    other_detectable = other_agent.sensor.get_detectable_grids(
                        other_pos, other_action.direction, self.grid_map
                    )
                    # 重叠网格越多，惩罚越大
                    overlap = len(my_detectable & other_detectable)
                    if overlap > 0:
                        duplicate_penalty += 0.1 * overlap
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
                detectable_grids = agent.sensor.get_detectable_grids(
                    next_pos, action.direction, self.grid_map
                )
                if detectable_grids:
                    # 直接判断边界索引
                    count = 0
                    for (gi, gj) in detectable_grids:
                        if (gi < 2 or gi >= self.grid_map.grid_height-2 or 
                            gj < 2 or gj >= self.grid_map.grid_width-2):
                            count += 1
                    boundary_reward += 0.5 * count
        
        return boundary_reward

    def update_boundary_weight(self):
        """更新边界权重（随时间衰减）"""
        self.boundary_weight = max(0.1, self.boundary_weight * (1 - self.boundary_decay_rate))

    def distributed_optimize(self, agent: 'Agent', 
                            paths: List[List['Action']], 
                            rewards: Dict[int, float]) -> List['Action']:
        """分布式优化控制序列 - 完善版"""
        # 选择最优路径索引
        if not rewards:
            return []
        best_path_idx = max(rewards, key=rewards.get)
        best_path = paths[best_path_idx]
        
        # 考虑协同约束（避让其他智能体）
        optimized_path = self.adjust_path_for_collaboration(agent, best_path)
        
        return optimized_path[:self.control_horizon]

    def run_step(self):
        """执行单步MPC控制流程"""
        # 同步时间步到地图（用于概率更新时间戳）
        self.grid_map.time_step = self.time_step
        
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
        self.grid_map.time_step = self.time_step

    def adjust_path_for_collaboration(self, agent: 'Agent', path: List['Action']) -> List['Action']:
        """调整路径以考虑协同约束"""
        adjusted_path = []
        
        for i, action in enumerate(path):
            next_pos = agent.predict_position(action)
            
            # 检查是否有碰撞风险
            collision_risk = False
            for other_agent in self.agents:
                if other_agent.id != agent.id:
                    if other_agent.planned_path and len(other_agent.planned_path) > i:
                        other_pos = other_agent.predict_position(other_agent.planned_path[i])
                        
                        distance = np.linalg.norm(np.array(next_pos) - np.array(other_pos))
                        
                        min_safe_distance = agent.safe_distance + other_agent.safe_distance
                        if distance < min_safe_distance:
                            collision_risk = True
                            break
            
            if collision_risk:
                # 生成避让动作（使用带转弯约束的可行动作）
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
        # 使用Agent的可行动作（含转弯约束）
        possible_actions = agent.generate_possible_actions()
        
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
            detectable = agent.sensor.get_detectable_grids(next_pos, action.direction, self.grid_map)
            if detectable:
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
        self.assigned_waypoint: Tuple[float, float] = None  # USV拦截任务目标点

    def can_detect_position(self, position: Tuple[float, float]) -> bool:
        """检查是否能探测到指定位置（基于当前姿态）"""
        distance = np.linalg.norm(np.array(self.position) - np.array(position))
        
        if self.agent_type == 'drone':
            dx = position[0] - self.position[0]
            dy = position[1] - self.position[1]
            angle = math.degrees(math.atan2(dy, dx))
            angle_diff = abs((angle - self.heading + 180) % 360 - 180)
            
            return distance <= self.sensor.range and angle_diff <= self.sensor.fov/2
        else:
            return distance <= self.sensor.range

    def receive_shared_info(self, sender_id: str, info: Dict):
        """接收其他智能体共享的信息"""
        self.shared_info[sender_id] = info

    def generate_possible_actions(self) -> List['Action']:
        """生成可能的动作集合 - 考虑转弯半径约束"""
        actions = []
        
        if self.agent_type == 'drone':
            directions = [0, 90, 180, 270]
            for direction in directions:
                if self.is_turn_feasible(direction, from_heading=self.heading):
                    end_pos = self.motion_model.calculate_end_position(self.position, direction)
                    actions.append(Action(
                        start_position=self.position,
                        end_position=end_pos,
                        direction=direction,
                        duration=1
                    ))
        else:
            directions = [0, 45, 90, 135, 180, 225, 270, 315]
            for direction in directions:
                if self.is_turn_feasible(direction, from_heading=self.heading):
                    end_pos = self.motion_model.calculate_end_position(self.position, direction)
                    actions.append(Action(
                        start_position=self.position,
                        end_position=end_pos,
                        direction=direction,
                        duration=1
                    ))
        
        return actions

    def is_turn_feasible(self, new_direction: float, from_heading: float = None) -> bool:
        """检查转弯是否满足最小转弯半径约束（允许指定起始航向）"""
        base_heading = self.heading if from_heading is None else from_heading
        angle_diff = abs((new_direction - base_heading + 180) % 360 - 180)
        
        if angle_diff < 1:
            return True
        
        if self.agent_type == 'drone':
            min_turn_radius = 100  # 无人机最小转弯半径100m
        else:
            min_turn_radius = 20   # 无人艇最小转弯半径20m
        
        # 简化：转弯半径与角度差相关（避免tan(0)）
        try:
            required_radius = self.motion_model.max_speed**2 / (9.8 * math.tan(math.radians(angle_diff)))
        except ZeroDivisionError:
            required_radius = float('inf')
        
        return required_radius >= min_turn_radius

    def generate_possible_paths(self, horizon: int, grid_map: 'SearchGrid') -> List[List['Action']]:
        """生成可能的路径集合（Beam Search）"""
        # Beam宽度根据平台类型自适应
        beam_width = 8 if self.agent_type == 'usv' else 12
        
        # 节点：(path, current_position, current_heading, score)
        initial_node = ([], self.position, self.heading, 0.0)
        beam = [initial_node]
        
        for depth in range(horizon):
            candidates = []
            for path, cur_pos, cur_heading, cur_score in beam:
                # 生成下一步动作（基于当前航向判断转弯可行）
                if self.agent_type == 'drone':
                    directions = [0, 90, 180, 270]
                else:
                    directions = [0, 45, 90, 135, 180, 225, 270, 315]
                for direction in directions:
                    if not self.is_turn_feasible(direction, from_heading=cur_heading):
                        continue
                    action = Action(
                        start_position=cur_pos,
                        end_position=self.motion_model.calculate_end_position(cur_pos, direction),
                        direction=direction,
                        duration=1
                    )
                    next_pos = action.end_position
                    # 局部启发式评分：可探测不确定度 + 目标概率偏好 - 转角代价
                    detectable = self.sensor.get_detectable_grids(next_pos, direction, grid_map)
                    local_score = 0.0
                    for grid_idx in detectable:
                        pos_m = grid_map.index_to_position(grid_idx)
                        local_score += grid_map.get_uncertainty(pos_m) + 0.5 * grid_map.get_target_probability(pos_m)
                    turn_cost = abs((direction - cur_heading + 180) % 360 - 180) / 180.0
                    local_score -= 0.2 * turn_cost
                    # UAV边界巡逻策略：前3600s优先覆盖边界、向最近边界靠拢并沿边行进
                    if self.agent_type == 'drone':
                        if getattr(grid_map, 'time_step', 0) < 3600:
                            # 距离边界越近加分
                            width_m = grid_map.width * 1852
                            height_m = grid_map.height * 1852
                            x, y = next_pos
                            dist_to_edges = min(x, y, width_m - x, height_m - y)
                            boundary_gain = max(0.0, (3000.0 - dist_to_edges) / 3000.0)  # 3km内有增益
                            local_score += 3.0 * boundary_gain
                            # 沿边方向加分（平行于x轴或y轴）
                            aligned = min(abs((direction % 90)), 90 - abs((direction % 90)))
                            local_score += 0.5 * (1.0 - aligned / 45.0)
                    # USV若有分配的拦截点，朝向该点加分
                    if self.agent_type == 'usv' and self.assigned_waypoint is not None:
                        dist_wp = np.linalg.norm(np.array(next_pos) - np.array(self.assigned_waypoint))
                        local_score += 5.0 / (1.0 + dist_wp / 100.0)
                    candidates.append((path + [action], next_pos, direction, cur_score + local_score))
            
            # 选取前K个候选
            if not candidates:
                break
            candidates.sort(key=lambda x: x[3], reverse=True)
            beam = candidates[:beam_width]
        
        # 只返回路径
        return [node[0] for node in beam] if beam else [[]]

    def update_local_state(self, grid_map: 'SearchGrid'):
        """更新智能体本地状态信息"""
        i, j = grid_map.position_to_index(self.position)
        
        self.local_target_prob = grid_map.target_probability[i, j]
        self.local_uncertainty = grid_map.uncertainty[i, j]
        self.on_boundary = grid_map.is_boundary_position(self.position)

    def predict_position(self, action: 'Action') -> Tuple[float, float]:
        """预测执行动作后的位置"""
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
        # 可探测网格缓存：(center_i, center_j, heading_int, agent_type) -> set[(i,j)]
        self._grid_cache: Dict[Tuple[int, int, int, str], Set[Tuple[int, int]]] = {}
        self._cache_hits = 0
        self._cache_misses = 0

    def scan(self, grid_map: 'SearchGrid', position: Tuple[float, float], 
            heading: float) -> Dict[Tuple[int, int], float]:
        """
        执行传感器扫描 - 同时支持静态与移动目标；输出包含正例与负例
        
        返回:
        探测到的网格位置和累计探测时间（未命中但在视场内的格点返回0）
        """
        detected_grids: Dict[Tuple[int, int], float] = {}
        
        # 处理所有目标
        for target in grid_map.targets:
            # 更新目标所在格
            if isinstance(target, MovingTarget):
                target.update_grid_position(grid_map)
            elif isinstance(target, Target):
                target.update_grid_position(grid_map)
            else:
                continue
            
            # 检查可探测性
            if self.is_target_detectable(target, position, heading):
                grid_pos = target.grid_position
                current_duration = grid_map.get_detection_duration(grid_pos)
                new_duration = current_duration + 1
                detected_grids[grid_pos] = new_duration
                # 持续10s确认
                if new_duration >= 10:
                    grid_map.confirm_target(grid_pos)
        
        # 负例：将当前视场内未命中的格点标注为0，便于概率衰减（下采样以提速，加入时间偏移避免棋盘格）
        fov_cells = self.get_detectable_grids(position, heading, grid_map)
        if fov_cells:
            stride = 2
            offset = (int(grid_map.time_step) % stride)
            for (i, j) in fov_cells:
                if ((i + j + offset) % stride) != 0:
                    continue
                if (i, j) not in detected_grids:
                    detected_grids[(i, j)] = 0.0
                    # 低概率轻微抖动，打散图案（不影响阈值判断）
                    # 注意：这里只写回detected_grids的值，实际概率更新在grid中进行
        
        return detected_grids

    def is_target_detectable(self, target: 'MovingTarget', 
                           sensor_position: Tuple[float, float], 
                           sensor_heading: float) -> bool:
        """检查目标是否在传感器探测范围内"""
        if self.agent_type == 'drone':
            dx = target.position[0] - sensor_position[0]
            dy = target.position[1] - sensor_position[1]
            distance = math.sqrt(dx**2 + dy**2)
            if distance > self.range:
                return False
                
            angle = math.degrees(math.atan2(dy, dx)) % 360
            angle_diff = abs((angle - sensor_heading + 180) % 360 - 180)
            return angle_diff <= self.fov/2
        else:
            dx = target.position[0] - sensor_position[0]
            dy = target.position[1] - sensor_position[1]
            distance = math.sqrt(dx**2 + dy**2)
            return distance <= self.range

    def get_detectable_grids(self, position: Tuple[float, float], 
                            heading: float, 
                            grid_map: 'SearchGrid') -> Set[Tuple[int, int]]:
        """
        获取传感器可探测的网格位置（带缓存）
        """
        center_i, center_j = grid_map.position_to_index(position)
        heading_int = int(round(heading)) % 360
        cache_key = (center_i, center_j, heading_int, self.agent_type)
        if cache_key in self._grid_cache:
            self._cache_hits += 1
            return self._grid_cache[cache_key]
        self._cache_misses += 1
        
        grid_range = int(self.range / (grid_map.resolution * 1852))
        
        detectable_grids: Set[Tuple[int, int]] = set()
        
        if self.agent_type == 'usv' or self.fov == 360:
            for i in range(max(0, center_i-grid_range), min(grid_map.grid_height, center_i+grid_range+1)):
                for j in range(max(0, center_j-grid_range), min(grid_map.grid_width, center_j+grid_range+1)):
                    if np.hypot(i-center_i, j-center_j) <= grid_range:
                        detectable_grids.add((i, j))
        else:
            angle_min = heading_int - self.fov/2
            angle_max = heading_int + self.fov/2
            
            for i in range(max(0, center_i-grid_range), min(grid_map.grid_height, center_i+grid_range+1)):
                for j in range(max(0, center_j-grid_range), min(grid_map.grid_width, center_j+grid_range+1)):
                    dx = j - center_j
                    dy = i - center_i
                    distance = np.hypot(dx, dy)
                    
                    if distance <= grid_range:
                        angle = math.degrees(math.atan2(dy, dx)) % 360
                        if angle_min <= angle <= angle_max or angle_min <= angle+360 <= angle_max:
                            detectable_grids.add((i, j))
        
        # 写入缓存（限制大小，简单LRU可后续加入；当前仅限制上限）
        if len(self._grid_cache) > 200000:
            self._grid_cache.clear()
        self._grid_cache[cache_key] = detectable_grids
        
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
            self.max_speed = 33.33  # 无人机最大速度 120km/h ≈ 33.33m/s
        else:  # 'usv'
            self.max_speed = 10.29  # 无人艇最大速度 20节 ≈ 10.29m/s
    
    def calculate_end_position(self, start: Tuple[float, float], direction: float) -> Tuple[float, float]:
        """计算动作结束位置 - 考虑速度约束"""
        angle_rad = math.radians(direction)
        
        # 0°朝北，x向右为正、y向上为正
        dx = self.max_speed * math.sin(angle_rad)
        dy = self.max_speed * math.cos(angle_rad)
        
        end_x = start[0] + dx
        end_y = start[1] + dy
        
        return (end_x, end_y)
    
    def generate_possible_actions(self, position: Tuple[float, float]) -> List['Action']:
        """生成可能的动作集合 - 考虑转弯半径约束（不直接使用，改为Agent生成）"""
        actions = []
        
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
        1. 基础概率场不随机：未更新格保持原值或缓慢回归到背景先验
        2. 探测到目标：概率显著增加 (70%-100%)
        3. 探测但未发现目标（负例）：概率降低至5%-15%
        """
        background_min = 0.1
        background_max = 0.2
        background_prior = 0.15
        
        # 处理当前探测结果
        for grid_idx, duration in detections.items():
            i, j = grid_idx
            self.detection_duration[i, j] = duration
            
            if duration > 0:  # 探测到目标
                self.target_probability[i, j] = min(1.0, 
                    max(0.7, self.target_probability[i, j] * 1.5 + 0.2))
            else:  # 负例：已扫描但未命中
                self.target_probability[i, j] = max(0.05,
                    min(0.15, self.target_probability[i, j] * 0.5))
            
            # 更新不确定性（熵）
            p = self.target_probability[i, j]
            if p in (0, 1):
                self.uncertainty[i, j] = 0.0
            else:
                self.uncertainty[i, j] = -p*math.log2(p) - (1-p)*math.log2(1-p)
            self.last_update_time[i, j] = self.time_step
        
        # 对于本步未更新的格点，缓慢回归到背景先验（去除随机性）
        for i in range(self.grid_height):
            for j in range(self.grid_width):
                if self.last_update_time[i, j] < self.time_step:
                    # 指数回归到背景先验
                    self.target_probability[i, j] = 0.98 * self.target_probability[i, j] + 0.02 * background_prior
                    p = self.target_probability[i, j]
                    if p in (0, 1):
                        self.uncertainty[i, j] = 0.0
                    else:
                        self.uncertainty[i, j] = -p*math.log2(p) - (1-p)*math.log2(1-p)

    def add_target_object(self, target: 'Target'):
        """添加目标对象"""
        self.targets.append(target)
        i, j = self.position_to_index(target.position)
        self.target_probability[i, j] = 0.8
        self.uncertainty[i, j] = 0.5
        self.confirmed_targets[i, j] = False
        self.detection_duration[i, j] = 0.0
        # 若为移动目标且未设置出生时间，则初始化为当前时间步
        if isinstance(target, MovingTarget) and not hasattr(target, 't0'):
            try:
                target.t0 = int(self.time_step)
            except Exception:
                target.t0 = 0

    def confirm_target(self, grid_idx: Tuple[int, int]):
        """确认目标位置"""
        i, j = grid_idx
        self.target_probability[i, j] = 1.0
        self.uncertainty[i, j] = 0.0
        self.confirmed_targets[i, j] = True
        self.detection_duration[i, j] = 10.0

    def get_target_probability(self, position: Tuple[float, float]) -> float:
        """获取目标存在概率（输入为实际坐标）"""
        i, j = self.position_to_index(position)
        return self.target_probability[i, j]

    def get_uncertainty(self, position: Tuple[float, float]) -> float:
        """获取环境不确定度（输入为实际坐标）"""
        i, j = self.position_to_index(position)
        return self.uncertainty[i, j]

    def get_detection_duration(self, position: Tuple[float, float]) -> float:
        """获取累计探测时间（输入为网格索引或实际坐标均可）"""
        if isinstance(position, tuple) and len(position) == 2 and all(isinstance(v, int) for v in position):
            i, j = position
        else:
            i, j = self.position_to_index(position)
        return self.detection_duration[i, j]

    def is_boundary_position(self, position: Tuple[float, float]) -> bool:
        """检查位置是否在边界区域"""
        i, j = self.position_to_index(position)
        return (i < 2 or i >= self.grid_height-2 or 
                j < 2 or j >= self.grid_width-2)

    def get_high_probability_targets(self, threshold: float = 0.6) -> List[Tuple[float, float]]:
        """获取高概率目标位置（实际坐标）"""
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
        grid_resolution_m = self.resolution * 1852
        
        j = int(position[0] / grid_resolution_m)
        i = int(position[1] / grid_resolution_m)
        
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
        i, j = index
        
        grid_resolution_m = self.resolution * 1852
        x = j * grid_resolution_m + grid_resolution_m / 2
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
                existing_target = None
                for target_id, target_data in self.targets.items():
                    if target_data['position'] == grid_idx:
                        existing_target = target_id
                        break
                
                if existing_target:
                    self.targets[existing_target]['last_seen'] = time_step
                    self.targets[existing_target]['duration'] = duration
                    
                    if duration >= 10:
                        self.targets[existing_target]['status'] = 'confirmed'
                    elif duration > 0:
                        self.targets[existing_target]['status'] = 'partial'
                else:
                    target_id = f"target_{self.next_target_id}"
                    self.next_target_id += 1
                    self.targets[target_id] = {
                        'position': grid_idx,
                        'last_seen': time_step,
                        'duration': duration,
                        'status': 'partial' if duration > 0 else 'new'
                    }
            else:
                # 负例：降低该位置的持续观测
                for target_id, target_data in self.targets.items():
                    if target_data['position'] == grid_idx:
                        self.targets[target_id]['duration'] = max(0, target_data['duration'] - 1)
                        if self.targets[target_id]['duration'] == 0:
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
        self.detection_time = 0
        self.reported = False
    
    def move(self, time_step: float):
        """移动目标"""
        if random.random() < 0.1:
            self.direction = (self.direction + random.uniform(-30, 30)) % 360
        
        angle_rad = math.radians(self.direction)
        dx = self.speed * time_step * math.cos(angle_rad)
        dy = self.speed * time_step * math.sin(angle_rad)
        self.position = (self.position[0] + dx, self.position[1] + dy)
    
    def check_detection(self, agents: List['Agent'], time_step: int):
        """供仿真框架调用的简易检测累积接口（与传感器一致阈值10s）"""
        for agent in agents:
            if agent.can_detect_position(self.position):
                self.detection_time += 1
                if self.detection_time >= 10:
                    self.detected = True
                return
        # 未被任何平台看到时衰减
        self.detection_time = max(0, self.detection_time - 1)

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
        self.detection_count = 0
        self.grid_position = None
    
    def update_grid_position(self, grid_map: 'SearchGrid'):
        """更新网格位置"""
        self.grid_position = grid_map.position_to_index(self.position)
    
    def is_in_detection_range(self, agent: 'Agent') -> bool:
        """检查是否在智能体探测范围内"""
        return agent.can_detect_position(self.position)
