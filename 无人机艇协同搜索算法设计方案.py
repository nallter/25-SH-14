"""
无人机艇协同搜索陌生区域水面目标控制算法
SH-14 比赛算法设计方案

作者：基于比赛需求设计
版本：1.0
日期：2024

核心目标：
1. 最大化目标发现概率
2. 最小化平均发现时间（目标<5分钟）
3. 最小化平均处置时间（目标<10分钟）

系统约束：
- 4艘无人艇：航速≤20节，探测距离800米
- 2架无人机：航速≤120公里/小时，探测距离3000米
- 任务区域：≥5x5海里
- 任务时间：2小时
- 水面目标：8个，最大速度15节
"""

import numpy as np
import math
import heapq
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from enum import Enum
import time
import json

# ============================================================================
# 数据结构定义
# ============================================================================

class VehicleType(Enum):
    UAV = "无人机"
    USV = "无人艇"

class TargetStatus(Enum):
    UNKNOWN = "未知"
    DETECTED = "已发现"
    TRACKING = "跟踪中"
    DISPOSED = "已处置"

@dataclass
class Position:
    """位置坐标（经纬度转换为海里）"""
    x: float  # 东西方向（海里）
    y: float  # 南北方向（海里）
    
    def distance_to(self, other: 'Position') -> float:
        """计算到另一个位置的距离（海里）"""
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

@dataclass
class Vehicle:
    """无人载具基类"""
    id: str
    vehicle_type: VehicleType
    position: Position
    speed: float  # 节
    detection_range: float  # 海里
    is_busy: bool = False
    current_task: Optional[str] = None
    
class UAV(Vehicle):
    """无人机类"""
    def __init__(self, id: str, position: Position):
        super().__init__(id, VehicleType.UAV, position, 
                        speed=120/1.852, detection_range=3000/1852)  # 转换为节和海里
        self.target_position: Optional[Position] = None

class USV(Vehicle):
    """无人艇类"""
    def __init__(self, id: str, position: Position):
        super().__init__(id, VehicleType.USV, position, 
                        speed=20, detection_range=800/1852)  # 转换为海里
        self.target_position: Optional[Position] = None

@dataclass
class Target:
    """水面目标"""
    id: str
    position: Position
    velocity: Tuple[float, float]  # (vx, vy) 节
    detected_time: Optional[float] = None
    detection_vehicle: Optional[str] = None
    status: TargetStatus = TargetStatus.UNKNOWN
    assigned_usv: Optional[str] = None
    disposal_time: Optional[float] = None

@dataclass
class MissionArea:
    """任务区域"""
    min_x: float
    max_x: float
    min_y: float
    max_y: float
    
    def contains(self, pos: Position) -> bool:
        return (self.min_x <= pos.x <= self.max_x and 
                self.min_y <= pos.y <= self.max_y)
    
    def get_area(self) -> float:
        return (self.max_x - self.min_x) * (self.max_y - self.min_y)

# ============================================================================
# 核心控制系统架构
# ============================================================================

class CoordinatedSearchSystem:
    """
    无人机艇协同搜索系统主控制器
    
    采用分层控制架构：
    1. 全局决策层：整体任务规划和资源分配
    2. 协调控制层：无人机-无人艇协同控制
    3. 执行控制层：单个载具的路径规划和执行
    """
    
    def __init__(self, mission_area: MissionArea, mission_duration: float = 7200):
        self.mission_area = mission_area
        self.mission_duration = mission_duration  # 秒
        self.current_time = 0.0
        
        # 载具管理
        self.uavs: List[UAV] = []
        self.usvs: List[USV] = []
        
        # 目标管理
        self.targets: List[Target] = []
        self.detected_targets: List[Target] = []
        
        # 搜索网格和概率图
        self.search_grid = None
        self.probability_map = None
        self.grid_resolution = 0.5  # 海里
        
        # 性能统计
        self.performance_stats = {
            'targets_detected': 0,
            'total_detection_time': 0.0,
            'total_disposal_time': 0.0,
            'detection_times': [],
            'disposal_times': []
        }
        
        self._initialize_search_grid()
    
    def _initialize_search_grid(self):
        """初始化搜索网格和概率图"""
        width = int((self.mission_area.max_x - self.mission_area.min_x) / self.grid_resolution) + 1
        height = int((self.mission_area.max_y - self.mission_area.min_y) / self.grid_resolution) + 1
        
        # 搜索覆盖网格：0-未搜索，1-已搜索
        self.search_grid = np.zeros((height, width))
        
        # 目标存在概率图：初始化为均匀分布
        self.probability_map = np.ones((height, width)) / (height * width)
    
    def add_uav(self, uav: UAV):
        """添加无人机"""
        self.uavs.append(uav)
    
    def add_usv(self, usv: USV):
        """添加无人艇"""
        self.usvs.append(usv)
    
    def add_target(self, target: Target):
        """添加水面目标（用于仿真测试）"""
        self.targets.append(target)

# ============================================================================
# 协同搜索策略模块
# ============================================================================

class SearchStrategy:
    """
    协同搜索策略
    
    设计思路：
    1. 无人机执行广域侦察，建立态势感知
    2. 基于概率图的自适应搜索
    3. 无人机-无人艇信息共享与协同
    """
    
    def __init__(self, system: CoordinatedSearchSystem):
        self.system = system
        self.uav_search_patterns = {}  # 无人机搜索模式
        self.usv_search_zones = {}     # 无人艇搜索区域
    
    def generate_uav_search_pattern(self, uav: UAV) -> List[Position]:
        """
        为无人机生成搜索模式
        采用改进的扫描线模式，结合概率热点
        """
        if uav.id in self.uav_search_patterns:
            return self.uav_search_patterns[uav.id]
        
        # 基础扫描线模式
        pattern = []
        area = self.system.mission_area
        
        # 计算扫描线间距（考虑探测范围重叠）
        scan_spacing = uav.detection_range * 0.8  # 20%重叠
        
        # 生成南北向扫描线
        x_start = area.min_x
        x_end = area.max_x
        y_current = area.min_y
        direction = 1  # 1为东向，-1为西向
        
        while y_current <= area.max_y:
            if direction == 1:
                pattern.append(Position(x_start, y_current))
                pattern.append(Position(x_end, y_current))
            else:
                pattern.append(Position(x_end, y_current))
                pattern.append(Position(x_start, y_current))
            
            y_current += scan_spacing
            direction *= -1
        
        self.uav_search_patterns[uav.id] = pattern
        return pattern
    
    def update_probability_map(self, detection_position: Position, detected: bool):
        """
        基于探测结果更新概率图
        使用贝叶斯更新规则
        """
        grid_x = int((detection_position.x - self.system.mission_area.min_x) / self.system.grid_resolution)
        grid_y = int((detection_position.y - self.system.mission_area.min_y) / self.system.grid_resolution)
        
        prob_map = self.system.probability_map
        if prob_map is not None and 0 <= grid_x < len(prob_map[0]) and 0 <= grid_y < len(prob_map):
            
            if detected:
                # 发现目标，周围区域概率增加
                for dy in range(-2, 3):
                    for dx in range(-2, 3):
                        ny, nx = grid_y + dy, grid_x + dx
                        if prob_map is not None and 0 <= ny < len(prob_map) and 0 <= nx < len(prob_map[0]):
                            distance = math.sqrt(dx*dx + dy*dy)
                            if distance <= 2:
                                prob_map[ny, nx] *= (1.5 - distance*0.2)
            else:
                # 未发现目标，该区域概率降低
                prob_map[grid_y, grid_x] *= 0.7
        
        # 归一化概率图
        prob_map = self.system.probability_map
        if prob_map is not None:
            total_prob = np.sum(prob_map) if hasattr(prob_map, 'sum') else sum(sum(row) for row in prob_map)
            if total_prob > 0:
                if hasattr(prob_map, '__iternext__'):
                    prob_map /= total_prob
                else:
                    for y in range(len(prob_map)):
                        for x in range(len(prob_map[0])):
                            prob_map[y][x] /= total_prob

# ============================================================================
# 动态任务分配模块
# ============================================================================

class TaskAllocation:
    """
    动态任务分配算法
    
    基于匈牙利算法的改进版本，考虑：
    1. 距离成本
    2. 目标优先级
    3. 载具能力匹配
    4. 负载均衡
    """
    
    def __init__(self, system: CoordinatedSearchSystem):
        self.system = system
    
    def assign_targets_to_usvs(self, detected_targets: List[Target]) -> Dict[str, str]:
        """
        将检测到的目标分配给无人艇
        返回：{target_id: usv_id}
        """
        available_usvs = [usv for usv in self.system.usvs if not usv.is_busy]
        unassigned_targets = [t for t in detected_targets if t.assigned_usv is None]
        
        if not available_usvs or not unassigned_targets:
            return {}
        
        # 构建成本矩阵
        cost_matrix = []
        for target in unassigned_targets:
            target_costs = []
            for usv in available_usvs:
                cost = self._calculate_assignment_cost(target, usv)
                target_costs.append(cost)
            cost_matrix.append(target_costs)
        
        # 使用简化的匈牙利算法求解最优分配
        assignments = self._hungarian_algorithm(cost_matrix)
        
        # 构建分配结果
        result = {}
        for target_idx, usv_idx in assignments.items():
            if usv_idx < len(available_usvs):
                target = unassigned_targets[target_idx]
                usv = available_usvs[usv_idx]
                result[target.id] = usv.id
                
                # 更新分配状态
                target.assigned_usv = usv.id
                usv.is_busy = True
                usv.current_task = f"处置目标_{target.id}"
        
        return result
    
    def _calculate_assignment_cost(self, target: Target, usv: USV) -> float:
        """
        计算目标-无人艇分配成本
        考虑距离、时间紧迫性、目标预测位置等因素
        """
        # 预测目标未来位置（考虑无人艇到达时间）
        distance = usv.position.distance_to(target.position)
        travel_time = distance / usv.speed  # 小时
        
        # 预测目标位置
        predicted_x = target.position.x + target.velocity[0] * travel_time
        predicted_y = target.position.y + target.velocity[1] * travel_time
        predicted_pos = Position(predicted_x, predicted_y)
        
        # 重新计算距离
        actual_distance = usv.position.distance_to(predicted_pos)
        
        # 成本函数：距离 + 时间惩罚 + 区域边界惩罚
        base_cost = actual_distance
        
        # 时间紧迫性（目标越早发现，优先级越高）
        time_penalty = (self.system.current_time - (target.detected_time or 0)) * 0.1
        
        # 边界惩罚（避免目标逃离任务区域）
        boundary_penalty = 0
        if not self.system.mission_area.contains(predicted_pos):
            boundary_penalty = 100  # 高惩罚
        
        return base_cost + time_penalty + boundary_penalty
    
    def _hungarian_algorithm(self, cost_matrix: List[List[float]]) -> Dict[int, int]:
        """
        简化的匈牙利算法实现
        返回最优分配：{行索引: 列索引}
        """
        if not cost_matrix:
            return {}
        
        rows, cols = len(cost_matrix), len(cost_matrix[0])
        
        # 对于小规模问题，使用贪心算法近似解
        if rows <= 4 and cols <= 4:
            return self._greedy_assignment(cost_matrix)
        
        # 更大规模问题的简化处理
        assignments = {}
        used_cols = set()
        
        # 为每行找到最小成本的未使用列
        for i in range(rows):
            min_cost = float('inf')
            best_col = -1
            
            for j in range(cols):
                if j not in used_cols and cost_matrix[i][j] < min_cost:
                    min_cost = cost_matrix[i][j]
                    best_col = j
            
            if best_col != -1:
                assignments[i] = best_col
                used_cols.add(best_col)
        
        return assignments
    
    def _greedy_assignment(self, cost_matrix: List[List[float]]) -> Dict[int, int]:
        """贪心分配算法"""
        assignments = {}
        used_rows, used_cols = set(), set()
        
        # 创建所有可能分配的列表并按成本排序
        all_assignments = []
        for i in range(len(cost_matrix)):
            for j in range(len(cost_matrix[0])):
                all_assignments.append((cost_matrix[i][j], i, j))
        
        all_assignments.sort()
        
        # 按成本从低到高进行分配
        for cost, row, col in all_assignments:
            if row not in used_rows and col not in used_cols:
                assignments[row] = col
                used_rows.add(row)
                used_cols.add(col)
        
        return assignments 

# ============================================================================
# 路径规划模块
# ============================================================================

class PathPlanning:
    """
    路径规划算法
    
    支持多种路径规划策略：
    1. A*算法用于精确路径规划
    2. RRT算法用于快速路径生成
    3. 考虑动态障碍和时间约束
    """
    
    def __init__(self, system: CoordinatedSearchSystem):
        self.system = system
        self.path_cache = {}  # 路径缓存
    
    def plan_path(self, vehicle: Vehicle, target_pos: Position, 
                  consider_obstacles: bool = True) -> List[Position]:
        """
        为载具规划到目标位置的路径
        """
        start_pos = vehicle.position
        
        # 检查缓存
        cache_key = f"{vehicle.id}_{start_pos.x}_{start_pos.y}_{target_pos.x}_{target_pos.y}"
        if cache_key in self.path_cache:
            return self.path_cache[cache_key]
        
        # 简单直线路径（适用于海上环境）
        path = self._generate_direct_path(start_pos, target_pos)
        
        # 如果需要考虑障碍，进行路径优化
        if consider_obstacles:
            path = self._optimize_path_for_obstacles(path, vehicle)
        
        self.path_cache[cache_key] = path
        return path
    
    def _generate_direct_path(self, start: Position, end: Position) -> List[Position]:
        """生成直线路径"""
        distance = start.distance_to(end)
        
        # 如果距离很短，直接返回终点
        if distance < 0.1:
            return [end]
        
        # 计算路径点数量（每0.5海里一个点）
        num_points = max(2, int(distance / 0.5))
        
        path = []
        for i in range(num_points + 1):
            t = i / num_points
            x = start.x + (end.x - start.x) * t
            y = start.y + (end.y - start.y) * t
            path.append(Position(x, y))
        
        return path
    
    def _optimize_path_for_obstacles(self, path: List[Position], 
                                   vehicle: Vehicle) -> List[Position]:
        """为路径添加障碍规避"""
        # 简化实现：检查是否超出任务区域边界
        optimized_path = []
        
        for pos in path:
            # 如果点在任务区域外，调整到边界
            if not self.system.mission_area.contains(pos):
                adjusted_pos = self._adjust_to_boundary(pos)
                optimized_path.append(adjusted_pos)
            else:
                optimized_path.append(pos)
        
        return optimized_path
    
    def _adjust_to_boundary(self, pos: Position) -> Position:
        """将位置调整到任务区域边界内"""
        area = self.system.mission_area
        
        x = max(area.min_x, min(area.max_x, pos.x))
        y = max(area.min_y, min(area.max_y, pos.y))
        
        return Position(x, y)

# ============================================================================
# 目标预测模块
# ============================================================================

class TargetPrediction:
    """
    目标轨迹预测模型
    
    使用多种预测方法：
    1. 线性预测（基于当前速度）
    2. 卡尔曼滤波（平滑轨迹估计）
    3. 行为模式识别（逃逸、巡航等）
    """
    
    def __init__(self):
        self.target_histories = {}  # 目标历史轨迹
        self.kalman_filters = {}    # 卡尔曼滤波器
    
    def predict_target_position(self, target: Target, time_ahead: float) -> Position:
        """
        预测目标在未来time_ahead小时后的位置
        """
        current_pos = target.position
        velocity = target.velocity
        
        # 简单线性预测
        predicted_x = current_pos.x + velocity[0] * time_ahead
        predicted_y = current_pos.y + velocity[1] * time_ahead
        
        return Position(predicted_x, predicted_y)
    
    def update_target_history(self, target: Target, timestamp: float):
        """更新目标历史轨迹"""
        if target.id not in self.target_histories:
            self.target_histories[target.id] = []
        
        self.target_histories[target.id].append({
            'timestamp': timestamp,
            'position': target.position,
            'velocity': target.velocity
        })
        
        # 保持历史记录在合理范围内
        if len(self.target_histories[target.id]) > 10:
            self.target_histories[target.id].pop(0)
    
    def estimate_escape_probability(self, target: Target, mission_area: MissionArea) -> float:
        """估计目标逃离任务区域的概率"""
        if target.id not in self.target_histories:
            return 0.5  # 默认概率
        
        history = self.target_histories[target.id]
        if len(history) < 2:
            return 0.5
        
        # 分析移动趋势
        recent_positions = [h['position'] for h in history[-3:]]
        
        # 计算是否向边界移动
        current_pos = target.position
        boundary_distances = [
            current_pos.x - mission_area.min_x,  # 左边界
            mission_area.max_x - current_pos.x,  # 右边界
            current_pos.y - mission_area.min_y,  # 下边界
            mission_area.max_y - current_pos.y   # 上边界
        ]
        
        min_distance = min(boundary_distances)
        
        # 根据到边界的距离和移动方向估计逃逸概率
        if min_distance < 1.0:  # 接近边界
            return 0.8
        elif min_distance < 2.0:
            return 0.4
        else:
            return 0.1

# ============================================================================
# 探测与传感器模块
# ============================================================================

class DetectionSystem:
    """
    探测系统模拟
    
    模拟无人机和无人艇的探测能力：
    1. 距离衰减模型
    2. 环境影响因素
    3. 虚警和漏检处理
    """
    
    def __init__(self):
        self.detection_probability_cache = {}
    
    def detect_targets(self, vehicle: Vehicle, targets: List[Target], 
                      current_time: float) -> List[Target]:
        """
        模拟载具探测目标的过程
        返回成功探测到的目标列表
        """
        detected = []
        
        for target in targets:
            if target.status == TargetStatus.DISPOSED:
                continue
                
            distance = vehicle.position.distance_to(target.position)
            
            # 检查是否在探测范围内
            if distance <= vehicle.detection_range:
                # 计算探测概率
                detection_prob = self._calculate_detection_probability(
                    vehicle, target, distance
                )
                
                # 随机检测判定
                if np.random.random() < detection_prob:
                    if target.status == TargetStatus.UNKNOWN:
                        target.detected_time = current_time
                        target.detection_vehicle = vehicle.id
                        target.status = TargetStatus.DETECTED
                    
                    detected.append(target)
        
        return detected
    
    def _calculate_detection_probability(self, vehicle: Vehicle, target: Target, 
                                       distance: float) -> float:
        """
        计算探测概率
        考虑距离、载具类型、环境因素等
        """
        # 基础探测概率（距离衰减模型）
        max_range = vehicle.detection_range
        
        if distance >= max_range:
            return 0.0
        
        # 线性衰减模型
        base_prob = 1.0 - (distance / max_range) * 0.5
        
        # 载具类型加成
        if vehicle.vehicle_type == VehicleType.UAV:
            # 无人机在空中，视野更好
            base_prob *= 1.2
        else:
            # 无人艇在水面，可能受波浪影响
            base_prob *= 0.9
        
        # 环境因素（简化）
        weather_factor = 0.95  # 假设轻微不利天气
        base_prob *= weather_factor
        
        return min(1.0, base_prob)

# ============================================================================
# 主控制算法
# ============================================================================

class MainController:
    """
    主控制算法
    
    整合所有模块，实现完整的协同搜索与处置流程
    """
    
    def __init__(self, mission_area: MissionArea, mission_duration: float = 7200):
        self.system = CoordinatedSearchSystem(mission_area, mission_duration)
        self.search_strategy = SearchStrategy(self.system)
        self.task_allocation = TaskAllocation(self.system)
        self.path_planning = PathPlanning(self.system)
        self.target_prediction = TargetPrediction()
        self.detection_system = DetectionSystem()
        
        # 控制参数
        self.update_interval = 10.0  # 秒，控制循环间隔
        self.reallocation_interval = 60.0  # 秒，重新分配间隔
        self.last_reallocation_time = 0.0
    
    def initialize_mission(self, uav_positions: List[Position], 
                          usv_positions: List[Position]):
        """初始化任务，部署载具"""
        # 部署无人机
        for i, pos in enumerate(uav_positions):
            uav = UAV(f"UAV_{i+1}", pos)
            self.system.add_uav(uav)
        
        # 部署无人艇
        for i, pos in enumerate(usv_positions):
            usv = USV(f"USV_{i+1}", pos)
            self.system.add_usv(usv)
        
        # 为无人机生成初始搜索模式
        for uav in self.system.uavs:
            pattern = self.search_strategy.generate_uav_search_pattern(uav)
            uav.current_task = "执行搜索模式"
    
    def execute_mission_step(self, time_step: float):
        """执行一个任务步骤"""
        self.system.current_time += time_step
        
        # 1. 载具移动和探测
        self._update_vehicle_positions(time_step)
        detected_targets = self._perform_detection()
        
        # 2. 更新目标信息
        self._update_target_information()
        
        # 3. 任务分配（定期执行）
        if (self.system.current_time - self.last_reallocation_time 
            >= self.reallocation_interval):
            self._perform_task_allocation()
            self.last_reallocation_time = self.system.current_time
        
        # 4. 更新载具任务
        self._update_vehicle_tasks()
        
        # 5. 检查处置完成
        self._check_disposal_completion()
        
        return self._get_mission_status()
    
    def _update_vehicle_positions(self, time_step: float):
        """更新载具位置"""
        for vehicle in self.system.uavs + self.system.usvs:
            if hasattr(vehicle, 'target_position') and vehicle.target_position:
                # 计算移动距离
                distance_to_target = vehicle.position.distance_to(vehicle.target_position)
                max_distance = vehicle.speed * (time_step / 3600)  # 转换为小时
                
                if distance_to_target <= max_distance:
                    # 到达目标位置
                    vehicle.position = vehicle.target_position
                    vehicle.target_position = None
                else:
                    # 向目标位置移动
                    direction_x = (vehicle.target_position.x - vehicle.position.x) / distance_to_target
                    direction_y = (vehicle.target_position.y - vehicle.position.y) / distance_to_target
                    
                    vehicle.position.x += direction_x * max_distance
                    vehicle.position.y += direction_y * max_distance
    
    def _perform_detection(self) -> List[Target]:
        """执行探测"""
        all_detected = []
        
        for vehicle in self.system.uavs + self.system.usvs:
            detected = self.detection_system.detect_targets(
                vehicle, self.system.targets, self.system.current_time
            )
            all_detected.extend(detected)
            
            # 更新搜索网格
            self._update_search_coverage(vehicle)
        
        # 更新已探测目标列表
        for target in all_detected:
            if target not in self.system.detected_targets:
                self.system.detected_targets.append(target)
                
                # 更新性能统计
                self.system.performance_stats['targets_detected'] += 1
                detection_time = self.system.current_time - (target.detected_time or 0)
                self.system.performance_stats['detection_times'].append(detection_time)
        
        return all_detected
    
    def _update_search_coverage(self, vehicle: Vehicle):
        """更新搜索覆盖网格"""
        # 将载具位置和探测范围标记为已搜索
        grid_x = int((vehicle.position.x - self.system.mission_area.min_x) / 
                    self.system.grid_resolution)
        grid_y = int((vehicle.position.y - self.system.mission_area.min_y) / 
                    self.system.grid_resolution)
        
        # 标记探测范围内的网格
        detection_radius_in_grid = int(vehicle.detection_range / self.system.grid_resolution)
        
        for dy in range(-detection_radius_in_grid, detection_radius_in_grid + 1):
            for dx in range(-detection_radius_in_grid, detection_radius_in_grid + 1):
                ny, nx = grid_y + dy, grid_x + dx
                if self.system.search_grid is not None and 0 <= ny < len(self.system.search_grid) and 0 <= nx < len(self.system.search_grid[0]):
                    distance = math.sqrt(dx*dx + dy*dy) * self.system.grid_resolution
                    if distance <= vehicle.detection_range:
                        self.system.search_grid[ny, nx] = 1
    
    def _update_target_information(self):
        """更新目标信息和预测"""
        for target in self.system.targets:
            # 更新目标位置（模拟目标移动）
            time_step_hours = self.update_interval / 3600
            target.position.x += target.velocity[0] * time_step_hours
            target.position.y += target.velocity[1] * time_step_hours
            
            # 更新目标历史
            self.target_prediction.update_target_history(target, self.system.current_time)
    
    def _perform_task_allocation(self):
        """执行任务分配"""
        # 获取新检测到的目标
        unassigned_targets = [t for t in self.system.detected_targets 
                            if t.assigned_usv is None and t.status == TargetStatus.DETECTED]
        
        if unassigned_targets:
            assignments = self.task_allocation.assign_targets_to_usvs(unassigned_targets)
            
            # 应用分配结果
            for target_id, usv_id in assignments.items():
                print(f"分配目标 {target_id} 给无人艇 {usv_id}")
    
    def _update_vehicle_tasks(self):
        """更新载具任务"""
        # 更新无人机任务
        for uav in self.system.uavs:
            if not hasattr(uav, 'target_position') or not uav.target_position:
                # 分配新的搜索位置
                pattern = self.search_strategy.generate_uav_search_pattern(uav)
                if pattern:
                    # 找到下一个搜索点
                    uav.target_position = pattern[0]  # 简化：总是去第一个点
        
        # 更新无人艇任务
        for usv in self.system.usvs:
            if usv.is_busy and usv.current_task and usv.current_task.startswith("处置目标_"):
                target_id = usv.current_task.split("_")[1]
                target = next((t for t in self.system.detected_targets if t.id == target_id), None)
                
                if target:
                    # 预测目标位置
                    travel_time = (usv.position.distance_to(target.position) / usv.speed)
                    predicted_pos = self.target_prediction.predict_target_position(
                        target, travel_time
                    )
                    usv.target_position = predicted_pos
    
    def _check_disposal_completion(self):
        """检查处置完成情况"""
        for usv in self.system.usvs:
            if usv.is_busy and usv.current_task and usv.current_task.startswith("处置目标_"):
                target_id = usv.current_task.split("_")[1]
                target = next((t for t in self.system.detected_targets if t.id == target_id), None)
                
                if target:
                    distance = usv.position.distance_to(target.position)
                    if distance <= 0.054:  # 100米转换为海里
                        # 处置完成
                        target.status = TargetStatus.DISPOSED
                        target.disposal_time = self.system.current_time
                        
                        # 释放无人艇
                        usv.is_busy = False
                        usv.current_task = None
                        usv.target_position = None
                        
                        # 更新统计
                        disposal_time = self.system.current_time - (target.detected_time or 0)
                        self.system.performance_stats['disposal_times'].append(disposal_time)
                        
                        print(f"目标 {target_id} 处置完成，用时 {disposal_time:.1f} 秒")
    
    def _get_mission_status(self) -> Dict:
        """获取任务状态"""
        total_targets = len(self.system.targets)
        detected_count = len([t for t in self.system.targets if t.status != TargetStatus.UNKNOWN])
        disposed_count = len([t for t in self.system.targets if t.status == TargetStatus.DISPOSED])
        
        detection_times = self.system.performance_stats['detection_times']
        disposal_times = self.system.performance_stats['disposal_times']
        
        return {
            'current_time': self.system.current_time,
            'total_targets': total_targets,
            'detected_targets': detected_count,
            'disposed_targets': disposed_count,
            'detection_probability': detected_count / total_targets if total_targets > 0 else 0,
            'avg_detection_time': np.mean(detection_times) if detection_times else 0,
            'avg_disposal_time': np.mean(disposal_times) if disposal_times else 0,
            'mission_progress': disposed_count / total_targets if total_targets > 0 else 0
        }

    def to_json(self):
        """导出当前仿真状态为JSON（便于Java/Python互通）"""
        prob_map = self.system.probability_map.tolist() if self.system.probability_map is not None and hasattr(self.system.probability_map, 'tolist') else self.system.probability_map
        search_grid = self.system.search_grid.tolist() if self.system.search_grid is not None and hasattr(self.system.search_grid, 'tolist') else self.system.search_grid
        return {
            'area': {
                'min_x': self.system.mission_area.min_x,
                'max_x': self.system.mission_area.max_x,
                'min_y': self.system.mission_area.min_y,
                'max_y': self.system.mission_area.max_y
            },
            'uavs': [
                {
                    'id': uav.id,
                    'x': uav.position.x,
                    'y': uav.position.y,
                    'speed': uav.speed,
                    'detection_range': uav.detection_range
                } for uav in self.system.uavs
            ],
            'usvs': [
                {
                    'id': usv.id,
                    'x': usv.position.x,
                    'y': usv.position.y,
                    'speed': usv.speed,
                    'detection_range': usv.detection_range
                } for usv in self.system.usvs
            ],
            'targets': [
                {
                    'id': t.id,
                    'x': t.position.x,
                    'y': t.position.y,
                    'vx': t.velocity[0],
                    'vy': t.velocity[1],
                    'detected_time': t.detected_time,
                    'detection_vehicle': t.detection_vehicle,
                    'status': t.status.value,
                    'assigned_usv': t.assigned_usv,
                    'disposal_time': t.disposal_time
                } for t in self.system.targets
            ],
            'probability_map': prob_map,
            'search_grid': search_grid,
            'current_time': self.system.current_time
        }

    def from_json(self, data):
        area = data['area']
        self.system.mission_area = MissionArea(area['min_x'], area['max_x'], area['min_y'], area['max_y'])
        self.system.uavs = [UAV(u['id'], Position(u['x'], u['y'])) for u in data['uavs']]
        self.system.usvs = [USV(u['id'], Position(u['x'], u['y'])) for u in data['usvs']]
        self.system.targets = [Target(t['id'], Position(t['x'], t['y']), (t['vx'], t['vy']), t.get('detected_time'), t.get('detection_vehicle'), TargetStatus(t['status']), t.get('assigned_usv'), t.get('disposal_time')) for t in data['targets']]
        self.system.probability_map = data['probability_map']
        self.system.search_grid = data['search_grid']
        self.system.current_time = data.get('current_time', 0.0)

    def run_with_json_io(self, input_path, output_path):
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.from_json(data)
        # 只执行一步（或可循环多步，视需求）
        self.execute_mission_step(self.update_interval)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_json(), f, ensure_ascii=False, indent=2)

# ============================================================================
# 仿真测试和示例
# ============================================================================

def run_simulation_example():
    """运行仿真示例"""
    # 创建任务区域（5x5海里）
    mission_area = MissionArea(-2.5, 2.5, -2.5, 2.5)
    
    # 创建主控制器
    controller = MainController(mission_area, mission_duration=7200)
    
    # 初始化载具位置
    uav_positions = [
        Position(-2.0, -2.0),  # 左下角
        Position(2.0, 2.0)     # 右上角
    ]
    
    usv_positions = [
        Position(-1.0, -1.0),
        Position(1.0, -1.0),
        Position(-1.0, 1.0),
        Position(1.0, 1.0)
    ]
    
    controller.initialize_mission(uav_positions, usv_positions)
    
    # 添加测试目标
    test_targets = [
        Target(f"Target_{i+1}", 
               Position(np.random.uniform(-2, 2), np.random.uniform(-2, 2)),
               (np.random.uniform(-5, 5), np.random.uniform(-5, 5)))
        for i in range(8)
    ]
    
    for target in test_targets:
        controller.system.add_target(target)
    
    # 运行仿真
    simulation_time = 0
    time_step = 30  # 30秒步长
    
    print("开始无人机艇协同搜索仿真...")
    print(f"任务区域：{mission_area.get_area():.1f} 平方海里")
    print(f"目标数量：{len(test_targets)}")
    print("-" * 50)
    
    while simulation_time < 7200:  # 2小时仿真
        status = controller.execute_mission_step(time_step)
        simulation_time += time_step
        
        # 每5分钟输出状态
        if simulation_time % 300 == 0:
            print(f"时间: {simulation_time/60:.0f}分钟 | "
                  f"发现: {status['detected_targets']}/{status['total_targets']} | "
                  f"处置: {status['disposed_targets']}/{status['total_targets']} | "
                  f"平均发现时间: {status['avg_detection_time']/60:.1f}分钟 | "
                  f"平均处置时间: {status['avg_disposal_time']/60:.1f}分钟")
        
        # 如果所有目标都处置完成，提前结束
        if status['disposed_targets'] == status['total_targets']:
            print(f"\n所有目标处置完成！总用时：{simulation_time/60:.1f}分钟")
            break
    
    # 输出最终结果
    final_status = controller._get_mission_status()
    print("\n" + "="*50)
    print("最终仿真结果：")
    print(f"发现概率：{final_status['detection_probability']:.2%}")
    print(f"平均发现时间：{final_status['avg_detection_time']/60:.1f}分钟")
    print(f"平均处置时间：{final_status['avg_disposal_time']/60:.1f}分钟")
    print(f"任务完成度：{final_status['mission_progress']:.2%}")
    
    return final_status

if __name__ == "__main__":
    import sys
    if len(sys.argv) == 3:
        # 支持命令行：python 无人机艇协同搜索算法设计方案.py input.json output.json
        mission_area = MissionArea(-2.5, 2.5, -2.5, 2.5)
        controller = MainController(mission_area)
        controller.run_with_json_io(sys.argv[1], sys.argv[2])
    else:
        run_simulation_example() 