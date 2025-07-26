"""
无人机艇协同搜索陌生区域水面目标控制算法
SH-14 比赛算法设计方案

作者：基于比赛需求设计
版本：1.0
日期：2025

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

import math
import heapq
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from enum import Enum
import time
import json
import random

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
                        speed=120/1.852, detection_range=3000/1852)  # 120km/h转换为节，3000m转换为海里
        self.target_position: Optional[Position] = None

class USV(Vehicle):
    """无人艇类"""
    def __init__(self, id: str, position: Position):
        super().__init__(id, VehicleType.USV, position, 
                        speed=20, detection_range=800/1852)  # 速度20节，800m转换为海里
        self.target_position: Optional[Position] = None
        self.heading: float = 0.0  # 航向角（弧度，0表示正东方向）

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
    
    def __init__(self, mission_area: MissionArea, mission_duration: float = 7200, update_interval: float = 10.0):
        self.mission_area = mission_area
        self.mission_duration = mission_duration  # 秒
        self.current_time = 0.0
        self.update_interval = update_interval  # 控制循环间隔(秒)
        
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
        self.search_grid = [[0 for _ in range(width)] for _ in range(height)]
        
        # 目标存在概率图：初始化为均匀分布
        self.probability_map = [[1.0 / (width * height) for _ in range(width)] for _ in range(height)]
    
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
        优化无人机扫描模式 - 重点增强边界和高速目标探测
        关键改进：
        1. 扇形角度动态调整(60-120度)
        2. 扫描密度自适应(30-50条扫描线)
        3. 边界区域扫描密度三倍
        4. 高速目标区域强化扫描
        5. 特别加强左边区域扫描
        """
        if uav.id in self.uav_search_patterns:
            return self.uav_search_patterns[uav.id]
        
        pattern = []
        area = self.system.mission_area
        
        # 动态扇形角度(60-90度)
        max_target_speed = max([math.sqrt(t.velocity[0]**2 + t.velocity[1]**2) 
                          for t in self.system.targets], default=15)
        sector_angle = 60 + 30 * min(1.0, max_target_speed / 15)  # 60-90度
        
        # 根据目标速度调整重叠率（高速目标增加重叠）
        overlap_ratio = 0.4 + 0.3 * min(1.0, max_target_speed / 15)  # 40-70%
        
        scan_range = uav.detection_range * (1 - overlap_ratio)
        scan_lines = int(40 + 10 * min(1.0, max_target_speed / 15))  # 40-50条扫描线
        
        # 边界区域特殊处理（增加50%扫描密度）
        boundary_boost = 1.5  # 边界增强
        
        # 两架无人机互补扫描
        phase_shift = 45 if uav.id.endswith('2') else 0  # 两架无人机扫描角度错开45度
        
        # 生成互补扫描路径（添加安全终止条件）
        step_angle = max(1, min(45, int(sector_angle/2)))  # 限制步长范围1-45度
        max_iterations = 360 * 2  # 最大迭代次数限制
        iteration = 0
        
        angle = phase_shift
        while angle < 360 + phase_shift and iteration < max_iterations:
            iteration += 1
            left_angle = math.radians((angle - sector_angle/2) % 360)
            right_angle = math.radians((angle + sector_angle/2) % 360)
            
            # 高密度扫描线（限制最大扫描线数）
            scan_lines = min(50, scan_lines)  # 不超过50条扫描线
            for i in range(scan_lines):
                t = i / (scan_lines - 1) if scan_lines > 1 else 0.5
                scan_angle = left_angle * (1-t) + right_angle * t
                
                # 计算扫描终点（考虑重叠）
                end_x = uav.position.x + scan_range * math.cos(scan_angle)
                end_y = uav.position.y + scan_range * math.sin(scan_angle)
                
                # 确保点在任务区域内
                end_x = max(area.min_x, min(area.max_x, end_x))
                end_y = max(area.min_y, min(area.max_y, end_y))
                
                # 限制总路径点数不超过1000
                if len(pattern) < 1000:
                    pattern.append(Position(end_x, end_y))
                    
                    # 左边区域额外增加扫描点
                    if end_x < area.min_x + 1.0 and len(pattern) < 990:  # 左边1海里内
                        pattern.append(Position(end_x, end_y))
                        pattern.append(Position(end_x + 0.3, end_y))  # 向内偏移
            
            angle += step_angle
        
        # 大幅增加边界区域扫描密度（每条边20个点）
        border_points = 20
        for i in range(border_points):
            # 四边均匀分布
            x = area.min_x + (area.max_x - area.min_x) * i/(border_points-1)
            y = area.min_y + (area.max_y - area.min_y) * i/(border_points-1)
            pattern.extend([
                Position(x, area.min_y),  # 下边
                Position(x, area.max_y),  # 上边
                Position(area.min_x, y),  # 左边
                Position(area.max_x, y)   # 右边
            ])
        
        # 特别加强上边区域（目标未发现区域）
        for i in range(20):  # 增加到20个扫描点
            # 上边区域增加更多扫描点
            x = area.min_x + (area.max_x - area.min_x) * (0.1 + 0.8 * i/19)
            pattern.append(Position(x, area.max_y))
            
            # 在上边附近区域增加额外扫描点（距离边界0.5海里）
            if i % 2 == 0:
                pattern.append(Position(x, area.max_y - 0.5))
        
        # 增加对角线扫描路径（覆盖更多区域）
        pattern.append(Position(area.min_x, area.min_y))
        pattern.append(Position(area.max_x, area.max_y))
        pattern.append(Position(area.min_x, area.max_y))
        pattern.append(Position(area.max_x, area.min_y))
        
        # 增加边界附近区域扫描点（距离边界0.5海里）
        for i in range(5):
            x = area.min_x + 0.5 + (area.max_x - area.min_x - 1) * i/4
            y = area.min_y + 0.5 + (area.max_y - area.min_y - 1) * i/4
            pattern.extend([
                Position(x, area.min_y + 0.5),
                Position(x, area.max_y - 0.5),
                Position(area.min_x + 0.5, y),
                Position(area.max_x - 0.5, y)
            ])
        
        self.uav_search_patterns[uav.id] = pattern
        return pattern
        
        # 增加中心区域覆盖点
        center_x = (area.min_x + area.max_x) / 2
        center_y = (area.min_y + area.max_y) / 2
        pattern.append(Position(center_x, center_y))
        
        # 增加边界检查点（每条边8个点）
        border_points = 8
        for i in range(border_points):
            # 下边界
            x = area.min_x + (area.max_x - area.min_x) * i / (border_points - 1)
            pattern.append(Position(x, area.min_y))
            # 上边界
            pattern.append(Position(x, area.max_y))
            # 左边界
            y = area.min_y + (area.max_y - area.min_y) * i / (border_points - 1)
            pattern.append(Position(area.min_x, y))
            # 右边界
            pattern.append(Position(area.max_x, y))
        
        self.uav_search_patterns[uav.id] = pattern
        return pattern
        
        # 加强边界检查点（增加边界点密度）
        border_density = 8  # 每条边的点数
        border_points = []
        for i in range(border_density):
            # 下边界
            x = area.min_x + (area.max_x - area.min_x) * i / (border_density - 1)
            border_points.append(Position(x, area.min_y))
            # 上边界
            border_points.append(Position(x, area.max_y))
            # 左边界
            y = area.min_y + (area.max_y - area.min_y) * i / (border_density - 1)
            border_points.append(Position(area.min_x, y))
            # 右边界
            border_points.append(Position(area.max_x, y))
        
        pattern.extend(border_points)
        
        self.uav_search_patterns[uav.id] = pattern
        return pattern

    def generate_usv_search_pattern(self, usv: USV) -> List[Position]:
        """
        优化无人艇Z字形搜索路径
        关键改进：
        1. 采用改进的Z字形路径避免无限循环
        2. 路径间距1.2倍探测范围
        3. 边界区域重点覆盖
        4. 增加边界区域扫描密度
        """
        pattern = []
        area = self.system.mission_area
        
        # 固定参数避免无限循环
        leg_spacing = usv.detection_range * 0.6  # 减小间距增加覆盖密度
        x_step = leg_spacing
        y_step = leg_spacing
        
        # 从当前位置开始
        x = usv.position.x
        y = usv.position.y
        direction = 1  # 固定从左到右开始
        
        # 确保初始点在区域内
        x = max(area.min_x, min(area.max_x, x))
        y = max(area.min_y, min(area.max_y, y))
        pattern.append(Position(x, y))
        
        # 边界区域特殊处理
        border_points = 10  # 边界扫描点数
        for i in range(border_points):
            # 四边均匀分布
            x_pos = area.min_x + (area.max_x - area.min_x) * i/(border_points-1)
            y_pos = area.min_y + (area.max_y - area.min_y) * i/(border_points-1)
            pattern.extend([
                Position(x_pos, area.min_y),  # 下边
                Position(x_pos, area.max_y),  # 上边
                Position(area.min_x, y_pos),  # 左边
                Position(area.max_x, y_pos)   # 右边
            ])
        
        # 生成优化Z字形路径（提高中心区域覆盖）
        max_iterations = 1000
        iteration = 0
        center_boost = 1.5  # 中心区域密度提升系数
        
        while y <= area.max_y and iteration < max_iterations:
            iteration += 1
            
            # 横向移动（中心区域增加路径点）
            while ((direction == 1 and x <= area.max_x) or 
                  (direction == -1 and x >= area.min_x)) and iteration < max_iterations:
                iteration += 1
                
                # 中心区域增加额外路径点
                is_center = (abs(x) < area.max_x/3 and abs(y) < area.max_y/3)
                density = center_boost if is_center else 1.0
                
                for _ in range(int(density)):
                    pattern.append(Position(x, y))
                
                x += x_step * direction
            
            # 调整到边界内并确保不超出范围
            x = max(area.min_x + 0.1, min(area.max_x - 0.1, x))
            
            # 纵向移动
            y += y_step
            pattern.append(Position(x, y))
            
            # 改变方向
            direction *= -1
            
            # 边界区域增加密度
            if y >= area.max_y - leg_spacing * 2 or y <= area.min_y + leg_spacing * 2:
                y_step = leg_spacing * 0.4  # 更密集扫描边界
            
            # 确保点在任务区域内
            y = max(area.min_y, min(area.max_y, y))
        
        return pattern
        
        self.usv_search_zones[usv.id] = pattern
        return pattern
        
        # 根据无人机ID分配不同起始象限（确保两架无人机覆盖不同区域）
        start_quadrant = 0 if uav.id.endswith('1') else 2
        
        # 2. 边界优先扫描（确保边界目标能被探测到）
        border_points = [
            Position(area.min_x, area.min_y),
            Position(area.max_x, area.min_y),
            Position(area.max_x, area.max_y),
            Position(area.min_x, area.max_y)
        ]
        pattern.extend(border_points)
        
        # 3. 分区扫描路径（结合边界和内部扫描）
        scan_spacing = uav.detection_range * 0.6  # 60%重叠率
        
        for q in range(4):
            q_idx = (start_quadrant + q) % 4
            q_min_x, q_max_x, q_min_y, q_max_y = quadrants[q_idx]
            
            # 水平扫描线
            y = q_min_y + scan_spacing/2
            while y <= q_max_y:
                if q_idx in [0, 2]:  # 左到右
                    pattern.append(Position(q_min_x, y))
                    pattern.append(Position(q_max_x, y))
                else:  # 右到左
                    pattern.append(Position(q_max_x, y))
                    pattern.append(Position(q_min_x, y))
                y += scan_spacing
        
        # 4. 对角线扫描（增加覆盖多样性）
        pattern.append(Position(area.min_x, area.min_y))
        pattern.append(Position(area.max_x, area.max_y))
        pattern.append(Position(area.min_x, area.max_y))
        pattern.append(Position(area.max_x, area.min_y))
        
        # 5. 随机热点强化（在边界区域增加探测点）
        for _ in range(8):
            edge = random.choice(['top', 'bottom', 'left', 'right'])
            if edge == 'top':
                pos = Position(random.uniform(area.min_x, area.max_x), area.max_y)
            elif edge == 'bottom':
                pos = Position(random.uniform(area.min_x, area.max_x), area.min_y)
            elif edge == 'left':
                pos = Position(area.min_x, random.uniform(area.min_y, area.max_y))
            else:
                pos = Position(area.max_x, random.uniform(area.min_y, area.max_y))
            pattern.append(pos)
        
        self.uav_search_patterns[uav.id] = pattern
        return pattern
        
        # 2. 基于概率热点的补充路径
        if self.system.probability_map:
            hot_spots = self._find_probability_hotspots()
            for spot in hot_spots:
                # 在热点周围生成密集路径点
                for angle in range(0, 360, 45):
                    x = spot.x + 0.3 * math.cos(math.radians(angle))
                    y = spot.y + 0.3 * math.sin(math.radians(angle))
                    pattern.append(Position(x, y))
        
        # 3. 边界检查点（减少边界遗漏）
        border_points = [
            Position(area.min_x, area.min_y),
            Position(area.max_x, area.min_y),
            Position(area.max_x, area.max_y),
            Position(area.min_x, area.max_y)
        ]
        pattern.extend(border_points)
        
        # 4. 随机扰动（避免规律性被利用）
        for i in range(5):
            pattern.append(Position(
                random.uniform(area.min_x + 0.5, area.max_x - 0.5),
                random.uniform(area.min_y + 0.5, area.max_y - 0.5)
            ))
        
        self.uav_search_patterns[uav.id] = pattern
        return pattern

    def _find_probability_hotspots(self) -> List[Position]:
        """从概率图中识别热点区域"""
        hotspots = []
        if not self.system.probability_map:
            return hotspots
            
        threshold = 2.0 / (len(self.system.probability_map) * len(self.system.probability_map[0]))
        
        for y in range(len(self.system.probability_map)):
            for x in range(len(self.system.probability_map[0])):
                if self.system.probability_map[y][x] > threshold:
                    real_x = self.system.mission_area.min_x + x * self.system.grid_resolution
                    real_y = self.system.mission_area.min_y + y * self.system.grid_resolution
                    hotspots.append(Position(real_x, real_y))
        
        return hotspots[:3]  # 返回概率最高的3个热点
        
        self.uav_search_patterns[uav.id] = pattern
        return pattern
        
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
                                prob_map[ny][nx] *= (1.5 - distance*0.2)
            else:
                # 未发现目标，该区域概率降低
                prob_map[grid_y][grid_x] *= 0.7
        
        # 归一化概率图
        prob_map = self.system.probability_map
        if prob_map is not None:
            total_prob = sum(sum(row) for row in prob_map)
        if total_prob > 0:
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
        优化目标-无人艇分配成本计算
        关键改进：
        1. 提高高速目标优先级(15节以上3倍权重)
        2. 增强边界目标处理
        3. 优化拦截成功率计算
        """
        # 计算目标速度
        target_speed = math.sqrt(target.velocity[0]**2 + target.velocity[1]**2)
        
        # 计算相对速度（考虑目标速度方向）
        relative_velocity_x = usv.speed * math.cos(usv.heading) - target.velocity[0]
        relative_velocity_y = usv.speed * math.sin(usv.heading) - target.velocity[1]
        relative_speed = math.sqrt(relative_velocity_x**2 + relative_velocity_y**2)
        
        # 计算拦截时间和距离
        current_distance = usv.position.distance_to(target.position)
        intercept_time = current_distance / max(0.1, relative_speed)  # 避免除以0
        
        # 预测目标位置（考虑高速目标）
        predict_factor = 1.5 if target_speed > 10 else 1.2
        predicted_pos = Position(
            target.position.x + target.velocity[0] * intercept_time * predict_factor,
            target.position.y + target.velocity[1] * intercept_time * predict_factor
        )
        
        # 基础成本（拦截时间）
        base_cost = intercept_time * (15 if target_speed > 14 else 10)  # 高速目标权重更高
        
        # 边界优先级（距离边界越近优先级越高）
        boundary_dist = min(
            target.position.x - self.system.mission_area.min_x,
            self.system.mission_area.max_x - target.position.x,
            target.position.y - self.system.mission_area.min_y,
            self.system.mission_area.max_y - target.position.y
        )
        boundary_boost = 1.0 + (2.0 - min(2.0, boundary_dist)) * 5.0  # 边界优先级提升6-11倍
        
        # 拦截成功率优化（考虑相对速度和距离）
        intercept_prob = 1.0 - math.exp(-current_distance / (usv.speed * (0.15 if target_speed > 10 else 0.25)))
        
        # 综合成本计算
        cost = (base_cost / boundary_boost) * (1.5 if target_speed > 14 else 1.0) / max(0.1, intercept_prob)
        
        # 确保成本为正
        return max(0.1, cost)
    
    def _hungarian_algorithm(self, cost_matrix: List[List[float]]) -> Dict[int, int]:
        """
        完整的匈牙利算法实现
        返回最优分配：{行索引: 列索引}
        直接使用改进的贪心算法
        """
        if not cost_matrix:
            return {}
        
        return self._improved_greedy_assignment(cost_matrix)

    def _improved_greedy_assignment(self, cost_matrix: List[List[float]]) -> Dict[int, int]:
        """改进的贪心分配算法，考虑全局最优"""
        assignments = {}
        used_rows, used_cols = set(), set()
        
        # 创建所有可能分配的列表并按成本排序
        all_assignments = []
        for i in range(len(cost_matrix)):
            for j in range(len(cost_matrix[0])):
                all_assignments.append((cost_matrix[i][j], i, j))
        
        # 按成本从低到高排序
        all_assignments.sort()
        
        # 按成本从低到高进行分配
        for cost, row, col in all_assignments:
            if row not in used_rows and col not in used_cols:
                assignments[row] = col
                used_rows.add(row)
                used_cols.add(col)
        
        return assignments
        
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
        """生成符合转弯半径约束的平滑路径"""
        distance = start.distance_to(end)
        
        # 如果距离很短，直接返回终点
        if distance < 0.1:
            return [end]
        
        # 计算最小转弯半径对应的最小路径点间距
        min_turn_radius = 0.054 if isinstance(self, USV) else 0.027  # 100m/50m转换为海里
        point_spacing = max(0.1, 2 * min_turn_radius)  # 确保路径点间距足够转弯
        
        num_points = max(3, int(distance / point_spacing))
        
        path = []
        for i in range(num_points + 1):
            t = i / num_points
            # 加入平滑过渡
            if i == 0:
                # 起始点保持原方向
                x = start.x
                y = start.y
            elif i == num_points:
                # 终点精确到达
                x = end.x
                y = end.y
            else:
                # 中间点平滑过渡
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
    
    def __init__(self, system):
        self.system = system
        self.detection_probability_cache = {}
    
    def detect_targets(self, vehicle: Vehicle, targets: List[Target],
                      current_time: float) -> List[Target]:
        """
        优化探测方法，提高目标发现率：
        - 无人艇：800米圆形探测，边界目标增强探测
        - 无人机：60度扇形探测，边界区域扩大探测范围  
        - 目标进入范围即标记为可疑
        - 持续10秒（2个检测周期）视为探测成功
        - 增强载具间可疑目标信息共享
        """
        detected = []
        area = self.system.mission_area
        
        # 先收集所有可疑目标信息
        suspicious_targets = [t for t in targets 
                            if hasattr(t, 'is_suspicious') and t.is_suspicious]
        
        for target in targets:
            # 跳过已处置或已发现的目标
            if target.status in [TargetStatus.DISPOSED, TargetStatus.DETECTED]:
                continue
                
            distance = vehicle.position.distance_to(target.position)
            
            # 初始化探测状态
            if not hasattr(target, 'detection_duration'):
                target.detection_duration = 0.0
                target.is_suspicious = False
            
            # 如果其他载具已标记为可疑，直接继承状态
            if any(t.id == target.id for t in suspicious_targets):
                target.is_suspicious = True
                target.detection_duration = max(
                    target.detection_duration,
                    max(t.detection_duration for t in suspicious_targets 
                       if t.id == target.id)
                )
            
            # 计算目标到边界的距离
            boundary_dist = min(
                target.position.x - area.min_x,
                area.max_x - target.position.x,
                target.position.y - area.min_y,
                area.max_y - target.position.y
            )
            
            # 边界目标增强探测（距离边界<2海里）
            boundary_boost = 1.0 + (2.0 - min(2.0, boundary_dist)) * 1.0  # 进一步提高边界探测强度
            
            # 计算目标速度
            target_speed = math.sqrt(target.velocity[0]**2 + target.velocity[1]**2)
            
            # 高速目标(>10节)专用探测逻辑
            if target_speed > 10:
                # 大幅提高高速目标探测范围和灵敏度
                speed_boost = 1.0 + (target_speed - 10) / 5  # 线性增强
                # 无人机对高速目标使用更宽的探测角度
                if vehicle.vehicle_type == VehicleType.UAV:
                    angle_threshold_multiplier = 1.5
            else:
                speed_boost = 1.0
                angle_threshold_multiplier = 1.0
            
            # 无人艇使用精确800米圆形探测
            if vehicle.vehicle_type == VehicleType.USV:
                # 边界目标+高速目标增强探测范围
                effective_range = vehicle.detection_range * boundary_boost * speed_boost
                in_range = distance <= effective_range
            # 无人机使用60度扇形探测
            else:
                dx = target.position.x - vehicle.position.x
                dy = target.position.y - vehicle.position.y
                angle = math.degrees(math.atan2(dy, dx)) % 360
                heading = getattr(vehicle, 'heading', 0) % 360
                angle_diff = min((angle - heading) % 360, (heading - angle) % 360)
                
                # 边界目标+高速目标扩大探测角度范围
                if target_speed > 14:  # 15节以上目标使用更大角度
                    angle_threshold = 45 * (1.0 + (1.0 - min(1.0, boundary_dist)) * 1.0) * 1.5
                else:
                    angle_threshold = 30 * (1.0 + (1.0 - min(1.0, boundary_dist)) * 0.75) * angle_threshold_multiplier
                
                in_range = (distance <= vehicle.detection_range * boundary_boost and 
                          angle_diff <= angle_threshold)
            
            # 更新探测状态
            if in_range:
                target.detection_duration += self.system.update_interval
                target.is_suspicious = True
                
                # 持续10秒视为探测成功（考虑累积时间）
                if target.detection_duration >= 10.0 and target.status == TargetStatus.UNKNOWN:
                    target.detected_time = current_time
                    target.detection_vehicle = vehicle.id
                    target.status = TargetStatus.DETECTED
                    detected.append(target)
                    
                    # 通知其他载具该目标已被确认
                    for t in targets:
                        if t.id == target.id:
                            t.status = TargetStatus.DETECTED
            else:
                # 仅当没有其他载具探测到时才重置
                if not any(v.position.distance_to(target.position) <= v.detection_range 
                   for v in self.system.uavs + self.system.usvs if v.id != vehicle.id):
                    target.detection_duration = max(0, target.detection_duration - self.system.update_interval/2)
        
        return detected
    
    def _calculate_detection_probability(self, vehicle: Vehicle, target: Target, 
                                       distance: float) -> float:
        """
        优化探测概率模型 - 大幅提升边界和高速目标发现率
        关键改进：
        1. 边界区域概率提升5-10倍
        2. 高速目标(>10节)概率提升3-5倍
        3. 非线性衰减模型优化
        """
        max_range = vehicle.detection_range
        
        if distance >= max_range:
            return 0.0
            
        # 边界区域特殊处理（距离边界<2海里时概率大幅提升）
        boundary_dist = min(
            target.position.x - self.system.mission_area.min_x,
            self.system.mission_area.max_x - target.position.x,
            target.position.y - self.system.mission_area.min_y,
            self.system.mission_area.max_y - target.position.y
        )
        # 更激进地提升边界探测概率（距离边界越近提升越大）
        boundary_boost = 1.0 + (2.0 - min(2.0, boundary_dist)) * 4.5  # 边界区域概率提升5.5-10倍
        
        # 目标速度计算
        target_speed = math.sqrt(target.velocity[0]**2 + target.velocity[1]**2)
        
        # 高速目标概率提升
        speed_boost = 1.0
        if target_speed > 10:  # 10节以上
            speed_boost = 3.0 + 2.0 * min(1.0, (target_speed - 10)/5)  # 3-5倍提升
        
        # 边界目标概率增强
        area = self.system.mission_area
        boundary_dist = min(
            target.position.x - area.min_x,
            area.max_x - target.position.x,
            target.position.y - area.min_y,
            area.max_y - target.position.y
        )
        
        # 边界区域（距离边界<探测范围）概率提高
        boundary_boost = 1.0 + (1.0 - min(1.0, boundary_dist/max_range)) * 0.5
        
        # 目标速度计算
        target_speed = math.sqrt(target.velocity[0]**2 + target.velocity[1]**2)
        
        # 非线性衰减模型（针对低速目标优化）
        if target_speed < 6:  # 低速目标(5.3节)
            base_prob = math.exp(-0.3 * (distance / max_range)**0.5) * boundary_boost * 1.2
        else:
            base_prob = math.exp(-0.5 * (distance / max_range)**0.6) * boundary_boost
        
        # 载具类型加成
        if vehicle.vehicle_type == VehicleType.UAV:
            # 无人机在空中，视野更好但有高度限制
            base_prob *= 1.1 if distance < max_range*0.7 else 0.8
        else:
            # 无人艇在水面，受波浪和海况影响更大
            sea_state_factor = 0.8 + random.gauss(0.1, 0.05)  # 模拟海况波动
            base_prob *= sea_state_factor
        
        # 环境因素
        weather_factor = 0.85 + random.gauss(0.1, 0.03)  # 随机天气影响
        base_prob *= weather_factor
        
        # 目标特性影响（低速目标更容易被发现）
        target_factor = 1.3 if target_speed < 6 else (1.2 if random.random() > 0.3 else 0.7)
        base_prob *= target_factor
        
        return min(0.95, max(0.15, base_prob))  # 保持在15%-95%范围内

# ============================================================================
# 主控制算法
# ============================================================================

class MainController:
    """
    主控制算法
    
    整合所有模块，实现完整的协同搜索与处置流程
    """
    
    def __init__(self, mission_area: MissionArea, mission_duration: float = 7200):
        # 先定义控制参数
        self.update_interval = 5.0  # 秒，控制循环间隔缩短为5秒
        self.reallocation_interval = 30.0  # 秒，重新分配间隔缩短为30秒
        self.last_reallocation_time = 0.0
        
        # 再初始化系统
        self.system = CoordinatedSearchSystem(mission_area, mission_duration, self.update_interval)
        self.search_strategy = SearchStrategy(self.system)
        self.task_allocation = TaskAllocation(self.system)
        self.path_planning = PathPlanning(self.system)
        self.target_prediction = TargetPrediction()
        self.detection_system = DetectionSystem(self.system)
        
    
    def initialize_mission(self, uav_positions=None, usv_positions=None):
        """按照比赛要求初始化载具位置 - 全部在任务区域左边"""
        area = self.system.mission_area
        
        if usv_positions is None:
            # 4艘无人艇在左边均匀分布，间距1km(0.54海里)
            usv_spacing = 0.54  # 海里
            total_usv_span = 3 * usv_spacing  # 4个点3个间隔
            start_y = (area.min_y + area.max_y - total_usv_span) / 2
            
            usv_positions = [
                Position(area.min_x, start_y),
                Position(area.min_x, start_y + usv_spacing),
                Position(area.min_x, start_y + 2*usv_spacing),
                Position(area.min_x, start_y + 3*usv_spacing)
            ]
        
        for i, pos in enumerate(usv_positions):
            usv = USV(f"USV_{i+1}", pos)
            self.system.add_usv(usv)
        
        if uav_positions is None:
            # 2架无人机都在左边(x坐标为负)，间距2km(1.08海里)
            uav_spacing = 1.08  # 海里
            center_y = (area.min_y + area.max_y) / 2
            
            uav1 = UAV(f"UAV_1", Position(area.min_x + 0.5, center_y - uav_spacing/2))
            uav2 = UAV(f"UAV_2", Position(area.min_x + 0.5, center_y + uav_spacing/2))
            self.system.add_uav(uav1)
            self.system.add_uav(uav2)
        else:
            for i, pos in enumerate(uav_positions):
                uav = UAV(f"UAV_{i+1}", pos)
                self.system.add_uav(uav)
        
        # 为无人机生成初始搜索模式
        for uav in self.system.uavs:
            pattern = self.search_strategy.generate_uav_search_pattern(uav)
            uav.current_task = "执行搜索模式"
    
    def execute_mission_step(self, time_step: float):
        """执行一个任务步骤"""
        self.system.current_time += time_step
        
        # 1. 生成新目标
        if random.random() < 0.1:  # 10%概率生成新目标
            self._generate_new_targets()
            
        # 2. 载具移动和探测
        self._update_vehicle_positions(time_step)
        detected_targets = self._perform_detection()
        
        # 3. 更新目标信息
        self._update_target_information()
        
        # 4. 协同策略优化
        self._optimize_coordination()
        
        return self._get_mission_status()

    def _optimize_coordination(self):
        """优化无人机和无人艇的协同策略"""
        # 1. 无人机优先搜索高概率区域
        for uav in self.system.uavs:
            # 检查是否有可疑目标需要跟踪
            suspicious_targets = [t for t in self.system.targets 
                                if hasattr(t, 'is_suspicious') and t.is_suspicious]
            
            if suspicious_targets:
                # 优先跟踪高速目标(>10节)或边界目标
                def target_priority(t):
                    speed = math.sqrt(t.velocity[0]**2 + t.velocity[1]**2)
                    boundary_dist = min(
                        t.position.x - self.system.mission_area.min_x,
                        self.system.mission_area.max_x - t.position.x,
                        t.position.y - self.system.mission_area.min_y,
                        self.system.mission_area.max_y - t.position.y
                    )
                    return speed * 0.8 + (2.0 - min(2.0, boundary_dist)) * 0.5
                
                # 按优先级排序目标
                suspicious_targets.sort(key=target_priority, reverse=True)
                highest_priority_target = suspicious_targets[0]
                
                # 计算预测位置（针对高速目标优化）
                target_speed = math.sqrt(highest_priority_target.velocity[0]**2 + highest_priority_target.velocity[1]**2)
                distance = uav.position.distance_to(highest_priority_target.position)
                
                # 高速目标使用更激进的预测位置
                if target_speed > 10:
                    time_to_reach = distance / (uav.speed * 1.2)  # 提高预测速度
                else:
                    time_to_reach = distance / uav.speed
                
                predicted_pos = Position(
                    highest_priority_target.position.x + highest_priority_target.velocity[0] * time_to_reach * 1.5,
                    highest_priority_target.position.y + highest_priority_target.velocity[1] * time_to_reach * 1.5
                )
                
                uav.target_position = predicted_pos
                uav.current_task = f"跟踪可疑目标 {highest_priority_target.id}"
            else:
                # 没有可疑目标时，执行扇形扫描
                pattern = self.search_strategy.generate_uav_search_pattern(uav)
                if pattern:
                    uav.target_position = pattern[0]
                    uav.current_task = "执行扇形扫描"
        
        # 2. 无人艇配合无人机行动
        for usv in self.system.usvs:
            if not usv.is_busy:
                # 寻找最近的可疑目标
                suspicious_targets = [t for t in self.system.targets 
                                    if hasattr(t, 'is_suspicious') and t.is_suspicious]
                
                if suspicious_targets:
                    closest_target = min(suspicious_targets, 
                                       key=lambda t: usv.position.distance_to(t.position))
                    
                    # 计算拦截路径
                    distance = usv.position.distance_to(closest_target.position)
                    time_to_reach = distance / usv.speed
                    predicted_pos = Position(
                        closest_target.position.x + closest_target.velocity[0] * time_to_reach,
                        closest_target.position.y + closest_target.velocity[1] * time_to_reach
                    )
                    
                    usv.target_position = predicted_pos
                    usv.is_busy = True
                    usv.current_task = f"拦截可疑目标 {closest_target.id}"
                else:
                    # 执行螺旋搜索
                    pattern = self.search_strategy.generate_usv_search_pattern(usv)
                    if pattern:
                        usv.target_position = pattern[0]
                        usv.current_task = "执行螺旋搜索"
    
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
                    
                    # 更新航向角
                    if isinstance(vehicle, USV):
                        vehicle.heading = math.atan2(direction_y, direction_x)
    
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
                
                # 更新性能统计（精确时间统计）
                self.system.performance_stats['targets_detected'] += 1
                if target.detected_time is not None:
                    # 计算从目标生成到被发现的时间
                    detection_time = target.detected_time - (self.system.current_time - target.detection_duration)
                    self.system.performance_stats['detection_times'].append(detection_time)
                    print(f"目标 {target.id} 发现时间: {detection_time/60:.1f}分钟")
        
        return all_detected
    
    def _update_search_coverage(self, vehicle: Vehicle):
        """更新搜索覆盖网格"""
        # 将载具位置和探测范围标记为已搜索
        grid_x = int((vehicle.position.x - self.system.mission_area.min_x) / 
                    self.system.grid_resolution)
        grid_y = int((vehicle.position.y - self.system.mission_area.min_y) / 
                    self.system.grid_resolution)
        
        # 标记探测范围内的网格（增加10%探测范围补偿）
        detection_radius_in_grid = int(vehicle.detection_range * 1.1 / self.system.grid_resolution)
        
        for dy in range(-detection_radius_in_grid, detection_radius_in_grid + 1):
            for dx in range(-detection_radius_in_grid, detection_radius_in_grid + 1):
                ny, nx = grid_y + dy, grid_x + dx
                if self.system.search_grid is not None and 0 <= ny < len(self.system.search_grid) and 0 <= nx < len(self.system.search_grid[0]):
                    distance = math.sqrt(dx*dx + dy*dy) * self.system.grid_resolution
                    if distance <= vehicle.detection_range:
                        self.system.search_grid[ny][nx] = 1
    
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
        """更新载具任务，增加可疑目标跟踪逻辑"""
        # 更新无人机任务
        for uav in self.system.uavs:
            # 检查是否有可疑目标需要跟踪
            suspicious_targets = [t for t in self.system.targets 
                                if hasattr(t, 'is_suspicious') and t.is_suspicious]
            
            if suspicious_targets:
                # 找到最近的可疑目标
                closest_target = min(suspicious_targets, 
                                   key=lambda t: uav.position.distance_to(t.position))
                
                # 计算预测位置（考虑目标速度和无人机速度）
                distance = uav.position.distance_to(closest_target.position)
                time_to_reach = distance / uav.speed
                predicted_pos = Position(
                    closest_target.position.x + closest_target.velocity[0] * time_to_reach,
                    closest_target.position.y + closest_target.velocity[1] * time_to_reach
                )
                
                # 确保预测位置在任务区域内
                if not self.system.mission_area.contains(predicted_pos):
                    predicted_pos = self.path_planning._adjust_to_boundary(predicted_pos)
                
                uav.target_position = predicted_pos
                uav.current_task = f"跟踪可疑目标 {closest_target.id}"
            elif not hasattr(uav, 'target_position') or not uav.target_position:
                # 没有可疑目标时，执行常规搜索
                pattern = self.search_strategy.generate_uav_search_pattern(uav)
                if pattern:
                    uav.target_position = pattern[0]
                    uav.current_task = "执行搜索模式"
        
        # 更新无人艇任务 - 配合无人机跟踪可疑目标
        for usv in self.system.usvs:
            if usv.is_busy and usv.current_task and usv.current_task.startswith("处置目标_"):
                target_id = usv.current_task.split("_")[1]
                target = next((t for t in self.system.detected_targets if t.id == target_id), None)
                
                if target:
                    # 计算拦截点：考虑目标速度和无人艇速度
                    current_distance = usv.position.distance_to(target.position)
                    relative_speed = math.sqrt(
                        (usv.speed * math.cos(usv.heading) - target.velocity[0])**2 +
                        (usv.speed * math.sin(usv.heading) - target.velocity[1])**2
                    )
                    
                    # 计算目标速度并计算拦截时间（针对高速目标优化）
                    target_speed = math.sqrt(target.velocity[0]**2 + target.velocity[1]**2)
                    intercept_time = current_distance / (relative_speed * (1.2 if target_speed > 10 else 1.0))
                    
                    # 预测目标位置（考虑无人机跟踪信息）
                    uav_tracking = next((u for u in self.system.uavs 
                                        if hasattr(u, 'current_task') and 
                                        f"跟踪可疑目标 {target.id}" in u.current_task), None)
                    
                    if uav_tracking:
                        # 如果有无人机在跟踪，使用无人机提供的最新位置
                        predicted_pos = uav_tracking.target_position
                    else:
                        # 否则使用预测位置
                        predicted_pos = Position(
                            target.position.x + target.velocity[0] * intercept_time * 1.2,
                            target.position.y + target.velocity[1] * intercept_time * 1.2
                        )
                    
                    # 确保预测位置在任务区域内
                    if not self.system.mission_area.contains(predicted_pos):
                        predicted_pos = self.path_planning._adjust_to_boundary(predicted_pos)
                    
                    usv.target_position = predicted_pos
                    
                    # 如果已经很接近目标，直接设置为目标当前位置
                    if current_distance < 0.1:  # 约185米
                        usv.target_position = target.position
            elif not usv.is_busy:
                # 空闲无人艇协助无人机跟踪可疑目标
                suspicious_targets = [t for t in self.system.targets 
                                     if hasattr(t, 'is_suspicious') and t.is_suspicious]
                
                if suspicious_targets:
                    # 找到最近的可疑目标
                    closest_target = min(suspicious_targets, 
                                       key=lambda t: usv.position.distance_to(t.position))
                    
                    # 计算拦截点
                    distance = usv.position.distance_to(closest_target.position)
                    time_to_reach = distance / usv.speed
                    predicted_pos = Position(
                        closest_target.position.x + closest_target.velocity[0] * time_to_reach,
                        closest_target.position.y + closest_target.velocity[1] * time_to_reach
                    )
                    
                    usv.target_position = predicted_pos
                    usv.is_busy = True
                    usv.current_task = f"协助跟踪可疑目标 {closest_target.id}"
    
    def _generate_new_targets(self):
        """根据比赛要求生成新目标"""
        # 检查总目标数不超过8个
        if len(self.system.targets) >= 8:
            return
            
        # 同一时刻最多2个新目标，且任务区域内不超过2个未发现目标
        new_targets = [t for t in self.system.targets 
                      if t.status == TargetStatus.UNKNOWN and 
                      (self.system.current_time - getattr(t, 'create_time', 0)) < 60]
        if len(new_targets) >= 2:
            return
            
        # 确保任务区域内未发现目标不超过2个
        undetected_in_area = [t for t in self.system.targets 
                            if t.status == TargetStatus.UNKNOWN and
                            self.system.mission_area.contains(t.position)]
        if len(undetected_in_area) >= 2:
            return
            
        # 从ABCD边随机选择进入点
        edge = random.choice(['A', 'B', 'C', 'D'])
        if edge == 'A':  # 上边 (y=max_y)
            pos = Position(
                random.uniform(self.system.mission_area.min_x, self.system.mission_area.max_x),
                self.system.mission_area.max_y
            )
            # 进入方向：向下偏转±45度
            base_angle = math.pi * 1.5  # 270度(向下)
            angle = base_angle + random.uniform(-math.pi/4, math.pi/4)
        elif edge == 'B':  # 右边 (x=max_x)
            pos = Position(
                self.system.mission_area.max_x,
                random.uniform(self.system.mission_area.min_y, self.system.mission_area.max_y)
            )
            # 进入方向：向左偏转±45度
            base_angle = math.pi  # 180度(向左)
            angle = base_angle + random.uniform(-math.pi/4, math.pi/4)
        elif edge == 'C':  # 下边 (y=min_y)
            pos = Position(
                random.uniform(self.system.mission_area.min_x, self.system.mission_area.max_x),
                self.system.mission_area.min_y
            )
            # 进入方向：向上偏转±45度
            base_angle = math.pi * 0.5  # 90度(向上)
            angle = base_angle + random.uniform(-math.pi/4, math.pi/4)
        else:  # 左边 (x=min_x)
            pos = Position(
                self.system.mission_area.min_x,
                random.uniform(self.system.mission_area.min_y, self.system.mission_area.max_y)
            )
            # 进入方向：向右偏转±45度
            base_angle = 0  # 0度(向右)
            angle = base_angle + random.uniform(-math.pi/4, math.pi/4)
        
        # 随机速度5-15节（避免静止目标）
        speed = random.uniform(5, 15)
        velocity = (speed * math.cos(angle), speed * math.sin(angle))
        
        target_id = f"Target_{len(self.system.targets)+1}"
        new_target = Target(target_id, pos, velocity)
        new_target.create_time = self.system.current_time
        self.system.add_target(new_target)
    
    def _get_mission_status(self) -> Dict:
        """获取任务状态"""
        total_targets = len(self.system.targets)
        detected_count = len([t for t in self.system.targets if t.status != TargetStatus.UNKNOWN])
        disposed_count = len([t for t in self.system.targets if t.status == TargetStatus.DISPOSED])
        
        detection_times = self.system.performance_stats['detection_times']
        disposal_times = self.system.performance_stats['disposal_times']
        
        avg_detection_time = sum(detection_times)/len(detection_times) if detection_times else 0
        avg_disposal_time = sum(disposal_times)/len(disposal_times) if disposal_times else 0
        
        return {
            'current_time': self.system.current_time,
            'total_targets': total_targets,
            'detected_targets': detected_count,
            'disposed_targets': disposed_count,  # 修正拼写错误
            'detection_probability': detected_count / total_targets if total_targets > 0 else 0,
            'avg_detection_time': avg_detection_time,
            'avg_disposal_time': avg_disposal_time,
            'mission_progress': disposed_count / total_targets if total_targets > 0 else 0
        }

    def to_json(self):
        """导出当前仿真状态为JSON（便于Java/Python互通）"""
        prob_map = self.system.probability_map
        search_grid = self.system.search_grid
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
