import json
import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional

# ===================== 基础数据结构 =====================

@dataclass
class Position:
    x: float
    y: float
    def distance_to(self, other: 'Position') -> float:
        return ((self.x - other.x)**2 + (self.y - other.y)**2) ** 0.5

@dataclass
class UAV:
    id: str
    position: Position
    detection_range: float  # 海里
    speed: float  # 节
    busy: bool = False
    search_dir: int = 0

@dataclass
class USV:
    id: str
    position: Position
    detection_range: float  # 海里
    speed: float  # 节
    busy: bool = False
    target_id: Optional[str] = None

@dataclass
class Target:
    id: str
    position: Position
    velocity: Tuple[float, float]  # (vx, vy) 单位:节
    detected: bool = False
    disposed: bool = False
    detected_time: Optional[float] = None
    disposed_time: Optional[float] = None

@dataclass
class MissionArea:
    min_x: float
    max_x: float
    min_y: float
    max_y: float
    def in_area(self, pos: Position) -> bool:
        return self.min_x <= pos.x <= self.max_x and self.min_y <= pos.y <= self.max_y

# ===================== 概率地图与贝叶斯更新 =====================

class ProbMap:
    def __init__(self, min_x, max_x, min_y, max_y, res=0.2, p0=0.01):
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y
        self.res = res
        self.nx = int((max_x - min_x) / res) + 1
        self.ny = int((max_y - min_y) / res) + 1
        self.grid = [[p0 for _ in range(self.nx)] for _ in range(self.ny)]
    def get(self, x, y):
        ix = int((x - self.min_x) / self.res)
        iy = int((y - self.min_y) / self.res)
        if 0 <= ix < self.nx and 0 <= iy < self.ny:
            return self.grid[iy][ix]
        return 0.0
    def set(self, x, y, p):
        ix = int((x - self.min_x) / self.res)
        iy = int((y - self.min_y) / self.res)
        if 0 <= ix < self.nx and 0 <= iy < self.ny:
            self.grid[iy][ix] = p
    def entropy(self):
        H = 0.0
        for row in self.grid:
            for p in row:
                if p > 0 and p < 1:
                    H -= p * math.log(p + 1e-8)
        return H
    def update_bayes(self, x, y, detected, pd=0.8, pf=0.2):
        ix = int((x - self.min_x) / self.res)
        iy = int((y - self.min_y) / self.res)
        if 0 <= ix < self.nx and 0 <= iy < self.ny:
            p = self.grid[iy][ix]
            if detected:
                p_new = pd * p / (pd * p + pf * (1 - p) + 1e-8)
            else:
                p_new = (1 - pd) * p / ((1 - pd) * p + (1 - pf) * (1 - p) + 1e-8)
            self.grid[iy][ix] = min(max(p_new, 0.001), 0.999)

# ===================== 多无人机艇协同搜索控制 =====================

class MultiAgentSearchController:
    def __init__(self):
        # 区域5x5海里
        self.area = MissionArea(-2.5, 2.5, -2.5, 2.5)
        # UAV参数
        self.uavs: List[UAV] = [
            UAV(f"UAV1", Position(-2.0, -2.0), 3000/1852, 120/1.852),
            UAV(f"UAV2", Position(2.0, 2.0), 3000/1852, 120/1.852)
        ]
        # USV参数
        self.usvs: List[USV] = [
            USV(f"USV1", Position(-1.0, -1.0), 800/1852, 20),
            USV(f"USV2", Position(1.0, -1.0), 800/1852, 20),
            USV(f"USV3", Position(-1.0, 1.0), 800/1852, 20),
            USV(f"USV4", Position(1.0, 1.0), 800/1852, 20)
        ]
        # 目标
        self.targets: List[Target] = [
            Target(f"T{i+1}", Position(random.uniform(-2,2), random.uniform(-2,2)),
                   (random.uniform(-5,5), random.uniform(-5,5))) for i in range(8)
        ]
        self.prob_map = ProbMap(-2.5, 2.5, -2.5, 2.5, res=0.2)
        self.time = 0.0
        self.time_step = 30.0  # 秒
        self.max_time = 7200.0 # 2小时
        self.detected_targets = set()
        self.performance = []  # (发现时间, 处置时间)

    def step(self):
        # 1. UAV概率搜索（最大信息增益）
        for uav in self.uavs:
            best_dir = (0, 0)
            best_gain = -float('inf')
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = uav.position.x + dx * uav.detection_range, uav.position.y + dy * uav.detection_range
                    if not self.area.in_area(Position(nx, ny)):
                        continue
                    p = self.prob_map.get(nx, ny)
                    gain = p * (1 - p)
                    for other in self.uavs:
                        if other is not uav and abs(other.position.x - nx) < 0.2 and abs(other.position.y - ny) < 0.2:
                            gain -= 0.1
                    if gain > best_gain:
                        best_gain = gain
                        best_dir = (dx, dy)
            # 移动
            move_dist = uav.speed * (self.time_step / 3600)
            uav.position.x = min(max(uav.position.x + best_dir[0] * move_dist, self.area.min_x), self.area.max_x)
            uav.position.y = min(max(uav.position.y + best_dir[1] * move_dist, self.area.min_y), self.area.max_y)
        # 2. UAV探测目标，贝叶斯更新，记录发现时间
        for uav in self.uavs:
            for t in self.targets:
                if not t.detected and uav.position.distance_to(t.position) <= uav.detection_range:
                    t.detected = True
                    t.detected_time = self.time
                    self.detected_targets.add(t.id)
                # 概率地图贝叶斯更新
                self.prob_map.update_bayes(t.position.x, t.position.y, t.detected)
        # 3. USV分配与路径规划
        for usv in self.usvs:
            if usv.target_id is None or any(t.id == usv.target_id and t.disposed for t in self.targets):
                # 分配最近未处置目标
                min_dist = float('inf')
                best_target = None
                for t in self.targets:
                    if t.detected and not t.disposed:
                        # 预测目标未来位置
                        time_to_reach = usv.position.distance_to(t.position) / usv.speed
                        pred_x = t.position.x + t.velocity[0] * time_to_reach
                        pred_y = t.position.y + t.velocity[1] * time_to_reach
                        pred_pos = Position(pred_x, pred_y)
                        dist = usv.position.distance_to(pred_pos)
                        if dist < min_dist:
                            min_dist = dist
                            best_target = t
                if best_target:
                    usv.target_id = best_target.id
                    usv.busy = True
                else:
                    usv.target_id = None
                    usv.busy = False
        # 4. USV移动与目标处置
        for usv in self.usvs:
            if usv.target_id:
                t = next((tt for tt in self.targets if tt.id == usv.target_id), None)
                if t and not t.disposed:
                    # 预测目标未来位置
                    time_to_reach = usv.position.distance_to(t.position) / usv.speed
                    pred_x = t.position.x + t.velocity[0] * time_to_reach
                    pred_y = t.position.y + t.velocity[1] * time_to_reach
                    pred_pos = Position(pred_x, pred_y)
                    # 直线移动
                    dist = usv.position.distance_to(pred_pos)
                    move_dist = usv.speed * (self.time_step / 3600)
                    if dist <= move_dist or dist < 0.054:  # 100米
                        usv.position = Position(pred_pos.x, pred_pos.y)
                        t.disposed = True
                        t.disposed_time = self.time
                        usv.busy = False
                        usv.target_id = None
                    else:
                        dx = (pred_pos.x - usv.position.x) / dist
                        dy = (pred_pos.y - usv.position.y) / dist
                        usv.position.x += dx * move_dist
                        usv.position.y += dy * move_dist
                        usv.position.x = min(max(usv.position.x, self.area.min_x), self.area.max_x)
                        usv.position.y = min(max(usv.position.y, self.area.min_y), self.area.max_y)
        # 5. 目标运动（线性+边界反弹）
        for t in self.targets:
            if not t.disposed:
                next_x = t.position.x + t.velocity[0] * (self.time_step / 3600)
                next_y = t.position.y + t.velocity[1] * (self.time_step / 3600)
                if not (self.area.min_x < next_x < self.area.max_x):
                    t.velocity = (-t.velocity[0], t.velocity[1])
                if not (self.area.min_y < next_y < self.area.max_y):
                    t.velocity = (t.velocity[0], -t.velocity[1])
                t.position.x += t.velocity[0] * (self.time_step / 3600)
                t.position.y += t.velocity[1] * (self.time_step / 3600)
                t.position.x = min(max(t.position.x, self.area.min_x), self.area.max_x)
                t.position.y = min(max(t.position.y, self.area.min_y), self.area.max_y)
        self.time += self.time_step

    def run(self):
        print("无人机艇协同概率搜索仿真开始...")
        while self.time < self.max_time:
            self.step()
            if int(self.time) % 300 == 0:
                detected = sum(1 for t in self.targets if t.detected)
                disposed = sum(1 for t in self.targets if t.disposed)
                print(f"{int(self.time/60)}分钟: 发现{detected}/{len(self.targets)} 处置{disposed}/{len(self.targets)}")
            if all(t.disposed for t in self.targets):
                print(f"所有目标处置完成，总用时{self.time/60:.1f}分钟")
                break
        self.print_performance()
        print("仿真结束。")

    def print_performance(self):
        detected_times = [t.detected_time for t in self.targets if t.detected_time is not None]
        disposed_times = [t.disposed_time for t in self.targets if t.disposed_time is not None]
        total_targets = len(self.targets)
        detected_count = len(detected_times)
        disposed_count = len(disposed_times)
        avg_detection_time = sum(detected_times)/detected_count/60 if detected_times else 0
        avg_disposal_time = sum(disposed_times)/disposed_count/60 if disposed_times else 0
        detection_probability = detected_count / total_targets if total_targets else 0
        print(f"发现概率: {detection_probability:.2%}")
        print(f"平均发现时间: {avg_detection_time:.1f}分钟")
        print(f"平均处置时间: {avg_disposal_time:.1f}分钟")
        # 评分
        detection_score = 20 if avg_detection_time <= 5 else max(0, 20 - (avg_detection_time-5)*2)
        disposal_score = 20 if avg_disposal_time <= 10 else max(0, 20 - (avg_disposal_time-10)*4)
        probability_score = detection_probability * 20
        total_score = detection_score + disposal_score + probability_score
        print(f"评分: 发现概率{probability_score:.1f}/20, 发现时间{detection_score:.1f}/20, 处置时间{disposal_score:.1f}/20, 总分{total_score:.1f}/60")

    def to_json(self):
        return {
            'area': {'min_x': self.area.min_x, 'max_x': self.area.max_x, 'min_y': self.area.min_y, 'max_y': self.area.max_y},
            'uavs': [{'id': u.id, 'x': u.position.x, 'y': u.position.y} for u in self.uavs],
            'usvs': [{'id': u.id, 'x': u.position.x, 'y': u.position.y} for u in self.usvs],
            'targets': [
                {'id': t.id, 'x': t.position.x, 'y': t.position.y, 'vx': t.velocity[0], 'vy': t.velocity[1],
                 'detected': t.detected, 'disposed': t.disposed, 'detected_time': t.detected_time, 'disposed_time': t.disposed_time}
                for t in self.targets
            ],
            'prob_map': self.prob_map.grid,
            'time': self.time
        }
    def from_json(self, data):
        a = data['area']
        self.area = MissionArea(a['min_x'], a['max_x'], a['min_y'], a['max_y'])
        self.uavs = [UAV(u['id'], Position(u['x'], u['y']), 3000/1852, 120/1.852) for u in data['uavs']]
        self.usvs = [USV(u['id'], Position(u['x'], u['y']), 800/1852, 20) for u in data['usvs']]
        self.targets = [Target(t['id'], Position(t['x'], t['y']), (t['vx'], t['vy']), t['detected'], t['disposed'], t.get('detected_time'), t.get('disposed_time')) for t in data['targets']]
        self.prob_map = ProbMap(self.area.min_x, self.area.max_x, self.area.min_y, self.area.max_y, res=0.2)
        self.prob_map.grid = data['prob_map']
        self.time = data.get('time', 0.0)
    def run_with_json_io(self, input_path, output_path):
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.from_json(data)
        self.step()
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_json(), f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    import sys
    if len(sys.argv) == 3:
        sim = MultiAgentSearchController()
        sim.run_with_json_io(sys.argv[1], sys.argv[2])
    else:
        sim = MultiAgentSearchController()
        sim.run() 