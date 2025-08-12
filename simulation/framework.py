import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Wedge
import time
from core.algorithm import DistributedMPC, Agent, MovingTarget  # 添加MovingTarget导入
import math
import random

class Simulation:
    """无人机艇协同搜索仿真框架"""
    
    def __init__(self, mpc: DistributedMPC, duration: int = 600, fps: int = 1, spawn_config: dict = None):
        """
        初始化仿真环境
        
        参数:
        mpc: 分布式MPC控制器实例
        duration: 仿真时长(秒)
        fps: 帧率(每秒更新次数)
        spawn_config: 目标生成配置，可选，如:
            {
                'enable': True,
                'total_targets': 8,
                'max_spawn_per_step': 2,
                'spawn_interval_seconds': 900,  # 每15分钟尝试生成
            }
        """
        self.mpc = mpc
        self.duration = duration
        self.fps = fps
        self.grid_map = mpc.grid_map
        self.agents = mpc.agents
        self.time_step = 0
        self.spawn_config = spawn_config or {'enable': False}
        self.spawned_count = 0
        
        self.metrics = {
            'discovery_times': [],
            'boundary_coverage': [],
            'collision_events': 0,
            'targets_found': 0,
            'moving_targets_found': 0,  # 新增移动目标发现统计
            # 评分相关
            'P': 0.0,
            'S1': 0.0,
            'S2': 0.0,
            'score': 0.0
        }
        
        # 初始化移动目标列表
        self.moving_targets = []
        for target in self.grid_map.targets:
            if isinstance(target, MovingTarget):
                self.moving_targets.append(target)
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        
        # 初始化可视化
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.ax.set_xlim(0, self.grid_map.width * 1852)
        self.ax.set_ylim(0, self.grid_map.height * 1852)
        self.ax.set_title('无人机艇协同搜索仿真')
        self.ax.set_xlabel('X (米)')
        self.ax.set_ylabel('Y (米)')
        
        # 绘制网格
        self.draw_grid()
        
        # 创建智能体可视化对象
        self.agent_plots = []
        for agent in self.agents:
            if agent.agent_type == 'drone':
                plot = self.ax.plot([], [], '^', markersize=10, color='blue')[0]
                wedge = Wedge((0, 0), agent.sensor.range, 
                              agent.heading - agent.sensor.fov/2, 
                              agent.heading + agent.sensor.fov/2, 
                              alpha=0.2, color='cyan')
                wedge.set_antialiased(False)
                self.ax.add_patch(wedge)
                self.agent_plots.append({'plot': plot, 'wedge': wedge})
            else:
                plot = self.ax.plot([], [], 'o', markersize=8, color='green')[0]
                circle = Circle((0, 0), agent.sensor.range, alpha=0.2, color='cyan')
                circle.set_antialiased(False)
                self.ax.add_patch(circle)
                self.agent_plots.append({'plot': plot, 'circle': circle})
        
        # 目标概率热力图（imshow），替代逐格矩形与文字
        width_m = self.grid_map.width * 1852
        height_m = self.grid_map.height * 1852
        cmap = plt.cm.get_cmap('hot').copy()
        cmap.set_bad(alpha=0.0)  # 屏蔽低于阈值的网格
        data = np.ma.masked_less(self.grid_map.target_probability, 0.1)
        self.heatmap = self.ax.imshow(
            data,
            cmap=cmap,
            origin='lower',
            extent=[0, width_m, 0, height_m],
            vmin=0.0,
            vmax=1.0,
            alpha=0.3,
            interpolation='nearest'
        )
        
        # 目标散点集合（复用，不每帧创建/删除）
        self.targets_scatter = self.ax.scatter([], [], s=[], c=[])
        
        # 文本：时间与指标（复用）
        self.time_text = self.ax.text(0.02, 0.98, "", transform=self.ax.transAxes, fontsize=12, va='top')
        self.metrics_text = self.ax.text(0.98, 0.98, "", transform=self.ax.transAxes, fontsize=10, va='top', ha='right')
        
        # blit需要：维护艺术家列表
        self._blit_artists = [self.heatmap, self.time_text, self.metrics_text, *[ap['plot'] for ap in self.agent_plots]]
        for ap in self.agent_plots:
            if 'wedge' in ap:
                self._blit_artists.append(ap['wedge'])
            if 'circle' in ap:
                self._blit_artists.append(ap['circle'])
        self._blit_artists.append(self.targets_scatter)
        
    def spawn_targets_if_needed(self):
        """按配置在边界随机生成移动目标，单步生成不超过配置数量"""
        cfg = self.spawn_config
        if not cfg.get('enable'):
            return
        total = cfg.get('total_targets', 8)
        if self.spawned_count >= total:
            return
        interval = max(1, cfg.get('spawn_interval_seconds', 900))
        if self.time_step % interval != 0:
            return
        to_spawn = min(cfg.get('max_spawn_per_step', 2), total - self.spawned_count)
        width_m = self.grid_map.width * 1852
        height_m = self.grid_map.height * 1852
        for _ in range(to_spawn):
            side = random.randint(0, 3)
            if side == 0:
                x = random.uniform(0, width_m); y = 0.0
            elif side == 1:
                x = width_m; y = random.uniform(0, height_m)
            elif side == 2:
                x = random.uniform(0, width_m); y = height_m
            else:
                x = 0.0; y = random.uniform(0, height_m)
            target = MovingTarget(start_position=(x, y))
            cx, cy = width_m/2.0, height_m/2.0
            angle = math.degrees(math.atan2(cy - y, cx - x)) % 360
            target.direction = (angle + random.uniform(-20, 20)) % 360
            target.t0 = self.time_step
            self.grid_map.add_target_object(target)
            self.moving_targets.append(target)
            self.spawned_count += 1
    
    def step(self, frame):
        """执行单步仿真"""
        self.mpc.run_step()
        self.spawn_targets_if_needed()
        
        for target in self.moving_targets:
            target.move(1.0/self.fps)
            target.check_detection(self.agents, self.time_step)
            if getattr(target, 'detected', False) and getattr(target, 't_detect', None) is None:
                target.t_detect = self.time_step
            if getattr(target, 't_dispose', None) is None:
                for agent in self.agents:
                    if agent.agent_type == 'usv':
                        dist = np.linalg.norm(np.array(agent.position) - np.array(target.position))
                        if dist <= 100.0:
                            target.t_dispose = self.time_step
                            break
        
        self.time_step += 1
        
        self.collect_metrics()
        
        if hasattr(self, 'fig'):
            return self.update_visualization()
        
        return self._blit_artists
    
    def collect_metrics(self):
        """收集性能指标"""
        for i in range(self.grid_map.grid_height):
            for j in range(self.grid_map.grid_width):
                if self.grid_map.confirmed_targets[i, j]:
                    if (i, j) not in self.metrics['discovery_times']:
                        self.metrics['discovery_times'].append((i, j, self.time_step))
                        self.metrics['targets_found'] += 1
        
        for target in self.moving_targets:
            if getattr(target, 'detected', False) and not getattr(target, 'reported', False):
                self.metrics['moving_targets_found'] += 1
                self.metrics['targets_found'] += 1
                target.reported = True
        
        for i in range(len(self.agents)):
            for j in range(i+1, len(self.agents)):
                dist = np.linalg.norm(np.array(self.agents[i].position) - 
                                     np.array(self.agents[j].position))
                min_dist = self.agents[i].safe_distance + self.agents[j].safe_distance
                if dist < min_dist:
                    self.metrics['collision_events'] += 1
        
        boundary_cells = 0
        covered_boundary = 0
        for i in range(self.grid_map.grid_height):
            for j in range(self.grid_map.grid_width):
                if self.grid_map.is_boundary_position(self.grid_map.index_to_position((i, j))):
                    boundary_cells += 1
                    if self.grid_map.detection_duration[i, j] > 0:
                        covered_boundary += 1
        if boundary_cells > 0:
            self.metrics['boundary_coverage'].append(covered_boundary / boundary_cells)
 
    def draw_grid(self):
        """绘制网格线"""
        grid_resolution_m = self.grid_map.resolution * 1852
        for i in range(self.grid_map.grid_height + 1):
            y = i * grid_resolution_m
            self.ax.axhline(y, color='gray', linestyle='-', alpha=0.15, linewidth=0.5)
        for j in range(self.grid_map.grid_width + 1):
            x = j * grid_resolution_m
            self.ax.axvline(x, color='gray', linestyle='-', alpha=0.15, linewidth=0.5)
    
    def update_visualization(self):
        """更新可视化（尽量复用艺术家，支持blit）"""
        # 智能体
        for i, agent in enumerate(self.agents):
            x, y = agent.position
            self.agent_plots[i]['plot'].set_data([x], [y])
            if agent.agent_type == 'drone':
                wedge = self.agent_plots[i]['wedge']
                wedge.set_center(agent.position)
                display_heading = 90 - agent.heading
                wedge.set_theta1(display_heading - agent.sensor.fov/2)
                wedge.set_theta2(display_heading + agent.sensor.fov/2)
            else:
                circle = self.agent_plots[i]['circle']
                circle.set_center(agent.position)
        
        # 热力图（只更新数据）
        data = np.ma.masked_less(self.grid_map.target_probability, 0.1)
        self.heatmap.set_data(data)
        
        # 目标散点（一次性更新坐标/颜色/大小）
        if self.moving_targets:
            offsets = np.array([t.position for t in self.moving_targets])
            colors = []
            sizes = []
            for t in self.moving_targets:
                if getattr(t, 'detected', False):
                    colors.append('green'); sizes.append(50)
                elif getattr(t, 'detection_time', 0) > 0:
                    colors.append('yellow'); sizes.append(35)
                else:
                    colors.append('red'); sizes.append(25)
            self.targets_scatter.set_offsets(offsets)
            self.targets_scatter.set_color(colors)
            self.targets_scatter.set_sizes(sizes)
        else:
            self.targets_scatter.set_offsets(np.empty((0, 2)))
            self.targets_scatter.set_sizes([])
            self.targets_scatter.set_color([])
        
        # 文本（复用对象）
        self.time_text.set_text(f"Time: {self.time_step}s")
        self.metrics_text.set_text(
            f"Targets Found: {self.metrics['targets_found']}\n"
            f"Moving Targets Found: {self.metrics['moving_targets_found']}\n"
            f"Collisions: {self.metrics['collision_events']}"
        )
        
        return self._blit_artists
    
    def run(self, visualize=True):
        """运行仿真"""
        start_time = time.time()
        
        if visualize:
            frames = int(self.duration * self.fps)
            ani = FuncAnimation(self.fig, self.step, frames=frames,
                                interval=int(1000/self.fps), repeat=False, blit=True)
            plt.show()
        else:
            total_steps = int(self.duration * self.fps)
            for step in range(total_steps):
                self.step(step)
                if step % max(50, self.fps) == 0:
                    print(f"Step {step}: Targets found {self.metrics['targets_found']}")
        
        end_time = time.time()
        print(f"仿真完成! 用时: {end_time-start_time:.2f}秒")
        self.report_metrics()
    
    def compute_score(self):
        """根据典型评分规则计算P、S1、S2与总分"""
        t_detect_list = []
        t_dispose_list = []
        total_targets = self.spawn_config.get('total_targets', len(self.moving_targets)) if self.spawn_config.get('enable') else len(self.moving_targets)
        detected_count = 0
        for target in self.moving_targets:
            if getattr(target, 't0', None) is None:
                continue
            if getattr(target, 't_detect', None) is not None:
                detected_count += 1
                t_detect_list.append((target.t_detect - target.t0) / self.fps)
            if getattr(target, 't_dispose', None) is not None:
                t_dispose_list.append((target.t_dispose - target.t0) / self.fps)
        P = (detected_count / total_targets) if total_targets > 0 else 0.0
        
        def piecewise_score(avg_time_sec: float, t_full: float, t_half: float) -> float:
            if avg_time_sec <= t_full:
                return 20.0
            if avg_time_sec <= t_half:
                return 10.0 + 10.0 * (t_half - avg_time_sec) / (t_half - t_full)
            return 0.0
        
        avg_detect = np.mean(t_detect_list) if t_detect_list else float('inf')
        S1 = piecewise_score(avg_detect, t_full=5*60, t_half=10*60)
        avg_dispose = np.mean(t_dispose_list) if t_dispose_list else float('inf')
        S2 = piecewise_score(avg_dispose, t_full=10*60, t_half=15*60)
        score = (S1 + S2) * P
        self.metrics['P'] = P
        self.metrics['S1'] = S1
        self.metrics['S2'] = S2
        self.metrics['score'] = score
    
    def report_metrics(self):
        """生成性能报告"""
        self.compute_score()
        
        print("\n===== 仿真性能报告 =====")
        print(f"总仿真时间: {self.time_step}秒")
        print(f"发现静态目标数量: {self.metrics['targets_found']}")
        print(f"发现移动目标数量: {self.metrics['moving_targets_found']}")
        
        if self.metrics['discovery_times']:
            times = [t for _, _, t in self.metrics['discovery_times']]
            print(f"平均目标发现时间(静态): {np.mean(times):.1f}秒")
            print(f"最短发现时间(静态): {min(times)}秒")
            print(f"最长发现时间(静态): {max(times)}秒")
        
        print(f"碰撞事件次数: {self.metrics['collision_events']}")
        
        if self.metrics['boundary_coverage']:
            print(f"最终边界覆盖率: {self.metrics['boundary_coverage'][-1]*100:.1f}%")
            print(f"最高边界覆盖率: {max(self.metrics['boundary_coverage'])*100:.1f}%")
        
        print(f"发现概率P: {self.metrics['P']*100:.1f}%")
        print(f"S1(发现时间得分): {self.metrics['S1']:.1f}")
        print(f"S2(处置时间得分): {self.metrics['S2']:.1f}")
        print(f"总分 (S1+S2)*P: {self.metrics['score']:.1f}")
        
        if self.metrics['boundary_coverage']:
            plt.figure(figsize=(12, 8))
            
            plt.subplot(2, 2, 1)
            plt.plot(self.metrics['boundary_coverage'])
            plt.title('边界覆盖率变化')
            plt.xlabel('时间步')
            plt.ylabel('覆盖率')
            
            plt.subplot(2, 2, 2)
            if self.metrics['discovery_times']:
                times = [t for _, _, t in self.metrics['discovery_times']]
                plt.hist(times, bins=20)
                plt.title('目标发现时间分布(静态)')
                plt.xlabel('发现时间(秒)')
                plt.ylabel('目标数量')
            
            plt.subplot(2, 2, 3)
            plt.imshow(self.grid_map.target_probability, cmap='hot', origin='lower')
            plt.colorbar(label='目标概率')
            plt.title('最终目标概率分布')
            
            plt.tight_layout()
            plt.show()
