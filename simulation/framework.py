import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Wedge
import time
from core.algorithm import DistributedMPC, Agent, MovingTarget  # 添加MovingTarget导入

class Simulation:
    """无人机艇协同搜索仿真框架"""
    
    def __init__(self, mpc: DistributedMPC, duration: int = 600, fps: int = 1):
        """
        初始化仿真环境
        
        参数:
        mpc: 分布式MPC控制器实例
        duration: 仿真时长(秒)
        fps: 帧率(每秒更新次数)
        """
        self.mpc = mpc
        self.duration = duration
        self.fps = fps
        self.grid_map = mpc.grid_map
        self.agents = mpc.agents
        self.time_step = 0
        self.metrics = {
            'discovery_times': [],
            'boundary_coverage': [],
            'collision_events': 0,
            'targets_found': 0,
            'moving_targets_found': 0  # 新增移动目标发现统计
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
                # 无人机用三角形表示
                plot = self.ax.plot([], [], '^', markersize=10, 
                                     color='blue' if agent.agent_type == 'drone' else 'green')[0]
                # 添加探测范围扇形
                wedge = Wedge((0, 0), agent.sensor.range, 
                              agent.heading - agent.sensor.fov/2, 
                              agent.heading + agent.sensor.fov/2, 
                              alpha=0.2, color='cyan')
                self.ax.add_patch(wedge)
                self.agent_plots.append({'plot': plot, 'wedge': wedge})
            else:
                # 无人艇用圆形表示
                plot = self.ax.plot([], [], 'o', markersize=8, 
                                    color='blue' if agent.agent_type == 'drone' else 'green')[0]
                # 添加探测范围圆形
                circle = Circle((0, 0), agent.sensor.range, alpha=0.2, color='cyan')
                self.ax.add_patch(circle)
                self.agent_plots.append({'plot': plot, 'circle': circle})
        
        # 目标可视化
        self.target_plots = []
        self.target_texts = []
        self.moving_target_plots = []
    
    def step(self, frame):
        """执行单步仿真"""
        # 运行MPC控制器
        self.mpc.run_step()
        
        # 更新移动目标
        for target in self.moving_targets:
            target.move(1.0/self.fps)  # 按时间步移动
            target.check_detection(self.agents, self.time_step)
        
        self.time_step += 1
        
        # 收集指标
        self.collect_metrics()
        
        # 更新可视化
        if hasattr(self, 'fig'):
            self.update_visualization()
        
        return self.agent_plots
    
    def collect_metrics(self):
        """收集性能指标"""
        # 检查静态目标发现
        for i in range(self.grid_map.grid_height):
            for j in range(self.grid_map.grid_width):
                if self.grid_map.confirmed_targets[i, j]:
                    if (i, j) not in self.metrics['discovery_times']:
                        self.metrics['discovery_times'].append((i, j, self.time_step))
                        self.metrics['targets_found'] += 1
        
        # 检查移动目标发现
        for target in self.moving_targets:
            # 检查目标是否在探测范围内
            for agent in self.agents:
                if agent.can_detect_position(target.position):
                    target.detection_time += 1
                    if target.detection_time >= 5:  # 持续5次探测确认
                        target.detected = True
                    break
            else:
                target.detection_time = 0
            
            if target.detected and not target.reported:
                self.metrics['moving_targets_found'] += 1
                self.metrics['targets_found'] += 1  # 同时计入总目标数
                target.reported = True
        
        # 检查碰撞
        for i in range(len(self.agents)):
            for j in range(i+1, len(self.agents)):
                dist = np.linalg.norm(np.array(self.agents[i].position) - 
                                     np.array(self.agents[j].position))
                min_dist = self.agents[i].safe_distance + self.agents[j].safe_distance
                if dist < min_dist:
                    self.metrics['collision_events'] += 1
        
        # 计算边界覆盖率
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
            self.ax.axhline(y, color='gray', linestyle='-', alpha=0.3)
        for j in range(self.grid_map.grid_width + 1):
            x = j * grid_resolution_m
            self.ax.axvline(x, color='gray', linestyle='-', alpha=0.3)
    
    def update_visualization(self):
        """更新可视化"""
        # 更新智能体位置
        for i, agent in enumerate(self.agents):
            x, y = agent.position
            self.agent_plots[i]['plot'].set_data([x], [y])
            
            if agent.agent_type == 'drone':
                # 更新无人机探测扇形
                wedge = self.agent_plots[i]['wedge']
                wedge.set_center(agent.position)
                # 修正朝向显示，将数学角度转换为绘图角度(90度朝东)
                display_heading = 90 - agent.heading
                wedge.set_theta1(display_heading - agent.sensor.fov/2)
                wedge.set_theta2(display_heading + agent.sensor.fov/2)
            else:
                # 更新无人艇探测圆形
                circle = self.agent_plots[i]['circle']
                circle.set_center(agent.position)
        
        # 更新目标位置和状态
        for plot in self.target_plots:
            plot.remove()
        self.target_plots = []
        
        for text in self.target_texts:
            text.remove()
        self.target_texts = []
        
        # 绘制目标概率热力图
        grid_resolution_m = self.grid_map.resolution * 1852
        for i in range(self.grid_map.grid_height):
            for j in range(self.grid_map.grid_width):
                prob = self.grid_map.target_probability[i, j]
                if prob > 0.1:  # 只显示概率大于0.1的网格
                    x = j * grid_resolution_m + grid_resolution_m/2
                    y = i * grid_resolution_m + grid_resolution_m/2
                    color = (1, 0, 0, min(1.0, prob))  # 红色表示目标概率
                    rect = plt.Rectangle((j*grid_resolution_m, i*grid_resolution_m), 
                                        grid_resolution_m, grid_resolution_m, 
                                        facecolor=color, alpha=0.3)
                    self.ax.add_patch(rect)
                    self.target_plots.append(rect)
                    
                    # 添加概率文本
                    text = self.ax.text(x, y, f"{prob:.1f}", 
                                      ha='center', va='center', fontsize=8)
                    self.target_texts.append(text)
        
        # 添加时间步文本
        time_text = self.ax.text(0.02, 0.98, f"Time: {self.time_step}s", 
                                transform=self.ax.transAxes, fontsize=12,
                                verticalalignment='top')
        self.target_texts.append(time_text)
        
        # 添加指标文本
        metrics_text = self.ax.text(0.98, 0.98, 
                                   f"Targets Found: {self.metrics['targets_found']}\n"
                                   f"Moving Targets Found: {self.metrics['moving_targets_found']}\n"
                                   f"Collisions: {self.metrics['collision_events']}",
                                   transform=self.ax.transAxes, fontsize=10,
                                   verticalalignment='top', horizontalalignment='right')
        self.target_texts.append(metrics_text)
        
        # 绘制移动目标状态
        for target in self.moving_targets:
            x, y = target.position
            if target.detected:
                color = 'green'  # 已确认目标
                size = 10
            elif target.detection_time > 0:
                color = 'yellow'  # 部分检测到
                size = 8
            else:
                color = 'red'  # 未检测到
                size = 6
            plot = self.ax.plot(x, y, 'o', markersize=size, color=color)[0]
            self.target_plots.append(plot)
        
        plt.draw()
    
    def run(self, visualize=True):
        """运行仿真"""
        start_time = time.time()
        
        if visualize:
            # 确保frames为整数
            frames = int(self.duration * self.fps)
            ani = FuncAnimation(self.fig, self.step, frames=frames,
                                interval=int(1000/self.fps), repeat=False)
            plt.show()
        else:
            # 无可视化运行
            for step in range(self.duration//self.fps):
                self.step(step)
                if step % 10 == 0:
                    print(f"Step {step}: Targets found {self.metrics['targets_found']}")
        
        end_time = time.time()
        print(f"仿真完成! 用时: {end_time-start_time:.2f}秒")
        self.report_metrics()
    
    def report_metrics(self):
        """生成性能报告"""
        print("\n===== 仿真性能报告 =====")
        print(f"总仿真时间: {self.time_step}秒")
        print(f"发现静态目标数量: {self.metrics['targets_found']}")
        print(f"发现移动目标数量: {self.metrics['moving_targets_found']}")
        
        if self.metrics['discovery_times']:
            times = [t for _, _, t in self.metrics['discovery_times']]
            print(f"平均目标发现时间: {np.mean(times):.1f}秒")
            print(f"最短发现时间: {min(times)}秒")
            print(f"最长发现时间: {max(times)}秒")
        
        print(f"碰撞事件次数: {self.metrics['collision_events']}")
        
        if self.metrics['boundary_coverage']:
            print(f"最终边界覆盖率: {self.metrics['boundary_coverage'][-1]*100:.1f}%")
            print(f"最高边界覆盖率: {max(self.metrics['boundary_coverage'])*100:.1f}%")
        
        # 绘制指标变化曲线
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
                plt.title('目标发现时间分布')
                plt.xlabel('发现时间(秒)')
                plt.ylabel('目标数量')
            
            plt.subplot(2, 2, 3)
            # 绘制目标概率热力图
            plt.imshow(self.grid_map.target_probability, cmap='hot', origin='lower')
            plt.colorbar(label='目标概率')
            plt.title('最终目标概率分布')
            
            plt.tight_layout()
            plt.show()
