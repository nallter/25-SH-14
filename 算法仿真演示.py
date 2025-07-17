#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
无人机艇协同搜索算法仿真演示
简化版本，用于快速验证算法效果

运行方法：python 算法仿真演示.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import random
import time
from datetime import datetime

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  
plt.rcParams['axes.unicode_minus'] = False

class SimpleSimulation:
    """简化的仿真演示类"""
    
    def __init__(self):
        # 任务区域 (5x5海里)
        self.area_size = 5.0
        self.area_bounds = [-2.5, 2.5, -2.5, 2.5]
        
        # 载具初始位置
        self.uav_positions = np.array([
            [-2.0, -2.0],  # UAV1 左下
            [2.0, 2.0]     # UAV2 右上
        ])
        
        self.usv_positions = np.array([
            [-1.0, -1.0],  # USV1 左下区域
            [1.0, -1.0],   # USV2 右下区域  
            [-1.0, 1.0],   # USV3 左上区域
            [1.0, 1.0]     # USV4 右上区域
        ])
        
        # 目标随机生成
        self.num_targets = 8
        self.targets = self._generate_random_targets()
        
        # 载具参数
        self.uav_speed = 120 / 1.852  # 120km/h转节
        self.usv_speed = 20           # 20节
        self.uav_detection_range = 3000 / 1852  # 3000m转海里
        self.usv_detection_range = 800 / 1852   # 800m转海里
        
        # 仿真状态
        self.current_time = 0.0
        self.time_step = 30.0  # 30秒步长
        self.mission_duration = 7200.0  # 2小时
        
        # 性能统计
        self.detected_targets = []
        self.disposed_targets = []
        self.detection_times = []
        self.disposal_times = []
        
    def _generate_random_targets(self):
        """生成随机分布的目标"""
        targets = []
        for i in range(self.num_targets):
            x = random.uniform(-2.0, 2.0)
            y = random.uniform(-2.0, 2.0)
            vx = random.uniform(-5, 5)  # 节
            vy = random.uniform(-5, 5)  # 节
            
            targets.append({
                'id': f'T{i+1}',
                'position': np.array([x, y]),
                'velocity': np.array([vx, vy]),
                'detected': False,
                'disposed': False,
                'detection_time': None,
                'disposal_time': None,
                'assigned_usv': None
            })
        
        return targets
    
    def _move_vehicles(self):
        """移动载具（简化的搜索模式）"""
        time_hours = self.time_step / 3600.0
        
        # 无人机采用简单的巡逻模式
        for i, uav_pos in enumerate(self.uav_positions):
            if i == 0:  # UAV1 水平巡逻
                self.uav_positions[i][0] += self.uav_speed * time_hours * 0.5
                if self.uav_positions[i][0] > 2.5:
                    self.uav_positions[i][0] = -2.5
            else:  # UAV2 垂直巡逻
                self.uav_positions[i][1] += self.uav_speed * time_hours * 0.5
                if self.uav_positions[i][1] > 2.5:
                    self.uav_positions[i][1] = -2.5
        
        # 无人艇朝最近未处置目标移动
        for i, usv_pos in enumerate(self.usv_positions):
            nearest_target = self._find_nearest_unassigned_target(usv_pos)
            if nearest_target:
                direction = nearest_target['position'] - usv_pos
                distance = np.linalg.norm(direction)
                if distance > 0:
                    direction = direction / distance
                    max_move = self.usv_speed * time_hours
                    if distance <= max_move:
                        self.usv_positions[i] = nearest_target['position'].copy()
                    else:
                        self.usv_positions[i] += direction * max_move
    
    def _find_nearest_unassigned_target(self, usv_pos):
        """找到最近的未分配目标"""
        min_distance = float('inf')
        nearest_target = None
        
        for target in self.targets:
            if target['detected'] and not target['disposed'] and not target['assigned_usv']:
                distance = np.linalg.norm(target['position'] - usv_pos)
                if distance < min_distance:
                    min_distance = distance
                    nearest_target = target
        
        return nearest_target
    
    def _update_target_positions(self):
        """更新目标位置"""
        time_hours = self.time_step / 3600.0
        
        for target in self.targets:
            target['position'] += target['velocity'] * time_hours
            
            # 边界反弹
            if target['position'][0] < -2.5 or target['position'][0] > 2.5:
                target['velocity'][0] *= -1
                target['position'][0] = np.clip(target['position'][0], -2.5, 2.5)
            
            if target['position'][1] < -2.5 or target['position'][1] > 2.5:
                target['velocity'][1] *= -1
                target['position'][1] = np.clip(target['position'][1], -2.5, 2.5)
    
    def _perform_detection(self):
        """执行探测"""
        # 无人机探测
        for i, uav_pos in enumerate(self.uav_positions):
            for target in self.targets:
                if not target['detected']:
                    distance = np.linalg.norm(target['position'] - uav_pos)
                    if distance <= self.uav_detection_range:
                        # 简化的探测概率
                        detection_prob = max(0.5, 1.0 - distance / self.uav_detection_range)
                        if random.random() < detection_prob:
                            target['detected'] = True
                            target['detection_time'] = self.current_time
                            self.detected_targets.append(target)
                            self.detection_times.append(self.current_time)
        
        # 无人艇探测和处置
        for i, usv_pos in enumerate(self.usv_positions):
            for target in self.targets:
                distance = np.linalg.norm(target['position'] - usv_pos)
                
                # 探测
                if not target['detected'] and distance <= self.usv_detection_range:
                    detection_prob = max(0.7, 1.0 - distance / self.usv_detection_range)
                    if random.random() < detection_prob:
                        target['detected'] = True
                        target['detection_time'] = self.current_time
                        self.detected_targets.append(target)
                        self.detection_times.append(self.current_time)
                
                # 处置（距离100米内）
                if (target['detected'] and not target['disposed'] and 
                    distance <= 0.054):  # 100米转海里
                    target['disposed'] = True
                    target['disposal_time'] = self.current_time
                    target['assigned_usv'] = f'USV{i+1}'
                    self.disposed_targets.append(target)
                    self.disposal_times.append(self.current_time)
    
    def run_simulation(self):
        """运行完整仿真"""
        print("开始无人机艇协同搜索仿真演示...")
        print(f"任务区域：{self.area_size}×{self.area_size} 海里")
        print(f"目标数量：{self.num_targets}个")
        print(f"载具配置：2架无人机，4艘无人艇")
        print("-" * 50)
        
        step_count = 0
        report_interval = 300  # 5分钟报告一次
        
        while self.current_time < self.mission_duration:
            # 执行仿真步骤
            self._move_vehicles()
            self._update_target_positions()
            self._perform_detection()
            
            self.current_time += self.time_step
            step_count += 1
            
            # 定期报告
            if self.current_time % report_interval < self.time_step:
                self._print_status_report()
            
            # 检查是否所有目标都已处置
            if len(self.disposed_targets) == self.num_targets:
                print(f"\n🎉 所有目标处置完成！总用时：{self.current_time/60:.1f}分钟")
                break
        
        # 最终结果
        self._print_final_results()
        return self._calculate_performance_metrics()
    
    def _print_status_report(self):
        """打印状态报告"""
        detected_count = len(self.detected_targets)
        disposed_count = len(self.disposed_targets)
        
        avg_detection_time = (np.mean(self.detection_times) / 60 
                            if self.detection_times else 0)
        avg_disposal_time = (np.mean(self.disposal_times) / 60 
                           if self.disposal_times else 0)
        
        print(f"时间: {self.current_time/60:5.0f}分钟 | "
              f"发现: {detected_count:2d}/{self.num_targets} | "
              f"处置: {disposed_count:2d}/{self.num_targets} | "
              f"平均发现时间: {avg_detection_time:4.1f}分钟 | "
              f"平均处置时间: {avg_disposal_time:4.1f}分钟")
    
    def _print_final_results(self):
        """打印最终结果"""
        metrics = self._calculate_performance_metrics()
        
        print("\n" + "="*60)
        print("🏆 最终仿真结果")
        print("="*60)
        print(f"发现概率 (P):      {metrics['detection_probability']:.1%}")
        print(f"发现评分 (S1):     {metrics['detection_score']:.0f}/20分")
        print(f"处置评分 (S2):     {metrics['disposal_score']:.0f}/20分")
        print(f"平均发现时间:      {metrics['avg_detection_time']:.1f}分钟")
        print(f"平均处置时间:      {metrics['avg_disposal_time']:.1f}分钟")
        print(f"任务完成度:        {metrics['mission_completion']:.1%}")
        print(f"总体性能得分:      {metrics['total_objective_score']:.0f}/60分")
        
        # 评价等级
        total_score = metrics['total_objective_score']
        if total_score >= 55:
            grade = "优秀 🏆"
        elif total_score >= 45:
            grade = "良好 👍"
        elif total_score >= 35:
            grade = "合格 ✓"
        else:
            grade = "需改进 ⚠️"
        
        print(f"算法评价等级:      {grade}")
        print("="*60)
    
    def _calculate_performance_metrics(self):
        """计算性能指标"""
        detected_count = len(self.detected_targets)
        disposed_count = len(self.disposed_targets)
        
        # 发现概率
        detection_probability = detected_count / self.num_targets
        
        # 平均时间（分钟）
        avg_detection_time = (np.mean(self.detection_times) / 60 
                            if self.detection_times else 0)
        avg_disposal_time = (np.mean(self.disposal_times) / 60 
                           if self.disposal_times else 0)
        
        # 评分计算（基于比赛标准）
        # S1: 发现时间评分
        if avg_detection_time <= 5:
            detection_score = 20
        elif avg_detection_time <= 10:
            detection_score = 20 - (avg_detection_time - 5) * 2
        else:
            detection_score = 0
        
        # S2: 处置时间评分
        if avg_disposal_time <= 10:
            disposal_score = 20
        elif avg_disposal_time <= 15:
            disposal_score = 20 - (avg_disposal_time - 10) * 4
        else:
            disposal_score = 0
        
        # P: 发现概率评分（满分20分）
        probability_score = detection_probability * 20
        
        return {
            'detection_probability': detection_probability,
            'avg_detection_time': avg_detection_time,
            'avg_disposal_time': avg_disposal_time,
            'detection_score': detection_score,
            'disposal_score': disposal_score,
            'probability_score': probability_score,
            'total_objective_score': probability_score + detection_score + disposal_score,
            'mission_completion': disposed_count / self.num_targets,
            'detected_count': detected_count,
            'disposed_count': disposed_count
        }
    
    def visualize_mission(self, save_plot=True):
        """可视化任务执行情况"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('无人机艇协同搜索任务可视化分析', fontsize=16, fontweight='bold')
        
        # 1. 任务区域和载具位置
        ax1.set_xlim(-3, 3)
        ax1.set_ylim(-3, 3)
        ax1.set_title('任务区域和载具部署', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # 绘制任务区域
        mission_area = patches.Rectangle((-2.5, -2.5), 5, 5, 
                                       linewidth=2, edgecolor='blue', 
                                       facecolor='lightblue', alpha=0.3)
        ax1.add_patch(mission_area)
        
        # 绘制无人机
        for i, pos in enumerate(self.uav_positions):
            circle = patches.Circle(pos, self.uav_detection_range, 
                                  alpha=0.2, color='red')
            ax1.add_patch(circle)
            ax1.plot(pos[0], pos[1], '^', markersize=12, color='red', 
                    label=f'UAV{i+1}' if i == 0 else "")
        
        # 绘制无人艇
        for i, pos in enumerate(self.usv_positions):
            circle = patches.Circle(pos, self.usv_detection_range, 
                                  alpha=0.2, color='green')
            ax1.add_patch(circle)
            ax1.plot(pos[0], pos[1], 's', markersize=10, color='green',
                    label=f'USV{i+1}' if i == 0 else "")
        
        # 绘制目标
        for target in self.targets:
            color = 'orange' if target['disposed'] else ('yellow' if target['detected'] else 'gray')
            marker = 'X' if target['disposed'] else ('o' if target['detected'] else '.')
            ax1.plot(target['position'][0], target['position'][1], 
                    marker, markersize=8, color=color)
        
        ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax1.set_xlabel('东西方向 (海里)')
        ax1.set_ylabel('南北方向 (海里)')
        
        # 2. 性能指标雷达图
        metrics = self._calculate_performance_metrics()
        categories = ['发现概率', '发现速度', '处置速度', '任务完成度']
        values = [
            metrics['detection_probability'],
            max(0, (10 - metrics['avg_detection_time']) / 10),  # 归一化发现速度
            max(0, (15 - metrics['avg_disposal_time']) / 15),   # 归一化处置速度
            metrics['mission_completion']
        ]
        
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        values += values[:1]  # 闭合图形
        angles += angles[:1]
        
        ax2.plot(angles, values, 'o-', linewidth=2, color='blue')
        ax2.fill(angles, values, alpha=0.25, color='blue')
        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(categories)
        ax2.set_ylim(0, 1)
        ax2.set_title('算法性能雷达图', fontweight='bold')
        ax2.grid(True)
        
        # 3. 时间线分析
        if self.detection_times:
            ax3.hist(np.array(self.detection_times) / 60, bins=10, alpha=0.7, 
                    color='orange', label='目标发现时间')
        if self.disposal_times:
            ax3.hist(np.array(self.disposal_times) / 60, bins=10, alpha=0.7, 
                    color='green', label='目标处置时间')
        
        ax3.axvline(x=5, color='red', linestyle='--', alpha=0.7, label='发现时间目标(5分钟)')
        ax3.axvline(x=10, color='purple', linestyle='--', alpha=0.7, label='处置时间目标(10分钟)')
        ax3.set_xlabel('时间 (分钟)')
        ax3.set_ylabel('目标数量')
        ax3.set_title('目标发现和处置时间分布', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 评分对比
        scores = [metrics['probability_score'], metrics['detection_score'], metrics['disposal_score']]
        score_names = ['发现概率\n(P)', '发现时间\n(S1)', '处置时间\n(S2)']
        colors = ['skyblue', 'lightgreen', 'lightcoral']
        
        bars = ax4.bar(score_names, scores, color=colors, alpha=0.8, edgecolor='black')
        ax4.axhline(y=20, color='red', linestyle='--', alpha=0.7, label='满分线(20分)')
        ax4.set_ylim(0, 22)
        ax4.set_ylabel('得分')
        ax4.set_title('客观评分结果', fontweight='bold')
        ax4.legend()
        
        # 在柱状图上显示具体分值
        for i, (bar, score) in enumerate(zip(bars, scores)):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{score:.1f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        if save_plot:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'仿真结果_{timestamp}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"\n📊 可视化结果已保存为: {filename}")
        
        plt.show()
        return fig

def main():
    """主函数"""
    print("🚁🚢 无人机艇协同搜索算法仿真演示")
    print("=" * 50)
    
    # 创建仿真实例
    sim = SimpleSimulation()
    
    # 运行仿真
    start_time = time.time()
    metrics = sim.run_simulation()
    end_time = time.time()
    
    print(f"\n⏱️  仿真运行时间: {end_time - start_time:.2f}秒")
    
    # 生成可视化
    print("\n📈 正在生成可视化图表...")
    sim.visualize_mission()
    
    print("\n✅ 仿真演示完成！")
    
    return metrics

if __name__ == "__main__":
    main() 