#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
优化版无人机艇协同搜索算法演示
针对性能问题进行优化改进

主要优化：
1. 更激进的搜索策略
2. 改进的载具协调机制  
3. 更快的任务分配频率
4. 预测式目标拦截
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import time
from datetime import datetime

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  
plt.rcParams['axes.unicode_minus'] = False

class OptimizedSimulation:
    """优化版仿真演示类"""
    
    def __init__(self):
        # 任务区域 (5x5海里)
        self.area_size = 5.0
        self.area_bounds = [-2.5, 2.5, -2.5, 2.5]
        
        # 载具初始位置 - 优化部署
        self.uav_positions = np.array([
            [-2.0, 0.0],   # UAV1 中心位置巡逻
            [2.0, 0.0]     # UAV2 中心位置巡逻
        ])
        
        self.usv_positions = np.array([
            [-1.5, -1.5],  # USV分布更集中，便于快速响应
            [1.5, -1.5],
            [-1.5, 1.5],
            [1.5, 1.5]
        ])
        
        # 载具搜索状态
        self.uav_search_directions = np.array([[1.0, 0.5], [-1.0, -0.5]])  # 初始搜索方向
        self.usv_target_assignments = [None] * 4  # USV目标分配
        
        # 目标随机生成
        self.num_targets = 8
        self.targets = self._generate_random_targets()
        
        # 载具参数 - 优化参数
        self.uav_speed = 120 / 1.852  # 120km/h转节
        self.usv_speed = 20           # 20节
        self.uav_detection_range = 3000 / 1852  # 3000m转海里
        self.usv_detection_range = 800 / 1852   # 800m转海里
        
        # 优化的仿真参数
        self.current_time = 0.0
        self.time_step = 15.0  # 缩短到15秒步长，提高响应速度
        self.mission_duration = 7200.0
        self.reallocation_interval = 30.0  # 更频繁的重新分配
        self.last_reallocation = 0.0
        
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
            vx = random.uniform(-8, 8)  # 增加目标速度范围
            vy = random.uniform(-8, 8)
            
            targets.append({
                'id': f'T{i+1}',
                'position': np.array([x, y]),
                'velocity': np.array([vx, vy]),
                'detected': False,
                'disposed': False,
                'detection_time': None,
                'disposal_time': None,
                'assigned_usv': None,
                'last_seen_time': None,
                'predicted_position': np.array([x, y])
            })
        
        return targets
    
    def _move_uavs_optimized(self):
        """优化的无人机移动策略"""
        time_hours = self.time_step / 3600.0
        
        for i, uav_pos in enumerate(self.uav_positions):
            # 如果有未探测区域，朝未探测区域移动
            # 否则执行扫描模式
            
            # 简化的螺旋扫描模式
            if i == 0:  # UAV1 顺时针螺旋
                center = np.array([0.0, 0.0])
                radius = 2.0
                angle = (self.current_time / 60.0) * 0.5  # 较快的扫描速度
                target_x = center[0] + radius * np.cos(angle)
                target_y = center[1] + radius * np.sin(angle)
                
            else:  # UAV2 逆时针螺旋
                center = np.array([0.0, 0.0])
                radius = 1.5
                angle = -(self.current_time / 60.0) * 0.7
                target_x = center[0] + radius * np.cos(angle)
                target_y = center[1] + radius * np.sin(angle)
            
            # 朝目标位置移动
            target_pos = np.array([target_x, target_y])
            direction = target_pos - uav_pos
            distance = np.linalg.norm(direction)
            
            if distance > 0.1:
                direction = direction / distance
                max_move = self.uav_speed * time_hours
                move_distance = min(max_move, distance)
                self.uav_positions[i] += direction * move_distance
            
            # 边界约束
            self.uav_positions[i] = np.clip(self.uav_positions[i], -2.5, 2.5)
    
    def _move_usvs_optimized(self):
        """优化的无人艇移动策略"""
        time_hours = self.time_step / 3600.0
        
        # 重新分配目标
        if self.current_time - self.last_reallocation >= self.reallocation_interval:
            self._reassign_targets()
            self.last_reallocation = self.current_time
        
        for i, usv_pos in enumerate(self.usv_positions):
            target_assigned = self.usv_target_assignments[i]
            
            if target_assigned and not target_assigned['disposed']:
                # 朝分配的目标移动，使用预测位置
                predicted_pos = self._predict_target_position(target_assigned, time_hours)
                direction = predicted_pos - usv_pos
                distance = np.linalg.norm(direction)
                
                if distance > 0.1:
                    direction = direction / distance
                    max_move = self.usv_speed * time_hours
                    move_distance = min(max_move, distance)
                    self.usv_positions[i] += direction * move_distance
            else:
                # 没有分配目标，执行区域搜索
                self._area_search_movement(i, time_hours)
            
            # 边界约束
            self.usv_positions[i] = np.clip(self.usv_positions[i], -2.5, 2.5)
    
    def _predict_target_position(self, target, time_ahead_hours):
        """预测目标位置"""
        if target['last_seen_time']:
            time_since_seen = (self.current_time - target['last_seen_time']) / 3600.0
            predicted = target['predicted_position'] + target['velocity'] * time_since_seen
        else:
            predicted = target['position'] + target['velocity'] * time_ahead_hours
        
        return predicted
    
    def _area_search_movement(self, usv_index, time_hours):
        """区域搜索移动模式"""
        # 每个USV负责一个象限
        quadrants = [
            np.array([-1.25, -1.25]),  # 左下
            np.array([1.25, -1.25]),   # 右下
            np.array([-1.25, 1.25]),   # 左上
            np.array([1.25, 1.25])     # 右上
        ]
        
        center = quadrants[usv_index]
        # 在象限内做小范围搜索
        search_radius = 0.8
        angle = (self.current_time / 30.0 + usv_index * np.pi/2) % (2 * np.pi)
        
        target_x = center[0] + search_radius * np.cos(angle)
        target_y = center[1] + search_radius * np.sin(angle)
        target_pos = np.array([target_x, target_y])
        
        direction = target_pos - self.usv_positions[usv_index]
        distance = np.linalg.norm(direction)
        
        if distance > 0.1:
            direction = direction / distance
            max_move = self.usv_speed * time_hours * 0.5  # 搜索时速度减半
            move_distance = min(max_move, distance)
            self.usv_positions[usv_index] += direction * move_distance
    
    def _reassign_targets(self):
        """重新分配目标给USV"""
        # 获取已发现但未处置的目标
        available_targets = [t for t in self.targets 
                           if t['detected'] and not t['disposed']]
        
        # 清空当前分配
        self.usv_target_assignments = [None] * 4
        
        # 贪心分配：为每个目标找最近的USV
        for target in available_targets:
            best_usv = None
            min_distance = float('inf')
            
            for i, usv_pos in enumerate(self.usv_positions):
                if self.usv_target_assignments[i] is None:  # USV空闲
                    predicted_pos = self._predict_target_position(target, 0.1)
                    distance = np.linalg.norm(predicted_pos - usv_pos)
                    
                    if distance < min_distance:
                        min_distance = distance
                        best_usv = i
            
            if best_usv is not None:
                self.usv_target_assignments[best_usv] = target
                target['assigned_usv'] = f'USV{best_usv+1}'
    
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
    
    def _perform_detection_optimized(self):
        """优化的探测功能"""
        # 无人机探测 - 提高探测效率
        for i, uav_pos in enumerate(self.uav_positions):
            for target in self.targets:
                if not target['detected']:
                    distance = np.linalg.norm(target['position'] - uav_pos)
                    if distance <= self.uav_detection_range:
                        # 提高探测概率
                        detection_prob = max(0.8, 1.0 - distance / self.uav_detection_range * 0.3)
                        if random.random() < detection_prob:
                            target['detected'] = True
                            target['detection_time'] = self.current_time
                            target['last_seen_time'] = self.current_time
                            target['predicted_position'] = target['position'].copy()
                            self.detected_targets.append(target)
                            self.detection_times.append(self.current_time)
                            print(f"⚡ UAV{i+1} 发现目标 {target['id']} 在位置 ({target['position'][0]:.1f}, {target['position'][1]:.1f})")
        
        # 无人艇探测和处置
        for i, usv_pos in enumerate(self.usv_positions):
            for target in self.targets:
                distance = np.linalg.norm(target['position'] - usv_pos)
                
                # 探测
                if not target['detected'] and distance <= self.usv_detection_range:
                    detection_prob = max(0.9, 1.0 - distance / self.usv_detection_range * 0.2)
                    if random.random() < detection_prob:
                        target['detected'] = True
                        target['detection_time'] = self.current_time
                        target['last_seen_time'] = self.current_time
                        target['predicted_position'] = target['position'].copy()
                        self.detected_targets.append(target)
                        self.detection_times.append(self.current_time)
                        print(f"⚡ USV{i+1} 发现目标 {target['id']}")
                
                # 更新目标信息（已发现的目标）
                if target['detected'] and distance <= self.usv_detection_range:
                    target['last_seen_time'] = self.current_time
                    target['predicted_position'] = target['position'].copy()
                
                # 处置（距离100米内）
                if (target['detected'] and not target['disposed'] and 
                    distance <= 0.054):  # 100米转海里
                    target['disposed'] = True
                    target['disposal_time'] = self.current_time
                    target['assigned_usv'] = f'USV{i+1}'
                    self.disposed_targets.append(target)
                    self.disposal_times.append(self.current_time)
                    print(f"🎯 USV{i+1} 处置目标 {target['id']} 完成！")
    
    def run_simulation(self):
        """运行优化仿真"""
        print("🚀 开始优化版无人机艇协同搜索仿真...")
        print(f"任务区域：{self.area_size}×{self.area_size} 海里")
        print(f"目标数量：{self.num_targets}个")
        print(f"载具配置：2架无人机，4艘无人艇")
        print(f"优化参数：{self.time_step}秒步长，{self.reallocation_interval}秒重分配间隔")
        print("-" * 60)
        
        step_count = 0
        report_interval = 180  # 3分钟报告一次
        
        while self.current_time < self.mission_duration:
            # 执行仿真步骤
            self._move_uavs_optimized()
            self._move_usvs_optimized()
            self._update_target_positions()
            self._perform_detection_optimized()
            
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
        detected_count = len([t for t in self.targets if t['detected']])
        disposed_count = len([t for t in self.targets if t['disposed']])
        
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
        print("🏆 优化版算法最终结果")
        print("="*60)
        print(f"发现概率 (P):      {metrics['detection_probability']:.1%}")
        print(f"发现评分 (S1):     {metrics['detection_score']:.0f}/20分")
        print(f"处置评分 (S2):     {metrics['disposal_score']:.0f}/20分")
        print(f"平均发现时间:      {metrics['avg_detection_time']:.1f}分钟")
        print(f"平均处置时间:      {metrics['avg_disposal_time']:.1f}分钟")
        print(f"任务完成度:        {metrics['mission_completion']:.1%}")
        print(f"总体性能得分:      {metrics['total_objective_score']:.0f}/60分")
        
        # 性能改进分析
        if metrics['avg_detection_time'] <= 5:
            detection_level = "🏆 优秀"
        elif metrics['avg_detection_time'] <= 10:
            detection_level = "👍 良好"
        else:
            detection_level = "⚠️ 需改进"
            
        if metrics['avg_disposal_time'] <= 10:
            disposal_level = "🏆 优秀"
        elif metrics['avg_disposal_time'] <= 15:
            disposal_level = "👍 良好"
        else:
            disposal_level = "⚠️ 需改进"
        
        print(f"发现时间评价:      {detection_level}")
        print(f"处置时间评价:      {disposal_level}")
        
        # 总体评价
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
        detected_count = len([t for t in self.targets if t['detected']])
        disposed_count = len([t for t in self.targets if t['disposed']])
        
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

def main():
    """主函数"""
    print("🚁🚢 优化版无人机艇协同搜索算法演示")
    print("=" * 50)
    
    # 创建优化仿真实例
    sim = OptimizedSimulation()
    
    # 运行仿真
    start_time = time.time()
    metrics = sim.run_simulation()
    end_time = time.time()
    
    print(f"\n⏱️  仿真运行时间: {end_time - start_time:.2f}秒")
    
    print("\n✅ 优化版仿真演示完成！")
    
    return metrics

if __name__ == "__main__":
    main() 