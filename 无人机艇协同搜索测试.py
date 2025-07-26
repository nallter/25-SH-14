"""
无人机艇协同搜索算法测试脚本
测试不同目标出现场景下算法的探测性能
"""

import math
import random
from 无人机艇协同搜索算法设计方案 import *

def test_random_targets():
    """测试随机目标出现场景"""
    print("\n=== 测试随机目标 ===")
    area = MissionArea(-2.5, 2.5, -2.5, 2.5)
    controller = MainController(area)
    
    # 部署载具（使用自动初始化）
    print("正在初始化载具...")
    controller.initialize_mission()
    print(f"初始化完成 - UAV数量: {len(controller.system.uavs)}, USV数量: {len(controller.system.usvs)}")
    
    # 使用controller.initialize_mission()中配置的位置
    
    # 记录目标进入时间
    target_enter_time = {}
    
    # 运行测试直到所有目标都被发现
    step = 0
    max_steps = 120  # 最多运行120步（2小时）
    # 添加异常处理和性能监控
    try:
        while step <= max_steps:
            # 添加循环保护，防止无限循环
            if step > 0 and step % 10 == 0:
                print(f"检查点：已运行{step}步")
                # 检查是否有进展
                detected_count = len([t for t in controller.system.targets if t.status != TargetStatus.UNKNOWN])
                if step > 30 and detected_count == 0:
                    print("警告：长时间未发现目标，可能存在问题")
                    break
            # 生成新目标（严格限制不超过8个）
            if len(controller.system.targets) < 8 and random.random() < 0.2:  # 降低生成概率
                # 先检查当前未发现目标数
                undetected = [t for t in controller.system.targets if t.status == TargetStatus.UNKNOWN]
                if len(undetected) < 4:  # 未发现目标少于4个时才生成新目标
                    controller._generate_new_targets()
                    # 确保记录所有目标的进入时间
                    for target in controller.system.targets[-1:]:  # 只处理最新生成的目标
                        if target.id not in target_enter_time:
                            target_enter_time[target.id] = step * 60  # 转换为秒
                        
            # 额外检查确保所有目标都有进入时间记录
            for target in controller.system.targets:
                if target.id not in target_enter_time:
                    target_enter_time[target.id] = step * 60  # 保守估计

            # 添加调试输出
            print(f"Step {step}: 当前目标数 {len(controller.system.targets)}, 已发现 {len([t for t in controller.system.targets if t.status == TargetStatus.DETECTED])}")
                
            print(f"\n=== Step {step} 详细执行日志 ===")
            print("1. 正在执行载具移动...")
            for uav in controller.system.uavs:
                print(f"  UAV {uav.id} 当前位置: {uav.position}")
            for usv in controller.system.usvs:
                print(f"  USV {usv.id} 当前位置: {usv.position}")
                
            print("2. 正在执行探测计算...")
            status = controller.execute_mission_step(60)  # 60秒步长
            print(f"3. 任务步骤执行结果: {status}")
            
            print("4. 系统状态检查:")
            print(f"  活动目标数: {len(controller.system.targets)}")
            print(f"  已发现目标: {len([t for t in controller.system.targets if t.status == TargetStatus.DETECTED])}")
            
            # 计算实际发现时间（从进入区域到被发现）
            detection_times = []
            for target in controller.system.targets:
                if target.status == TargetStatus.DETECTED and target.detected_time:
                    # 安全获取进入时间，如果不存在则使用保守估计
                    enter_time = target_enter_time.get(target.id, target.detected_time - 60)  # 默认1分钟前
                    detection_time = target.detected_time - enter_time
                    detection_times.append(detection_time)
            
            avg_detection_time = sum(detection_times)/len(detection_times) if detection_times else 0
            
            print(f"时间 {step}分钟 - 目标数: {len(controller.system.targets)} | "
                  f"已发现: {len([t for t in controller.system.targets if t.status == TargetStatus.DETECTED])} | "
                  f"平均发现时间: {avg_detection_time/60:.1f}分钟")
            
            # 输出目标状态
            for target in controller.system.targets:
                edge = ""
                pos = target.position
                area = controller.system.mission_area
                if abs(pos.y - area.max_y) < 0.1:
                    edge = "上边进入"
                elif abs(pos.y - area.min_y) < 0.1:
                    edge = "下边进入" 
                elif abs(pos.x - area.max_x) < 0.1:
                    edge = "右边进入"
                elif abs(pos.x - area.min_x) < 0.1:
                    edge = "左边进入"
                
                status = "未发现" if target.status == TargetStatus.UNKNOWN else "已发现"
                print(f"目标 {target.id}: {status} {edge} 位置({pos.x:.2f},{pos.y:.2f}) "
                      f"速度{math.sqrt(target.velocity[0]**2 + target.velocity[1]**2):.1f}节")
            
            # 检查是否所有目标都被发现或超时
            if (all(t.status != TargetStatus.UNKNOWN for t in controller.system.targets) and len(controller.system.targets) >= 8) or step >= max_steps or controller.system.current_time >= controller.system.mission_duration:
                # 输出未发现目标的信息
                for target in controller.system.targets:
                    if target.status == TargetStatus.UNKNOWN:
                        print(f"警告: 目标 {target.id} 未被发现 - 最后位置({target.position.x:.2f},{target.position.y:.2f}) 速度{math.sqrt(target.velocity[0]**2 + target.velocity[1]**2):.1f}节")
                break
                
            step += 1
            
    except Exception as e:
        print(f"运行时异常: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        print("测试结束，清理资源...")

if __name__ == "__main__":
    # 只运行随机目标测试
    test_random_targets()
