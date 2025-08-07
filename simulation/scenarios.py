
from core.algorithm import Agent, DistributedMPC,SearchGrid,MovingTarget
from simulation.framework import Simulation

def create_basic_scenario():
    """创建基础功能验证场景"""
    grid = SearchGrid(width=5, height=5, resolution=0.1)  # 5海里×5海里
    
    # 计算初始位置 (区域左侧边中段)
    center_y = 4630  # 区域高度9260m(5海里)，中点为4630m
    
    # 无人艇配置 (间距1km)
    usv_spacing = 1000
    usv1_y = center_y - usv_spacing*1.5  # 3130m
    usv2_y = center_y - usv_spacing*0.5  # 4130m
    usv3_y = center_y + usv_spacing*0.5  # 5130m
    usv4_y = center_y + usv_spacing*1.5  # 6130m
    
    # 无人机配置 (间距2km)
    uav_spacing = 2000
    uav1_y = center_y - uav_spacing*0.5  # 3630m
    uav2_y = center_y + uav_spacing*0.5  # 5630m
    
    agents = [
        Agent("UAV1", position=(0, uav1_y), heading=90, agent_type="drone", safe_distance=50),
        Agent("UAV2", position=(0, uav2_y), heading=90, agent_type="drone", safe_distance=50),
        Agent("USV1", position=(0, usv1_y), heading=90, agent_type="usv", safe_distance=100),
        Agent("USV2", position=(0, usv2_y), heading=90, agent_type="usv", safe_distance=100),
        Agent("USV3", position=(0, usv3_y), heading=90, agent_type="usv", safe_distance=100),
        Agent("USV4", position=(0, usv4_y), heading=90, agent_type="usv", safe_distance=100),
    ]
    
    mpc = DistributedMPC(agents, grid)
    
    # 添加静态目标
    grid.add_target(position=(1000, 1000))  # 西北角
    grid.add_target(position=(4000, 4000))  # 东南角
    
    return mpc

def create_boundary_priority_scenario():
    """创建边界优先策略验证场景"""
    grid = SearchGrid(width=5, height=5, resolution=0.1)
    agents = [
        Agent("UAV1", position=(0, 2500), heading=90, agent_type="drone", safe_distance=50),
        Agent("UAV2", position=(0, 4500), heading=90, agent_type="drone", safe_distance=50),
    ]
    mpc = DistributedMPC(agents, grid)
    
    # 在边界添加多个目标
    grid.add_target(position=(500, 500))         # 西北
    grid.add_target(position=(500, 4500))       # 东北
    grid.add_target(position=(4500, 500))        # 西南
    grid.add_target(position=(4500, 4500))      # 东南
    
    return mpc

def create_collision_avoidance_scenario():
    """创建避撞机制验证场景"""
    grid = SearchGrid(width=5, height=5, resolution=0.1)
    
    # 创建密集部署的智能体
    agents = [
        Agent("UAV1", position=(1000, 1000), heading=90, agent_type="drone", safe_distance=50),
        Agent("UAV2", position=(1050, 1050), heading=90, agent_type="drone", safe_distance=50),
        Agent("USV1", position=(3000, 3000), heading=90, agent_type="usv", safe_distance=100),
        Agent("USV2", position=(3050, 3050), heading=90, agent_type="usv", safe_distance=100),
    ]
    
    return DistributedMPC(agents, grid)

def create_moving_targets_scenario():
    """创建移动目标验证场景"""
    grid = SearchGrid(width=5, height=5, resolution=0.1)
    agents = [
        Agent("UAV1", position=(0, 2500), heading=90, agent_type="drone", safe_distance=50),
        Agent("UAV2", position=(0, 4500), heading=90, agent_type="drone", safe_distance=50),
        # 添加无人艇
        Agent("USV1", position=(0, 1500), heading=90, agent_type="usv", safe_distance=100),
        Agent("USV2", position=(0, 3500), heading=90, agent_type="usv", safe_distance=100),
    ]
    mpc = DistributedMPC(agents, grid)
    
    # 添加移动目标
    target1 = MovingTarget(start_position=(1000, 1000))  # 15节=7.72m/s
    target2 = MovingTarget(start_position=(4000, 1000))
    grid.add_target(target1.position)
    grid.add_target(target2.position)
    
    return mpc
