import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.algorithm import SearchGrid, Agent, DistributedMPC, StaticTarget
from simulation.framework import Simulation

def test_basic_functionality():
    """基础功能验证测试"""
    # 创建5海里×5海里的搜索区域
    grid = SearchGrid(width=5, height=5, resolution=0.1)
    
    # 配置智能体 (2无人机+4无人艇)
    agents = [
        Agent("UAV1", position=(0, 3630), heading=90, agent_type="drone", safe_distance=50),
        Agent("UAV2", position=(0, 5630), heading=90, agent_type="drone", safe_distance=50),
        Agent("USV1", position=(0, 3130), heading=90, agent_type="usv", safe_distance=100),
        Agent("USV2", position=(0, 4130), heading=90, agent_type="usv", safe_distance=100),
        Agent("USV3", position=(0, 5130), heading=90, agent_type="usv", safe_distance=100),
        Agent("USV4", position=(0, 6130), heading=90, agent_type="usv", safe_distance=100),
    ]
    
    # 创建MPC控制器
    mpc = DistributedMPC(agents, grid)
    
    # 添加多个静态目标对象
    targets = [
        StaticTarget(position=(2000, 2000)),  # 区域中心附近
        StaticTarget(position=(4000, 4000)),  # 东南区域
        StaticTarget(position=(1000, 5000)),  # 西南区域
        StaticTarget(position=(5000, 1000))   # 东北区域
    ]
    for target in targets:
        grid.add_target_object(target)
    
    # 运行仿真
    sim = Simulation(mpc, duration=1200, fps=5)
    sim.run(visualize=True)

if __name__ == "__main__":
    test_basic_functionality()
