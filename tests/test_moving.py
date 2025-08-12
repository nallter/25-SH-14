import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from simulation.scenarios import create_moving_targets_scenario
from simulation.framework import Simulation

def test_moving_targets():
    """移动目标测试"""
    mpc = create_moving_targets_scenario()
    
    sim = Simulation(mpc, duration=200, fps=5)
    sim.run(visualize=True)
    
    # 验证所有目标被发现
    assert sim.metrics['targets_found'] == 2, "未能跟踪移动目标"
    
    # 检查无人艇是否存在
    assert any(agent.agent_type == 'usv' for agent in mpc.agents), "场景中缺少无人艇"

if __name__ == "__main__":
    test_moving_targets()
