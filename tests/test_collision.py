from simulation.scenarios import create_collision_avoidance_scenario
from simulation.framework import Simulation

def test_collision_avoidance():
    """避撞机制测试"""
    mpc = create_collision_avoidance_scenario()
    sim = Simulation(mpc, duration=120, fps=2)
    sim.run(visualize=True)
    
    # 验证无碰撞发生
    assert sim.metrics['collision_events'] == 0, "避撞机制失效"

if __name__ == "__main__":
    test_collision_avoidance()