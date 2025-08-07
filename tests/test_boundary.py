from simulation.scenarios import create_boundary_priority_scenario
from simulation.framework import Simulation

def test_boundary_priority():
    """边界优先策略测试"""
    mpc = create_boundary_priority_scenario()
    sim = Simulation(mpc, duration=300, fps=2)
    sim.run(visualize=True)
    
    # 验证边界覆盖率在前期较高
    assert max(sim.metrics['boundary_coverage']) > 0.7, "边界优先策略未生效"

if __name__ == "__main__":
    test_boundary_priority()