import argparse
from simulation.scenarios import create_basic_scenario, create_boundary_priority_scenario, create_collision_avoidance_scenario, create_moving_targets_scenario
from simulation.framework import Simulation

def main():
    parser = argparse.ArgumentParser(description='无人机艇协同搜索算法仿真')
    parser.add_argument('--scenario', type=str, default='basic',
                        choices=['basic', 'boundary', 'collision', 'moving'],
                        help='选择测试场景 (basic, boundary, collision, moving)')
    parser.add_argument('--duration', type=int, default=600,
                        help='仿真时长(秒)')
    parser.add_argument('--fps', type=int, default=2,
                        help='帧率(每秒更新次数)')
    parser.add_argument('--visualize', action='store_true',
                        help='启用可视化')
    parser.add_argument('--enable-typical', action='store_true',
                        help='启用典型场景：5x5海里、2小时、8个目标、同一时刻不超过2个进入')
    
    args = parser.parse_args()
    
    # 根据选择的场景创建MPC控制器
    if args.scenario == 'basic':
        mpc = create_basic_scenario()
    elif args.scenario == 'boundary':
        mpc = create_boundary_priority_scenario()
    elif args.scenario == 'collision':
        mpc = create_collision_avoidance_scenario()
    elif args.scenario == 'moving':
        mpc = create_moving_targets_scenario()
    
    # 典型场景配置：2小时=7200秒
    spawn_cfg = None
    duration = args.duration
    if args.enable_typical:
        duration = 7200
        spawn_cfg = {
            'enable': True,
            'total_targets': 8,
            'max_spawn_per_step': 2,
            'spawn_interval_seconds': 900  # 每15分钟尝试生成（总计8个，均匀分布）
        }
    
    # 创建并运行仿真
    sim = Simulation(mpc, duration=duration, fps=args.fps, spawn_config=spawn_cfg)
    sim.run(visualize=args.visualize)
    
    # 保存结果
    if args.save:
        sim.save_results(args.save)

if __name__ == "__main__":
    main()