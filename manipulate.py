# main.py

from collimator.simulation import Simulator
from franka_load_env import build_diagram

def main():
    diagram = build_diagram()
    simulator = Simulator(diagram)

    print("Starting the Collimator simulation. Press Ctrl+C to stop.")
    try:
        simulator.initialize()
        simulator.run_for(duration=5.0)
        print("Finished 5 seconds of simulation.")
    except KeyboardInterrupt:
        print("Simulation interrupted by user.")

    print("Exiting main.")

if __name__ == "__main__":
    main()
