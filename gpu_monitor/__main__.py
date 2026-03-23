"""
__main__.py — Entry point: python -m gpu_monitor
"""
from gpu_monitor.app import GpuMonitorApp


def main() -> None:
    GpuMonitorApp().run()


if __name__ == "__main__":
    main()
