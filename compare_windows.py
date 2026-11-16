"""兼容旧批处理命令；当前仅运行轨道根数预测及其基线。"""
import sys
from compare_elements import main

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
