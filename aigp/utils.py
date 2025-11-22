# aigp/utils.py
import time
import contextlib

@contextlib.contextmanager
def Timer(name="任务"):
    """
    上下文管理器，用于记录任务运行时间
    """
    start = time.time()
    print("开始 {}...".format(name))
    yield
    elapsed = time.time() - start
    print("{} 完成，耗时：{:.2f}秒".format(name, elapsed))
