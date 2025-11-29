import matlab.engine
import atexit
import os
import sys


class MatlabEngineManager:
    """
    MATLAB Engine 的单例管理器。
    确保只启动一个 Engine 实例，并在 Python 退出时自动关闭。
    """
    _instance = None
    _engine = None
    _is_engine_available = False  # 标记Engine是否成功启动

    def __new__(cls):
        """实现单例模式：确保只创建一个实例"""
        if cls._instance is None:
            cls._instance = super(MatlabEngineManager, cls).__new__(cls)
        return cls._instance

    def _initialize_engine(self):
        """内部方法：负责启动 MATLAB Engine 实例"""
        if self._engine is None:
            print("正在尝试启动 MATLAB Engine...")
            try:
                # 尝试启动一个新的 MATLAB 进程
                self._engine = matlab.engine.start_matlab()
                self._is_engine_available = True
                print("MATLAB Engine 启动成功。")

                # 注册关闭函数，确保程序正常或异常退出时，Engine 都能被关闭
                atexit.register(self.stop_engine)

            except Exception as e:
                self._is_engine_available = False
                self._engine = None
                print("-" * 50)
                print("🚨 警告：无法启动 MATLAB Engine。")
                print("请确认您已完成 MATLAB Engine API for Python 的安装和配置。")
                print(f" 详细错误: {e}")
                print("-" * 50)

    def get_engine(self):
        """
        获取 MATLAB Engine 实例。如果尚未启动，则惰性启动它。

        :return: MATLAB Engine 实例，如果启动失败则返回 None。
        """
        if self._engine is None and not self._is_engine_available:
            self._initialize_engine()

        return self._engine

    def is_available(self) -> bool:
        """检查 MATLAB Engine 是否可用"""
        # 尝试惰性启动
        if self._engine is None and not self._is_engine_available:
            self._initialize_engine()

        return self._is_engine_available

    def stop_engine(self):
        """停止 MATLAB Engine (由 atexit 自动调用或手动调用)"""
        if self._engine is not None:
            print("正在关闭 MATLAB Engine...")
            try:
                # 使用 quit() 命令关闭 MATLAB 进程
                self._engine.quit()
            except Exception as e:
                # 捕获可能的在退出时发生的错误
                print(f"关闭 MATLAB Engine 时发生错误: {e}")
            finally:
                self._engine = None
                self._is_engine_available = False
                print("MATLAB Engine 已关闭。")

    def add_algorithm_path(self, path_to_m_files: str):
        """
        将包含您的 .m 算法文件的路径添加到 MATLAB 搜索路径。

        :param path_to_m_files: .m 文件所在的本地目录路径。
        """
        if self.is_available():
            # 使用 os.path.isdir 检查路径是否存在
            if not os.path.isdir(path_to_m_files):
                print(f"路径 '{path_to_m_files}' 不存在，跳过路径添加。")
                return

            # 使用 MATLAB 的 addpath 命令
            self._engine.addpath(path_to_m_files, nargout=0)
            print(f"已将路径 '{path_to_m_files}' 添加到 MATLAB 搜索路径。")
        else:
            print("MATLAB Engine 不可用，无法添加路径。")


# 实例化管理器，供外部模块导入和调用
ENGINE_MANAGER = MatlabEngineManager()