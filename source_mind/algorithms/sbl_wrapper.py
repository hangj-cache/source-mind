import numpy as np
from typing import Tuple, Dict, List
from source_mind.algorithms.sbl_algorithm import SBL_solver  # <-- 导入新的 Python 求解器


class SBLSourceLocalizer:
    # 构造函数现在非常简单
    def __init__(self, cortex_data: Dict = None, **kwargs):
        self.cortex = cortex_data
        print("SBLSourceLocalizer (Python) 初始化成功。")

    def compute_kernel(self,
                       Gain: np.ndarray,
                       L: np.ndarray,
                       B: np.ndarray,
                       Cortex: Dict,
                       InverseMethod: str,
                       Reg: int = 1,
                       **kwargs) -> Tuple[np.ndarray, Dict, List, List]:

        if B.ndim != 3 or B.shape[0] == 0:
            raise ValueError("B 必须是 (N_片段, 传感器, TFs) 的三维数组。")

        # n_segments = B.shape[0]
        n_segments = 1 if B.ndim == 2 else B.shape[0]
        n_sensor = B.shape[0] if B.ndim == 2 else B.shape[1]
        # n_sensor = B.shape[1]
        all_kernels = []
        all_s_reco = []
        par_last = {}
        NoiseCov = np.eye(n_sensor)

        print(f"-> 正在调用 Python SBL_solver，循环 {n_segments} 个片段...")

        # 循环处理 50 个变换域片段
        for i in range(n_segments):

            # 根据B的维度选择不同的取值方式
            if B.ndim == 2:
                B_i = B  # 2维时直接取整个B
            else:
                B_i = B[i, :, :].astype(np.float64)  # 非2维时取第i个切片并转float64

            try:
                # 直接调用 Python 求解器
                Kernel_i, par_last = SBL_solver(
                    B=B_i,
                    L=L,
                    epsilon=1e-4,  # 停止条件
                    flags=1,  # MacKay updatas (flags=1)
                    prune=[1, 1e-6],  # 启用剪枝，阈值 1e-6
                    Cov_n=NoiseCov,  # 传入噪声协方差矩阵
                    print_progress=1  # 打印迭代进度,
                )

                s_reco = Kernel_i @ B_i
                all_s_reco.append(s_reco)
                all_kernels.append(Kernel_i)

                print(f"-> 片段 {i + 1}/{n_segments} 计算完成。")

            except Exception as e:
                print(f"🚨 片段 {i + 1} 调用 SBL_solver 函数失败！错误: {e}")
                raise

        # 汇总结果
        if all_kernels:
            # 在最后一个维度上堆叠所有核，然后计算平均值
            Kernel_avg = np.mean(np.stack(all_kernels, axis=-1), axis=-1)
            print(f"✅ {n_segments} 个片段计算完成。平均溯源核形状: {Kernel_avg.shape}")
        else:
            Kernel_avg = np.array([])

        return Kernel_avg, par_last, all_kernels, all_s_reco