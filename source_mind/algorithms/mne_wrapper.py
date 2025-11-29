
import numpy as np
from typing import Tuple, Dict, List
from source_mind.algorithms.mne_algorithm import MNE_solver  # <-- 导入新的 Python 求解器


class MNESourceLocalizer:
    # 构造函数现在非常简单
    def __init__(self, cortex_data: Dict = None, **kwargs):
        self.cortex = cortex_data
        print("MNESourceLocalizer (Python) 初始化成功。")

    def compute_kernel(self,
                       Gain: np.ndarray,
                       L: np.ndarray,
                       B: np.ndarray,
                       Cortex: Dict,
                       InverseMethod: str,
                       Reg: int = 1,
                       **kwargs) -> Tuple[np.ndarray, Dict, List, List]:

        # n_segments = B.shape[0]
        n_segments = 1 if B.ndim == 2 else B.shape[0]
        all_kernels = []
        all_s_reco = []
        par_last = {}

        print(f"-> 正在调用 Python MNE_solver，循环 {n_segments} 个片段...")

        # 循环处理 50 个变换域片段
        for i in range(n_segments):

            # 根据B的维度选择不同的取值方式
            if B.ndim == 2:
                B_i = B  # 2维时直接取整个B
            else:
                B_i = B[i, :, :].astype(np.float64)  # 非2维时取第i个切片并转float64

            try:
                # 直接调用 Python 求解器
                Kernel_i, par_last = MNE_solver(
                    B=B_i,
                    Gain=Gain,
                    L=L,
                    Cortex=Cortex,
                    InverseMethod=InverseMethod,
                    Reg=Reg,
                    **kwargs
                )

                s_reco = Kernel_i @ B_i
                all_s_reco.append(s_reco)
                all_kernels.append(Kernel_i)

                print(f"-> 片段 {i + 1}/{n_segments} 计算完成。")

            except Exception as e:
                print(f"🚨 片段 {i + 1} 调用 MNE_solver 函数失败！错误: {e}")
                raise

        # 汇总结果
        if all_kernels:
            # 在最后一个维度上堆叠所有核，然后计算平均值
            Kernel_avg = np.mean(np.stack(all_kernels, axis=-1), axis=-1)
            print(f"✅ {n_segments} 个片段计算完成。平均溯源核形状: {Kernel_avg.shape}")
        else:
            Kernel_avg = np.array([])

        return Kernel_avg, par_last, all_kernels, all_s_reco