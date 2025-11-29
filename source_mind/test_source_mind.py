import os
import sys
import numpy as np
from scipy.io import savemat

# ********** 请务必修改以下路径，指向您的实际文件位置 **********
# 1. 设置包含 溯源模拟器/ 目录的父目录路径
PROJECT_ROOT = "D:/code/Python/溯源模拟器"
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# 2. 设置包含 MNE.m 文件的目录路径 (您的 matlab_scripts 目录)
MATLAB_SCRIPTS_PATH = "D:/code/Python/溯源模拟器/source_mind/matlab_scripts"

# 3. 设置您的 .mat 数据文件所在的目录
DATA_DIR = "D:/code/Python/溯源模拟器/source_mind/data"
# **********************************************************

# 4. 设置结果保存路径
save_path = r"D:\code\Python\溯源模拟器\source_mind\result"


try:
    # 导入我们封装的类和加载器
    from source_mind.algorithms.mne_wrapper import MNESourceLocalizer
    from source_mind.algorithms.sbl_wrapper import SBLSourceLocalizer
    # load_mat_matrix 位于 data_io/file_loaders.py
    from source_mind.data_io.file_loaders import load_mat_matrix

except ImportError as e:
    print("🚨 导入错误：请确认 source_mind 包结构和路径设置正确。")
    print(f"详细错误: {e}")
    sys.exit(1)


def run_mat_data_test():
    """使用本地 .mat 文件数据运行测试流程，并循环调用 MNE.m。"""

    print("=" * 60)
    print("        🧠 溯源智芯 (SourceMind) MATLAB 数据测试开始")
    print("=" * 60)
    print(f"MATLAB 脚本路径: {MATLAB_SCRIPTS_PATH}")

    # --- 1. 加载数据 ---
    try:
        print(f"正在从 {DATA_DIR} 加载数据...")

        channelselect = list(range(0, 32)) + list(range(33, 42)) + list(range(43, 64))

        # Gain: 从 model.mat 中加载 (假设变量名为 'Gain')
        model = load_mat_matrix(os.path.join(DATA_DIR, 'model.mat'), var_name='model')
        Gain_data = model['Gain'][0][0]
        selected_gain = Gain_data[channelselect, :]
        # L: 从 L.mat 中加载 (变量名 '62x6002d')
        # 注意：这里假设 load_mat_matrix 能够处理这个非标准变量名
        L_data = load_mat_matrix(os.path.join(DATA_DIR, 'L.mat'), var_name='L')


        # B_dataStorage: 从 datayu_1.mat 中加载 (变量名 'B_dataStorage', 形状: 50x62x6)
        Btrans_storage = load_mat_matrix(os.path.join(DATA_DIR, 'datayu_1.mat'), var_name='B_dataStorage')
        TBFs_storage = load_mat_matrix(os.path.join(DATA_DIR, 'datayu_1.mat'), var_name='TBFs_dataStorage')

        B_storage = np.matmul(Btrans_storage, TBFs_storage)

        # Cortex: 从 Cortex.mat 中加载 (变量名 'Cortex')
        Cortex_dict = load_mat_matrix(os.path.join(DATA_DIR, 'Cortex.mat'), var_name='Cortex')[0][0]

        # 确保数据形状与 MATLAB MNE.m 预期一致
        print(f"✅ 数据加载成功。")
        print(f"   L 矩阵形状: {L_data.shape}")
        print(f"   Gain 矩阵形状: {selected_gain.shape}")
        print(f"   B_Storage 形状 (N_片段, 传感器, TFs): {B_storage.shape}")
        print(f"   TBFs_dataStorage 形状 (N_片段, 传感器, TFs): {TBFs_storage.shape}")

    except (FileNotFoundError, KeyError, IOError, ValueError) as e:
        print(f"❌ 数据加载失败，请检查路径和变量名: {e}")
        return

    # --- 2. 初始化封装器 ---
    try:
        # 初始化时会触发 MATLAB Engine 的启动
        mne_localizer = MNESourceLocalizer(
            cortex_data=Cortex_dict  # 传递 Cortex 结构体
        )

        sbl_localizer = SBLSourceLocalizer(
            cortex_data=Cortex_dict
        )
    except RuntimeError as e:
        print(f"\n❌ 初始化失败：{e}")
        return


    ratio = 1
    all_kernels = []
    # --- 3. 运行算法 ---
    print("\n--- 测试(循环 50 个片段) ---")
    try:
    ## ========================================wMNE=========================================
        # Kernel_avg, params_last, all_kernels, all_swMNE = mne_localizer.compute_kernel(
        #     B=B_storage,
        #     Gain=selected_gain,
        #     L=L_data,
        #     Cortex=Cortex_dict,
        #     InverseMethod='wMNE',
        #     Reg=1
        # )
        # print("✅ swMNE 50 个片段调用成功！")
        # print(f"   返回平均溯源核形状: {Kernel_avg.shape}")
        # print(f"   总共计算了 {len(all_kernels)} 个溯源核。")

    ## ========================================LORETA=========================================
        # Kernel_avg, params_last, all_kernels, all_sLORETA = mne_localizer.compute_kernel(
        #     B=B_storage,
        #     Gain=selected_gain,
        #     L=L_data,
        #     Cortex=Cortex_dict,
        #     InverseMethod='LORETA',
        #     Reg=1
        # )
        # print("✅ sLORETA 50 个片段调用成功！")
        # print(f"   返回平均溯源核形状: {Kernel_avg.shape}")
        # print(f"   总共计算了 {len(all_kernels)} 个溯源核。")

    ## ========================================SBL=========================================
        Kernel_avg, params_last, all_kernels, all_sSBL = sbl_localizer.compute_kernel(
            B=B_storage,
            Gain=selected_gain,
            L=L_data,
            Cortex=Cortex_dict,
            InverseMethod='SBL',
            Reg=1
        )
        print("✅ sSBL 50 个片段调用成功！")
        print(f"   返回平均溯源核形状: {Kernel_avg.shape}")
        print(f"   总共计算了 {len(all_kernels)} 个溯源核。")



        # savemat(os.path.join(save_path,"s_reco.mat"), {"sloreta": all_sLORETA, "swmne": all_swMNE, "ssbl": all_sSBL})


    except Exception as e:
        print(f"❌ 调用失败，请检查代码和数据格式。详细错误已在上方打印。")
        # 由于错误已在 compute_kernel 内部打印，这里不再重复打印详细 e

    print("\n" + "=" * 60)


if __name__ == "__main__":
    # 强制将 test_source_mind.py 所在的目录也加入系统路径，以便导入
    # sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    run_mat_data_test()