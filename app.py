import os
os.environ["USE_SHM"] = "0"
import io
import tempfile
import numpy as np
import scipy.io as sio
from typing import Tuple, Dict
from flask import Flask, request, jsonify, send_file, after_this_request
# 必须导入 CORS，因为前端 index.html 是本地文件，与服务器端口不同，会产生跨域问题
import flask_cors
from flask_cors import CORS
import atexit
import shutil # 导入 shutil 用于更强大的目录删除
from os.path import join
import torch

# 从已提供的算法文件中导入求解器
# 假设文件结构为: app.py 和 source_mind/ 文件夹在同一目录下
try:
    # 导入 MNE/LORETA 求解器。请确保 source_mind/algorithms/mne_algorithm.py 存在
    from source_mind.algorithms.mne_algorithm import MNE_solver
    # 导入 SBL 求解器。请确保 source_mind/algorithms/sbl_algorithm.py 存在
    from source_mind.algorithms.sbl_algorithm import SBL_solver
    from source_mind.algorithms.ADMM_Network import ESINetADMMLayer

except ImportError as e:
    print(f"🚨 导入算法文件失败，请检查文件结构: {e}")
    print("如果您尚未创建算法文件，请在 source_mind/algorithms/ 下创建它们，并定义 MNE_solver 和 SBL_solver 函数。")


# =================================================================
# === Flask 应用设置 ===
# =================================================================

app = Flask(__name__)
# 允许前端跨域调用 API，默认端口 5000
CORS(app)

# 临时目录用于保存上传的文件
TEMP_DIR = tempfile.mkdtemp()
# atexit.register(lambda: os.path.isdir(TEMP_DIR) and os.removedirs(TEMP_DIR))
atexit.register(lambda: os.path.isdir(TEMP_DIR) and shutil.rmtree(TEMP_DIR))

# RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "result")
RESULT_DIR = os.path.join("source_mind/", "result")
os.makedirs(RESULT_DIR, exist_ok=True)


@app.route('/run_localization', methods=['POST'])
def run_localization():
    """
    API 接口：接收文件和算法选择，运行源定位算法，并返回 .mat 文件。
    """
    # 文件路径字典，用于存储上传文件的临时路径
    temp_file_paths = {}
    try:
        # 1. 获取算法名称和文件
        algorithm = request.form.get('algorithm')

        file_model = request.files.get('model_file')
        file_b_storage = request.files.get('data_file')
        file_cortex = request.files.get('cortex_file')
        file_l = request.files.get('l_file')

        if not algorithm or not file_model or not file_b_storage or not file_cortex or not file_l:
            return jsonify({"error": "缺少必需的参数 (算法名称或 Gain/B_storage/cortex/l 文件)"}), 400

        # 定义临时路径
        temp_file_paths['model'] = os.path.join(TEMP_DIR, file_model.filename or 'model.mat')
        temp_file_paths['data'] = os.path.join(TEMP_DIR, file_b_storage.filename or 'data.mat')
        temp_file_paths['cortex'] = os.path.join(TEMP_DIR, file_cortex.filename or 'cortex.mat')
        temp_file_paths['l'] = os.path.join(TEMP_DIR, file_l.filename or 'l.mat')

        # 保存文件
        file_model.save(temp_file_paths['model'])
        file_b_storage.save(temp_file_paths['data'])
        file_cortex.save(temp_file_paths['cortex'])
        file_l.save(temp_file_paths['l'])

        print(f"✔ 所有文件已保存到临时路径: {TEMP_DIR}")



        channelselect = list(range(0, 32)) + list(range(33, 42)) + list(range(43, 64))
        # -------- 1️⃣ Gain 从 model_file 读取 --------
        model_data = sio.loadmat(temp_file_paths['model'])
        if "model" not in model_data:
            return jsonify({"error": "model_file 缺少变量 'model'"}), 400
        model_struct = model_data.get('model')
        if model_struct is not None and model_struct.shape == (1, 1) and 'Gain' in model_struct[0][0].dtype.fields:
            Gain_data = model_struct[0][0]['Gain']
        else:
            # 如果不是结构体，则尝试直接从顶层变量 'Gain' 提取
            Gain_data = model_data.get('Gain')

        selected_gain = Gain_data[channelselect, :]
        print(f"✔ Gain 加载成功: {selected_gain.shape}")

        # -------- 2️⃣ B_dataStorage + TBFs_dataStorage 从 data_file 读取 --------
        data_dict = sio.loadmat(temp_file_paths['data'])

        # if "B_dataStorage" not in data_dict or "TBFs_dataStorage" not in data_dict:
        #     return jsonify({"error": "data_file 缺少 B_dataStorage 或 TBFs_dataStorage"}), 400

        if "B" not in data_dict or "TBFs" not in data_dict:
            return jsonify({"error": "data_file 缺少 B 或 TBFs"}), 400

        B_storage = data_dict["B"]
        TBFs_storage = data_dict["TBFs"]
        print(f"✔ 数据加载成功: B={B_storage.shape} TBFs={TBFs_storage.shape}")

        # 批量矩阵乘法 → 得到 50x62x300
        # B_storage = np.matmul(B_storage, TBFs_storage)
        print(f"✔ B_storage 计算完成: {B_storage.shape}")

        # -------- 3️⃣ L 矩阵从 l_file 获取 --------
        L_dict = sio.loadmat(temp_file_paths['l'])
        L_data = next((v for k, v in L_dict.items() if isinstance(v, np.ndarray) and not k.startswith('__')), None)
        if L_data is None:
            return jsonify({"error": "l_file 无法识别主要的 L 矩阵变量"}), 400

        print(f"✔ L 加载成功: {L_data.shape}")

        # -------- 4️⃣ Cortex 信息读取 --------
        Cortex_dict = sio.loadmat(temp_file_paths['cortex'])
        Cortex_data = Cortex_dict.get('Cortex')
        # 尝试解包 MATLAB 结构体
        if Cortex_data is not None and Cortex_data.ndim >= 2 and Cortex_data.shape[0] > 0 and Cortex_data.shape[1] > 0:
            Cortex_dict_unpacked = Cortex_data[0][0]
        else:
            Cortex_dict_unpacked = Cortex_data  # 尝试直接使用顶层变量

        if Cortex_dict_unpacked is None:
            return jsonify({"error": "cortex_file 无法提取 Cortex 结构体"}), 400


        if selected_gain is None or B_storage is None or Cortex_dict is None or L_data is None:
            return jsonify({"error": "无法从上传文件中解析出 Gain 或 B_storage或 L 矩阵或Cortex。请检查变量名是否正确。"}), 400

        Gain = selected_gain

        # 确保 B_storage 是三维 (N_片段 x nSensor x nSnap)
        if B_storage.ndim == 2:
            B_storage = B_storage[np.newaxis, :, :]

        n_segments = B_storage.shape[0]
        n_sensor = B_storage.shape[1]

        # 3. 初始化结果存储字典
        results_data = {}
        ratio = 1

        all_s_reco = []
        all_kernels = []

        # 4. 算法选择和执行

        if algorithm == 'sbl':
            print(f"--- 正在运行 SBL (Sparse Bayesian Learning)，共 {n_segments} 个片段 ---")

            for i in range(n_segments):
                B_i = B_storage[i, :, :].astype(np.float64)

                # 调用 SBL_solver
                Kernel_sbl, par_sbl = SBL_solver(
                    B=B_i,
                    L=L_data,
                    epsilon=1e-4,  # 停止条件
                    flags=1,
                    prune=[1, 1e-6],
                    Cov_n=np.eye(n_sensor),
                    print_progress=1
                )
                print(f"B_i的形状：{B_i.shape}")
                print(f"核的形状：{Kernel_sbl.shape}")
                all_kernels.append(Kernel_sbl)
                S_SBL = Kernel_sbl @ B_i * ratio
                all_s_reco.append(S_SBL)


            results_data[f'S_{algorithm.upper()}'] = all_s_reco

        elif algorithm in ['wmne', 'sloreta']:
            print(f"--- 正在运行 MNE/LORETA 算法 ({algorithm})，共 {n_segments} 个片段 ---")


            for i in range(n_segments):
                B_i = B_storage[i, :, :]

                # 调用 MNE_solver
                Kernel_i, params_i = MNE_solver(
                    B=B_i,
                    Gain=Gain,
                    L=L_data,  # 假设 L_whitened = Gain
                    Cortex=Cortex_dict,
                    InverseMethod=algorithm,
                    Reg=1
                )
                all_kernels.append(Kernel_i)
                S_SBL = Kernel_i @ B_i * ratio
                all_s_reco.append(S_SBL)

            results_data[f'S_{algorithm.upper()}'] = all_s_reco

        elif algorithm == 'duvl1n':
            print(f"--- 正在运行 DUVL1N 算法 ({algorithm})，共 {n_segments} 个片段 ---")
            # === 模型路径处理 ===
            model_dir = os.path.join(os.path.dirname(__file__), "source_mind\log_duvl1n")
            model_filename = "0.0037918177-DUV-lam0.00001rho600000-L1N-2d-2d-600-0.001-_model_20251124_191904_78.pth"
            model_path = os.path.join(model_dir, model_filename)

            if not os.path.exists(model_path):
                return jsonify({"error": f"DUVL1N 模型文件未找到: {model_path}"}), 500
            L_tensor = torch.from_numpy(L_data.astype(np.float32))
            # === 加载模型 ===
            try:
                model = ESINetADMMLayer(L_tensor)
                params_load = torch.load(model_path, map_location='cpu')  # 兼容无 GPU 环境
                model.load_state_dict(params_load)
                model.eval()
            except Exception as e:
                return jsonify({"error": f"加载 DUVL1N 模型失败: {str(e)}"}), 500

            # === 验证 TBFs 维度 ===
            if TBFs_storage.ndim != 2:
                return jsonify({"error": "TBFs 必须是二维矩阵 (K x nSnap)"}), 400
            n_snap = B_storage.shape[2]
            if TBFs_storage.shape[1] != n_snap:
                return jsonify({
                    "error": f"TBFs 时间维度 ({TBFs_storage.shape[1]}) 与 B 数据 ({n_snap}) 不匹配"
                }), 400

            with torch.no_grad():
                for i in range(n_segments):
                    B_i = B_storage[i, :, :]  # shape: (n_sensor, n_snap)

                    # 转换为 PyTorch 张量
                    B_i_tensor = torch.from_numpy(B_i.astype(np.float32))  # (62, 300)
                    TBFs_tensor = torch.from_numpy(TBFs_storage.astype(np.float32))  # (K, 300)

                    # 计算 B_trans = B_i @ TBFs^T → (62, K)
                    B_trans = torch.matmul(B_i_tensor, TBFs_tensor.t())  # 注意：.t() 是 PyTorch 的转置

                    # 构建输入字典
                    x = {'B_trans': B_trans.unsqueeze(0)}  # 添加 batch 维度 → (1, 62, K)

                    # 前向传播
                    s_gen_trans = model(x)  # 输出 shape: (1, n_dipole, K)

                    # 去除 batch 维度
                    s_gen_temp = s_gen_trans.squeeze(0)  # (n_dipole, K)

                    # 重构时间序列: S = s_gen_temp @ TBFs → (n_dipole, n_snap)
                    s_gen = torch.matmul(s_gen_temp, TBFs_tensor)  # (n_dipole, 300)

                    # 转为 NumPy 并保存
                    S_L1N = s_gen.cpu().numpy().astype(np.float64)
                    all_s_reco.append(S_L1N)

            results_data[f'S_{algorithm.upper()}'] = all_s_reco

        else:
            return jsonify({"error": f"不支持的算法: {algorithm}"}), 400

        print(f"✅ 算法 {algorithm} 计算完成。")


        # filename = f"{algorithm}_source_results.mat"
        # save_path = os.path.join(RESULT_DIR, filename)
        #
        # # 保存 MATLAB 数据文件
        # sio.savemat(save_path, results_data)
        # 6. ⭐ 关键步骤：将结果写入内存缓冲区并发送给客户端
        output = io.BytesIO()
        sio.savemat(output, results_data)
        output.seek(0)  # 将指针重置到文件开头

        filename = f"{algorithm}_source_results.mat"

        # 使用 send_file 从内存缓冲区直接下载文件
        response = send_file(
            output,
            as_attachment=True,
            download_name=filename,
            mimetype="application/x-matlab"
        )

        return response
    except Exception as e:
        # 打印详细错误到服务器控制台
        import traceback
        traceback.print_exc()
        print(f"🚨 致命错误: {e}")
        # 返回 500 错误码和详细错误信息给前端
        return jsonify({"error": f"服务器内部错误，请检查控制台输出。详细信息: {str(e)}"}), 500

    finally:
        # -------------------------------------------------------------
        # ✅ 清理机制：使用 finally 块确保上传的临时输入文件被删除
        # -------------------------------------------------------------
        for key, path in temp_file_paths.items():
            if os.path.exists(path):
                try:
                    os.remove(path)
                    print(f"✔ 请求结束，已清理临时输入文件: {path}")
                except OSError as e:
                    print(f"⚠️ 无法删除临时输入文件 {path}: {e}")


# -------------------------------------------------------------
# 启动服务器
if __name__ == '__main__':
    print("=" * 40)
    print("--- 启动 Flask 服务器 ---")
    print("请访问 http://127.0.0.1:5000/ 运行前端。")
    print("=" * 40)
    app.run(host='0.0.0.0', port=5000, debug=True)