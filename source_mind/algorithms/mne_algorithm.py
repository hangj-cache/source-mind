import numpy as np
import scipy.linalg as la
from scipy.sparse import identity, diags, csr_matrix, spdiags, coo_matrix
from typing import Tuple, Dict, List, Union


# ===================================================================
# === 辅助函数：tess_vertconn (皮层网格连接) ===
# ===================================================================

def tess_vertconn(Vertices: np.ndarray, Faces: np.ndarray) -> csr_matrix:
    """
    TESS_VERTCONN: 计算顶点连接矩阵 (VertConn).

    基于 Brainstorm 的 MATLAB 逻辑实现。

    :param Vertices: Mx3 矩阵 (顶点坐标)
    :param Faces: Nx3 矩阵 (面片/三角形，存储顶点索引)
    :return: VertConn: 稀疏连接矩阵 (nVertices x nVertices), 1 表示连接。
    """
    # Check matrices orientation
    if Vertices.shape[1] != 3 or Faces.shape[1] != 3:
        raise ValueError('Faces and Vertices must have 3 columns (X,Y,Z).')

    n = Vertices.shape[0]

    # MATLAB 逻辑：将所有面片边的连接关系展开成 rowno 和 colno 向量
    # 对于每个面片 (v1, v2, v3):
    # (v1, v2), (v1, v3)
    # (v2, v1), (v2, v3)
    # (v3, v1), (v3, v2)

    # 转换为 0-based 索引 (MATLAB 是 1-based)
    Faces_zero_based = Faces - 1

    # 1. 构造行索引 (rowno)
    # Faces_zero_based[:, 0] 重复两次 (v1 -> v2, v1 -> v3)
    # Faces_zero_based[:, 1] 重复两次 (v2 -> v1, v2 -> v3)
    # Faces_zero_based[:, 2] 重复两次 (v3 -> v1, v3 -> v2)
    rowno = np.concatenate([
        Faces_zero_based[:, 0], Faces_zero_based[:, 0],
        Faces_zero_based[:, 1], Faces_zero_based[:, 1],
        Faces_zero_based[:, 2], Faces_zero_based[:, 2]
    ])

    # 2. 构造列索引 (colno)
    colno = np.concatenate([
        Faces_zero_based[:, 1], Faces_zero_based[:, 2],
        Faces_zero_based[:, 0], Faces_zero_based[:, 2],
        Faces_zero_based[:, 0], Faces_zero_based[:, 1]
    ])

    # 3. 构造数据 (data)，所有连接权重为 1
    data = np.ones_like(rowno, dtype=np.int8)

    # 4. 构建 COOrdinate 稀疏矩阵
    VertConn_coo = coo_matrix((data, (rowno, colno)), shape=(n, n))

    # 5. 转换为 CSR 格式，并确保是逻辑连接 (即合并重复连接，只保留非零值)
    # MATLAB 的 logical(sparse(...)) 在 Python 中等价于将 COOP 转换为 CSR/CSC
    # 并使用 .astype(bool) 或 .astype(int) 确保数值是 1/0
    VertConn = VertConn_coo.tocsr()
    # 确保连接值是布尔值（1或0），并返回 CSR 格式
    VertConn[VertConn > 0] = 1

    return VertConn


# ===================================================================
# === 辅助函数：bst_bsxfun_rdivide (NumPy 广播实现) ===
# ===================================================================

def bst_bsxfun_rdivide(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """ 模拟 MATLAB 的 bst_bsxfun(@rdivide, Kernel, diag) """
    # NumPy 的广播机制自动处理列向量 B 和矩阵 A 的逐元素除法
    return A / B


# ===================================================================
# === MNE 主求解器 (MNE_solver) ===
# ===================================================================

def MNE_solver(
        B: np.ndarray,
        Gain: np.ndarray,
        L: np.ndarray,
        Cortex: Dict,
        InverseMethod: str,
        Reg: int = 1,
        pQ: Union[List, None] = None
) -> Tuple[np.ndarray, Dict]:
    """
    基于 L2 范数的 MNE 源定位算法实现 (Python/NumPy 版本).

    :param B: 白化后的 M/EEG 数据 (nSensor x nSnap)
    # ... (参数和文档不变)
    """

    # 将所有输入转换为 float64
    B = B.astype(np.float64)
    Gain = Gain.astype(np.float64)
    L = L.astype(np.float64)

    # ===== DEFINE DEFAULT OPTIONS =====
    nSensor, nSource = L.shape
    nSnap = B.shape[1]
    SNR = 3
    OPTIONS = {
        'depth': 1,
        'weightlimit': 10,
        'weightexp': 0.5,
        'fMRI_prior': None,
    }

    # ... (fMRI 先验处理逻辑不变) ...
    if pQ is not None and len(pQ) > 0:
        supra_threshold = set()
        for q_list in pQ:
            # 假设 pQ 是列表的列表或包含 ndarray 的列表
            if isinstance(q_list, list):
                for q in q_list:
                    if isinstance(q, np.ndarray):
                        supra_threshold.update(np.where(q != 0)[0])
            elif isinstance(q_list, np.ndarray):
                supra_threshold.update(np.where(q_list != 0)[0])

        OPTIONS['fMRI_prior'] = 0.1 * np.ones(nSource)
        if supra_threshold:
            indices = np.array(list(supra_threshold))
            OPTIONS['fMRI_prior'][indices] = 1

    # 默认权重 (w)
    w = np.ones(nSource)

    # ===== Depth compensation & wMNE/LORETA weighting =====
    method_lower = InverseMethod.lower()

    if method_lower == 'wmne':
        if Gain.shape[1] == nSource:
            # Gain 是单方向 LFM
            if OPTIONS['fMRI_prior'] is None:
                w = np.ones(nSource)
            else:
                w = OPTIONS['fMRI_prior']
        else:
            # Gain 是三方向 LFM (nSensor x 3*nSource)
            if OPTIONS['fMRI_prior'] is None:
                G_reshaped = Gain.reshape(nSensor, 3, nSource)
                # 计算每个源点的范数平方和，并取倒数
                w_sq = np.sum(np.sum(G_reshaped ** 2, axis=0), axis=0)
                w = 1.0 / w_sq
            else:
                w = OPTIONS['fMRI_prior']

            # ===== APPLY DEPTH WEIGHTHING =====
            if OPTIONS['depth']:
                # APPLY WEIGHT LIMIT
                print('Applying weight limit.')
                weightlimit2 = OPTIONS['weightlimit'] ** 2
                limit = np.min(w) * weightlimit2
                w[w > limit] = limit

                # APPLY WEIGHT EXPONENT
                print('Applying weight exponent.')
                w = w ** OPTIONS['weightexp']

    # ===== ADJUSTING SOURCE COVARIANCE MATRIX (C_J) =====
    print(f'Adjusting source covariance matrix for {InverseMethod}...')

    # 1. 构造 C_J 矩阵
    # 初始 C_J 为对角矩阵，对角线元素为 w
    C_J = diags(w, 0, shape=(nSource, nSource), format='csc')

    # 2. LORETA 特殊处理
    if method_lower == 'loreta':
        vertconn_needed = True
        VertConn = None

        # 尝试安全地从 Cortex 中获取 VertConn
        try:
            VertConn = Cortex['VertConn']
        except (KeyError, TypeError, AttributeError):
            # 字段不存在或 Cortex 不是可索引的对象
            pass

        if VertConn is not None:
            # 检查 VertConn 是否为稀疏矩阵且非空
            if hasattr(VertConn, 'nnz') and VertConn.nnz > 0:
                vertconn_needed = False
            # 检查 VertConn 是否为稠密数组且非空
            elif hasattr(VertConn, 'size') and VertConn.size > 0:
                vertconn_needed = False

        if vertconn_needed:
            print('LORETA: Computing Vertices Connectivity (VertConn)...')
            # Vertices 和 Faces 必须在 Cortex 字典中
            # 假设 Cortex['Vertices'] 和 Cortex['Faces'] 已被正确解包成 2D 数组
            # 尝试通过 try-except 再次安全获取 Vertices/Faces，以防 Cortex 仍有问题
            try:
                Vertices = Cortex['Vertices']
                Faces = Cortex['Faces']
            except (KeyError, TypeError, AttributeError) as e:
                raise ValueError(f"Failed to access Vertices or Faces from Cortex after unwrap: {e}")

            VertConn = tess_vertconn(Vertices, Faces)
            Cortex['VertConn'] = VertConn
        else:
            # VertConn 已经存在且非空，使用它
            # 注意：VertConn 在 try-except 中已经被赋值
            pass  # VertConn 已是正确的值

        # M = I - D_inv * Adj
        VertConn_sparse = VertConn.tocsr()
        degrees = np.array(VertConn_sparse.sum(axis=1)).flatten()

        # 构建 M 矩阵
        # 处理度为零的点：避免除以零
        degrees[degrees == 0] = 1
        A_inv = 1.0 / degrees

        # M = I - D_inv * Adj
        M = identity(nSource, format='csc') - diags(A_inv, 0) @ VertConn_sparse

        # Employ Depth Compensation for LORETA (MATLAB代码中的w是ones)
        M = diags(w, 0, format='csc') @ M

        # C_J = (M'*M + 0.0001*trace(M'*M)/nSource*I)^-1
        MtM = M.T @ M
        trace_MtM = np.trace(MtM.todense())  # 稀疏矩阵求迹
        C_J_inv_sparse = MtM + 0.0001 * (trace_MtM / nSource) * identity(nSource, format='csc')

        # 求逆
        C_J_dense = la.inv(C_J_inv_sparse.todense())
        C_J = csr_matrix(C_J_dense)  # 转换回稀疏矩阵，以保持代码一致性

    # 3. 归一化 (适用于所有方法)
    if 'dot' in dir(C_J):  # 稀疏矩阵
        L_C_J = L @ C_J
        trclcl = np.trace(L_C_J @ L.T)
    else:  # 稠密矩阵
        trclcl = np.trace(L @ C_J @ L.T)

    C_J = C_J * (nSensor / trclcl)

    # 4. Cholesky 分解 (Rc) 和加权导联矩阵 (LW)
    if 'todense' in dir(C_J):
        C_J_dense = C_J.todense()
    else:
        C_J_dense = C_J

    # Cholesky 分解
    try:
        Rc = la.cholesky(C_J_dense, lower=True)
    except la.LinAlgError:
        print("Warning: C_J is not positive definite. Adding regularization.")
        epsilon = 1e-6
        C_J_dense += epsilon * np.eye(nSource)
        Rc = la.cholesky(C_J_dense, lower=True)

    LW = L @ Rc

    # ===== Estimate the Regularization Parameter via Data-Driven Process =====
    par = {}
    if Reg:
        # ... (证据最大化逻辑不变)
        MAX_iter = 100
        gamma = 1.0
        cost_old = 0

        for iter_num in range(1, MAX_iter + 1):
            Cov_B = np.eye(nSensor) + gamma * (LW @ LW.T)
            Inv_Cov_B = la.inv(Cov_B)

            term1 = gamma * LW.T @ Inv_Cov_B
            norm_term = la.norm(term1 @ B, 'fro') ** 2
            trace_term = np.trace(term1 @ LW)

            # 防止除以零
            if np.abs(trace_term) < 1e-12:
                print("Warning: Trace term near zero, exiting regularization.")
                break

            gamma = norm_term / (nSnap * trace_term)

            # 计算 cost
            log_det_Cov_B = np.log(la.det(Cov_B))
            trace_term2 = np.trace(B @ B.T @ Inv_Cov_B)
            cost = -(nSnap * log_det_Cov_B + trace_term2 + nSnap * nSensor * np.log(2 * np.pi)) / 2

            MSE = (cost - cost_old) / cost
            cost_old = cost

            if np.abs(MSE) < 1e-5:
                break

        lambda2 = 1.0 / gamma
    else:
        # 固定正则化
        lambda2 = SNR ** (-2)
        Cov_B = np.eye(nSensor) + (LW @ LW.T) / lambda2

        # 计算 cost (如果需要)
        try:
            Inv_Cov_B = la.inv(Cov_B)
            log_det_Cov_B = np.log(la.det(Cov_B))
            trace_term2 = np.trace(B @ B.T @ Inv_Cov_B)
            cost = -(nSnap * log_det_Cov_B + trace_term2 + nSnap * nSensor * np.log(2 * np.pi)) / 2
        except la.LinAlgError:
            cost = np.nan  # 矩阵不可逆
            print("Warning: Cov_B is singular during fixed regularization cost calculation.")

    par['Cost'] = cost

    # ===== Compute SVD and Kernel =====
    print('Computing SVD of whitened and weighted lead field matrix.')
    U, s_diag, Vh = la.svd(LW, full_matrices=False)
    V = Vh.T

    s_sq = s_diag ** 2
    ss = s_diag / (s_sq + lambda2)

    # Kernel = Rc * V * diag(ss) * U';
    Kernel = Rc @ V @ np.diag(ss) @ U.T

    # ===== Post-processing (dSPM / sLORETA) =====
    if method_lower == 'dspm':
        print('Computing dSPM inverse operator.')
        # dspmdiag 是一个列向量 (N_source, 1)
        dspmdiag = np.sqrt(np.sum(Kernel ** 2, axis=1, keepdims=True))
        Kernel = bst_bsxfun_rdivide(Kernel, dspmdiag)

    elif method_lower == 'sloreta':
        print('Computing sLORETA inverse operator.')
        # sloretadiag 是一个列向量 (N_source, 1)
        sloretadiag = np.sqrt(np.sum(Kernel * L.T, axis=1, keepdims=True))
        Kernel = bst_bsxfun_rdivide(Kernel, sloretadiag)

    return Kernel, par