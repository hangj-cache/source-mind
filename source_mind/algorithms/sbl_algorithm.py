import numpy as np
import scipy.linalg as la
from typing import Tuple, Dict, List, Union


def SBL_solver(
        B: np.ndarray,
        L: np.ndarray,
        flags: Union[int, List[int]] = 2,
        epsilon: float = 1e-6,
        max_iter: int = 1000,
        prune: List[Union[int, float]] = [0, 1e-6],
        NLFM: int = 0,
        beta: float = 1.0,
        Cov_n: Union[np.ndarray, None] = None,
        print_progress: int = 1
) -> Tuple[np.ndarray, Dict]:
    """
    SBL (Sparse Bayesian Learning) 求解器.

    根据 David Wipf 和 Srikantan Nagarajan 的统一贝叶斯框架实现。
    用于估计源电流 S，其中模型为 B = L*s + noise。

    :param B: 观测数据 (nSensor x nSnap)
    :param L: 导联场矩阵 (nSensor x nSource)
    :param flags: 迭代更新算法 (0: EM, 1: MacKay, 2: Convexity-based, [3, p]: SMAP/MFOCUSS)
    :param epsilon: 停止迭代的阈值 (MSE)
    :param max_iter: 最大迭代次数
    :param prune: [是否剪枝 (0/1), 剪枝阈值 (e.g., 1e-6)]
    :param NLFM: 是否归一化导联场矩阵 (0/1)
    :param beta: 传感器噪声方差的逆 (precision)，即 1/sigma^2
    :param Cov_n: 传感器噪声协方差矩阵 (如果为 None, 则默认为单位矩阵)
    :param print_progress: 是否打印迭代进度 (0/1)
    :return: (ImagingKernel, par)
        - ImagingKernel: 反演算子 (nSource x nSensor)
        - par: 包含 gamma, evidence, beta, keeplist 的字典
    """

    # 强制类型转换
    B = B.astype(np.float64)
    L = L.astype(np.float64)

    # 维度
    nSensor, nSnap = B.shape
    nSource = L.shape[1]

    # 处理 flags 列表
    if not isinstance(flags, (list, np.ndarray)):
        flags = [flags]

    # 默认噪声协方差
    if Cov_n is None:
        Cov_n = np.eye(nSensor, dtype=np.float64)
    else:
        Cov_n = Cov_n.astype(np.float64)

    if print_progress:
        print('\nRunning Sparse Bayesian Learning...')

    # --------------- Initial states --------------
    # gamma 初始值: (nSource,) 向量
    trace_L_LT = np.trace(L @ L.T)
    gamma = (np.ones(nSource) + np.random.randn(nSource) * 1e-4) * (nSensor / trace_L_LT)

    cost_old = 1.0
    # keep_list 存储原始索引 (0-based)
    keep_list = np.arange(nSource)
    evidence = np.zeros(max_iter)

    # =======================================
    # Leadfield Matrix normalization (NLFM)
    # =======================================
    if NLFM:
        # LfvW = (mean(L.^2,1)).^0.5; L = L.*kron(ones(nSensor,1),1./LfvW);
        LfvW = np.sqrt(np.mean(L ** 2, axis=0))
        # 避免除以零或无穷大
        LfvW[LfvW == 0] = np.min(LfvW[LfvW != 0])
        # 使用广播机制进行列归一化
        L = L / LfvW[np.newaxis, :]

    # =========================================================================
    #                                  Iteration
    # =========================================================================
    for iter_num in range(max_iter):

        # ******** Prune things as hyperparameters go to zero **********
        if prune[0]:
            if not keep_list.size:
                break

            threshold = np.max(gamma) * prune[1]
            # 找到需要保留的 gamma 索引 (在当前 gamma 向量中的索引)
            index_to_keep = np.where(gamma > threshold)[0]

            # 剪枝操作
            gamma = gamma[index_to_keep]
            L = L[:, index_to_keep]
            keep_list = keep_list[index_to_keep]

            if not keep_list.size:
                if print_progress:
                    print(f"SBLiters: {iter_num + 1}, Pruned all sources. Stopping.")
                break

        Ns = len(keep_list)

        # 1. 计算 Cov_b = beta*Cov_n + L*Gamma*L'
        # L_Gamma: L 的列由 gamma 缩放 (nSensor x Ns)
        L_Gamma = L * gamma[np.newaxis, :]
        # L*Gamma*L' (nSensor x nSensor)
        L_Gamma_LT = L_Gamma @ L.T

        Cov_b = beta * Cov_n + L_Gamma_LT

        # 预先计算 Cov_b 的逆
        try:
            Inv_Cov_b = la.inv(Cov_b)
        except la.LinAlgError:
            if print_progress:
                print(f"SBLiters: {iter_num + 1}, Cov_b is singular. Stopping.")
            break

        # ========== Update hyperparameters ===========

        # L_T_Inv_Cov_b: L.T @ Inv_Cov_b (Ns x nSensor)
        L_T_Inv_Cov_b = L.T @ Inv_Cov_b

        # 辅助计算: L_T_Inv_Cov_b @ B (Ns x nSnap)
        L_T_Inv_Cov_b_B = L_T_Inv_Cov_b @ B

        # 迭代更新每个 gamma_k
        for k in range(Ns):
            l_k = L[:, k]  # 当前源点的导联场 (nSensor, 1)
            gamma_k = gamma[k]  # 当前 gamma 值

            # M = gamma(k)*L(:,k)'/Cov_b;
            # M_k = gamma_k * l_k.T @ Inv_Cov_b (1 x nSensor)
            M_k = gamma_k * L_T_Inv_Cov_b[k, :]  # 已经包含了 gamma_k

            # 计算 Trace (M_k * l_k)
            # M_k * l_k 是标量: l_k.T @ (M_k.T) * gamma_k^{-1}
            trace_term = np.dot(M_k, l_k)

            # 计算 Norm (M_k * B, 'fro')^2
            # M_k @ B (1 x nSnap)
            # norm_term_sq = la.norm(M_k @ B, 'fro') ** 2
            norm_term_sq = la.norm(M_k @ B, 2) ** 2
            if flags[0] == 0:
                # ------------- EM algorithm ---------------
                # gamma(k) = (norm(M*B,'fro'))^2/nSnap + gamma(k) - trace(M*L(:,k)*gamma(k));
                gamma_new = norm_term_sq / nSnap + gamma_k - trace_term
            elif flags[0] == 1:
                # ----------- MacKay updatas --------------
                # gamma(k) = (norm(M*B,'fro'))^2 / (nSnap * trace(M*L(:,k)));
                if trace_term != 0:
                    gamma_new = norm_term_sq / (nSnap * trace_term)
                else:
                    gamma_new = gamma_k  # 避免除以零
            elif flags[0] == 2:
                # -------- Convexity_based approach --------
                # |l_k.T @ Inv_Cov_b @ B|_F
                norm_lT_Inv_CovB_B = la.norm(L_T_Inv_Cov_b[k, :] @ B, 'fro')

                # trace((l_k.T @ Inv_Cov_b) @ l_k)
                trace_lT_Inv_CovB_l = np.dot(L_T_Inv_Cov_b[k, :], l_k)

                if trace_lT_Inv_CovB_l > 0 and nSnap > 0:
                    gamma_new = gamma_k * norm_lT_Inv_CovB_B / np.sqrt(nSnap * trace_lT_Inv_CovB_l)
                else:
                    gamma_new = gamma_k  # 避免除以零或负数
            elif flags[0] == 3:
                # ------------ S-MAP (MFOCUSS) ----------------
                p = flags[1] if len(flags) > 1 else 1.0  # 默认 p=1
                # gamma(k) = ((norm(M*B,'fro'))^2/nSnap)^((2-p)/2);
                term = norm_term_sq / nSnap
                gamma_new = term ** ((2 - p) / 2)
            else:
                raise ValueError(f"Unsupported SBL flags value: {flags[0]}")

            # 防止 gamma 变为负数 (SBL 假设 gamma >= 0)
            gamma[k] = max(1e-12, gamma_new)

            # 计算 Cost/Evidence (log marginal likelihood)
        # cost = trace(B*B'/Cov_b)+nSnap*log(det(Cov_b))+nSnap*nSensor*log(2*pi);
        # cost = -0.5*cost;

        # 1. trace(B*B'/Cov_b) = trace(B.T @ Inv_Cov_b @ B) = norm(Inv_Cov_b^1/2 @ B, 'fro')^2
        trace_B_Inv_CovB = np.trace(B.T @ Inv_Cov_b @ B)

        # 2. log(det(Cov_b))
        try:
            # _, log_det_Cov_b = la.slogdet(Cov_b)
            _, log_det_Cov_b = np.linalg.slogdet(Cov_b)
        except la.LinAlgError:
            if print_progress:
                print(f"SBLiters: {iter_num + 1}, Determinant calculation failed. Stopping.")
            break

        # 总证据的负值 (简化计算)
        cost_total = trace_B_Inv_CovB + nSnap * log_det_Cov_b + nSnap * nSensor * np.log(2 * np.pi)
        cost = -0.5 * cost_total

        # 检查收敛
        if cost_old != 0:
            MSE = (cost - cost_old) / cost
        else:
            MSE = 1.0  # 首次迭代

        cost_old = cost

        if print_progress:
            print(f"SBLiters: {iter_num + 1}, num voxels: {Ns}, MSE: {MSE:.8f}")

        if abs(MSE) < epsilon:
            break

        evidence[iter_num] = cost

    # =========================================================================
    #                             Compute ImagingKernel
    # =========================================================================

    # 剪枝后实际迭代次数
    iter_finished = iter_num + 1 if iter_num < max_iter - 1 else max_iter
    evidence = evidence[:iter_finished]

    # ImagingKernel = (repmat(gamma,1,nSensor).*L')/Cov_b;
    # 对应: ImagingKernel = Gamma * L.T * Inv_Cov_b

    # L_Gamma_T: L.T 的行由 gamma 缩放 (Ns x nSensor)
    L_Gamma_T = L.T * gamma[:, np.newaxis]

    # Kernel (Ns x nSensor)
    ImagingKernel_compact = L_Gamma_T @ Inv_Cov_b

    # 将 Kernel 映射回原始的 nSource 维度
    ImagingKernel_full = np.zeros((nSource, nSensor), dtype=np.float64)
    if Ns > 0:
        # keep_list 是原始索引
        ImagingKernel_full[keep_list, :] = ImagingKernel_compact

    # 构建返回参数字典
    par = {
        'gamma': gamma,
        'evidence': evidence,
        'beta': beta,
        'keeplist': keep_list
    }

    if print_progress:
        print(f'\nSparse Bayesian Learning Finished in {iter_finished} iterations!')

    return ImagingKernel_full, par