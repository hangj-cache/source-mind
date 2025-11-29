# 脑源定位模拟器 

这是一个基于 Python Flask 框架构建的后端服务，用于接收用户上传的脑源定位模拟数据文件（如 Gain 矩阵、数据存储、皮层信息等），根据指定算法（SBL, MNE, LORETA）进行计算，并以 `.mat` 文件的形式将结果流式传输返回给客户端。

## 核心特性

- **跨域支持 (CORS)：** 允许前端（如本地 HTML 文件或 Web 应用）安全地调用 API。
- **文件安全处理：** 所有上传的输入文件被保存到临时的系统目录 (`TEMP_DIR`)，并在请求处理完毕后通过 `finally` 块确保彻底清理，防止数据残留。
- **无磁盘结果输出：** 计算结果（`.mat` 文件）直接写入内存缓冲区 (`io.BytesIO`)，然后流式传输给客户端，彻底避免了 Windows 环境下由于文件锁 (`WinError 32`) 导致的清理失败问题。
- **多种算法支持：** 集成了稀疏贝叶斯学习 (SBL) 和最小范数估计 (MNE/LORETA) 等算法求解器。

## 环境要求与设置

### 1. 先决条件

- Python 3.7+
- 必需的 Python 包

### 2. 安装依赖

请确保您的环境中已安装所有必需的库：

```
pip install flask numpy scipy flask-cors
```

### 3. 项目结构

为了使 `app.py` 能够正确导入求解器，您的项目文件结构必须如下所示：

```
.
├── app.py              # Flask 后端主程序
├── data/               # 🆕 示例数据存储目录
│   ├── model.mat       # 示例 Gain/Head Model 文件
│   ├── data.mat        # 示例 EEG/MEG 数据文件
│   └── ...
└── source_mind/
    ├── algorithms/
    │   ├── __init__.py
    │   ├── mne_algorithm.py   # 包含 MNE_solver 的文件
    │   └── sbl_algorithm.py   # 包含 SBL_solver 的文件
    └── result/
```

您需要确保在 `source_mind/algorithms/` 目录下创建并实现了 `MNE_solver` 和 `SBL_solver` 函数。

## 数据文件来源与格式 (核心)

**所有输入 `.mat` 文件均来源于或兼容于 MATLAB 软件的 Brainstorm 工具箱数据结构。**

为方便测试和使用，已将简单示例数据文件放在 `data/` 目录。

| API 参数名    | 文件描述         | Brainstorm 对应数据           | 必需的 MATLAB 变量名及结构                                   |
| ------------- | ---------------- | ----------------------------- | ------------------------------------------------------------ |
| `model_file`  | 头模型/增益矩阵  | Head model                    | **Gain** 矩阵 或 包含 **Gain** 字段的 **model** 结构体。     |
| `data_file`   | 传感器数据存储   | Time-Frequency (TF) 或 Epochs | 必须包含 **B** (脑电数据) 和 **s_real**(脑源数据)            |
| `cortex_file` | 皮层/源空间几何  | Surface/Cortex file           | 必须包含名为 **Cortex** 的结构体。                           |
| `l_file`      | L 矩阵 (L-Curve) | Tikhonov 规则化参数           | 必须包含 L 矩阵 (通常为对角矩阵)，作为文件中的主要 `numpy.ndarray` 变量。 |

**重要说明：**

- 后端在处理 `model_file` 时会自动尝试从结构体 `model[0][0]['Gain']` 或顶层变量 `Gain` 中提取增益矩阵。
- `data_file` 中的 `B_dataStorage` 和 `TBFs_dataStorage` 将在服务器端进行矩阵乘法 (`np.matmul(Btrans_storage, TBFs_storage)`)，以生成最终的传感器数据 `B_storage`。

## API 接口说明

### `/run_localization`

该接口负责接收所有输入文件，执行源定位计算，并将结果以文件形式返回。

- **方法:** `POST`
- **Content-Type:** `multipart/form-data`

| 参数名        | 类型   | 描述                                                    | 必需性 |
| ------------- | ------ | ------------------------------------------------------- | ------ |
| `algorithm`   | String | 要执行的算法名称。目前支持 `sbl`, `wmne`, `sloreta`。   | 是     |
| `model_file`  | File   | 包含 Gain 矩阵的 `.mat` 文件。                          | 是     |
| `data_file`   | File   | 包含 B_dataStorage 和 TBFs_dataStorage 的 `.mat` 文件。 | 是     |
| `cortex_file` | File   | 包含 Cortex 结构体的 `.mat` 文件。                      | 是     |
| `l_file`      | File   | 包含 L 矩阵（源空间约束）的 `.mat` 文件。               | 是     |

### 成功响应

- **状态码:** `200 OK`
- **内容:** 直接返回计算得到的 `.mat` 文件作为附件下载。文件命名格式为：`<algorithm>_source_results.mat`。

### 失败响应

- **状态码:** `400 Bad Request` 或 `500 Internal Server Error`

- **内容:** JSON 对象，包含错误原因。

  ```
  {
    "error": "服务器内部错误，请检查控制台输出。详细信息: <错误信息>"
  }
  ```

## 清理机制说明 (重点)

为了解决文件锁问题，`app.py` 采用了双重清理机制：

1. **输入文件清理:**
   - 上传的 `model`, `data`, `cortex`, `l` 文件被保存到系统临时目录 (`TEMP_DIR`)。
   - `@app.route` 装饰器中的 **`finally` 块**确保无论请求成功还是失败，这些临时文件都会被 `os.remove()` 删除。
2. **结果文件处理:**
   - 结果数据通过 `scipy.io.savemat(output, results_data)` 写入 `output = io.BytesIO()` **内存缓冲区**。
   - `send_file()` 直接从内存发送数据。
   - 由于结果文件从未写入磁盘，因此**无需清理**，完美规避了文件被占用 (WinError 32) 的风险。