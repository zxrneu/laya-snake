# 🐍 Laya Neural Snake (神经决策贪吃蛇)

基于开源非自回归决策大模型 **[Laya](https://github.com/NandhaKishorM/laya)** (`convaiinnovations/laya`) 构建的实时空间感知与自动驾驶贪吃蛇游戏。

不同于传统生成式 LLM 缓慢的逐字采样与幻觉问题，Laya 采用 **ModernBERT-large (421M)** 编码器底座与专门的决策原语（`choice`），单步前向推演即可输出高标定性、鲁棒的动作概率分布。

---

## 🌟 核心特性

- **非自回归秒级决策**：基于 Laya 的 `choice` 动作原语，避免文本生成的延迟，在 Mac 上原生启用 **Apple Metal (MPS)** 硬件加速，单步推理仅需几十毫秒。
- **高级空间态势感知与防陷阱算法（Anti-Trap & Tail-Reachability）**：
  - **真实避障路径（True BFS Distance）**：用真实避开身体的 BFS 最短路径代替曼哈顿距离，让长蛇不再误判直线障碍。
  - **吃食逃逸连通性（Post-Food Tail Reachability）**：虚拟演练吃完食物后的身体状态，检测是否仍留有通往蛇尾的安全逃生通道，防止“自杀式贪吃”。
  - **安全追尾漫游（Tail-Chasing Wandering）**：在食物周围通道拥堵时，主动放弃直冲，沿自身外延安全游走待机。
- **全功能赛博风 Web 控制台与升级动效**：
  - **动态大网格支持**：提供 **16x16**、**20x20 (默认)**、**24x24**、**28x28** 及 **32x32 (超大)** 宽阔对战地图。
  - **灵动仿生细节**：蛇眼瞳孔动态注视食物方位，蛇头周期性吐出分叉红信。
  - **炫彩粒子与计分浮字**：吃食瞬间触发 18 颗霓虹光斑炸裂与 `+10` 浮空升华特效。
  - **蛇头动作概率罗盘**：在蛇头周围以动态长度与安全着色（翠绿/青蓝/红叉）实时投射 Laya 动作向量。
  - **实时概率柱状图 & 态势监测**：呈现模型置信度、各方向安全属性（`EAT(SAFE)` / `CLOSER` / `CHASE TAIL` / `POCKET TRAP`）。
  - **单步推演 & 速度调节**：支持单帧调试与 50ms ~ 600ms 动态步频滑块。

---

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                      前端 Web 控制台                         │
│  (HTML5 Canvas 渲染 + 实时概率柱状图 + 决策罗盘 + 遥测审计流)    │
└───────────────────────────▲─────────────────────────────────┘
                            │ WebSocket / REST API
┌───────────────────────────▼─────────────────────────────────┐
│                    FastAPI 空间后端 (server.py)             │
│  - 状态构建: 蛇头/食物/蛇身坐标、距离计算                      │
│  - BFS 泛洪空间检测 (BFS Flood-Fill Space Analysis)          │
│  - 动态准则生成 (Dynamic Movement Criteria Generation)       │
└───────────────────────────▲─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│               Laya 决策模型 (convaiinnovations/laya)         │
│  - ModernBERT-large 421M                                    │
│  - 硬件加速: Apple Silicon MPS / CUDA / CPU                  │
│  - 动作原语: type="choice" (UP, DOWN, LEFT, RIGHT)           │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 快速上手

本项目使用现代 Python 包管理器 **[uv](https://docs.astral.sh/uv/)** 进行管理。

### 1. 克隆与安装依赖

```bash
# 进入项目目录
cd 0921

# 使用 uv 自动安装环境与依赖
uv sync
```

> **提示**：首次运行时，Laya 会自动从 Hugging Face 下载模型权重（约 800MB）并缓存至本地。

### 2. 启动服务

```bash
uv run python server.py
```

服务启动后，默认监听在 **`http://localhost:8088`**。

### 3. 打开游戏

使用浏览器访问：
👉 **`http://localhost:8088/`**

- 点击 **“启动 Laya 自动驾驶”**：观看 Laya 模型自主操控贪吃蛇觅食成长。
- 点击 **“单步推演”**：每点一次前进一格，观察右侧面板的概率分布与置信度变化。
- 切换到 **“手动操控”**：使用键盘 `W` `A` `S` `D` 或方向键亲自体验，同时可实时观察 Laya 对你每一步走法的建议与打分。

---

## 📁 项目结构

```
0921/
├── server.py              # FastAPI 后端服务（模型加载、态势感知、WS/REST 端点）
├── pyproject.toml         # uv 项目配置与依赖声明
├── uv.lock                # 锁定依赖版本
├── README.md              # 项目文档说明
├── static/
│   ├── index.html         # 游戏主页面结构
│   ├── style.css          # 暗黑赛博霓虹风格样式表
│   └── game.js            # 贪吃蛇客户端引擎、Canvas 绘制及 WebSocket 通信
└── test_snake_ai.py       # 本地决策逻辑验证脚本
```

---

## 📡 API 接口参考

### 1. 单步推演接口 (REST)
- **路径**：`POST /api/predict`
- **请求体**：
  ```json
  {
    "head": [5, 5],
    "food": [5, 8],
    "body": [[5, 4], [5, 3]],
    "grid_size": [14, 14],
    "current_direction": "DOWN"
  }
  ```
- **返回体示例**：
  ```json
  {
    "choice": "DOWN",
    "probabilities": {
      "UP": 0.0186,
      "DOWN": 0.9541,
      "LEFT": 0.0149,
      "RIGHT": 0.0123
    },
    "confidence": 0.8299,
    "inference_ms": 48.5,
    "is_safe": true,
    "analysis": {
      "UP": { "safe": false, "reason": "Reverse Neck-snap" },
      "DOWN": { "safe": true, "dist": 2, "space": 80, "is_food": false }
    }
  }
  ```

### 2. 实时通信接口 (WebSocket)
- **路径**：`ws://localhost:8088/ws/play`
- **通信格式**：发送与接收包含 `step_id` 的 JSON 报文，无额外 HTTP 握手开销，满足高帧率自动驾驶需求。

---

## 📜 依赖说明

- **模型**：[convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya) / [GitHub](https://github.com/NandhaKishorM/laya)
- **推理后端**：PyTorch (`mps` / `cuda` / `cpu`), Transformers, Safetensors
- **服务端**：FastAPI, Uvicorn, WebSockets
- **前端**：HTML5 Canvas + Vanilla CSS/JS
