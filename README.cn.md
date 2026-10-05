# Gear Insertion Policy（齒輪插入策略）

**[English version](README.en.md) | 繁體中文（本頁）**

本專案是 Isaac Lab 官方教學
[「Training a Gear Insertion Policy and ROS Deployment」](https://isaac-sim.github.io/IsaacLab/release/3.0.0/source/policy_deployment/02_gear_assembly/gear_assembly_policy.html)
的**獨立可編輯複本**：把原本位在 IsaacLab 內建 `isaaclab_tasks` 底層的
Manager-Based RL 環境（場景、觀測、獎勵、終止條件、事件、雙機械手臂設定、
RSL-RL PPO agent 設定）搬到這個資料夾，改用**純 `python` 指令**訓練，並新增一套
**ROS2 AI 模型驗證環境**——用 Isaac Sim 模擬機器人，在真的接上硬體之前，先驗證
匯出的策略在真實 ROS2 訊息迴圈下是否還能正常運作。所有物理/訓練參數都是官方在
真實 UR10e / Flexiv Rizon 4s 上驗證過的 sim-to-real 設定，原封不動保留。

> 本專案已經在真實 GPU（RTX 5070 Ti）上完整跑過一次訓練（800 次迭代、約 19.4
> 小時）與一次 ROS2 驗證迴圈，結果見下方「[訓練結果](#訓練結果)」。

---

## 目錄

1. [專案結構](#專案結構)
2. [訓練結果](#訓練結果)
3. [環境需求](#環境需求)
4. [快速開始（訓練 SOP）](#快速開始訓練-sop)
5. [model.pt 轉 policy.pt 匯出 SOP](#modelpt-轉-policypt-匯出-sop)
6. [訓練可設定的參數](#訓練可設定的參數)
7. [Sim-to-Real 三大一致性原則](#sim-to-real-三大一致性原則)
8. [程式碼架構分析](#程式碼架構分析)
9. [已註冊的 Gym 任務](#已註冊的-gym-任務)
10. [ROS2 AI 模型驗證（以 Isaac Sim 模擬機器人）](#ros2-ai-模型驗證以-isaac-sim-模擬機器人)
11. [換手臂 SOP：Swapping the Robot Arm](#換手臂-sopswapping-the-robot-arm)
12. [換物件 SOP：Swapping the Manipulated Object](#換物件-sopswapping-the-manipulated-object)
13. [疑難排解](#疑難排解)
14. [已知限制與驗證紀錄](#已知限制與驗證紀錄)
15. [參考資料](#參考資料)

---

## 專案結構

```
GearInsertionPolicy/
├── README.md / README.en.md
├── scripts/                               # 所有 python 入口腳本（訓練/播放/ROS2 橋接）
│   ├── train.py                           # 訓練入口（取代 isaaclab.sh train）
│   ├── play.py                            # 播放/評估入口（取代 isaaclab.sh play）
│   └── ros2_isaac_sim_bridge.py           # Isaac Sim 端 ROS2 橋接（需 GPU/Kit，用 python 執行）
├── train/                                 # 訓練相關：環境定義 + 訓練結果
│   ├── gear_insertion_policy/             # 可 import 的 Python 套件（底層任務實作）
│   │   ├── __init__.py
│   │   └── tasks/
│   │       ├── __init__.py                # import 兩個 config 子套件 -> 觸發 gym.register()
│   │       ├── gear_assembly_env_cfg.py   # 機器人無關的共用基底（場景/觀測/獎勵/終止條件骨架）
│   │       ├── mdp/                       # 任務專用 MDP 函式庫
│   │       │   ├── observations.py        #   gear_shaft_pos_w / gear_pos_w ...
│   │       │   ├── rewards.py             #   keypoint_entity_error(_exp) / keypoint_ee_grasp_error(_exp)
│   │       │   ├── terminations.py        #   reset_when_gear_dropped / *_orientation_exceeds_threshold
│   │       │   ├── events.py              #   randomize_gear_type / randomize_gears_and_base_pose / set_robot_to_grasp_pose
│   │       │   └── noise_models.py        #   ResetSampledConstantNoiseModel(Cfg) / *Quaternion*
│   │       └── config/
│   │           ├── ur_10e/                # UR10e（6 DOF）+ Robotiq 2F-140 / 2F-85
│   │           └── rizon_4s/              # Flexiv Rizon 4s（7 DOF）+ Grav 夾爪
│   └── result/                            # 已完成訓練的 TensorBoard 匯出圖表 + 影片
│       ├── clip_0000.mp4                  # 訓練中錄製的影片片段
│       ├── Train-mean_reward*.png         # 獎勵曲線
│       ├── Episode_Termination-*.png      # 回合結束原因統計
│       ├── Episode_Reward-*.png           # 各獎勵項拆解
│       ├── Loss-*.png                     # PPO 訓練損失
│       └── Perf-*.png / Policy-*.png      # 訓練效能 / 動作標準差
└── valid/                                 # 驗證相關：ROS2 AI 模型驗證
    └── ros2_validation_cpp/               # ROS2 ament_cmake 套件：gear_insertion_ros2_cpp（rclcpp + LibTorch）
        ├── package.xml / CMakeLists.txt
        ├── include/gear_insertion_ros2_cpp/tiny_json.hpp
        ├── src/policy_inference_node.cpp  # 「AI 模型」端：torch::jit::load 直接跑 TorchScript
        ├── src/success_monitor_node.cpp   # 自動化驗證：統計每回合成功/失敗、輸出成功率報告
        └── launch/gear_insertion_validation_cpp.launch.py
```

`logs/`（訓練時自動產生的 checkpoint / TensorBoard events / 影片，`.gitignore`
已排除）執行 `scripts/train.py` 後會重新產生在 `GearInsertionPolicy/logs/`
（相對於執行指令時的工作目錄）；`train/result/` 則是手動整理、保留下來的
「這次訓練跑完的結果快照」，兩者用途不同，不要搞混。

## 訓練結果

UR10e + Robotiq 2F-140（任務 `GearInsertion-UR10e-2F140-ROSInference-v0`）已經
完整訓練過一輪：**800/800 次迭代，耗時約 19.4 小時**（RTX 5070 Ti）。最終指標
（TensorBoard 平滑值 / 原始值，迭代 799）：

| 指標 | 平滑值 | 原始值 | 說明 |
|---|---|---|---|
| `Train/mean_reward` | 3.06 | 3.39 | 從 0 開始，前 ~150 次迭代快速爬升，之後持續緩慢上升到收斂，收斂曲線形狀健康 |
| `Episode_Termination/time_out` | 0.71 | 0.68 | 約 iteration 100 左右爬到高點後打平，代表「沒掉齒輪、沒翻過頭」的比例趨於穩定 |

完整曲線圖與一段訓練中錄製的影片在 [train/result/](train/result/)：

- `Train-mean_reward.png` / `Train-mean_reward-time.png`：獎勵隨迭代次數/時間的曲線
- `Episode_Termination-time_out.png` / `-gear_dropped.png` / `-gear_orientation_exceeded.png`：三種回合結束原因的比例
- `Episode_Reward-end_effector_gear_keypoint_tracking*.png`：插入精細度相關的獎勵項
- `Loss-value.png` / `Loss-surrogate.png` / `Loss-entropy.png` / `Loss-learning_rate.png`：PPO 訓練診斷指標
- `Perf-*.png`：每秒步數 / 收集時間 / 學習時間
- `Policy-mean_std.png`：動作標準差（探索程度）
- `clip_0000.mp4`：訓練中錄製的影片片段

**重要提醒**：`time_out` 比例高只代表「齒輪沒掉、姿態沒翻過頭」，**不代表齒輪
真的插到軸上了**——這也是為什麼下方「ROS2 AI 模型驗證」一節的
`ros2_isaac_sim_bridge.py` 額外用真實的齒輪-軸心距離做更嚴格的成功判定（見該節
「程式碼分析」）。這份訓練結果可以直接拿去跑 `scripts/play.py` 或接到 ROS2
驗證迴圈裡實測。

## 環境需求

這個專案**不是**一個全新的、脫離 Isaac Lab 的安裝——它 import 了 `isaaclab`、
`isaaclab_assets`、`isaaclab_tasks`、`isaaclab_rl`、`isaaclab_physx` 等套件，
因此必須用**已安裝好 Isaac Lab 的 Python 直譯器**執行，例如：

```bash
source /path/to/your/isaaclab-arena/.venv/bin/activate
```

之後底下所有指令的 `python` 都是指這個直譯器。`scripts/train.py` / `scripts/play.py`
/ `scripts/ros2_isaac_sim_bridge.py` 都會自動把 `GearInsertionPolicy/train/`
加進 `sys.path`，不需要額外 `pip install` 本專案。

ROS2 驗證部分（`valid/ros2_validation_cpp/`）另外需要：ROS2（本機驗證用的是
Jazzy）、LibTorch（C++ 版 PyTorch，含 `share/cmake/Torch`）、`cmake`/`g++`。

## 快速開始（訓練 SOP）

### Step 1：小規模視覺化檢查環境是否正確

```bash
cd GearInsertionPolicy

python scripts/train.py \
    --rl_library rsl_rl \
    --task GearInsertion-UR10e-2F140-ROSInference-v0 \
    --num_envs 4 \
    --visualizer kit
```

會打開 Isaac Sim 檢視器，應該可以看到多個手臂抓著齒輪亂動（此時還不會成功插入，
是正常現象）。確認場景、物理、抓取姿態合理後 `Ctrl+C` 停止，進入下一步。

### Step 2：Headless 全規模訓練 + 定期錄影

```bash
python scripts/train.py \
    --rl_library rsl_rl \
    --task GearInsertion-UR10e-2F140-ROSInference-v0 \
    --num_envs 256 \
    --video --video_length 200 --video_interval 76800
```

不加 `--visualizer` 就是 headless（這個版本的 Isaac Lab 已經沒有 `--headless`
旗標了，見下一節）。完整訓練約需 12–24 小時；本專案實測 UR10e 版本花了約 19.4
小時（見「訓練結果」）。影片與 checkpoint 存在
`logs/rsl_rl/gear_assembly_ur10e/<timestamp>/`。Rizon 4s 版本把 `--task` 換成
`GearInsertion-Rizon4s-Grav-ROSInference-v0`。

### 用 TensorBoard 監控訓練

```bash
python -m tensorboard.main --logdir logs/rsl_rl/gear_assembly_ur10e
```

重點看 `Train/mean_reward` 和 `Episode_Termination/time_out` 有沒有隨迭代次數
整體往上爬（允許中間震盪）。

### Step 3：播放/評估訓練好的策略

```bash
python scripts/play.py \
    --rl_library rsl_rl \
    --task GearInsertion-UR10e-2F140-v0 \
    --num_envs 16 \
    --checkpoint logs/rsl_rl/gear_assembly_ur10e/<run>/model_<iter>.pt
```

會順便把策略匯出成 `policy.pt`（TorchScript）與 `policy.onnx`——匯出位置與
完整 SOP 見下一節「[model.pt 轉 policy.pt 匯出 SOP](#modelpt-轉-policypt-匯出-sop)」。
`policy.pt` 就是 ROS2 驗證要用的檔案。

### Step 4：ROS2 模擬驗證 / 真實機器人部署

接上 ROS2 驗證迴圈（見「[ROS2 AI 模型驗證](#ros2-ai-模型驗證以-isaac-sim-模擬機器人)」
一節），或直接依
[Isaac ROS Manipulation DNN Policy 文件](https://nvidia-isaac-ros.github.io/reference_workflows/isaac_for_manipulation/packages/isaac_ros_manipulation_dnn_policy/index.html)
部署到真實硬體。Isaac ROS 推論管線直接吃 `.pt` checkpoint + `agent.yaml` +
`env.yaml`，不需要額外匯出步驟。

## model.pt 轉 policy.pt 匯出 SOP

訓練時 RSL-RL 存的 `model_<iter>.pt` 是**原始 checkpoint**——格式是給
`OnPolicyRunner.load()` 讀的，用來繼續訓練或在 `play.py` 裡重建完整的
actor/critic 網路，**不能**直接拿給 ROS2 推論節點或其他部署環境用。
`scripts/play.py` 在載入 checkpoint 播放的同時，會自動把它轉成兩個獨立、
可攜的匯出格式：

- **`policy.pt`**（TorchScript，`torch.jit.load()` 可讀，C++/LibTorch 也能直接
  讀——本專案 `valid/ros2_validation_cpp/` 的 ROS2 推論節點用的就是這個檔案）
- **`policy.onnx`**（ONNX，給 TensorRT 等其他推論框架用）

### Step 1：找到要轉換的 checkpoint

```bash
ls logs/rsl_rl/gear_assembly_ur10e/<run_timestamp>/model_*.pt
```

訓練每隔 `save_interval`（預設 `50`，見「訓練可設定的參數」）次迭代存一個
`model_<iteration>.pt`（iteration 從 0 算起）。完整跑完 800 次迭代的最後一個
通常是 `model_799.pt`。

### Step 2：執行 `play.py` 觸發匯出

```bash
cd GearInsertionPolicy

python scripts/play.py \
    --rl_library rsl_rl \
    --task GearInsertion-UR10e-2F140-v0 \
    --num_envs 16 \
    --checkpoint logs/rsl_rl/gear_assembly_ur10e/<run_timestamp>/model_799.pt
```

- `--task` 用跟訓練時相同的機器人/夾爪組合，一般訓練任務 ID（不帶
  `-ROSInference` 後綴）或 ROSInference 版本都可以播放，差別只在重置姿態是否
  隨機。
- `--checkpoint` 可以是完整路徑、`latest`（自動抓最新 run 裡最新的
  checkpoint）、或 `best`。
- `--num_envs` 這裡只是看效果用，設小一點即可（不需要訓練時的 1024）。
- Rizon 4s 版本把 `--task` 換成對應的 `GearInsertion-Rizon4s-Grav-v0`。

### Step 3：確認匯出結果（注意匯出路徑，不是跟 checkpoint 同一層！）

```bash
ls logs/rsl_rl/gear_assembly_ur10e/<run_timestamp>/exported/
# policy.pt  policy.onnx
```

**匯出位置是 checkpoint 所在資料夾底下的 `exported/` 子資料夾**，即：

```
logs/rsl_rl/gear_assembly_ur10e/<run_timestamp>/exported/policy.pt   ✓ 正確
logs/rsl_rl/gear_assembly_ur10e/<run_timestamp>/policy.pt            ✗ 錯誤，這裡不會有檔案
```

這是直接從 `isaaclab_rl/entrypoints/backends/play_rsl_rl.py` 原始碼核對出來的：

```python
export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
...
export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")
```

### Step 4：接到 ROS2 驗證（或其他部署）

把 Step 3 找到的 `.../exported/policy.pt` 完整路徑餵給 ROS2 推論節點的
`policy_path` 參數：

```bash
ros2 launch gear_insertion_ros2_cpp gear_insertion_validation_cpp.launch.py \
    policy_path:=/絕對路徑/logs/rsl_rl/gear_assembly_ur10e/<run_timestamp>/exported/policy.pt \
    num_arm_joints:=6
```

完整 SOP 見「[ROS2 AI 模型驗證](#ros2-ai-模型驗證以-isaac-sim-模擬機器人)」一節。
`policy_path` 一定要用**絕對路徑**（`ros2 launch` 執行時的工作目錄不一定是
`GearInsertionPolicy/`，相對路徑容易找不到檔案）。

## 訓練可設定的參數

`python scripts/train.py ...` 可設定的參數分三層，優先序：
**命令列參數 > `agents/rsl_rl_ppo_cfg.py`（RL 超參數預設值）> 環境 cfg 檔案**。

### A. 命令列參數

**任務 / 環境**

| 參數 | 類型 | 預設值 | 說明 |
|---|---|---|---|
| `--task` | str | 無（必填） | Gym 任務 ID，見「已註冊的 Gym 任務」 |
| `--num_envs` | int | cfg 裡的值（本任務 1024） | 平行環境數 |
| `--seed` | int | agent cfg 裡的值 | 隨機種子，`-1` 會自動隨機抽 |
| `--device` | str | `cuda:0` | `cpu` / `cuda` / `cuda:N` |
| `--distributed` | flag | `False` | 多 GPU / 多節點訓練 |

**畫面 / 錄影**

| 參數 | 類型 | 預設值 | 說明 |
|---|---|---|---|
| `--visualizer` / `--viz` | str（CSV） | 無（＝headless） | `kit`/`rerun`/`newton_gl`/`viser`；**不加就是 headless** |
| `--video` | flag | `False` | 訓練中定期錄影 |
| `--video_length` / `--video_interval` | int | cfg 裡的值 | 影片長度 / 間隔（env steps） |

**訓練長度 / Checkpoint**

| 參數 | 類型 | 預設值 | 說明 |
|---|---|---|---|
| `--max_iterations` | int | agent cfg（本任務 1500） | 覆寫總迭代數 |
| `--checkpoint` | str | `None` | 接續訓練：路徑 / `latest` / `best` |

**實驗記錄**：`--experiment_name` / `--run_name` / `--logger`
（`tensorboard`/`wandb`/`neptune`）/ `--log_project_name`。

> **沒有 `--headless` 這個旗標**（已實測確認 `AppLauncher.add_app_launcher_args()`
> 註冊的旗標清單裡沒有它）。現在用 `--visualizer` 控制：不加就是 headless，
> `--visualizer kit` 開視窗。`train.py`/`play.py` 背後的 RSL-RL backend 用
> `parse_known_args`，傳 `--headless` 會被安靜吞掉（無效但不報錯）；但
> `ros2_isaac_sim_bridge.py` 用嚴格的 `parse_args()`，傳 `--headless` 會直接
> 噴 `unrecognized arguments` 錯誤。

### B. RL 超參數（`train/gear_insertion_policy/tasks/config/<robot>/agents/rsl_rl_ppo_cfg.py`）

CLI 沒有提供覆寫旗標，要調整就直接改這個檔案：

| 欄位 | 目前值 | 說明 |
|---|---|---|
| `num_steps_per_env` | `512` | 每次 PPO 更新前，每個環境收集的 rollout 步數 |
| `max_iterations` | `1500` | 預設總訓練迭代數 |
| `save_interval` | `50` | 每多少次迭代存一次 checkpoint |
| `clip_actions` | `1.0` | 動作裁剪範圍 |
| `rnn_type` / `rnn_hidden_dim` / `rnn_num_layers` | `lstm` / `256` / `2` | Actor/Critic 的 LSTM 設定 |
| `hidden_dims` | `[256, 128, 64]` | LSTM 後接的 MLP |
| `learning_rate` / `schedule` / `desired_kl` | `5e-4` / `adaptive` / `0.008` | 學習率與自動調整 |
| `clip_param` / `entropy_coef` | `0.2` / `0.0` | PPO 裁剪範圍 / 熵獎勵係數 |
| `num_learning_epochs` / `num_mini_batches` | `8` / `16` | 每批 rollout 的訓練切分 |
| `gamma` / `lam` | `0.99` / `0.95` | 折扣因子 / GAE λ |

完整欄位可用
`python -c "import dataclasses; from isaaclab_rl.rsl_rl import RslRlPpoAlgorithmCfg as C; [print(f.name) for f in dataclasses.fields(C)]"`
列出對照。

### C. 環境層級參數（`gear_assembly_env_cfg.py` / `config/<robot>/joint_pos_env_cfg.py`）

| 參數 | 目前值 | 說明 |
|---|---|---|
| `scene.num_envs` | `1024` | 預設平行環境數 |
| `episode_length_s` | `6.66` | 每回合秒數 |
| `decimation` / `sim.dt` | `4` / `1/120` | 控制步對應物理子步 / 物理步長 |
| `joint_action_scale` | `0.025` | 網路輸出 × 這個值 = 關節角增量（弧度） |
| `solver_position_iteration_count` | `4` | 接觸求解器迭代次數 |
| `gpu_collision_stack_size` 等 | `2**30` 等 | PhysX GPU buffer 大小 |

## Sim-to-Real 三大一致性原則

| 原則 | 意義 | 對應程式碼 |
|---|---|---|
| **輸入一致性** | 策略在模擬中看到的觀測，要跟真機感測器量得到的一致，並加對應雜訊 | `joint_pos`/`joint_vel` 無雜訊；`gear_shaft_pos`/`gear_shaft_quat` 有雜訊模型（模擬 FoundationPose + RealSense 誤差） |
| **系統響應一致性** | 模擬中機器人對動作的反應要跟真機一樣 | 摩擦力材質、PD 增益隨機化、關節摩擦、接觸求解器迭代次數 |
| **輸出一致性** | Isaac Lab 對策略輸出的後處理，真機推論時也要做一樣的 | `RelativeJointPositionActionCfg(scale=..., use_zero_offset=True)`——真機端要用同樣的 `action_scale` |

**除錯技巧**：真機失敗時，先讓真機的初始觀測跟模擬一致，比較 controller/系統
反應，藉此判斷是「觀測不一致」還是「物理/控制器不一致」。

## 程式碼架構分析

### `train/gear_insertion_policy/tasks/gear_assembly_env_cfg.py`（機器人無關的共用基底）

- **`GearAssemblySceneCfg`**：地板、`factory_gear_base`（齒輪基座，`kinematic_enabled=True`）、
  `factory_gear_{small,medium,large}`（三種可動齒輪）、`robot: ArticulationCfg = MISSING`
  （由各機器人 config 填入，**ROBOT SWAP POINT**）、燈光、機台。
- **`ActionsCfg`**：`arm_action`（由各機器人填 `RelativeJointPositionActionCfg`）、
  `gripper_action`（齒輪已被夾爪抓住，訓練中不主動控制，通常為 `None`）。
- **`ObservationsCfg`**：`PolicyCfg`（真機能拿到、含雜訊：`joint_pos`/`joint_vel`/`gear_shaft_pos`/`gear_shaft_quat`）、
  `CriticCfg`（額外含 `gear_pos`/`gear_quat` 等無雜訊特權資訊，只給 critic 用，訓練加速收斂）。
- **`RewardsCfg`**：`keypoint_entity_error`（`-1.5`，齒輪-基座關鍵點距離懲罰）、
  `keypoint_entity_error_exp`（`+1.5`，指數型精細對準獎勵）、`action_rate_l2`（`-5e-6`，動作平滑）。
- **`TerminationsCfg`**：`time_out`；`gear_dropped`（末端-齒輪距離 >15cm）；
  `gear_orientation_exceeded`（姿態偏移超過角度門檻）。

### `train/gear_insertion_policy/tasks/mdp/`（任務專屬 MDP 函式庫）

| 檔案 | 內容 |
|---|---|
| `observations.py` | `gear_shaft_pos_w`/`gear_shaft_quat_w`（軸心世界座標）、`gear_pos_w`/`gear_quat_w`（齒輪自身姿態，critic 專用） |
| `rewards.py` | `keypoint_entity_error(_exp)`（關鍵點距離，比單純 pose 誤差更能同時捕捉平移+旋轉）、`keypoint_ee_grasp_error(_exp)`（末端 vs 正確抓取姿態，只有 Rizon 4s 用） |
| `terminations.py` | `reset_when_gear_dropped`、`reset_when_gear_orientation_exceeds_threshold` |
| `events.py` | `randomize_gear_type`（reset 時隨機選齒輪尺寸，記錄在 `env._gear_type_manager`）、`randomize_gears_and_base_pose`、`set_robot_to_grasp_pose`（差分 IK 把手臂移到抓好姿態）；依賴 `isaaclab_tasks.contrib.automate.factory_control`（IsaacLab 內建通用工具，非本專案複製範圍） |
| `noise_models.py` | `ResetSampledConstantNoiseModel(Cfg)`（每回合抽一次固定雜訊）、`ResetSampledQuaternionNoiseModel(Cfg)` |

### `config/ur_10e/` vs `config/rizon_4s/`

| | UR10e（6 DOF） | Rizon 4s（7 DOF）+ Grav |
|---|---|---|
| 末端 body | `wrist_3_link` | `link7` |
| 動作縮放 `joint_action_scale` | `0.025` | `0.025` |
| 觀測雜訊 | `gear_shaft_pos` ±5mm | `gear_shaft_pos` ±10mm + `gear_shaft_quat` ±2°（感知管線雜訊較大） |
| 額外 `EventCfg`：`robot_joint_stiffness_and_damping`/`joint_friction` | 有（模擬真機黏滯摩擦，PD 增益隨機化） | 無（真機控制器更準，不需要） |
| 額外 `RewardsCfg`：`end_effector_grasp_keypoint_tracking(_exp)` | 無 | 有（`weight_ramp_steps=512_000` 線性拉權重，避免一開始就過度懲罰） |
| `config/<robot>/ros_inference_env_cfg.py` | 繼承訓練版，固定重置姿態 + 暴露 ROS 中繼資料（`obs_order`/`action_space`/`fixed_asset_init_pos_center` 等） | 同左 |

### `config/<robot>/agents/rsl_rl_ppo_cfg.py`

兩機器人共用 RNN(LSTM)+PPO 架構：`HeteroscedasticGaussianDistributionCfg(std_type="log")`
讓動作標準差可學習；用 RNN 而非純 MLP，是因為策略需要對雜訊觀測與接觸狀態做時序濾波。

## 已註冊的 Gym 任務

| Gym 任務 ID | 機器人 | 夾爪 | 用途 |
|---|---|---|---|
| `GearInsertion-UR10e-2F140-v0` | UR10e (6DOF) | Robotiq 2F-140 | 訓練（隨機重置） |
| `GearInsertion-UR10e-2F85-v0` | UR10e (6DOF) | Robotiq 2F-85 | 訓練（隨機重置） |
| `GearInsertion-UR10e-2F140-ROSInference-v0` | UR10e (6DOF) | Robotiq 2F-140 | 固定重置 + ROS 中繼資料 |
| `GearInsertion-UR10e-2F85-ROSInference-v0` | UR10e (6DOF) | Robotiq 2F-85 | 固定重置 + ROS 中繼資料 |
| `GearInsertion-Rizon4s-Grav-v0` | Flexiv Rizon 4s (7DOF) | Grav 夾爪 | 訓練（隨機重置） |
| `GearInsertion-Rizon4s-Grav-ROSInference-v0` | Flexiv Rizon 4s (7DOF) | Grav 夾爪 | 固定重置 + ROS 中繼資料 |

這些 ID 與官方原版 `IsaacContrib-Deploy-GearAssembly-*` 不同，避免跟已安裝的
IsaacLab 內建任務衝突，兩者可在同一環境中並存。

## ROS2 AI 模型驗證（以 Isaac Sim 模擬機器人）

訓練完成、`play.py` 匯出 `policy.pt` 之後，真正上真機前還缺一個環節：**驗證
`policy.pt` 在真的透過 ROS2 傳輸觀測/動作的情況下是否還正常運作**（跨進程、
訊息序列化、RNN 隱狀態管理、動作後處理——這些在同進程直接呼叫的 `play.py`
測不到）。本專案用 **Isaac Sim 模擬機器人**，搭配一個真正的 **C++ ROS2 node**
跑 AI 推論，組成完整的 ROS2 閉迴圈。

### 架構總覽

```
┌─────────────────────────────┐   /gear_insertion/joint_states (sensor_msgs/JointState)
│  scripts/                   │   /gear_insertion/gear_shaft_pose (geometry_msgs/PoseStamped)
│  ros2_isaac_sim_bridge.py   │ ──────────────────────────────────────────────────┐
│  (python 執行，需 Kit/GPU)   │                                                    │
│  = 機器人 + 感知端(模擬)      │ <─────────────────────────────────────────────┐  │
└─────────────────────────────┘   /gear_insertion/policy_action (Float32MultiArray) │
         │ /gear_insertion/episode_event (JSON)                                   │  │
         v                                                                         │  │
┌─────────────────────────────┐                   ┌─────────────────────────────┐ │  │
│ valid/ros2_validation_cpp/   │                   │ valid/ros2_validation_cpp/   │ │  │
│ success_monitor_node         │                   │ policy_inference_node        │─┘  │
│ = 自動化驗證/成功率統計      │                   │ = AI 模型（torch::jit::load）│────┘
└─────────────────────────────┘                   └─────────────────────────────┘
```

機器人端（`scripts/ros2_isaac_sim_bridge.py`）需要 Kit/GPU，必須用 `python`
直接執行；AI 模型端與驗證端（`valid/ros2_validation_cpp/`）是純 rclcpp +
LibTorch，用 `ros2 run`/`ros2 launch` 啟動，可以在不同機器上跑。

### 事前準備

```bash
source /opt/ros/<distro>/setup.bash

cd GearInsertionPolicy/valid/ros2_validation_cpp
colcon build --cmake-args -DCMAKE_PREFIX_PATH=/path/to/libtorch
source install/setup.bash
export LD_LIBRARY_PATH=/path/to/libtorch/lib:$LD_LIBRARY_PATH   # 執行期也需要
```

`isaacsim.ros2.bridge` extension 能否在你的 ROS2 distro 下啟用，需要實際
GPU/Kit 環境才能驗證；其餘(colcon build、節點啟動、端到端訊息流)已在本機
ROS2 Jazzy + LibTorch 環境實測通過，見「已知限制與驗證紀錄」。

### SOP：跑一次完整的 ROS2 AI 模型驗證

**終端機 1（機器人端＝Isaac Sim，必須用 python）：**

```bash
source /opt/ros/<distro>/setup.bash   # 讓 Isaac Sim 進程內能 import rclpy
cd GearInsertionPolicy

python scripts/ros2_isaac_sim_bridge.py \
    --task GearInsertion-UR10e-2F140-ROSInference-v0 \
    --num_episodes 50 \
    --insertion_distance_threshold 0.01
```

不加 `--visualizer` 就是 headless；想看視窗加 `--visualizer kit`。

**終端機 2（AI 模型端 + 驗證端）：**

```bash
source /opt/ros/<distro>/setup.bash
source GearInsertionPolicy/valid/ros2_validation_cpp/install/setup.bash

ros2 launch gear_insertion_ros2_cpp gear_insertion_validation_cpp.launch.py \
    policy_path:=/path/to/logs/rsl_rl/gear_assembly_ur10e/<run>/exported/policy.pt \
    num_arm_joints:=6 \
    report_path:=/tmp/gear_insertion_validation_report.json
```

`policy.pt` 就是 `scripts/play.py` 匯出的 TorchScript 檔（匯出路徑與完整 SOP
見「[model.pt 轉 policy.pt 匯出 SOP](#modelpt-轉-policypt-匯出-sop)」）。終端機
2 會持續印出
`episodes=.. success=.. (xx.x%) failures={...}`，`report_path` 則是持續更新的
JSON 成功率報表。Rizon 4s 版本把 `--task`/`num_arm_joints` 換成對應值。

### 快速檢查工具

```bash
ros2 topic list                                   # 應看到 4 個 /gear_insertion/* topic
ros2 topic hz /gear_insertion/joint_states        # 驗證發布頻率
ros2 topic echo /gear_insertion/episode_event     # 看每回合結束事件 (JSON)
```

### Topic / 訊息格式表

| Topic（`<ns>` 預設 `gear_insertion`） | 訊息型別 | 發布者 | 內容 |
|---|---|---|---|
| `/<ns>/joint_states` | `sensor_msgs/JointState` | Isaac Sim 橋接節點 | 關節位置/速度，**無雜訊** |
| `/<ns>/gear_shaft_pose` | `geometry_msgs/PoseStamped` | Isaac Sim 橋接節點 | 齒輪軸心世界座標，**含訓練時同款雜訊模型** |
| `/<ns>/policy_action` | `std_msgs/Float32MultiArray` | AI 推論節點 | 策略原始動作，已夾到 `[-1, 1]`，長度 = `num_arm_joints` |
| `/<ns>/episode_event` | `std_msgs/String`（JSON） | Isaac Sim 橋接節點 | `{"event":"episode_end","episode_index":N,"outcome":"success"/"failure","reason":"time_out"/"time_out_not_inserted"/"gear_dropped"/"gear_orientation_exceeded","final_distance_m":0.00xxx}` |

### 程式碼分析

**`scripts/ros2_isaac_sim_bridge.py`（機器人端）**

- 跟 `train.py`/`play.py` 一樣的開頭模式：`argparse` 收參數 + `AppLauncher.add_app_launcher_args(parser)`，
  `AppLauncher(args_cli)` 開完 Kit 後，用
  `omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate("isaacsim.ros2.bridge", True)`
  啟用 ROS2 bridge extension（刻意不用 `isaacsim.core.utils.extensions.enable_extension`
  這類便利包裝——實測發現這層包裝的模組路徑在不同 Isaac Sim 版本間會搬家，
  `omni.kit.app` 才是兩種包裝底層都呼叫、不會變的那一層），再 `import rclpy`。
- `IsaacSimRobotBridgeNode.__init__`：用 `isaaclab_tasks.utils.parse_env_cfg(task_id, device=..., num_envs=1)`
  取得 env cfg 再 `gym.make(task_id, cfg=env_cfg)`——這是 Isaac Lab 標準建環境流程
  （直接 `gym.make(task_id, num_envs=1)` 不會生效）。
- `_publish_observations`：直接切 `self._obs["policy"][0]`（訓練用的同一個觀測張量，
  含雜訊），依 `[joint_pos(n), joint_vel(n), gear_shaft_pos(3), gear_shaft_quat(4)]`
  順序切回去。**四元數分量順序是 `(x, y, z, w)`**（Isaac Lab 3.0 的
  `QUAT_XYZW_ELEMENT_NAMES` 慣例，跟 `geometry_msgs/Quaternion` 剛好同序，
  不需要重排；`gear_shaft_quat_w()` 的 docstring 寫 `(w,x,y,z)` 是舊版 Isaac Lab
  留下的過期說明）。
- `_publish_episode_end`：**成功判定不是只看 time_out**。`time_out` 只是必要
  條件，還要另外用 `_obs_term_slice()`（從 `ObservationManager.active_terms`/
  `group_obs_term_dim` 動態算偏移量，因為 critic 群組的 `joint_pos`/`joint_vel`
  維度跟機器人/夾爪有關，不能寫死）取出 `critic` 觀測群組裡無雜訊的 `gear_pos`
  （齒輪實際位置）與 `gear_shaft_pos`（目標位置），算出真實距離，小於
  `--insertion_distance_threshold`（預設 1cm）才算 `success`；只撐到時間到但
  沒插進去，標成 `time_out_not_inserted`。`episode_event` 會帶上
  `final_distance_m` 這個實際數字。
- `step_once`：收不到新 `/policy_action`（推論節點還沒起來或掛了）超過
  `--hold_last_action_steps` 步，改成全 0 動作（原地不動），避免套用過期指令。

**`valid/ros2_validation_cpp/src/policy_inference_node.cpp`（AI 模型端）**

- `torch::jit::load(policy_path, device)` 載入 `play.py` 匯出的 TorchScript，
  介面對應 `isaaclab_rl/rsl_rl/exporter.py` 的 `_TorchPolicyExporter`：
  `policy_.forward({obs_tensor}).toTensor()` 回傳 actor 的（確定性）平均動作，
  **不會自動夾範圍**——RSL-RL 的 `clip_actions=1.0` 是訓練環境內部做的，匯出的
  TorchScript 模組沒有這段，程式裡用 `torch::clamp(raw_action, -1.0, 1.0)` 補回來。
- `policy_.get_method("reset")({})`：收到 `episode_end` 事件時清空 LSTM 隱狀態
  （對應 `exporter.py` 的 `reset_memory()`），避免新回合帶著上一回合的記憶開局。
- `tiny_json.hpp`：刻意不用通用 JSON 函式庫，寫了一個只認得本專案自己固定
  `{"key": "value", ...}` 格式的最小化字串擷取函式——因為生產端與消費端都是
  自己控制，格式固定，不需要通用解析器的相依性。

**`valid/ros2_validation_cpp/src/success_monitor_node.cpp`（自動化驗證）**

純粹的 `episode_event` 計數器：累加 `total`/`success`，依失敗原因分桶
（`gear_dropped`/`gear_orientation_exceeded`/`time_out_not_inserted`），每
`report_every_n` 回合印一次摘要，`report_path` 給的話持續覆寫 JSON 報表。

### 這段驗證測到什麼、測不到什麼

**測得到**：匯出的 `policy.pt` 在跨進程、真實 ROS2 訊息序列化/反序列化、含
雜訊觀測、RNN 隱狀態正確重置、動作裁剪邏輯正確還原這些條件下是否正常運作，
以及用 Isaac Sim 物理模擬當「機器人」時**真實的插入成功率**（用齒輪-軸心距離
門檻判定，不是只看時間到）。

**測不到**：真實感測器的雜訊特性（仍是模擬器算出來的雜訊模型）、真實機器人
控制器的動態響應（黏滯摩擦、通訊延遲）、真正的 cuMotion 運動規劃。這些仍需要
依「Step 4：真實機器人部署」接上正式的
[Isaac ROS Manipulation DNN Policy 管線](https://nvidia-isaac-ros.github.io/reference_workflows/isaac_for_manipulation/packages/isaac_ros_manipulation_dnn_policy/index.html)
才能完整驗證。

## 換手臂 SOP：Swapping the Robot Arm

以新增一支手臂（例如 Franka Panda）為例，照抄 `train/gear_insertion_policy/tasks/config/ur_10e/`
這個資料夾結構：

1. **建立新資料夾** `train/gear_insertion_policy/tasks/config/<your_robot>/`，
   複製 `ur_10e/` 底下所有檔案當範本。
2. **換 `ArticulationCfg`**（`joint_pos_env_cfg.py` 標記 `# --- ROBOT SWAP POINT`
   的那一行）：換成你的手臂設定（`isaaclab_assets` 現成的，或自己寫的）。
3. **更新機器人相關標量參數**（`<Robot>GearAssemblyEnvCfg.__post_init__` 裡）：
   - `self.end_effector_body_name`：末端執行器 body 名稱
   - `self.num_arm_joints`：手臂自由度（不含夾爪）
   - `self.grasp_rot_offset`：抓取姿態四元數偏移（用差分 IK 或 `--visualizer kit`
     小 `num_envs` 反覆調）
   - `self.observations.policy.joint_pos/joint_vel.params["asset_cfg"].joint_names`
   - `self.actions.arm_action = mdp.RelativeJointPositionActionCfg(...)`
4. **夾爪 actuator + joint setter**：
   - `self.scene.robot.actuators["gripper_..."] = ImplicitActuatorCfg(...)`
   - 寫 `set_finger_joint_pos_<your_gripper>(...)` 函式（參考
     `set_finger_joint_pos_robotiq_2f140`/`set_finger_joint_pos_grav`）
   - `self.gear_offsets_grasp` / `self.hand_grasp_width` / `self.hand_close_width`
5. **物理/動作調校**（依「系統響應一致性」，用真機錄影比對迭代調整）：
   `solver_position_iteration_count`；黏滯摩擦明顯就加
   `robot_joint_stiffness_and_damping`/`joint_friction`（像 UR10e）；`joint_action_scale`
   需要真機階躍響應量測確認夠大能克服摩擦。
6. **註冊 Gym 任務**：`config/<your_robot>/__init__.py` 呼叫
   `gym.register(id="GearInsertion-<YourRobot>-v0", ...)`，並在
   `tasks/__init__.py` 的 `from .config import ...` 加上新子套件。
7. **RSL-RL agent 設定**：通常直接沿用既有 RNN+PPO 架構，只改 `experiment_name`。
8. `python scripts/train.py --task GearInsertion-<YourRobot>-v0 --num_envs 4 --visualizer kit`
   小規模驗證，再進入全規模訓練。

## 換物件 SOP：Swapping the Manipulated Object

若要把「齒輪插入」換成別的插入/組裝任務，照下面清單修改
`train/gear_insertion_policy/tasks/gear_assembly_env_cfg.py`（標記
`# --- OBJECT SWAP POINT`）與相依檔案：

1. **場景資產**：把 `factory_gear_base`/`factory_gear_{small,medium,large}`
   這 4 個 `RigidObjectCfg` 換成你的 USD 資產（母件維持 `kinematic_enabled=True`）。
2. **偏移量常數**：`self.gear_offsets`（子件關鍵插入點相對母件參考點的偏移）。
3. **`mdp/events.py`**：`randomize_gear_type`（換 `gear_types` 列表）、
   `randomize_gears_and_base_pose`（位姿隨機化範圍）、`set_robot_to_grasp_pose`
   （若抓取方式不同，如吸盤而非夾爪，需對應修改）。
4. **`mdp/observations.py`**：`gear_shaft_pos_w`/`gear_quat_w` 等目前寫死用
   `factory_gear_*` 這幾個 scene entity 名稱，換物件要同步改。
5. **`mdp/rewards.py`/`mdp/terminations.py`**：邏輯本身通用，但
   `keypoint_scale`/`distance_threshold`/角度容忍門檻要依新物件尺度重調。
6. **`config/<robot>/joint_pos_env_cfg.py`**：`gear_offsets_grasp`/
   `hand_grasp_width`/`hand_close_width` 要重新標定。
7. **物理材質**：`EventCfg` 裡每個 `*_physics_material` 要換成新物件的摩擦係數
   （錄真機影片，模擬中肉眼比對，不需等完整訓練跑完）。
8. **`scripts/ros2_isaac_sim_bridge.py` 的 `_publish_episode_end`**：若母件/子件
   scene entity 名稱改了，`_obs_term_slice` 用的 `gear_pos`/`gear_shaft_pos`
   term 名稱也要同步改。

## 疑難排解

### PhysX Collision Stack Overflow

```
PhysX error: PxGpuDynamicsMemoryConfig::collisionStackSize buffer overflow detected,
please increase its size to at least 269452544 in the scene desc!
```

在 `gear_assembly_env_cfg.py` 的 `GearAssemblyEnvCfg.sim.physics`（`PhysxCfg`）調大：

```python
sim: SimulationCfg = SimulationCfg(
    physics=PhysxCfg(
        gpu_collision_stack_size=2**31,
        gpu_max_rigid_contact_count=2**23,
        gpu_max_rigid_patch_count=2**23,
    ),
)
```

### CUDA Out of Memory

依優先順序：① 降低 `--num_envs`（256→128→64…）；② 若曾調高
`solver_position_iteration_count`，調回預設 `4`；③ 拿掉 `--video`。

### `isaacsim.core.utils` ImportError

```
ModuleNotFoundError: No module named 'isaacsim.core.utils'
```

某些 Isaac Sim 版本把這個便利包裝移到 `extsDeprecated/` 底下。本專案的
`ros2_isaac_sim_bridge.py` 已改用不受版本影響的 `omni.kit.app` API（見上方
「程式碼分析」），不會再遇到這個問題；若你在別的腳本遇到類似錯誤，同樣改用
`omni.kit.app.get_app().get_extension_manager()` 即可。

## 已知限制與驗證紀錄

本專案的核心程式碼（`train/gear_insertion_policy/`、
`scripts/ros2_isaac_sim_bridge.py`、`valid/ros2_validation_cpp/`）都已在真實
環境實測過，不是只有靜態檢查：

- **環境定義**：6 個 Gym 任務、2 組 RSL-RL agent 設定，用 IsaacLab venv 靜態
  建構過（import + 實例化全部 cfg class）全數通過；並在真實 GPU（RTX 5070 Ti）
  上完整訓練一輪 UR10e 版本（800 次迭代，見「訓練結果」），Isaac Lab 自己印出
  的 Observation/Action/Termination/Reward Manager 表格跟設計完全吻合。
- **ROS2 驗證迴圈**：`valid/ros2_validation_cpp/` 在本機 ROS2 Jazzy + LibTorch
  環境 `colcon build` 成功，並用一個假 TorchScript 策略做過端到端測試（收發
  訊息、`reset()` 呼叫都正常）；`scripts/ros2_isaac_sim_bridge.py` 在使用者
  自己的 GPU 上實際跑過 50 回合，`policy_inference_node` 接上真實 checkpoint
  後，`episode_event` 正確出現 `success`/`gear_dropped`/`gear_orientation_exceeded`
  三種結果。
- **已知限制**：成功判定用「齒輪-軸心直線距離」而非完整關鍵點法，沒有考慮
  插入姿態角度；`isaacsim.ros2.bridge` extension 本身在不同 ROS2 distro 下的
  相容性、以及接上真正 FoundationPose/cuMotion 的完整 Isaac ROS 管線，仍需要
  依「Step 4：真實機器人部署」自行驗證。

## 參考資料

- [Isaac Lab 官方教學：Training a Gear Insertion Policy and ROS Deployment](https://isaac-sim.github.io/IsaacLab/release/3.0.0/source/policy_deployment/02_gear_assembly/gear_assembly_policy.html)
- [Isaac ROS Manipulation DNN Policy（真機部署管線）](https://nvidia-isaac-ros.github.io/reference_workflows/isaac_for_manipulation/packages/isaac_ros_manipulation_dnn_policy/index.html)
- rsl_rl TorchScript 匯出介面：`isaaclab_rl/isaaclab_rl/rsl_rl/exporter.py`（IsaacLab 原始碼）
