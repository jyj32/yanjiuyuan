# 优先级网络第一版本部署包

版本：完整优先级排序（`full_ranking`）

用途：对一个 RGB-D 场景中的全部瓶子候选输出连续 `priority_score`，并按分数从高到低给出完整抓取优先级。预测第 1 名作为最先抓取目标。

## 1. 边界

- 本模型只输出优先级，不负责检测瓶子。
- 本模型不加载、不修改第一个网络的权重，也不与第一个网络共享参数。
- 部署时可由第一个网络或人工标注提供 OBB、四关键点和遮挡状态；接口来源不影响两个网络在参数上的解耦。
- 不需要瓶子实例分割掩码。
- `best.pt` 是优先级网络自身训练得到的权重。
- `yolo11s.pt` 只是构造优先级网络双 Backbone 所需的官方通用初始化文件，不是第一个网络的权重。

## 2. 包内文件

- `best.pt`：第一版本最佳权重，最佳周期 17。
- `yolo11s.pt`：官方 YOLO11s 架构初始化文件。
- `priority_model.py`：双 RGB/深度 Backbone、融合 Neck、旋转 ROI 和 Priority Head。
- `priority_data.py`：与训练完全一致的 letterbox、深度归一化和 18 维几何编码。
- `infer_priority.py`：直接部署推理入口。
- `verify_bundle.py`：哈希、checkpoint 结构和严格参数加载自检。
- `deploy_config.json`：图像尺寸、深度范围和版本模式。
- `bundle_manifest.json`：模型身份、哈希和测试指标。
- `input_example.json`：输入字段示例，不包含真实图像和深度文件。
- `training_metrics.json`：正式训练结果摘要。

## 3. 环境

建议使用带 NVIDIA GPU 的 Python 3.10/3.11 环境。先按目标机器的 CUDA 版本安装 PyTorch，再安装其余依赖：

```bash
pip install -r requirements.txt
```

训练机已验证组合：PyTorch `2.12.0.dev+cu130`、Ultralytics `8.3.39`、OpenCV `4.13.0`、NumPy `1.26.4`。不要求部署机使用完全相同的 PyTorch 开发版，但 CUDA 与 PyTorch 必须匹配。

## 4. 解压后先自检

在本目录执行：

```bash
python verify_bundle.py
```

只有看到 `"status": "ok"` 和 `"strict_load": true` 才表示权重、版本和网络结构全部一致。

## 5. 输入格式

`infer_priority.py` 支持：

1. 单个场景 JSON；
2. 场景数组 JSON；
3. `{"scenes": [...]}`；
4. 每行一个场景的 JSONL。

每个场景必须包含：

- `scene_id`：场景唯一编号；
- `rgb_path`：无损 RGB 图像路径；
- `depth_path`：与 RGB 同尺寸、逐像素对齐的二维 `.npy` 深度，单位必须是毫米；
- `objects`：画面中全部瓶子候选。

每个候选必须包含：

- `instance_id`：场景内唯一编号；
- `obb_corners_px`：4 个 OBB 顶点的原 RGB 像素坐标，顶点顺序必须与训练标注接口一致；
- `keypoints`：固定名称 `C/N/L/B`，每点包含 `xy_px` 和 `point_visible`；
- `occlusion_state`：`0` 表示 clear，`1` 表示 occluded。

部署输入不需要人工 `priority_rank`。如果存在该字段，推理也不会把它作为模型输入。深度值 `0`、NaN 或无穷会被视为无效；模型根据完整深度图动态生成有效性通道，并通过 OBB 在融合特征上采样，因此不要只保存瓶子裁剪深度。

## 6. 推理

```bash
python infer_priority.py \
  --input your_scene.json \
  --output priority_result.json \
  --device cuda:0
```

没有可用 GPU 时可指定 `--device cpu`，但速度会明显下降。`--device auto` 会优先使用 `cuda:0`。

输出中的关键字段：

- `selected_instance_id`：预测最先抓取的候选；
- `priority_order`：所有候选的完整降序；
- `predicted_priority_rank`：预测名次；
- `priority_score`：Priority Head 连续原始分数，只在同一模型和同一场景内比较，不应当作概率。

## 7. 已验证结果与限制

独立测试集 30 个场景：

- Top-1 accuracy：73.33%；
- Pairwise accuracy：80.35%；
- MRR：83.56%。

该版本是两个版本中的主部署建议。它学习的是当前拍摄系统和人工偏好下的相对排序；换相机、深度单位、关键点定义、OBB 顶点顺序或操作员偏好后，需要重新验证，不能只凭模型能运行就判定抓取策略有效。
