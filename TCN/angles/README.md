# TCN 角度泛化：逐箱 1:1 与 yaw/pitch FiLM

本目录独立于原随机 8:2 和旧 3:1 角度实验，使用 `TCN/` 内的骨干、数据读取及指标函数，无父项目其他模块依赖。正式运行从头初始化，不加载历史 checkpoint。

## 实验矩阵

| yaw 箱宽 | `none` | `yaw` | `yaw_pitch` |
|---|---|---|---|
| 5° | 普通 TCN | TCN＋yaw FiLM | TCN＋yaw/pitch FiLM |
| 10° | 普通 TCN | TCN＋yaw FiLM | TCN＋yaw/pitch FiLM |
| 15° | 普通 TCN | TCN＋yaw FiLM | TCN＋yaw/pitch FiLM |

默认 9 个实验：6 个 FiLM 实验与 3 个普通 TCN 对照。只做 6 个时传 `--conditions yaw yaw_pitch`。每种宽度三个模型共用逐条一致的冻结划分。

## 数据与划分

BUAA internal：258403 条，10 类，每条 800 个距离单元，逐条 L2 归一化。yaw 沿用缓存中的相对方位角，范围 [0°,180°]；零起点箱编号为 0、1、2……，偶数箱进入训练候选池，奇数箱全量测试。所有区间左闭右开，最后一箱包含 180°。

| 箱宽 | 训练候选区间示例 | 测试区间示例 | train | validation | test |
|---|---|---|---:|---:|---:|
| 5° | 0–5，10–15，20–25…… | 5–10，15–20，25–30…… | 115533 | 12842 | 130028 |
| 10° | 0–10，20–30，40–50…… | 10–20，30–40，50–60…… | 116063 | 12902 | 129438 |
| 15° | 0–15，30–45，60–75…… | 15–30，45–60，75–90…… | 112674 | 12523 | 133206 |

1:1 指角度箱数量，不强制样本数量相等。训练池按类别留 10% 验证，使用 seed1056；各类训练条数向下取整。所有集合均有全部 10 类。固定索引、索引 hash、文件 SHA256 和逐类逐箱计数位于 `splits/yaw{5,10,15}/`，运行时重新生成索引并逐条核对。

不设置角度缓冲带，训练/测试可能共享源文件和航迹。这是当前 yaw 区间留出设置。yaw/pitch 联合条件实验仍复用相同 yaw 划分，回答加入 pitch 是否有助于跨 yaw 识别；不据此声称跨未见 pitch 泛化。不同箱宽使用不同测试样本，直接模型比较应在同一箱宽内进行。

pitch 直接采用 BUAA 缓存字段，当前范围约 0–32.24°，其物理含义及类别相关性沿用原数据。角度模型在训练、验证、测试、部署时都需要对应真实角度。类别角度覆盖并不均匀，计数 CSV 可用于检查角度与类别的相关性。

## 模型

保留原 6 层 TCN TemporalBlock：每块两个因果 Conv1d，32 通道，kernel7，dilation1/2/4/8/16/32，dropout0.1，weight_norm。分类头为 Conv1d32→8、Flatten、Linear6400→10，保留距离位置。

- `none`：不使用角度；144178 参数。
- `yaw`：条件 `[sin(yaw), cos(yaw)]`；每块独立 MLP2→32→64、SiLU；157426 参数。
- `yaw_pitch`：条件 `[sin(yaw), cos(yaw), sin(pitch), cos(pitch)]`；每块独立 MLP4→32→64、SiLU；157810 参数。

角度以度输入，在网络中转换为弧度。MLP 输出分为各 32 维 gamma、beta，作用于该块输出所有距离位置：`h'=(1+gamma)*h+beta`。最后线性层权重和偏置初始化为零，因此初始 FiLM 为恒等映射。相同 seed 下三个模型共享相同的骨干和分类头初始权重；额外 FiLM 层的初始化会消耗随机数，训练中的 dropout 随机轨迹不要求逐位相同。

## 训练与选优

旧服务器运行库 PyTorch1.12/cuDNN7 在 `cudnn.deterministic=True` 时会让本模型的膨胀卷积回退到很慢的实现。新服务器正式重跑应统一传 `--cudnn-mode fast`：开启cuDNN、关闭其deterministic限制、保持benchmark=False，并保留seed56。该开关在随机种子设置之后生效，所有实际后端/精度标志写入运行和checkpoint的provenance。fast模式不承诺逐位复现；模型结构、精度设置、划分和选优规则不变。`--cudnn-mode strict`保留历史模式，也是未指定该参数时的兼容默认值。独立val/test会恢复checkpoint记录的模式；旧checkpoint默认为strict。

统一 seed56，100 epochs，batch64，AdamW lr5e-4、weight_decay1e-4、gradient_clip5。无学习率调度、AMP、增强、均衡采样、蒸馏或额外损失；普通随机无放回遍历训练集，仅分类交叉熵。每轮只计算 validation，最低 validation CE 选 best，等全部训练结束后才计算 last/best test。

默认拒绝覆盖已有运行目录，不实现断点续训。`--smoke` 每类每集合 2 条、默认两轮，用于代码检查；独立评估 smoke checkpoint 时自动保持 smoke 标识和 20 条集合，不能误记为正式结果。

## 环境与数据路径

```bash
cd HRRP
python -m pip install -r TCN/requirements.txt
```

需要提供包含逐条yaw和pitch元数据的BUAA internal L2缓存。下列命令可用 `--data-dir /path/to/BUAA/cache/internal_l2` 指定位置；配置内相对路径以各配置文件目录为基准，命令行路径相对当前工作目录。checkpoint读取兼容旧版和新版PyTorch。

## 检查固定划分

```bash
python -m TCN.angles.split --config TCN/angles/configs/yaw5.json
python -m TCN.angles.split --config TCN/angles/configs/yaw10.json
python -m TCN.angles.split --config TCN/angles/configs/yaw15.json
```

默认数据 `HRRP/dataset/BUAA/cache/internal_l2`。数据放在别处时各命令可添加 `--data-dir /path/to/BUAA/cache/internal_l2`。

## 单个实验

```bash
# 小规模 GPU 流程检查；换新的 output 避免覆盖。
CUDA_VISIBLE_DEVICES=0 python -m TCN.angles.train \
  --config TCN/angles/configs/yaw5.json --conditioning yaw_pitch \
  --smoke --device cuda --output TCN/angles/runs/example_smoke

# 正式 5°、联合 yaw/pitch FiLM。
CUDA_VISIBLE_DEVICES=0 python -u -m TCN.angles.train \
  --config TCN/angles/configs/yaw5.json --conditioning yaw_pitch \
  --device cuda --output TCN/angles/runs/example_formal
```

## 全部实验

```bash
# 先列出全部命令，不创建运行目录。
bash TCN/angles/run.sh --gpus 0 1 2 \
  --output-root TCN/angles/runs/yaw11_seed56 --dry-run

# 9 组正式实验，最多三组同时运行，每张指定 GPU 最多一组。
bash TCN/angles/run.sh --gpus 0 1 2 \
  --output-root TCN/angles/runs/yaw11_seed56
```

指定 9 张卡即可同时跑 9 组，例如 `--gpus 0 1 2 3 4 5 6 7 8`。仅需 6 个 FiLM 实验加 `--conditions yaw yaw_pitch`；限定箱宽加 `--widths 5 15`。使用 `--cpu --smoke` 可串行检查全部 9 组。

后台启动示例：

```bash
mkdir -p logs
nohup bash TCN/angles/run.sh --gpus 0 1 2 \
  --output-root TCN/angles/runs/yaw11_seed56 \
  > logs/yaw11_seed56.log 2>&1 &
```

各组日志在输出根目录 `logs/`，启动状态在 `launch_manifest.json`。有一组失败后停止启动后续组，让已启动的组正常结束，并返回非零退出码。Ctrl-C 仅终止本次启动的子进程。

## 独立 val/test

```bash
python -m TCN.angles.val --config TCN/angles/configs/yaw5.json \
  --checkpoint TCN/angles/runs/example_formal/best.pt \
  --device cuda --output TCN/angles/evaluations/example_val

python -m TCN.angles.test --config TCN/angles/configs/yaw5.json \
  --checkpoint TCN/angles/runs/example_formal/best.pt \
  --device cuda --output TCN/angles/evaluations/example_test
```

模型条件类型从 checkpoint 读取，严格检查源签名、类别映射、箱宽、协议、固定划分及参数键名。每次评估需指定新的输出目录。

## 输出

磁盘空间紧张时，单个训练或矩阵启动器均可添加 `--compact`。训练和选优不变；仅保存一个不断覆盖更新的 `best.pt`（模型权重和来源信息，不含优化器），不保存 `last.pt`。最后一轮模型在内存中评估，仍保存 last 的完整指标、逐类/逐箱和混淆矩阵；best 额外保存逐样本预测。NPZ 中索引用无损 uint32，类别与预测用无损 uint8，yaw/pitch 可用索引从共享原缓存取回。每轮 validation history 仅保留总体指标，不重复逐类/混淆矩阵；输出体积写入 `storage_summary.json`。

九卡正式运行示例：

```bash
bash TCN/angles/run.sh --gpus 0 1 2 3 4 5 6 7 8 --compact --cudnn-mode fast \
  --output-root TCN/angles/runs/yaw11_seed56_compact_fast
```

该模式的 best 权重可直接使用原独立 val/test 入口。下面列出的完整保存格式适用于未加 `--compact` 的运行。

每组保存 `best.pt`、`last.pt`、`history.json`、`resolved_config.json`、实际 `split_indices.npz`、`result.json`。配置记录实际运行版本、条件类型、初始化 hash、共享骨干初始化 hash、源文件 hash、实际与正式划分 hash。

best/last test 及独立评估均保存：JSON 指标（Acc、Balanced Acc、Macro-F1、逐类指标、混淆矩阵、逐测试 yaw 箱准确率）；NPZ 逐条索引、标签、预测、yaw、pitch；逐类 CSV、逐箱 CSV 和混淆矩阵 CSV。空 yaw 箱准确率为空值，不算成 0；逐箱报告只计算该集合对应的箱。

## 测试

在项目根目录运行：

```bash
python -m unittest TCN.angles.tests.test_angles -v
```

测试覆盖区间边界、三份冻结划分、无泄漏与覆盖、跨环境稳定 NPZ、FiLM 恒等初始化/参数量/可训练性/角度敏感性、数据对齐、验证集选优顺序、拒绝覆盖与不匹配 checkpoint、独立评估。

TCN 骨干来源与 MIT 许可见上级目录。FiLM 机制参考 Perez 等的 [FiLM: Visual Reasoning with a General Conditioning Layer](https://arxiv.org/abs/1709.07871)；当前 yaw/pitch 接入及实验划分为本项目实现。
