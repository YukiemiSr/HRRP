# TCN：BUAA原8:2幅度识别基线

模型、数据读取、训练、验证、测试都在本目录，运行不依赖父项目的`open_baselines`或其他Python模块。默认使用父目录已有BUAA缓存，不重复复制约0.83GB信号数据。目录可以单独复制到服务器，使用`--data-dir`指定缓存。

```text
TCN/
├── model.py                 # TCNClassifier，骨干＋保留距离位置的分类头
├── tcn.py                   # 固定版本MIT原作者TCN骨干
├── train.py                 # 从头训练，验证CE选best，训练结束测试best/last
├── val.py                   # 独立验证入口＋公共evaluate函数
├── test.py                  # 独立测试入口，不选checkpoint
├── data.py                  # 原固定划分、索引检查、resident batch
├── dataset.py               # 原BUAA缓存读取、原始.dat预处理实现
├── prepare_data.py          # 可选的原始.dat→L2缓存入口
├── utils.py                 # 指标、配置、随机种子、checkpoint加载
├── config.json              # 原100轮/seed56配置及冻结数据/划分哈希
├── requirements.txt         # 仅NumPy和PyTorch
├── checkpoints/
│   ├── best.pt              # 原8:2 seed56、第99轮、Test Acc 85.82374%
│   └── original_result.json # 原结果，含10类指标和混淆矩阵
├── splits/split_indices.npz # 原train/validation/test逐条索引
├── licenses/TCN_LICENSE     # MIT原作者许可
├── provenance.json          # 来源版本及打包文件哈希
└── tests/test_package.py    # 划分/权重/训练顺序/独立目录运行检查
```

## 数据与原实验口径

BUAA internal共258403条、10类，每条800个距离单元，已逐条L2归一化。模型只输入幅度，不使用yaw、pitch或FiLM。

“8:2”指先分80%训练候选池和20%测试，再从训练池内留10%验证，即实际约72%/8%/20%。

| 集合 | 记录数 |
|---|---:|
| train | 186041 |
| validation | 20677 |
| test | 51685 |

固定split seed56；训练seed默认56。包内split为原实验逐条索引，并同时校验文件SHA256、索引hash、完整覆盖及不重叠。不能将随机记录划分的结果解释为跨文件/航迹或跨姿态泛化。

TCN为6个32通道TemporalBlock，每块两次因果膨胀Conv1d，kernel7，dilation1/2/4/8/16/32，dropout0.1，weight_norm；分类头Conv1d32→8＋Flatten＋Linear6400→10，共144178参数。保留原参数键名，严格加载原checkpoint，不转换权重。

## 环境与路径

```bash
cd TCN
python -m pip install -r requirements.txt
```

已有PyTorch和NumPy时直接运行即可。`config.json`中的相对路径以配置文件所在目录为基准，不随终端工作目录变化；命令行`--data-dir`、`--checkpoint`、`--output`相对当前工作目录。

默认缓存：`../dataset/BUAA/cache/internal_l2`。复制到服务器后可覆盖：

```bash
python data.py --data-dir /path/to/BUAA/cache/internal_l2
```

该命令核验数据、划分，并重新计算历史划分检查逐条一致，不训练。

## 训练

先做实际数据小规模流程检查（每类每集合2条，各20条；2轮，不代表正式性能）：

```bash
python train.py --smoke --device cpu
```

默认输出`runs/smoke_seed_56/`。正式从头训练：

```bash
CUDA_VISIBLE_DEVICES=1 python -u train.py \
  --device cuda \
  --data-dir /path/to/BUAA/cache/internal_l2 \
  --output runs/random82_seed56
```

100epochs、batch64、AdamW lr0.0005、weight_decay0.0001、clip5、无scheduler或增强。`--seed`只改变训练随机性，不改变固定划分。调试可传`--epochs`、`--batch-size`、`--threads`；其实际值写入resolved_config。已有输出目录会拒绝覆盖；换新目录运行，不支持断点续训。

每轮只前向validation，按最低CE保存best；100轮结束后才评估last和best的test。训练会生成`best.pt`、`last.pt`、`history.json`、`result.json`、`resolved_config.json`、`split_indices.npz`及best/last逐样本测试预测。

## 独立验证与测试

直接使用随目录附带的原best checkpoint（CPU/GPU均可）：

```bash
python val.py --device cpu
python test.py --device cpu
```

默认保存`evaluations/validation_best.json`、`evaluations/test_best.json`及对应`*_predictions.npz`。JSON包含Acc、macro_accuracy（Balanced Acc）、macroF1、每类指标及混淆矩阵，NPZ包含原索引、标签与预测。已有结果拒绝覆盖，可指定新的`--output`。

评估自己训练的checkpoint：

```bash
python val.py --checkpoint runs/random82_seed56/best.pt \
  --device cuda --output evaluations/new_validation.json
python test.py --checkpoint runs/random82_seed56/best.pt \
  --device cuda --output evaluations/new_test.json
```

在服务器运行这些命令也可追加`--data-dir`。快速验证入口用`--smoke`，只评估每类2条，不应作为正式成绩。实际准确率可能受PyTorch版本、设备及数值舍入影响，应以完整测试输出为准。

## 在代码中加载

在本目录运行：

```python
import torch
from model import TCNClassifier

checkpoint = torch.load("checkpoints/best.pt", map_location="cpu", weights_only=False)
model = TCNClassifier(input_size=800, num_classes=10)
model.load_state_dict(checkpoint["model_state_dict"], strict=True)
model.eval()
# 输入shape [batch, 1, 800]，须按样本L2归一化；输出10类logits。
```

在父项目也可`from TCN.model import TCNClassifier`。分类索引按raw label排序：2,3,4,5,9,13,14,15,20,81，分别为A319、A320、A321、A330、BY737、BY777、BY787、C909、C919、ATR-72。

## 原始.dat预处理（已有缓存时跳过）

```bash
python prepare_data.py --raw-dir /path/to/internal_data \
  --labels /path/to/raw_label_to_name.json --output /path/to/new_empty_cache
```

`labels`格式为`{"2":"A319", ...}`。新缓存若来自不同记录/顺序/文件时间，源签名会变化；默认配置有意锁定原实验，不能只改路径就将新数据套到旧测试索引。需要建立新实验并核验配置和划分。默认原BUAA缓存不受此操作影响。

## 检查

```bash
python -m unittest discover -s tests -v
```

测试包含复制本目录到独立临时目录、清空PYTHONPATH后，实际执行train/val/test的CPU smoke，确认无父项目代码依赖。测试可通过`TCN_DATA_DIR`环境变量指定现有原BUAA缓存。

来源：TCN骨干采用locuslab/TCN，commit `2f8c2b817050206397458dfd1f5a25ce8a32fe65`，MIT。`tcn.py`和原BUAA loader保持源文件逐字一致；其余为原项目纯幅度基线的独立整理。原实验所有代码和结果保留。
