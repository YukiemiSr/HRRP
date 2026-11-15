# HRRP

HRRP（高分辨距离像）识别实验代码。包含BUAA纯幅度TCN基线和基于yaw/pitch条件的TCN＋FiLM角度泛化实验。

## TCN

模型、训练、验证、测试和数据处理集中在[`TCN/`](TCN/README.md)，无需原项目其它Python模块。包含原随机8:2实验的`seed56`最佳checkpoint及固定划分索引。

```bash
git clone https://github.com/YukiemiSr/HRRP.git
cd HRRP/TCN
python -m pip install -r requirements.txt

# 指定原BUAA缓存；小规模流程检查，不代表正式成绩
python train.py --data-dir /path/to/BUAA/cache/internal_l2 --smoke --device cpu

# 使用随仓库附带的原checkpoint测试
python test.py --data-dir /path/to/BUAA/cache/internal_l2 --device cpu
```

完整训练、验证、测试和数据格式见[TCN运行说明](TCN/README.md)。原始BUAA数据不随仓库分发，需要自行提供缓存或原始数据。

## TCN＋FiLM角度泛化

代码、5°/10°/15°固定yaw划分、配置和测试位于[`TCN/angles/`](TCN/angles/README.md)。每个箱宽比较普通TCN、yaw FiLM和yaw/pitch FiLM，三个条件共用同一划分；模型均从头训练。FiLM在每个TCN块后使用真实角度调制特征，训练和推理均需逐条角度元数据。

```bash
# 在仓库根目录执行；指定本机BUAA缓存
python -m TCN.angles.train --config TCN/angles/configs/yaw5.json \
  --conditioning yaw_pitch --data-dir /path/to/BUAA/cache/internal_l2 \
  --smoke --device cpu --output TCN/angles/runs/smoke_yaw5
```

完整实验矩阵、正式训练、独立评估及划分核验见[角度泛化说明](TCN/angles/README.md)。`--smoke`只检查流程，不代表正式成绩。角度实验的运行输出、数据缓存和训练checkpoint不随本仓库发布。

## 原基线

- 输入：单条800-bin HRRP，逐条L2归一化；10类，无yaw/pitch输入。
- 原随机8:2划分：80%训练候选池内再留10%验证，实际train/validation/test为186041/20677/51685条。
- 100轮、seed56，按验证CE最低选best；第99轮best在原test上的Acc为85.82374%。
- 该记录随机划分结果不能解释为跨文件、跨航迹或未见姿态泛化。
- 目录整理已通过4项单元/集成检查，完整test逐条预测与原结果一致。

## 来源

TCN骨干来自[locuslab/TCN](https://github.com/locuslab/TCN)，固定commit `2f8c2b817050206397458dfd1f5a25ce8a32fe65`，原MIT许可证保留于[`TCN/licenses/TCN_LICENSE`](TCN/licenses/TCN_LICENSE)。checkpoint、划分和文件哈希见[`TCN/provenance.json`](TCN/provenance.json)。
