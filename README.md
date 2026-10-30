# HRRP

HRRP（高分辨距离像）识别实验代码。目前包含独立的BUAA纯幅度TCN基线。

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

## 原基线

- 输入：单条800-bin HRRP，逐条L2归一化；10类，无yaw/pitch输入。
- 原随机8:2划分：80%训练候选池内再留10%验证，实际train/validation/test为186041/20677/51685条。
- 100轮、seed56，按验证CE最低选best；第99轮best在原test上的Acc为85.82374%。
- 该记录随机划分结果不能解释为跨文件、跨航迹或未见姿态泛化。
- 目录整理已通过4项单元/集成检查，完整test逐条预测与原结果一致。

## 来源

TCN骨干来自[locuslab/TCN](https://github.com/locuslab/TCN)，固定commit `2f8c2b817050206397458dfd1f5a25ce8a32fe65`，原MIT许可证保留于[`TCN/licenses/TCN_LICENSE`](TCN/licenses/TCN_LICENSE)。checkpoint、划分和文件哈希见[`TCN/provenance.json`](TCN/provenance.json)。
