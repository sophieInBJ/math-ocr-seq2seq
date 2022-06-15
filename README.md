# 口算批改-识别模型
本方案是基于 2D-attention+Seq2Seq 的OCR识别模型，用于小学口算批改场景中的算式识别部分。（算式检测部分参看[https://github.com/sophieInBJ/math-detect](https://github.com/sophieInBJ/math-detect)）

## 环境要求
`pytorch >= 1.3 `

## 数据准备

参考示例文件`train_data/train.txt` 

```
./datasets/data_001/images/00055.jpg 3.4×0.5=1.7
./datasets/data_001/images/00022.jpg 45+2+50=97
./datasets/data_001/images/00006.jpg 18÷3+9=15
./datasets/data_001/images/00019.jpg 49÷7=7
./datasets/data_001/images/00008.jpg 0.35+0.45=0.8
./datasets/data_001/images/00009.jpg 28.56-14.76=13.8

``` 

## 训练

首先，在`cfgs.py` 文件中修改配置，设置字典集，训练参数等项目

然后，执行以下命令：

```
python train.py

```  
## 推理

```
python infer.py 

```