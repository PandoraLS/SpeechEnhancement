## 语音增强
语音增强模型的处理和实现

## 环境配置数据集准备
### 环境配置
基于Anaconda创建环境`speechenhance`
```bash
conda create -n speechenhance python=3.7
```
```bash
pip install -r requirement.txt
```

### 数据集
语音数据集:[TIMIT语音](https://github.com/philipperemy/timit)
训练集4620条,测试集1680条,共6300条
噪声数据集[Noisex-92](http://spib.linse.ufsc.br/noise.html)
TIMIT数据集下载之后可能无法使用,需要将原来的格式进行转换一下
```bash
python dataset/TIMIT_trans.py
```



## 数据预处理
TIMIT解压之后可能需要转换一下才能使用

## 模型构建

## 模型训练与测试


### 持续更新...


代办事项:
- 添加tensorboard可视化训练过程
