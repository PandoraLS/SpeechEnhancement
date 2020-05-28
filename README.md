## 语音增强
语音增强模型的处理和实现

## 环境配置数据集准备
### 环境配置
基于Anaconda创建环境`speech_enhance`
```bash
conda create -n speechenhance python=3.7
```
```bash
conda install tensorboard
pip install -r requirement.txt
```
环境依赖
```bash
Python 3.7.x
CUDA 10.1
Pytorch 1.3
```

### 数据集
语音数据集:[TIMIT语音](https://github.com/philipperemy/timit)
训练集4620条,测试集1680条,共6300条

噪声数据集[Noisex-92](http://spib.linse.ufsc.br/noise.html)

TIMIT数据集下载之后可能无法使用,需要将原来的格式进行转换一下，转换函数`timit_trans`在`util\utils.py`中

## Usage

### Clone

```shell script
git clone git@github.com:PandoraLS/SpeechEnhancement.git
```

### Train（train.py)

使用 `train.py` 训练模型，接收如下参数：

- `-h`，显示帮助信息
- `-C`, `--config`, 指定训练相关的配置文件。它们通常被存放于`config/train/`目录下，拓展名为`json5`
- `-R`, `--resume`, 从最近一次保存的模型断点处继续训练

语法：`python train.py [-h] -C CONFIG [-R]`，例如：

```shell script
python train.py -C config/20121212_noALoss.json5
# 训练模型所用的配置文件为 config/20121212_noALoss.json5
# 默认使用所有的GPU训练。若没有GPU，则使用CPU

CUDA_VISIBLE_DEVICES=1,2  python train.py -C config/20121212_noALoss.json5
# 训练模型所用的配置文件为 config/20121212_noALoss.json5
# 使用索引为1、2的GPU训练

CUDA_VISIBLE_DEVICES=1,2  python train.py -C config/20121212_noALoss.json5 -R
# 训练模型所用的配置文件为 config/20121212_noALoss.json5
# 使用索引为1、2的GPU训练
# 从最近一次保存的模型断点处继续训练
```

补充说明：
- 一般训练所需要的配置文件都放置在`config/`目录下，配置文件拓展名为`json5`
- 配置文件中的参数作用见“参数说明”部分

### Enhancement（enhancement.py）

TODO

### Test（test.py）

TODO

### Visualization

训练过程中产生的所有日志信息都会存储在`<config["root_dir"]>/<config filename>/`目录下。这里的`<config["root_dir"]>`指配置文件中的 `root_dir`参数的值，`<config filename>`指配置文件名。

假设用于训练的配置文件为`config/train/sample_16384.json5`，`sample_16384.json`中`root_dir`参数的值为`/home/Exp/UNetGAN/`，那么训练过程中产生的日志文件会存储在 `/home/Exp/UNetGAN/sample_16384/` 目录下。该目录包含以下内容：

- `logs/`: 存储`Tensorboard`相关的数据，包含损失曲线，语音波形，语音文件等
- `checkpoints/`: 存储模型训练过程中产生的所有断点，后续可通过这些断点来重启训练或进行语音增强
- `<Date>.json`文件: 训练时使用的配置文件的备份

## 参数说明

在训练，测试与增强时都需要指定具体的配置文件，本节来说明配置文件中的参数。

### 训练参数

训练过程中产生的日志信息会存放在`<config["root_dir"]>/<config filename>/`目录下

```json5
{
    "seed": 0,  // 为Numpy，PyTorch设置随机种子，尽可能保证实验的可重复性
    "description": "描述部分",  // 实验描述，会限制在 tensorboard 的 text 面板 中
    "root_dir": "~/uestc/xiaomi_code/SpeechEnhancement",    // 项目根目录
    "cudnn_deterministic": false,   // 开启可保证实验的可重复性，但影响性能
    "trainer": {
        "epochs": 1200, // 实验进行的总轮次
        "save_checkpoint_interval": 10, // 存储模型断点的间隔
        "joint": {
            "stoi_factor": 1., // 联合损失函数的stoi_factor
            "sdr_factor": 5000., // 联合损失函数 sdr_factor
            "rmse_factor": 10. // 联合损失函数 rmse_factor
        },
        "validation": {
            "interval": 10, // 在验证集上进行验证的间隔，每做一次验证都会消耗大量的时间
            "find_max": true,
            "custom": {
                "visualize_audio_limit": 20,    // 验证时 Tensorboard 中可视化音频文件的数量
                "visualize_waveform_limit": 20, // 验证时 Tensorboard 中可视化语音波形的数量
                "visualize_spectrogram_limit": 20,  // 验证时 Tensorboard 中可视化语谱图的数量
                "sample_length": 16384  // 验证时的采样长度，与模型训练时指定的采样长度有关
            }
        }
    },
    "model": {
        "module": "model.simple_cnn", // 存放模型的文件
        "main": "BaseCNN", // 模型的类
        "args": {} // 传递给模型的参数
    },
    "loss_function": {
        "module": "util.loss",  // 在joint train中不起作用，该部分需要进一步修改
        "main": "rmse_loss", // 具体损失函数
        "args": {}  // 传给该函数的参数
    },
    "optimizer": {
        "lr": 0.0006, // 模型学习率
        "beta1": 0.9,   // Adam动量参数1
        "beta2": 0.999  // Adam动量参数2
    },
    "train_dataset": {
        "module": "dataset.waveform_dataset",
        "main": "WaveformDataset",
        "args": {
            "dataset": "~/uestc/xiaomi_code/SpeechEnhancement/dataset/train.txt",
            "limit": null,
            "offset": 0,
            "sample_length": 16384,
            "train": true
        }
    },
    "validation_dataset": {
        "module": "dataset.waveform_dataset",
        "main": "WaveformDataset",
        "args": {
            "dataset": "~/uestc/xiaomi_code/SpeechEnhancement/dataset/val.txt",
            "limit": 400,
            "offset": 0,
            "train": false
        }
    },
    "train_dataloader": {
        "batch_size": 100,
        "num_workers": 40,
        "shuffle": true,
        "pin_memory": true
    }
}
```

## 可重复性

本项目已经将可以设置随机种子的位置抽离成了可配置的参数，这保证了基本的可重复性。
如果你使用 CuDNN 作为后端，还可以进一步指定确定性参数，但这会影响性能。

本项目抽离出了部分随机种子，来尽可能保证实验的可重复性。

- CPU 训练后，GPU 接着训练，反之亦然
- 使用 N 个 GPU 训练一定 epochs 后，使用 M 个 GPU 继续训练，这里的 N 不等于 M
- 项目使用了随机上插值，目前未找到任何方法来保证这个操作的可重复性。

## Contributing

This is a open-source project. Fork the project, complete the code and send pull request.
