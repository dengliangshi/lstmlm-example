## 语言建摸示例 - LSTM语言模型

神经网络语言模型的实现示例，采用Python语言，Tensorflow框架，长短期记忆（Long Short Term Memory, LSTM）循环神经网络（Recurrent Neural Network, RNN）的神经网络结构，实现了基本神经网络语言模型。通过该示例试图说明神经网络语言模型实现的细节。

## 使用说明
相关的Python依赖包通过pipenv进行安装和管理，下载模型后，在模型目录下，通过`pipenv install`命令安装依赖包，然后`pipenv shell`进入虚拟环境下，`python main.py`便可直接运行模型。模型的参数调整，通过`python main.py --help`命令进行查看。