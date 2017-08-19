##Emotional Chatting Machine: Emotional Conversation Generation with Internal and External Memory
###Introduction
This project is a tensorflow implement of [Emotional Conversation Generation](https://arxiv.org/abs/1704.01074). We revise the interface of the seq2seq model on tensorflow to realize our model.
###requirements
 - Python 3.5
 - Tensorflow 1.0.1
 - Numpy

###Usage
1. **train**
```Shell
python train.py -m [model name]
```
We provide three models, including isolated emotion embedding('embedding') and internal memory('internal') and the mixed one('ECM'). You can set the batch size and the learning rate in this script.
As long as we know, no optimizers other than SGD can perform well on this model, so there is no option for optimizers.

2. **test**
```Shell
python test.py -m [model name]
```
This script will generate a pickle file named "test_result_"+model_chosen+".pkl" consisting of posts, ground truth and the generated responses.

3. **apply**
```Shell
python apply.py -m [model name] -p [model path]
```
We provide a simple application script. When running this script, you need to enter what you want to say to the machine(which should be in Chinese and segmented) and the emotion category(1 for like, 2 for sadness, 3 for disgust, 4 for anger, 5 for happiness and 0 for other), then it will return you a response. We provide three pretrained models, and you can use your own trained model, too.

###TODO
 - Add external memory to the model.
 - Speed up the application.