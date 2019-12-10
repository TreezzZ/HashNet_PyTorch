# HashNet: Deep Learning to Hash by Continuation

## REQUIREMENTS
`pip install -r requirements.txt`

1. pytorch >= 1.0
2. loguru

## DATASETS
1. [CIFAR-10](http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)
2. [NUS-WIDE](https://pan.baidu.com/s/1f9mKXE2T8XpIq8p7y8Fa6Q) Password: uhr3
3. [Imagenet100](https://pan.baidu.com/s/1Vihhd2hJ4q0FOiltPA-8_Q) Password: ynwf

## USAGE
```
usage: run.py [-h] [--dataset DATASET] [--root ROOT] [--batch-size BATCH_SIZE]
              [--lr LR] [--code-length CODE_LENGTH] [--max-iter MAX_ITER]
              [--max-epoch MAX_EPOCH] [--num-query NUM_QUERY]
              [--num-train NUM_TRAIN] [--num-workers NUM_WORKERS]
              [--topk TOPK] [--gpu GPU] [--gamma GAMMA]

ADSH_PyTorch

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     Dataset name.
  --root ROOT           Path of dataset
  --batch-size BATCH_SIZE
                        Batch size.(default: 64)
  --lr LR               Learning rate.(default: 1e-4)
  --code-length CODE_LENGTH
                        Binary hash code length.(default: 12)
  --max-iter MAX_ITER   Number of iterations.(default: 50)
  --max-epoch MAX_EPOCH
                        Number of epochs.(default: 3)
  --num-query NUM_QUERY
                        Number of query data points.(default: 1000)
  --num-train NUM_TRAIN
                        Number of training data points.(default: 2000)
  --num-workers NUM_WORKERS
                        Number of loading data threads.(default: 0)
  --topk TOPK           Calculate map of top k.(default: all)
  --gpu GPU             Using gpu.(default: False)
  --gamma GAMMA         Hyper-parameter.(default: 200)
  ```

## EXPERIMENTS
cifar10: 1000 query images, 2000 sampling images.

nus-wide: Top 21 classes, 2100 query images, 2000 sampling images.

model: Alexnet

 | | 12 bits | 24 bits | 32 bits | 48 bits 
   :-:   |  :-:    |   :-:   |   :-:   |   :-:     
cifar-10 MAP@ALL | 0.9075 | 0.9047 | 0.9116 | 0.9045
nus-wide MAP@5000 | 0.8698 | 0.9022 | 0.9079 | 0.9133
