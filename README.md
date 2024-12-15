# MNIST Digit Classification Model

- **Parameters**: 7,544 (all trainable)
- **Model Size**: 0.68 MB
- **Architecture Overview**:
  - 8 Convolutional layers
  - Batch Normalization after each conv layer
  - Dropout for regularization
  - Global Average Pooling before final layer
  - No Fully Connected layers (except final prediction layer)

## Training Parameters

- **Optimizer**: SGD with Nesterov Momentum
  - Initial Learning Rate: 0.01
  - Momentum: 0.9
  - Weight Decay: 0.0001
  - Nesterov: True

- **Learning Rate Scheduler**: OneCycleLR
  - Max Learning Rate: 0.02
  - Epochs: 15
  - Pct Start: 0.2
  - Division Factor: 25
  - Final Division Factor: 1e4
  - Annealing Strategy: Cosine
  - Momentum Range: 0.85 to 0.95

- **Batch Size**: 
  - 128 (with CUDA)
  - 64 (without CUDA)

## Results

### Training Progress (Note the epoch is as is: epoch runs from 0 to 14)
- Epoch 1: 96.91% train accuracy, 98.12% test accuracy
- Epoch 5: 98.53% train accuracy, 98.89% test accuracy
- Epoch 10: 98.94% train accuracy, 99.28% test accuracy
- Epoch 14: 99.13% train accuracy, 99.44% test accuracy

### Key Achievements
- **Target Accuracy**: 99.40%
- **Achieved Accuracy**: 99.44%
- **Best Test Loss**: 0.0199
- **Convergence**: Model showed consistent improvement throughout training
- **Training Time**: ~20-35 iterations per second on GPU

### Notable Features
- Reached 96%+ accuracy in the first epoch (for both train and test)
- Crossed 99% training accuracy by epoch 6th (for test)
- Maintained stable performance without overfitting
- Achieved target accuracy with a lightweight model (<8K parameters)

## Logs:
```

 mnist_low_param_model_v2  python train.py
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 16, 26, 26]             144
              ReLU-2           [-1, 16, 26, 26]               0
       BatchNorm2d-3           [-1, 16, 26, 26]              32
           Dropout-4           [-1, 16, 26, 26]               0
            Conv2d-5           [-1, 10, 24, 24]           1,440
              ReLU-6           [-1, 10, 24, 24]               0
       BatchNorm2d-7           [-1, 10, 24, 24]              20
           Dropout-8           [-1, 10, 24, 24]               0
            Conv2d-9           [-1, 10, 24, 24]             100
        MaxPool2d-10           [-1, 10, 12, 12]               0
           Conv2d-11           [-1, 10, 10, 10]             900
             ReLU-12           [-1, 10, 10, 10]               0
      BatchNorm2d-13           [-1, 10, 10, 10]              20
          Dropout-14           [-1, 10, 10, 10]               0
           Conv2d-15             [-1, 10, 8, 8]             900
             ReLU-16             [-1, 10, 8, 8]               0
      BatchNorm2d-17             [-1, 10, 8, 8]              20
          Dropout-18             [-1, 10, 8, 8]               0
           Conv2d-19             [-1, 16, 6, 6]           1,440
             ReLU-20             [-1, 16, 6, 6]               0
      BatchNorm2d-21             [-1, 16, 6, 6]              32
          Dropout-22             [-1, 16, 6, 6]               0
           Conv2d-23             [-1, 16, 6, 6]           2,304
             ReLU-24             [-1, 16, 6, 6]               0
      BatchNorm2d-25             [-1, 16, 6, 6]              32
          Dropout-26             [-1, 16, 6, 6]               0
        AvgPool2d-27             [-1, 16, 1, 1]               0
           Conv2d-28             [-1, 10, 1, 1]             160
================================================================
Total params: 7,544
Trainable params: 7,544
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.65
Params size (MB): 0.03
Estimated Total Size (MB): 0.68
----------------------------------------------------------------
None
EPOCH: 0
Loss=0.1802 Batch_id=468 Accuracy=66.91: 100%|███████████████████████████████| 469/469 [00:13<00:00, 33.64it/s] 

Test set: Average loss: 0.2208, Accuracy: 9443/10000 (94.43%)

EPOCH: 1
Loss=0.1444 Batch_id=468 Accuracy=96.41: 100%|███████████████████████████████| 469/469 [00:13<00:00, 34.02it/s] 

Test set: Average loss: 0.0672, Accuracy: 9812/10000 (98.12%)

EPOCH: 2
Loss=0.0828 Batch_id=468 Accuracy=97.62: 100%|███████████████████████████████| 469/469 [00:13<00:00, 34.45it/s] 

Test set: Average loss: 0.0457, Accuracy: 9857/10000 (98.57%)

EPOCH: 3
Loss=0.0992 Batch_id=468 Accuracy=98.07: 100%|███████████████████████████████| 469/469 [00:13<00:00, 34.32it/s] 

Test set: Average loss: 0.0381, Accuracy: 9889/10000 (98.89%)

EPOCH: 4
Loss=0.0921 Batch_id=468 Accuracy=98.37: 100%|███████████████████████████████| 469/469 [00:15<00:00, 30.81it/s] 

Test set: Average loss: 0.0358, Accuracy: 9888/10000 (98.88%)

EPOCH: 5
Loss=0.1030 Batch_id=468 Accuracy=98.53: 100%|███████████████████████████████| 469/469 [00:12<00:00, 36.50it/s] 

Test set: Average loss: 0.0344, Accuracy: 9889/10000 (98.89%)

EPOCH: 6
Loss=0.0711 Batch_id=468 Accuracy=98.66: 100%|███████████████████████████████| 469/469 [00:13<00:00, 34.49it/s] 

Test set: Average loss: 0.0302, Accuracy: 9909/10000 (99.09%)

EPOCH: 7
Loss=0.0088 Batch_id=468 Accuracy=98.73: 100%|███████████████████████████████| 469/469 [00:13<00:00, 34.53it/s] 

Test set: Average loss: 0.0234, Accuracy: 9925/10000 (99.25%)

EPOCH: 8
Loss=0.0434 Batch_id=468 Accuracy=98.82: 100%|███████████████████████████████| 469/469 [00:14<00:00, 33.11it/s] 

Test set: Average loss: 0.0247, Accuracy: 9923/10000 (99.23%)

EPOCH: 9
Loss=0.0105 Batch_id=468 Accuracy=98.87: 100%|███████████████████████████████| 469/469 [00:12<00:00, 37.23it/s] 

Test set: Average loss: 0.0248, Accuracy: 9923/10000 (99.23%)

EPOCH: 10
Loss=0.0435 Batch_id=468 Accuracy=98.94: 100%|███████████████████████████████| 469/469 [00:15<00:00, 30.22it/s] 

Test set: Average loss: 0.0233, Accuracy: 9928/10000 (99.28%)

EPOCH: 11
Loss=0.0156 Batch_id=468 Accuracy=99.06: 100%|███████████████████████████████| 469/469 [00:13<00:00, 34.21it/s] 

Test set: Average loss: 0.0230, Accuracy: 9930/10000 (99.30%)

EPOCH: 12
Loss=0.0186 Batch_id=468 Accuracy=99.13: 100%|███████████████████████████████| 469/469 [00:13<00:00, 34.32it/s] 

Test set: Average loss: 0.0217, Accuracy: 9933/10000 (99.33%)

EPOCH: 13
Loss=0.0275 Batch_id=468 Accuracy=99.16: 100%|███████████████████████████████| 469/469 [00:13<00:00, 34.52it/s] 

Test set: Average loss: 0.0201, Accuracy: 9937/10000 (99.37%)

EPOCH: 14
Loss=0.0023 Batch_id=468 Accuracy=99.13: 100%|███████████████████████████████| 469/469 [00:12<00:00, 37.80it/s] 

Test set: Average loss: 0.0199, Accuracy: 9944/10000 (99.44%)

```
## Plots

![plots](./output.png)
