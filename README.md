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

### Training Progress
- Epoch 1: 96.60% train accuracy, 98.30% test accuracy
- Epoch 5: 98.49% train accuracy, 99.11% test accuracy
- Epoch 10: 98.92% train accuracy, 99.31% test accuracy
- Epoch 15: 99.21% train accuracy, 99.41% test accuracy

### Key Achievements
- **Target Accuracy**: 99.4%
- **Achieved Accuracy**: 99.41%
- **Best Test Loss**: 0.0186
- **Convergence**: Model showed consistent improvement throughout training
- **Training Time**: ~20-35 iterations per second on GPU

### Notable Features
- Reached 96%+ accuracy in the first epoch
- Crossed 99% training accuracy by epoch 11
- Maintained stable performance without overfitting
- Achieved target accuracy with a lightweight model (<8K parameters)
