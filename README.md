# MobileNetV2
TensorFlow implementation for MobileNetV2

According to the paper: [Inverted Residuals and Linear Bottlenecks: Mobile Networks for
Classification, Detection and Segmentation](https://arxiv.org/pdf/1801.04381.pdf)

## Testing

Trained with [TensorFlow-Slim image classification model library](https://github.com/tensorflow/models/tree/master/research/slim) using 2 GTX 1080 Ti.

| Dataset | Model        | GPUs | Sync gradients | Training time | Recall 5 | Accuracy |
| ------- | ------------ | ---- | -------------- | ------------- | -------- | -------- |
| Cifar10 | MobileNet V2 | 2    | False          | 3:50 hours    | 0.9967   | 0.9071   |
| Cifar10 | MobileNet V2 | 2    | False          | 15 hours      | 0.9976   | 0.9315   |

Added the trained checkpoint under 'pretrained' folder.
