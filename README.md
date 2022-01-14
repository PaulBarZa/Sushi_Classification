# Sushi classification

Machine learning model for sushi classification (50 classes).

Dataset ["Sushi-50 - A dataset from Jianing Qiu et al [28/12/2020]"](https://homepages.inf.ed.ac.uk/rbf/CVonline/Imagedbase.htm)

## Deep learning

Fine-tuning of the model - [MobileNetV2](https://arxiv.org/abs/1801.04381).

**Results:** 70% Top-1 accuracy & 93% Top-5 accuracy.

State of the art has better results but with bigger models (see this [paper](https://bmvc2019.org/wp-content/uploads/papers/0839-paper.pdf) for an example - 90% on [ResNet-101](https://arxiv.org/abs/1512.03385)).

## Deep learning as features extraction

I use my previous model as features extraction and trained two models on this features :

### Extra trees

**Results:** +2% accuracy (72% Top-1 accuracy).

### Support Vector Machine

**Results:** +5% accuracy (75% Top-1 accuracy).