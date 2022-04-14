# This is the PyTorch Implementation of EARN

## Introduction
This repository is Pytorch implementation of Entity-enhanced Adaptive Reconstruction Network for Weakly Supervised Referring Expression Grounding.

## Prerequisites

* Python 3.5
* Pytorch 0.4.1
* CUDA 8.0

## Data Prepare

Download the images from [MSCOCO](http://mscoco.org/dataset/#overview). Prepare the refcoco/refcoco+/refcocog annotations following [REFER](https://github.com/lichengunc/refer) API. Follow Steps 1 in [MattNet](https://github.com/lichengunc/MAttNet) training to get the index of training and evaluation data.

## Evaluation for Complex Relational Reasoning of Referring Expression Grounding
We gather the referring expressions with higher-order or multi-entity relationships (mainly based on the length of the referring expression and the number of entities) from the original RefCOCO, RefCOCO+ and RefCOCOg validation and test set to evaluate the ability of models to reason the complex relationship. You can download the validation set in [cache/prepro/](cache/prepro/). 

1) Examples.
   
   There are some examples for referring expressions with higher-order or multi-entity relationships. More examples can be seen in [visualization.ipynb](visualization.ipynb). 
   
 ![example1](./pics/example1.png)
 ![example2](./pics/example2.png)
 ![example3](./pics/example3.png)
<!-- <center>Some examples of the validation set with complex relationship.</center> -->


2) Our performance.
   
   Here we show the number (num) and its percentage (ratio) of the expressions with complex relationships in original validation and test set, and the accuracy (IoU > 0.5) comparison of the max-context pooling (mcxtp) and soft-context pooling (scxtp). The RefCOCOg dataset has longer queries, so the cases with complex relationships are much higher. From the results, we can see soft-context pooling can perform better on complex relational reasoning.
<center>
<table>
<tr><td>

|  | RefCOCO | RefCOCO+ | RefCOCOg|
|:--:|:--:|:--:|:--:|
| num   |  653    | 637     |  4233   |
| ratio | ~3\%    | ~3\%    |  ~44\%  |
| mcxtp | 17.46\% | 20.88\% | 43.11\% |
| scxtp | 21.75\% | 21.66\% | 46.47\% |

<!-- | num   |  653 （~3\%）   | 637 (~3\%)    |  4233 (~44\%)   |
| num   |  653 （~3\%）   | 637 (~3\%)    |  4233 (~44\%)   | -->
</td></tr> 
</table>
</center>
