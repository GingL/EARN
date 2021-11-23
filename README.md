# Evaluation for Complex Relationship Reasoning of Referring Expression Grounding

## Introduction

This repository provides a validation set to evaluate the complex relationship reasoning for Referring Expression Grounding (REG) task. 
REG aims to distinguish the target from other objects in an image, usually the same category. Since people tend to use context information to describe a particular object, relationship reasoning is vital for REG, which is also the main challenge. Under the complicated situation, the relationship can be very difficult for the REG model to learn. 
Hence, we gather the referring expressions with higher-order or multi-entity relationships (mainly based on the length of the referring expression) from the original RefCOCO, RefCOCO+ and RefCOCOg validation and test set to evaluate the ability of models to reason the complex relationship. You can download the validation set in [cache](cache/prepro/). 

## Prerequisites

* Python 3.5
* Pytorch 0.4.1
* CUDA 8.0

## Data Prepare

Please refer to [MattNet](https://github.com/lichengunc/MAttNet) to install [mask-faster-rcnn](https://github.com/lichengunc/mask-faster-rcnn), [REFER](https://github.com/lichengunc/refer) and [refer-parser2](https://github.com/lichengunc/refer-parser2).
Follow Steps 1 & 2 in Training to prepare the data.

## More Details
1) Some Examples.
   
   There are some examples for referring expressions with higher-order or multi-entity relationships. More examples can be seen in [visualization.ipynb](visualization.ipynb). 
   

![example1](./pics/example.png)
<!-- <center>Some examples of the validation set with complex relationship.</center> -->


1) Our Performance.
   
   Here we show the number and its percentage (num) of the expressions with complex relationships, and the accuracy (IoU > 0.5) comparison of the max-context pooling (mcxtp) and soft-context pooling (scxtp). The RefCOCOg dataset has longer queries, so the cases with complex relationships are much higher. From the results, we can see soft-context pooling can perform better on complex relational reasoning.
<center>
<table>
<tr><td>

|  | RefCOCO | RefCOCO+ | RefCOCOg|
|:--:|:--:|:--:|:--:|
| num   |  653 （~3\%）   | 637 (~3\%)    |  4233 (~44\%)   |
| mcxtp | 17.46\% | 20.88\% | 43.11\% |
| scxtp | 21.75\% | 21.66\% | 46.47\% |

</td></tr> 
</table>
</center>
