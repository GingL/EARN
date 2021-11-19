# PyTorch Implementation of KARN

## Introduction

This repository will be Pytorch implementation of **Knowledge-guided Adaptively Reconstruction Network for Weakly Supervised Referring Expression Grounding**.


## Prerequisites

* Python 3.5
* Pytorch 0.4.1
* CUDA 8.0

## Data Prepare

   Please refer to [MattNet](https://github.com/lichengunc/MAttNet) to install [mask-faster-rcnn](https://github.com/lichengunc/mask-faster-rcnn), [REFER](https://github.com/lichengunc/refer) and [refer-parser2](https://github.com/lichengunc/refer-parser2).
   Follow Step 1 & 2 in Training to prepare the data and features.

## Performance on complex relationship
1) Examples


2) Performance
<table>
<tr><td>

| exp_id | RefCOCO | RefCOCO+ | RefCOCOg|
|--|--|--|--|
| mcxtp | 17.46\% | 20.88\% | 43.11\% |
| scxtp | 21.75\% | 21.66\% | 46.47\% |
| num   |  653    | 637     |  4233   |

</td></tr> 
</table>



