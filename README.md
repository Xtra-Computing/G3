<p align="center">
  <a href="https://github.com/gunrock/gunrock/"><img src="https://user-images.githubusercontent.com/2193051/82186697-d54e4800-991d-11ea-9123-e0575a07bc4f.png" width="200"></a>
  <br>
</p>

## What's new
G3 is research prototype that is currently in active development. You are welcome to share ideas and contribute.

## Overview
**G<sup>**3**</sup>** is built as a programming framework for Graph Neural Network (GNN) applications that supports graph related operations natively and leverages on the computation power of Graphics Processing Units (GPU). It is named G3, which stands for Graph processing system but tailored for GNN processing on GPUs. The values of G3 can be summarized as follow:

- G3 thrives to fundamentally enhance the performance of GNN applications by incorporating graph oriented operators at the low level.
- G3 provides a set of existing basic modules that can be stacked up to build up user-customized GNN architectures.
- G3 provides ﬂexible C++ based APIs such that developers are able to deﬁne and implement their own GNN layers when such layers do not exist in our system.
- G3 frees the programmers from low level GPU thread management and memory management concerns, which is usually daunting and injects numerous unnoticeable bugs into the system.
- G3 automatically applies graph-centric optimizations on the GNN architectures during run time.

We have done some preliminary studies to show how such system can be used to build Graph Convolutional Network (GCN) , the state of the art in various GNN models, and how our implementation exceeds the performances of Tensorﬂow implementation (the original implementation) as well as PyTorch implementation by a signiﬁcant margin. Details are [here](https://www.comp.nus.edu.sg/~hebs/pub/vldb2020-G3-submission.pdf).

G3 is built based on **Gunrock**, a CUDA library for graph-processing designed specifically for the GPU.  For more details, see [Gunrock's Github repo](https://github.com/gunrock/gunrock/).

## Getting Started

### Prerequisites

### Quick Install


## How to cite **G<sup>**3**</sup>** 
If you use **G<sup>**3**</sup>**  in your paper, please cite our work ([full version](https://www.comp.nus.edu.sg/~hebs/pub/vldb2020-G3-submission.pdf)).
```
@article{husong2020g3,
 author = {Liu, Husong and Lu, Shengliang and Chen, Xinyu and He, Bingsheng},
 title = {G3: When Graph Neural Networks Meet Parallel Graph Processing Systems on GPUs},
 journal = {under submission},
 year = {2020}
}
```

## Related work
* Graph systems on GPU: [Gunrock](https://github.com/gunrock/gunrock/) | [Medusa](https://github.com/Xtra-Computing/Medusa)
* Graph neural network systems:[PyTorch Geometric (PyG)](https://github.com/rusty1s/pytorch_geometric) | [Tensorflow](https://github.com/tensorflow/tensorflow) 

