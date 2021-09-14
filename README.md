# Amazon Last-Mile Challenge

Code maintainer： Wentao Zhao (wz2543@columbia.edu)
Code contributors: Chen Jing, Yunhao Xing, Wentao Zhao

This code is for the [Amazon last-mile research challenge](https://routingchallenge.mit.edu/). 


We realized an attention model in the Transformer architecture for route optimization problems under realistically sized problem instances. 
The training is done via reinforcement learning and imitation learning.


## Dependencies

* Python>=3.8
* NumPy
* SciPy
* PyTorch
* torch-geometric
* tqdm

## Usage
```
python src/train.py
```

## Acknowledgements
This repository is built based on the following repositories:
* https://github.com/wouterkool/attention-learn-to-route
* https://github.com/Hanjun-Dai/graph_comb_opt

For more details, please see the papers [Attention, Learn to Solve Routing Problems!](https://arxiv.org/abs/1803.08475), 
[Learning Combinatorial Optimization Algorithms over Graphs](https://arxiv.org/abs/1704.01665).
```
@article{kool2018attention,
  title={Attention, learn to solve routing problems!},
  author={Kool, Wouter and Van Hoof, Herke and Welling, Max},
  journal={arXiv preprint arXiv:1803.08475},
  year={2018}
}
``` 
```
@article{dai2017learning,
  title={Learning combinatorial optimization algorithms over graphs},
  author={Dai, Hanjun and Khalil, Elias B and Zhang, Yuyu and Dilkina, Bistra and Song, Le},
  journal={arXiv preprint arXiv:1704.01665},
  year={2017}
}
```
