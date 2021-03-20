# HyperNetworks
PyTorch implementation of [HyperNetworks](https://arxiv.org/abs/1609.09106) (Ha et al., ICLR 2017) for ResNet. The code is primarily for CIFAR-10 but it's super easy to use it for any other dataset. It's also very easy to use it for ResNet architectures of different depths.

(Please cite this repository if you use any of the code/diagrams here, Thanks! 😊😊)

To cite, please use the following BibTeX entry:

```
@article{gaurav2018hypernetsgithub,
  title={HyperNetworks(Github)},
  author={{Mittal}, G.},
  howpublished = {\url{https://github.com/g1910/HyperNetworks}},
  year={2018}
}
```

## How to Run

```commandline
python train.py
```

## Working

![model_diagram](https://raw.githubusercontent.com/g1910/HyperNetworks/master/diagrams/model_diagram.png)

![model_diagram_simplified](https://raw.githubusercontent.com/g1910/HyperNetworks/master/diagrams/model_simplified.png)

![forward_backward_pass](https://raw.githubusercontent.com/g1910/HyperNetworks/master/diagrams/forward_backward_pass.png)
