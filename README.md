# 3DHumanGAN: 3D-Aware Human Image Generation with 3D Pose Mapping
<img src="./assets/teaser.jpg" width="96%" height="96%">

[Zhuoqian Yang](https://yzhq97.github.io/), [Shikai Li](mailto:lishikai@sensetime.com), [Wayne Wu](https://wywu.github.io/), [Bo Dai](https://daibo.info/) <br>
**[[Video Demo]](https://www.youtube.com/watch?v=-bUNfhNYj24)** | **[[Project Page]](https://3dhumangan.github.io/)** | **[[Paper]](https://arxiv.org/abs/2212.07378)**

**Abstract:** *We present 3DHumanGAN, a 3D-aware generative adversarial network (GAN) that synthesizes images of full-body humans with consistent appearances under different view-angles and body-poses. To tackle the representational and computational challenges in synthesizing the articulated structure of human bodies, we propose a novel generator architecture in which a 2D convolutional backbone is modulated by a 3D pose mapping network. The 3D pose mapping network is formulated as a renderable implicit function conditioned on a posed 3D human mesh. This design has several merits: i) it allows us to harness the power of 2D GANs to generate photo-realistic images; ii) it generates consistent images under varying view-angles and specifiable poses; iii) the model can benefit from the 3D human prior. Our model is adversarially learned from a collection of web images needless of manual annotation.* <br>

## Getting Started
Please see `doc/INSTALL.md` for setting up the project environment.
Please see `doc/GET_STARTED.md` for an inference tutorial.


## TODOs
- [x] Release technical report.
- [x] Release code and pretrained models for training and inference.
- [ ] Release preprocessed train-ready dataset.
- [ ] Add instructions and scripts for data preprocessing.
- [ ] Add instructions and code for evaluation.

## Related Work
* (ICCV 2023) **OrthoPlanes: A Novel Representation for Better 3D-Awareness of GANs**, Honglin He et al. [[Paper](https://arxiv.org/abs/2309.15830)], [[Project Page](https://orthoplanes.github.io/)]
* (ECCV 2022) **StyleGAN-Human: A Data-Centric Odyssey of Human Generation**, Jianglin Fu et al. [[Paper](https://arxiv.org/pdf/2204.11823.pdf)], [[Project Page](https://stylegan-human.github.io/)], [[Dataset](https://github.com/stylegan-human/StyleGAN-Human)]

## Citation

If you find this work useful for your research, please consider citing our paper:

```bibtex
@inproceedings{yang20233dhumangan,
  title={3DHumanGAN: 3D-Aware Human Image Generation with 3D Pose Mapping},
  author={Yang, Zhuoqian and Li, Shikai and Wu, Wayne and Dai, Bo},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={23008--23019},
  year={2023}
}
```

