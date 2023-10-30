# Get Started

## Inference with pretrained model

- Download the model from [here](https://huggingface.co/inrainbws/3DHumanGAN-SHHQ-512x256)
- Extract and put under `./checkpoints`
- Download example dataset from [here](https://huggingface.co/datasets/inrainbws/SHHQ-example-dataset)
- Extract and put under `./datasets`
```
.
├── datasets
│   ├── densepose_data.json
│   ├── shhq_example_dataset
│   └── SMPL_NEUTRAL.pkl
├── log
│   └── map3dbn512l
│       └──  00295000_generator_ema_state_dict.pth
...
```
- run the inference script
```bash
python apps/sample_from_generator.py --config MAP3DBN512L --checkpoint log/map3dbn512l/00295000_generator_ema_state_dict.pth --dataroot ./datasets/shhq_example_dataset/ --dataset_length 10 --seeds 1 2 3 --back_and_forth
```

## Training

- The command below is an example for training the `256x128` model. 
- Scripts and instructions for preprocessing the [SHHQ](https://stylegan-human.github.io/data.html) dataset will be added.
```
python -m torch.distributed.launch --nproc_per_node=${NGPUS} apps/train.py --config MAP3DBN
```