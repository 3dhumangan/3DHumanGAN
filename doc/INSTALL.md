# Installation


## Create environment with Anaconda
```bash
conda create -n 3dhumangan python=3.9 -y
eval "$(conda shell.bash hook)"
conda activate 3dhumangan

conda install -y gcc=9.5.0 cxx-compiler -c conda-forge
conda install -y pytorch=1.9.1 torchvision=0.10.1 cudatoolkit=11.1 numpy -c pytorch -c nvidia -c conda-forge
conda install -y pip click ninja requests tqdm matplotlib nvitop

pip install scipy opencv-python imageio-ffmpeg pyspng joblib smplx[all] chumpy clean-fid tensorboardX einops

conda install -y -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -y -c bottler nvidiacub
conda install -y pytorch3d=0.6.2 -c pytorch3d
```

## Obtain SMPL model
- Download SMPL version 1.1.0 from <a href="https://smpl.is.tue.mpg.de/download.php">SMPL download page</a>
- Put `SMPL_NEUTRAL.pkl` under `./datasets`
```
.
├── apps
├── configs
├── datasets
│   ├── densepose_data.json
│   └── SMPL_NEUTRAL.pkl
...
```