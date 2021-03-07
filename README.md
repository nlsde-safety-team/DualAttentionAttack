# DualAttentionAttack



## Running

### before running

you need:

- dataset
- 3d object `.obj`  and texture file `.mtl` (eg. `src/audi_et_te.obj` and `src/audi_et_te.mtl`)
- face id list `.txt` which need to be trained (eg. `src/all_faces.txt`)
- seed content texture and edge mask texture

### training

```shell
python train.py --datapath=[path to dataset] --content=[path to seed content] --canny=[path to edge mask]
```

results will be stored in `src/logs/`, include:

- output images
- `loss.txt`
- `texture.npy`  the trained texture file

### testing

```shell
python test.py --texture=[path to texture]
```

results will be stored in `src/acc.txt`