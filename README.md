# 3DSeg
Volumetric segmentation using a V-Net model

## Getting Started
### Prerequisites
We recommend on using the following docker image. Pull the following docker 
image using:

```
docker pull geromev/dsgzneuro:latest 
```

Alternatively, you can also:
```
pip install -r requirements.txt 
```

## Dataset
All volumes should be in `./vols_all/`.

## Training
We use `Segmentation3DTrainer` object in `trainer.py` which performs data loading, 
data augmentation, and training. As an example usage to train a model you can
 do the following:
```
python trainer_example_usage.py --train_model=1
```
Training will ouptut the following:
1. Save weights of the trained model at `./model/`.
2. Output tensorboard event logs at `./logs/`.
3. If `debug_dump_image` is set to True, will save augmented volumes at `
./augmented_vols`.


## Testing
When training is finished, the trained model will be used to evaluate the 
test set.. You can evaluate the test set 
using the following:
```
python trainer_example_usage.py --train_model=0
```
Predicted segmentation outputs will be in `./pred_output/`. 

## Contributing

## License
This project is licensed under the MIT License - see the 
[LICENSE.md](LICENSE.md) file for details