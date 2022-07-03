# DeepFry: Identifying Vocal Fry Using Deep Neural Networks
This repository is for the paper [DeepFry: Identifying Vocal Fry Using Deep Neural Networks](https://arxiv.org/abs/2203.17019) by Bronya R. Chernyak, Talia Ben Simon, Yael Segal, Jeremy Steffman, Eleanor Chodroff, Jennifer S. Cole, Joseph Keshet.  
It contains code for predicting creaky voice, as well as pre-trained models.


We provide two pre-trained models:
- [DeepFry - from the paper](https://github.com/bronichern/DeepFry/models/CREAK-220lr_0.001_decay_21_input_size_512_hidden_size_256_channels_512_normalize_False_measure_ff1_dropout_0.1_classes_3_.pth).
- [DeepFry - trained on both the Nuclear and Pre-Nuclear
datasets, that were described in the paper](https://github.com/bronichern/DeepFry/models/CREAK-74lr_0.001_decay_38_input_size_128_hidden_size_256_channels_512_normalize_False_measure_ff1_dropout_0.1_classes_3_logtxt_ff1_.pth).

This repository enables you to identify creaky frames in a given audio, see details below.
## Requirements and Installation
### Conda (Linux) ###
```
conda env create -f environment.yml
```

  
### Custom Installation ###
* To run this repoitory, your environment should have Python 3.8.
* You will need Pytorch (1.12.0), but you don't have to use GPU.
  Please refer to: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) and install the ***stable*** pytorch version most suitable for you environment specifications (OS, CUDA version, etc.).
* Finally, you will need the packages specified in the requirements file:
  ```
  pip install -r requirements.txt
  ```

  

## Identifying Creak (Running the repo) ##
There are two options to run this repository:
1. Run on a directory with wav files without corresponding annotated textgrids.
2. Run on a directory with wav files and their corresponding textgrids.

- Remove the following argument from the commands below to run on CPU:
```--cuda```
- If you get an error regarding the number of workers used in the test dataloader, due to you computers specs, you can change them by adding the following argument:
```--workers num_workers```  
- To run on a custom dataset, the .wav files (and optionally their corresponding TextGrid files) should be located under a folder named 'test' as follows(- See the 'allstar' folder in this repository for an example):
```
|-- CustomDataDIR
|   |-- test
|   |   |-- file1.wav
|   |   |-- file1.TextGrid
|   |   |-- file2.wav
|   |   |-- file2.TextGrid
```
and then you can specify the argument ```--data_dir CustomDataDir```

## Identify creak - ALLSTAR dataset
This options allows you to test the repository. In the folder 'allstar' you will find wav files with their corresponding textgrids, which we used to test our model on, as specified in the paper.  
**Note that the results in the paper were reported for 20ms to have a proper comparison between methods, while our model was trained on 5ms, so the measures here might differ slighly.**


 #### Run option #1: Only output measures:
```
python run.py --data_dir allstar --model_name model_path --cuda
```

#### Run option #2: Output measures & Write predictions to a textgrid:
```
python run.py --data_dir allstar --model_name model_path  --out_dir out_path --cuda
```

where `model_path` is the absolute path to the pre-trained model, and `out_dir` is the path to the directory in which the textgrids will be saved to with the predictions of the model.

## Identify Creak - custom dataset - no annotations
```
python run.py --data_dir data_path --model_name model_path --out_dir out_path --custom --cuda
```

Where `model_path` and `out_dir` is the same as above and `data_path` is the absolute path to a directory with wav files in which creak should be identified.

## Identify Creak - custom dataset - with annotations
- Annotated silences - optional: should be under 'Speaker - word' tier, marked as 'sp' or under 'creak-gold' without a mark.
- Annotated creak - optional: should be under 'creak-gold' tier, marked as 'c'.  
**You can refer to the TextGrid files in the "allstar" folder for an example**
```
python run.py --data_dir data_path --model_name model_path --out_dir out_path --cuda
```

Where `model_path` and `out_dir` is the same as above and `data_path` is the absolute path to a directory with wav files in which creak should be identified alongside their corresponding textgrids.


