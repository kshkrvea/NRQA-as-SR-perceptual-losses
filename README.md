 🔧[Install](#installation)  **|** 💻[Train](#training) **|** 📊[Test](#testing) **|** 🗓️[Roadmap](#roadmap)

## Getting Started
TLDR: взять академ

## Installation
1. Clone repo
```sh
  git clone --recursive git@vg-code.gml-team.ru:26e_kas/super-resolution-attacks.git
  cd super-resolution-attacks
```
2. Install dependent packages

All experiments are run under conda enviroment. For quick start create an enviroment using following commands:

  ```sh
  ./enviroments/basicsr.sh
  conda activate basicsr
  ```
This will create a _basicsr_ enviroment with ~~almost~~ all neccessary packages and buil cuda extensions.
In case of missing packages, please, write them down into _~/requirements.txt_


## Training
### Tensorboard
You can use integrated in VSCode tensorboard extension or follow this instructions:
1. Connect your port 16006 to the server free port. For examle, let's use port number 12345:
```sh
   ssh -L localhost:16006:localhost:12345 login@vg-intellect-1.lab.graphicon.ru
```
2. Start tmux session for tensorboard
```sh
  tmux
```
You are in tmux session now, to quit it press _Ctrl+B, D_

3. Start tensorboard session (once)
```sh
  conda create -n tensorboard
  conda activate tensorboard
  pip install tensorflow
  tensorboard --logdir_spec=26e_kas:/home/26e_kas@lab.graphicon.ru/sr_attacks/runs,25e_chi:/home/25e_chi@lab.graphicon.ru/super-resolution-attacks/runs --port=12345
```

4. Leave tmux session (press _Ctrl+B, D_)

5. If you want to come back to this session, run
```sh
  tmux attach-session -t 0
```
6. Open http://localhost:16006/#timeseries on your local machine

### Datasets
Datasets are stored on _calypso/26e_kas/datasets_.

### Yaml files
All training and testing options are stored in _.yaml_ files with the following structure (only important points are mentioned):
```yaml
   #General options
   task: #Experiment series, the name of folders with logs, checpoints etc.
   model: #SR model name
   weight_scale: #For pretrained models. Scale factor of loaded weights
   scale: #Scale factor of trained model
   
   #Generator parameters
   netG: 
      net_type: #SR model name
      arguments: 
         #__init__ arguments
      init_type: default
      freeze_blocks: #Number of freezed blocks (negative number mean number of unfreezed blocks)
  
  #Training parameters
  train: 
    G_lossfn_types: 
      lossfn_name_1:
        weight: #Ratio in weighted sum
        mode: #FR or NR or pseudo_FR (RR = NR(sr) - NR(gt), orientation is presserved)
        reverse: #In terms of loss: lower is better == not reversed
        args:
          #metric __init__ arguments
      lossfn_name_1:
        #Same
```

See example of testing _.yaml_ on ~/options/basicvsrpp/4x_train_vimeo_lpips_001.yaml

### Run
After all preparation steps just run the following command in a new tmux session:
```ssh
python train_model.py --opt path/to/the/options_file.yaml
```
Make sure that tqdm completed several first steps _Ctrl+B, D_ and chill out for the next ~10h

### GPU usage
You may use _nvitop_ to monitor free GPUs 
```ssh
pip install nvitop
nvitop
```

## Testing
### Yaml files
See example of testing _.yaml_ on ~/options/basicvsrpp/test_4x_vimeo_000.yaml

### Run
```ssh
python test_model.py --opt path/to/the/options_file.yaml
```
