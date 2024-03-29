# Project Mayhem

## Installation

Clone repo and create a conda environment using the following commands:

```shell
git clone git@vg-code.gml-team.ru:26e_kas/super-resolution-attacks.git
cd super-resolution-attacks
./enviroments/basicsr.sh
conda activate basicsr
```

We provide all instructions for BasicVSR++ (aka `basicsr`). Modify commands for other VSR methods.

Project files are stored in `//calypso/work/26e_kas` directory. Mount it before next steps:

```shell
sudo mount.cifs //calypso/work/26e_kas ~/mnt/calypso -o credentials=$HOME/mnt/credentials,uid=$(id -u),gid=$(id -g)
```

Credentials file should contain `username`, `password`, and `domain=Graphics2` fields.

## Training

We use tensorboard to manage logs. Setup integrated VS Code tensorboard extension or follow these instructions:

1. Connect your port 16006 to the server free port. For example, let's use port number 12345:
   ```shell
   ssh -L localhost:16006:localhost:12345 login@vg-intellect-1.lab.graphicon.ru
   ```
   
2. Start tmux session for the tensorboard:
   ```shell
   tmux
   pip install tensorboard
   tensorboard --port=12345 --logdir=...
   ```
   
   You can use deprecated `--logdir_spec` argument instead of `--logdir` argument to work with all runs:
   ```shell
   --logdir_spec=26e_kas:/home/26e_kas@lab.graphicon.ru/sr_attacks/runs,25e_chi:/home/25e_chi@lab.graphicon.ru/super-resolution-attacks/runs
   ```
   
   Detach from the session (press _Ctrl+B, D_)

3. (Optional) If you want to come back to this session later, run:
   ```shell
   tmux attach-session -t 0
   ```

4. Open http://localhost:16006/ to access the tensorboard

All training options are configured as `.yaml` files. See example in `options/basicvsrpp/4x_finetune_vimeo_clipiqa_001.yaml`.
Copy and modify one file (important points are mentioned here):

```yaml
task: # Experiment name. Should be unique across all runs

train: 
  G_lossfn_types: 
    lossfn_name_1:
      weight: # Weight in weighted loss sum
      mode: # FR, NR, or pseudo_FR (NR(sr) - NR(gt))
      reverse: # false when lower is better

    lossfn_name_2:
      # ...
```

Run the following command to start training:

```shell
python train_model.py --opt options/vsr-method-name/options_file.yaml
```

You also can use `run_several.py` script to schedule several runs.

## Testing

To test many runs for one VSR method modify the `test_config.yaml` file under `options/vsr-method-name` subdirectory.

```shell
python testing_pipeline.py --opt options/vsr-method-name/test_config.yaml
```

This script will save results to `.xlsx` file on mounted storage. Use the following script to plot heatmap and save it to `heatmap.pdf`.

```shell
python manyplots.py
```
