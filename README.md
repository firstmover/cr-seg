## Consistency Regularization Improves Placenta Segmentation in Fetal EPI MRI Time Series

This repo includes the code for the paper [Liu et al. 2023](https://arxiv.org/pdf/2310.03870.pdf).

![Python 3.9](https://img.shields.io/badge/Python-3.9%2B-brightgreen.svg)
[![GitHub Repo Stars](https://img.shields.io/github/stars/firstmover/cr-seg?label=Stars&logo=github&color=red)](https://github.com/firstmover/cr-seg)

<br>
<img src="https://people.csail.mit.edu/liuyingcheng/data/projects/fetal/cr_seg/overview_method.png">
<hr/>

### Environment and dependency

Create a conda environment with: 
```bash 
conda create --name cr_seg --file requirements.txt
```

Specify your paths to data, cache, and results directories in: 
1. `./envs/default`
2. `./configs/segm_release/spatial_temporal_cr.py`

### Results

#### Pre-process of data 

```bash 
srun --partition=gpu \
--job-name=segm \
--gres=gpu:1 \
--ntasks=1 \
--ntasks-per-node=1 \
--cpus-per-task=16 \
--time=24:00:00 \
python scripts/pre_compute_data.py
```

#### Training 

pretrain registration models for all cross-validation folds 
```bash 
python ./scripts/submit_job_registraion.py \
--exp-name regi_release \
--config-name voxelmorph \
--job-name=regi \
--num-gpus-per-node=1 \
--cpus-per-task=20 \
--num-nodes=1 \
--array-parallelism=5
```

Train UNet with spatial and temporal consistency regularization for all cross-validation folds
```bash 
python ./scripts/submit_job_segmentation.py \
--task-mode train \
--exp-name segm_release \
--config-name spatial_temporal_cr \
--lambda-list '0.001' \
--lambda-t-list '0.001' \
--job-name segm_regi \
--num-gpus-per-node 4 \
--cpus-per-task 8 \
--array-parallelism 5
```

#### Inference and evaluation 

Run inference for labeled data for all cross-validation folds 
```bash 
python ./scripts/submit_job_segmentation.py \
--task-mode inference_labeled \
--exp-name segm_release \
--config-name spatial_temporal_cr \
--lambda-list '0.001' \
--lambda-t-list '0.001' \
--tta \
--tta-cfg-path ./configs/segm_release/_base_/tta_all_crop.py \
--save-data-name-list 'img,pred_seg_map,gt_seg_map' \
--job-name inference \
--partition gpu \
--num-gpus-per-node 1 \
--cpus-per-task 16 \
--array-parallelism 5
```

Run inference for time series data (unlabeled and labeled data) for all cross-validation folds
```bash 
python ./scripts/submit_job_segmentation.py \
--task-mode inference_time_series \
--exp-name segm_release \
--config-name spatial_temporal_cr \
--lambda-list '0.001' \
--lambda-t-list '0.001' \
--tta \
--tta-cfg-path ./configs/segm_release/_base_/tta_all_crop.py \
--save-data-name-list 'pred_seg_map' \
--job-name inference \
--partition gpu \
--num-gpus-per-node 1 \
--cpus-per-task 16 \
--array-parallelism 5
```

#### Visualization 

Pre-compute and evulate time series data 
```bash
python scripts/visualization/pre_compute_time_series.py \
--result_root ./results/segm_release \
--job-name eval \
--partition gpu \
--num-gpus-per-node 1 \
--cpus-per-task 24 \
--array-parallelism 8
```

Visualize labeled results
```bash
streamlit run scripts/visualization/labeled.py -- --result_root ./results/segm_release --model_name epoch_100_all
```

Visualize time series results
```bash
streamlit run scripts/visualization/time_series.py -- --result_root ./results/segm_release --model_name epoch_100_all
```

### todos 
- [ ] Add commands for non-slurm users 
- [ ] Add more details for data set structures 
- [ ] Update citation

### Acknowledgement
- [mabulnaga/automatic-placenta-segmentation](https://github.com/mabulnaga/automatic-placenta-segmentation)
- [voxelmorph/voxelmorph](https://github.com/voxelmorph/voxelmorph)

### License

This repo is licensed under the MIT License and the copyright belongs to all authors - see the [LICENSE](https://github.com/firstmover/cr-seg/blob/master/LICENSE) file for details.

### Citation

```
```

### Contact

Email: liuyc@mit.edu
