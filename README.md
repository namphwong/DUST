# DUST
## Web project: Swin Transformer 1D

Phase 0 specification is complete. Start with [project scope](docs/project-scope.md),
[data contract](docs/data-contract.md), [acceptance criteria](docs/acceptance-criteria.md),
and [phase status](docs/phase-status.md). Format examples are in
[examples/phase0](examples/phase0/README.md).

Phase 1 browser feasibility has passed: the untrained Swin 1D exports to ONNX
and matches PyTorch in browser WASM on eight fixtures. See
[the feasibility report and run instructions](docs/browser-feasibility.md).
The historical untrained check remains available at `/?phase=1`.

Phase 2 data preparation is complete: official UCI HAR total acceleration,
subject-disjoint train/validation/test, train-only normalization and six sourced demo CSVs.
See [the data report and Kaggle usage](docs/data-report.md). Reproduce with
`python training/prepare_uci_har.py --download` and verify with
`python training/verify_phase2.py` (NumPy requirements: `training/requirements-data.txt`).

Phase 3 is complete on Kaggle GPU: six runs (CNN/Swin, seeds 42/43/44),
downloaded checkpoint hashes and test metrics independently verified.
Validation-selected Swin seed 44 has test accuracy 0.871734 / macro-F1 0.870489;
Swin three-seed macro-F1 is 0.868143 ± 0.008016 (sample std).
The 0.90 target is not met; this remains a clearly limited research demo.
See [the Kaggle report](docs/kaggle-training-report.md), [historical CPU report](docs/training-report.md),
[GPU notebook](notebooks/kaggle_train_har_gpu.ipynb) and [Kaggle/resume instructions](notebooks/README.md).
Phase 4 is complete: the default browser page uses the trained Kaggle checkpoint,
exported as float32 ONNX (8.62 MiB). All 2947 test predictions match the checkpoint;
macro-F1 degradation is zero. Browser WASM parity passes 14 fixtures with a measured
benchmark. See [artifact report and commands](docs/phase4-artifact-report.md),
[model card](docs/model-card.md) and [release manifest](web/public/models/har-swin-kaggle-v1/release.json).
Phase 5 UI is complete: the default page is MotionSense, with six sourced UCI train
demos, real ONNX inference, XYZ chart, six scores and demo result CSV downloads.
See the historical [UI report](docs/phase5-ui-report.md).
Phase 6 completes actual user CSV inference in a Web Worker: explicit g/m/s²/gravity,
segment-aware 50 Hz resampling, 128/64 windows, confirmation, progress/cancel,
source isolation and all-window result CSV exports. No CSV upload or paid backend.
17 synthetic CSV fixtures / 18 windows match independent NumPy + PyTorch golden
tensors/predictions. A 100,000-row session runs all 1,561 windows with a responsive UI.
See [pipeline report and reproduction commands](docs/phase6-pipeline-report.md).
The pre-Phase-7 [UI/UX redesign](docs/ui-ux-redesign.md) makes input/output the main
dashboard: file selection/drag-drop/pasted CSV, step guidance, whole-file window
distribution, selected-window timestamps/scores and full CSV export. Responsive
desktop/mobile layouts and all 32 browser tests are verified locally.
Technical checks remain at `/?phase=4` and `/?phase=1`. Tests use port 4174;
the user preview stays on 4173. Vercel deployment/release acceptance remain pending (Phase 7–8).

**Dual Swin Transformer for Video-Time-Series Fusion**

This project involves the use of a Dual Swin Transformer model to effectively fuse video and time-series data. Below are the steps for preparation, data generation, and training.

Related Paper:

[Shi, L., Chen, Y., Liu, M., & Guo, F. (2024). DuST: Dual Swin Transformer for Multi-modal Video and Time-Series Modeling. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 4537-4546).](https://openaccess.thecvf.com/content/CVPR2024W/WAD/html/Shi_DuST_Dual_Swin_Transformer_for_Multi-modal_Video_and_Time-Series_Modeling_CVPRW_2024_paper.html)

## Environment Setup

Follow these steps to set up the environment for the DUST project:

### Creating a Conda Environment

1. Create a new conda environment with Python 3.8:
   
   ```bash
   conda create -n dust python=3.8
   ```
2. Activate the newly created environment:
   
   ```bash
   conda activate dust
   ```
3. Install PyTorch, torchvision, torchaudio, and CUDA support using conda:

   ```bash
   conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
   ```
4. Install the necessary Python packages using pip:

   ```bash
   pip install timm
   pip install einops
   pip install opencv-python
   pip install pandas
   ```
   
## Generate Toy Data
To generate toy data for training and testing, run the following command:

`$ python gen_data.py`


## Training
The training process is divided into several stages:

1. **Stage 1: Train Video Model**
   - Train the video model by executing:
     ```
     python step_fusion_3d.py
     ```

   - step_fusion_3d.py: Script Configuration

   - The `step_fusion_3d.py` script is crucial for training the video model in the DUST framework. It includes several configuration options:

      - Training Parameters
     
         - `--batch_size`: Sets the batch size for training (default: 9).
         - `--num_epochs`: Determines the number of training epochs (default: 100).
         - `--lr`: Specifies the learning rate (default: 0.001).
         - `--seed`: Sets a random seed for reproducibility (default: 93728645).
         - `--print_every`: Defines the interval of epochs for validation and model saving (default: 3).

      - Model Configuration
        
         - `--num_frames`: Number of frames or time points (default: 77).
         - `--height`: Height of each frame (default: 224).
         - `--width`: Width of each frame (default: 224).

      - Input Paths
    
         - `--path_video_train`: Path to the video training data (default: './data/video_train.pkl').
         - `--path_video_val`: Path to the video validation data (default: './data/video_val.pkl').
         - `--path_label_train`: Location of the training label data (default: './data/label_train.pkl').
         - `--path_label_val`: Location of the validation label data (default: './data/label_val.pkl').

      - Output Paths
         - `--outpath`: Directory to save the trained models (default: './saved_models/step_3d/').
         - `--logpath`: Directory for saving logs (default: './logs/').

   - These configurations allow for fine-tuning of the video model training process in the DUST project.


2. **Stage 2: Train Time-Series Model**
   - Next, train the time-series model using:
     ```
     python step_fusion_1d.py
     ```
   - step_fusion_1d.py: Script Configuration

   - The `step_fusion_1d.py` script, a key part of the DUST model's training, utilizes several arguments:

      - Training Parameters
         - `--batch_size`: Input batch size (default: 2000).
         - `--num_epochs`: Number of epochs (default: 200).
         - `--lr`: Learning rate (default: 0.0003).
         - `--seed`: Random seed (default: 93728645).
         - `--print_every`: Epoch interval for validation/model saving (default: 1).

      - Model Settings
         - `--num_frames`: Number of frames (default: 51).
         - `--num_channels`: Number of channels (default: 3).

      - Input/Output Paths
         - `--path_kine_train`: Kinematic training data path (default: './data/kine_train.pkl').
         - `--path_kine_val`: Kinematic validation data path (default: './data/kine_val.pkl').
         - `--path_label_train`: Training labels path (default: './data/label_train.pkl').
         - `--path_label_val`: Validation labels path (default: './data/label_val.pkl').
         - `--outpath`: Model saving directory (default: './saved_models/step_1d/').
         - `--logpath`: Log directory (default: './logs/').

   - These configurations provide detailed control for the time-series model training in the DUST framework.


3. **Stage 3: Extract Video Features**
   - After training the video model, extract features from videos with:
     ```
     python step_fusion_3d_get_feature.py
     ```

   - step_fusion_3d_get_feature.py: Script Configuration

   - The `step_fusion_3d_get_feature.py` script is designed for extracting features from videos as part of the DUST model's training process. It utilizes several arguments for configuration:

      - Basic Settings
        
         - `--batch_size`: Sets the input batch size for training (default: 80).
         - `--seed`: Specifies a random seed for reproducibility (default: 93728645).

      - Model Configuration
         
         - `--path_3d_model`: Path to the pre-trained swin 3d model (default: './saved_models/step_3d/final_3d.pth').
         - `--num_frames`: Number of frames or time points (default: 77).
         - `--height`: Height of each frame (default: 224).
         - `--width`: Width of each frame (default: 224).

      - Input Paths
         
         - `--path_video_train`: Location of the video training data (default: './data/video_train.pkl').
         - `--path_video_val`: Location of the video validation data (default: './data/video_val.pkl').
         - `--path_label_train`: Path to training label data (default: './data/label_train.pkl').
         - `--path_label_val`: Path to validation label data (default: './data/label_val.pkl').

      - Output Paths
         
         - `--outpath`: Directory to save extracted video features (default: './video_features/').
         - `--logpath`: Directory for saving logs (default: './logs/').

   - These configurations enable the extraction of features from video data.


4. **Stage 4: Update Time-Series Model**
   - Finally, update the time-series model by running:
     ```
     python step_fusion.py
     ```
   - step_fusion.py: Script Configuration

   - The `step_fusion.py` script is an essential component for the final stage of the DUST model's training, focusing on the fusion of video and kinematic data. It is configured through the following arguments:

      - Training Parameters
         
         - `--batch_size`: Sets the batch size for training (default: 2000).
         - `--num_epochs`: Determines the number of training epochs (default: 200).
         - `--lr`: Specifies the learning rate (default: 0.0003).
         - `--seed`: Sets a random seed for reproducibility (default: 93728645).
         - `--print_every`: Defines the interval of epochs for validation and model saving (default: 1).

      - Model Configuration
         
         - `--path_1d_model`: Path to the pre-trained swin 1d model (default: './saved_models/step_1d/final_1d.pth').
         - `--num_frames`: Number of frames or time points (default: 51).
         - `--num_channels`: Number of channels (default: 3).

      - Input Paths
         
         - `--path_kine_train`: Location of the kinematic training data (default: './data/kine_train.pkl').
         - `--path_kine_val`: Location of the kinematic validation data (default: './data/kine_val.pkl').
         - `--path_vidfeat_train`: Path to the training video features (default: './video_features/train_3d_features.pkl').
         - `--path_vidfeat_val`: Path to the validation video features (default: './video_features/val_3d_features.pkl').
         - `--path_label_train`: Path to training label data (default: './data/label_train.pkl').
         - `--path_label_val`: Path to validation label data (default: './data/label_val.pkl').

      - Output Paths
         
         - `--outpath`: Directory to save the final fused models (default: './saved_models/step_fusion/').
         - `--logpath`: Directory for saving logs (default: './logs/').

This configuration caters to the intricate process of fusing video and kinematic data, representing the culmination of the DUST model's training protocol.

## Acknowledgments

This project has been greatly influenced and derives key components from the following repositories:

1. **Video Swin Transformer in PyTorch**: Our work incorporates elements from the Video Swin Transformer implementation available at [haofanwang/video-swin-transformer-pytorch](https://github.com/haofanwang/video-swin-transformer-pytorch). We extend our gratitude to Haofan Wang and contributors for their pioneering work in the field of video processing with transformers.

2. **Swin Transformer 1D**: Elements of our project are inspired by the Swin Transformer 1D, accessible at [meraks/Swin-Transformer-1D](https://github.com/meraks/Swin-Transformer-1D). We acknowledge the efforts of Merak and their contributors in adapting Swin Transformer architectures for 1D data processing, which has significantly informed our approach.

We thank these contributors for their open-source work, which has been instrumental in shaping our project.

