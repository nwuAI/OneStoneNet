## **Kill Two Birds with One Stone: Inconsistent Timestep Diffusion Models for Both Blind and Non-blind Image Inpainting**

Zhan Li,Yuning Wang

Here we provide the PyTorch implementation of our latest version.

### **Prerequisites**

Please refer to this link(openai\:guided-diffusion):<https://github.com/openai/guided-diffusion>

### **Non-blind inpainting**

For non-blind inpainting tasks, it is necessary to download non-blind inpainting code.(The difference between non-blind and blind inpainting lies solely in whether the loss function includes a mask.)

#### **Train**

**Command**

```python
python train.py --image_size 256 --num_channels 256 --num_res_blocks 2 --diffusion_steps 1000 --noise_schedule linear --lr 1e-4 --batch_size 2 --predict_xstart True --learn_sigma False --timestep_respacing "250"
```

**Parameter Description**

In the `train.py` file, `args.data_dir` is the path for the training data, `mask_dir` is the path for the training masks, `log_interval` specifies how many iterations to wait before saving logs, `save_interval` specifies how many iterations to wait before saving the model, and `resume_checkpoint` represents the path to a pre-existing model checkpoint for resuming training.

In the `gaussian_diffusion.py` file, within the `training_losses1` function, you can adjust the `name` variable to modify the save location for samples during the training process.

In the `train_util.py` file, modify the first argument of `bf.join("", filename)` in the `save` function to save the checkpoints during the training process.

#### **Test**

**Pre-trained Models**

<https://pan.baidu.com/s/1pVFWR98UWR46D9cY5HeV-A?pwd=kfdk> password: kfdk

**Command**

```python
python test.py --class_cond False --timestep_respacing "250"
```

**Parameter Description**

Download the checkpint for different dataset and modify the `model_path` parameter in `test.py` to point to the checkpoint of the SRN phase, and the `args1.model_path` parameter to point to the checkpoint of the IRN phase. In `test.py`, `test_datadir` is the directory for the original test images, and `test_maskdir` is the directory for the masks. `save_dir` is the location where the restored results are saved.

### **Blind inpainting**

For blind inpainting tasks, it is necessary to download blind inpainting code.

#### **Train**

The training steps are consistent with non-blind restoration steps; in the actual code, only the loss function is changed.

#### **Test**

**Pre-trained Models**

<https://pan.baidu.com/s/1pVFWR98UWR46D9cY5HeV-A?pwd=kfdk> password: kfdk

**Testing Steps**

The testing procedure is consistent with the non-blind restoration testing procedure.
