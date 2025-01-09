"""
This project's code is based on https://github.com/openai/guided-diffusion.
"""
from PIL import Image
import argparse
import os
from guided_diffusion.metrics_my1 import EdgeAccuracy
from torchvision.utils import save_image
import numpy as np
import torch as th
import functools
import torch.distributed as dist
from guided_diffusion.my_util import stitch_images,create_dir,stitch_images1
from guided_diffusion.dataset_my import load_data
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from guided_diffusion.gaussian_diffusion import GaussianDiffusion


def main():
    th.cuda.device_count()

    args = create_argparser().parse_args()
    args1 = create_argparser().parse_args()
    args1.in_channels = 7
    args1.out_channels = 3

    # image unet model path
    args1.model_path=""

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    state_dict = th.load(args.model_path,map_location=th.device('cuda'))
    model.load_state_dict(state_dict, strict=True)
    if len(model.load_state_dict(state_dict)) == 0:
        print("All parameters have been loaded successfully.")
    else:
        print("There are some parameters that were not loaded.")

    model.to(dist_util.dev())

    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()
    model1, diffusion1 = create_model_and_diffusion(
        **args_to_dict(args1, model_and_diffusion_defaults().keys())
    )

    state_dict1 = th.load(args1.model_path,map_location=th.device('cuda'))
    model1.load_state_dict(state_dict1, strict=True)
    print("len(model1.load_state_dict(state_dict1)):",len(model1.load_state_dict(state_dict1)))
    if len(model1.load_state_dict(state_dict1)) == 0:
        print("All parameters have been loaded successfully.")
    else:
        print("There are some parameters that were not loaded.")

    model1.to(dist_util.dev())

    if args.use_fp16:
        model1.convert_to_fp16()
    model1.eval()

    test_data = load_data(
        data_dir=args.test_datadir,
        mask_dir=args.test_maskdir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=False,
        deterministic=True,
        mask_train=False,
    )

    all_images = []
    all_labels = []

    while len(all_images) * args.batch_size < args.num_samples:

        image,mask,mask_image,edge,mask_edge,gray_maskimage,gray_image,_=next(test_data)
        image=image.to(dist_util.dev())
        gray_maskimage=gray_maskimage.to(dist_util.dev())
        mask_edge=mask_edge.to(dist_util.dev())
        gray_image=gray_image.to(dist_util.dev())
        mask_image=mask_image.to(dist_util.dev())

        image1=image
        mask1=mask
        mask_image1=mask_image
        edge1=edge
        mask_edge1=mask_edge
        gray_maskimage1=gray_maskimage
        gray_image1=gray_image

        image2=image
        mask2=mask
        mask_image2=mask_image
        edge2=edge
        mask_edge2=mask_edge
        gray_maskimage2=gray_maskimage
        gray_image2=gray_image

        model_kwargs = {}

        if args.class_cond:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
            )
            model_kwargs["y"] = classes
        logger.log("edge sampling...")

        qunsunxt=th.randn(1, 1, 256, 256)
        qunsunxt=qunsunxt.to("cuda")
        qunsunxt1=qunsunxt

        jiazao_time1 = th.tensor([249])
        tensor1 = jiazao_time1.long().to(dist_util.dev())

        compute_losses2 = functools.partial(
            diffusion.training_losses4,
            model,
            gray_maskimage,
            tensor1
        )
        sample,out = compute_losses2()

        sample_edge=sample

        print(th.max(((sample+1)/2) *255))
        print(th.min(((sample+1)/2) *255))

        sample1=out
        sample1 = (((sample1 + 1)/2) * 255).clamp(0, 255).to(th.uint8)
        sample1 = sample1.permute(0, 2, 3, 1)
        sample1 = sample1.contiguous()
        # save dir
        save_dir = ''
        image = Image.fromarray(sample1.squeeze().cpu().numpy(),mode="L")
        idx=len(all_images)
        file_name = f'edge_{idx:05}.jpg'
        file_path = os.path.join(save_dir, file_name)
        image.save(file_path)

        logger.log("image sampling...")

        jiazao_time1 = th.tensor([249])
        tensor1 = jiazao_time1.long().to(dist_util.dev())

        compute_losses1 = functools.partial(
            diffusion1.training_losses3,
            model1,
            sample_edge,
            mask_image,
            tensor1
        )

        sample3 = compute_losses1()

        image2 = ((image2 + 1) * 127.5).clamp(0, 255).to(th.uint8)
        image2 = image2.permute(0, 2, 3, 1)
        mask_image2 = ((mask_image2 + 1) * 127.5).clamp(0, 255).to(th.uint8)
        mask_image2 = mask_image2.permute(0, 2, 3, 1)
        mask_edge2 = ((mask_edge2 + 1) * 127.5).clamp(0, 255).to(th.uint8)
        mask_edge2 = mask_edge2.permute(0, 2, 3, 1)


        sample33=sample3
        sample33 = ((sample33+1)*127.5).clamp(0, 255).to(th.uint8)
        sample33 = sample33.permute(0, 2, 3, 1)
        sample33 = sample33.contiguous()
        # save dir
        save_dir = ''
        image1 = Image.fromarray(sample33.squeeze().cpu().numpy())
        idx=len(all_images)
        file_name = f'{idx:05}.jpg'
        file_path = os.path.join(save_dir, file_name)
        image1.save(file_path)
        vir_images1 = stitch_images1(
            mask_image2
        )
        # save dir
        name = os.path.join("", str(len(all_images)).zfill(5) + "_0.png")

        print('\nsaving sample ' + name)
        vir_images1.save(name)
        all_images.append("1")

    

    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=2,
        batch_size=1,
        use_ddim=False,
        # edge unet model path
        model_path="",
        # test data path
        test_datadir="",
        # test mask path
        test_maskdir="",
        image_size=256,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
