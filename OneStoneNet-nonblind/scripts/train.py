# -- coding: utf-8 --
"""
This project's code is based on https://github.com/openai/guided-diffusion.
"""

import argparse
from guided_diffusion.dataset_my import load_data
from guided_diffusion import dist_util, logger
# from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop
from torch.utils.tensorboard import SummaryWriter
import torch
from guided_diffusion.discriminator import Discriminator

def main():
    torch.autograd.set_detect_anomaly(True)
    torch.cuda.device_count()
    args = create_argparser().parse_args()
    args1 = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")

    args1.in_channels=7
    args1.out_channels=3

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model1, diffusion1 = create_model_and_diffusion(
        **args_to_dict(args1, model_and_diffusion_defaults().keys())
    )
    discriminator = Discriminator(in_channels=2, use_sigmoid=True)
    discriminator1 = Discriminator(in_channels=3, use_sigmoid=True)

    model.to(dist_util.dev())
    discriminator.to(dist_util.dev())
    model1.to(dist_util.dev())
    discriminator1.to(dist_util.dev())

    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)
    logger.log("creating data loader...")

    data = load_data(
        data_dir=args.data_dir,
        mask_dir=args.mask_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        deterministic=False,
        mask_train=True,
    )

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        discriminator=discriminator,

        model1=model1,
        diffusion1=diffusion1,
        discriminator1=discriminator1,

        data=data,

        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,

        log_interval=args.log_interval,
        save_interval=args.save_interval,

        resume_checkpoint1=args.resume_checkpoint1,
        resume_checkpoint2=args.resume_checkpoint2,
        resume_checkpoint3=args.resume_checkpoint3,
        resume_checkpoint4=args.resume_checkpoint4,
        resume_checkpoint5=args.resume_checkpoint5,

        resume_checkpoint11=args.resume_checkpoint11,
        resume_checkpoint21=args.resume_checkpoint21,
        resume_checkpoint31=args.resume_checkpoint31,
        resume_checkpoint41=args.resume_checkpoint41,
        resume_checkpoint51=args.resume_checkpoint51,

        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,

    ).run_loop()



def create_argparser():
    defaults = dict(
        # train data path
        data_dir="",
        # train mask path
        mask_dir="",

        schedule_sampler="uniform",

        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=2,
        microbatch=-1,
        ema_rate="0.9999",

        log_interval=5000,
        save_interval=5000,

        # edge unet model path
        resume_checkpoint1="",
        # edge unet ema path
        resume_checkpoint2="",
        # edge unet opt path
        resume_checkpoint3="",
        # edge discriminator model path
        resume_checkpoint4="",
        # edge discri opt path
        resume_checkpoint5="",

        # image unet model path
        resume_checkpoint11="",
        # image unet ema path
        resume_checkpoint21="",
        # image unet opt path
        resume_checkpoint31="",
        # image discriminator model pth
        resume_checkpoint41="",
        # image discri opt path
        resume_checkpoint51="",

        use_fp16=False,
        fp16_scale_growth=1e-3,

        resume_step="",

        # no
        model_path="",

        image_input_channels=7,
        image_output_channels=3,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
