import copy
import functools
import os

import blobfile as bf
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW,Adam

from . import dist_util, logger
from .fp16_util import MixedPrecisionTrainer
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler
from torch.utils.tensorboard import SummaryWriter


INITIAL_LOG_LOSS_SCALE = 20.0


class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        discriminator,

        model1,
        diffusion1,
        discriminator1,
        data,

        batch_size,
        microbatch,
        lr,
        ema_rate,

        log_interval,
        save_interval,
        resume_checkpoint1,
        resume_checkpoint2,
        resume_checkpoint3,
        resume_checkpoint4,
        resume_checkpoint5,
        resume_checkpoint11,
        resume_checkpoint21,
        resume_checkpoint31,
        resume_checkpoint41,
        resume_checkpoint51,

        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        step=0,
        resume_step=0,

    ):

        self.writer=SummaryWriter()
        self.best_loss = 100

        self.model = model
        self.diffusion = diffusion
        self.discriminator=discriminator

        self.model1 = model1
        self.diffusion1 = diffusion1
        self.discriminator1 = discriminator1

        self.data = data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint1 = resume_checkpoint1
        self.resume_checkpoint2 = resume_checkpoint2
        self.resume_checkpoint3 = resume_checkpoint3
        self.resume_checkpoint4 = resume_checkpoint4
        self.resume_checkpoint5 = resume_checkpoint5
        self.resume_checkpoint11 = resume_checkpoint11
        self.resume_checkpoint21 = resume_checkpoint21
        self.resume_checkpoint31 = resume_checkpoint31
        self.resume_checkpoint41 = resume_checkpoint41
        self.resume_checkpoint51 = resume_checkpoint51
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps

        self.step = 0
        self.resume_step = resume_step
        self.global_batch = self.batch_size * dist.get_world_size()

        self.sync_cuda = th.cuda.is_available()
        self._load_and_sync_parameters()
        self._load_and_sync_parameters1()














        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )
        self.opt=Adam(params=self.mp_trainer.master_params,lr=self.lr,betas=(0.0,0.9))
        self.mpdis_trainer = MixedPrecisionTrainer(
            model=self.discriminator,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )
        self.dis_opt = Adam(params=self.mpdis_trainer.master_params, lr=self.lr * 0.1, betas=(0.0, 0.9))

        self.mp_trainer1 = MixedPrecisionTrainer(
            model=self.model1,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )
        self.opt1 = Adam(params=self.mp_trainer1.master_params, lr=self.lr, betas=(0.0, 0.9))
        self.mpdis_trainer1 = MixedPrecisionTrainer(
            model=self.discriminator1,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )
        self.dis_opt1 = Adam(params=self.mpdis_trainer1.master_params, lr=self.lr * 0.1, betas=(0.0, 0.9))
        


        '''
        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )     
        '''

        '''
        self.dis_opt=AdamW(params=self.discriminator.parameters(),
                                        lr=self.lr * 0.1,
                                        weight_decay=self.weight_decay)
        '''

        if self.resume_step:

            self._load_optimizer_state()
            self._load_optimizer_state1()





            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
            self.ema_params1 = [
                self._load_ema_parameters1(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]
            self.ema_params1 = [
                copy.deepcopy(self.mp_trainer1.master_params)
                for _ in range(len(self.ema_rate))
            ]
        self.use_ddp = False
        self.ddp_model = self.model
        self.ddp_dis_model=self.discriminator
        self.ddp_model1 = self.model1
        self.ddp_dis_model1=self.discriminator1
    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint1

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                logger.log(f"loading edge unet model from checkpoint: {resume_checkpoint}...")
                self.model.load_state_dict(
                    dist_util.load_state_dict(
                        resume_checkpoint, map_location=dist_util.dev()
                    )
                )

        dist_util.sync_params(self.model.parameters())
        resume_checkpoint11 = self.resume_checkpoint11

        if resume_checkpoint11:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint11)
            if dist.get_rank() == 0:
                logger.log(f"loading iamge unet model from checkpoint: {resume_checkpoint11}...")
                self.model1.load_state_dict(
                    dist_util.load_state_dict(
                        resume_checkpoint11, map_location=dist_util.dev()
                    )
                )

        dist_util.sync_params(self.model1.parameters())
    def _load_and_sync_parameters1(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint4
        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                logger.log(f"loading edge discriminator model from checkpoint: {resume_checkpoint}...")
                self.discriminator.load_state_dict(
                    dist_util.load_state_dict(
                        resume_checkpoint, map_location=dist_util.dev()
                    )
                )

        dist_util.sync_params(self.discriminator.parameters())
        resume_checkpoint41 = find_resume_checkpoint() or self.resume_checkpoint41
        if resume_checkpoint41:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint41)
            if dist.get_rank() == 0:
                logger.log(f"loading edge discriminator model from checkpoint: {resume_checkpoint41}...")
                self.discriminator1.load_state_dict(
                    dist_util.load_state_dict(
                        resume_checkpoint41, map_location=dist_util.dev()
                    )
                )

        dist_util.sync_params(self.discriminator1.parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.mp_trainer.master_params)
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint2
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading edge EMA from checkpoint: {ema_checkpoint}...")
                state_dict = dist_util.load_state_dict(
                    ema_checkpoint, map_location=dist_util.dev()
                )
                ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)

        dist_util.sync_params(ema_params)
        return ema_params

    def _load_ema_parameters1(self, rate):
        ema_params = copy.deepcopy(self.mp_trainer1.master_params)
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint21
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading image EMA from checkpoint: {ema_checkpoint}...")
                state_dict = dist_util.load_state_dict(
                    ema_checkpoint, map_location=dist_util.dev()
                )
                ema_params = self.mp_trainer1.state_dict_to_master_params(state_dict)

        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint3
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )

        if bf.exists(opt_checkpoint):
            logger.log(f"loading edge unet optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)
        main_checkpoint31 = find_resume_checkpoint() or self.resume_checkpoint31
        opt_checkpoint31 = bf.join(
            bf.dirname(main_checkpoint31), f"opt{self.resume_step:06}.pt"
        )

        if bf.exists(opt_checkpoint31):
            logger.log(f"loading iamge unet optimizer state from checkpoint: {opt_checkpoint31}")
            state_dict31 = dist_util.load_state_dict(
                opt_checkpoint31, map_location=dist_util.dev()
            )
            self.opt1.load_state_dict(state_dict31)

    def _load_optimizer_state1(self):

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint5
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )

        if bf.exists(opt_checkpoint):
            logger.log(f"loading edge optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.dis_opt.load_state_dict(state_dict)
        main_checkpoint51 = find_resume_checkpoint() or self.resume_checkpoint51
        opt_checkpoint51 = bf.join(
            bf.dirname(main_checkpoint51), f"opt{self.resume_step:06}.pt"
        )

        if bf.exists(opt_checkpoint51):
            logger.log(f"loading image optimizer state from checkpoint: {opt_checkpoint51}")
            state_dict51 = dist_util.load_state_dict(
                opt_checkpoint51, map_location=dist_util.dev()
            )
            self.dis_opt1.load_state_dict(state_dict51)

    def run_loop(self):

        epoch=0
        while (
            not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps
        ):

            batch, mask, mask_image, edge, mask_edge,mask_grayimage,gray_image,cond = next(self.data)

            if((self.step*self.batch_size) % 21000==0 and self.step!=0):
                epoch = epoch + 1

            self.run_step1(edge, mask_edge,mask_grayimage,gray_image,mask,batch,mask_image,cond)
            self.step += 1

            if self.step % self.log_interval == 0:
                logger.dumpkvs()
            if self.step % self.save_interval == 0:
                self.save()
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return


            '''






































           
            '''
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step1(self,edge, mask_edge,mask_grayimage,gray_image, mask,batch,mask_image,cond):


        for i in range(0, mask_edge.shape[0], self.microbatch):


            micro = mask_edge[i: i + self.microbatch].to(dist_util.dev())
            micro1 = edge[i: i + self.microbatch].to(dist_util.dev())
            micro2 = mask_grayimage[i: i + self.microbatch].to(dist_util.dev())
            micro3 = gray_image[i: i + self.microbatch].to(dist_util.dev())
            micro4 = mask[i: i + self.microbatch].to(dist_util.dev())
            micro5= batch[i: i + self.microbatch].to(dist_util.dev())
            micro6=mask_image[i: i + self.microbatch].to(dist_util.dev())
            micro_cond = {
                k: v[i: i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            }
            last_batch = (i + self.microbatch) >= mask_edge.shape[0]


            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())
            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                self.ddp_dis_model,
                micro,
                micro1,
                micro2,
                micro3,
                micro4,
                t,
                model_kwargs=micro_cond,
                trainers = self.mp_trainer,
                weights=weights,
                opts=self.opt,
                ema_rate=self.ema_rate,
                ema_params=self.ema_params,
                step=self.step,
                writer=self.writer,
                dis_trainer=self.mpdis_trainer,
                dis_opt=self.dis_opt,
            )



            if last_batch or not self.use_ddp:
                losses,pred_x_t_minus_1_result = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses,pred_x_t_minus_1_result = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["eps_loss"].detach()
                )
            self.writer.add_scalar("edge_dis_loss-iter", losses['dis_loss'], self.step)
            self.writer.add_scalar("edge_gen_eps_loss-iter", losses['eps_loss'], self.step)
            self.writer.add_scalar("edge_gen_gan_loss-iter", losses['gen_gan_loss'], self.step)
            self.writer.add_scalar("edge_gen_fm_loss-iter", losses['gen_fm_loss'], self.step)
            self.writer.add_scalar("edge_gen_loss-iter", losses['gen_loss'], self.step)


            compute_losses1 = functools.partial(
                self.diffusion.training_losses1,
                self.ddp_model1,
                self.ddp_dis_model1,
                micro,
                micro1,
                micro2,
                micro3,
                micro4,
                micro5,
                micro6,
                pred_x_t_minus_1_result,
                t,
                model_kwargs=micro_cond,
                trainers=self.mp_trainer1,
                weights=weights,
                opts=self.opt1,
                ema_rate=self.ema_rate,
                ema_params=self.ema_params1,
                step=self.step,
                writer=self.writer,
                dis_trainer=self.mpdis_trainer1,
                dis_opt=self.dis_opt1,
            )
            if last_batch or not self.use_ddp:
                losses1 = compute_losses1()
            else:
                with self.ddp_model.no_sync():
                    losses1 = compute_losses1()
            self.writer.add_scalar("image_dis_loss-iter", losses1['dis_loss'], self.step)
            self.writer.add_scalar("image_gen_eps_loss-iter", losses1['eps_loss'], self.step)
            self.writer.add_scalar("image_gen_gan_loss-iter", losses1['gen_gan_loss'], self.step)
            self.writer.add_scalar("image_gen_perceptual_loss-iter", losses1['gen_perceptual_loss'], self.step)
            self.writer.add_scalar("image_gen_style_loss-iter", losses1['gen_style_loss'], self.step)
            self.writer.add_scalar("image_gen_loss-iter", losses1['gen_loss'], self.step)



        self._anneal_lr()
        self.log_step()




















































    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def save(self):

        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving edge unet model {rate}...")
                if not rate:
                    filename = f"edge_unet_model{(self.step+self.resume_step):06d}.pt"
                else:
                    filename = f"edge_unet_ema_{rate}_{(self.step+self.resume_step):06d}.pt"

                with bf.BlobFile(bf.join("", filename), "wb") as f:
                    th.save(state_dict, f)

        save_checkpoint(0, self.mp_trainer.master_params)

        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if dist.get_rank() == 0:
            with bf.BlobFile(
                bf.join("", f"edge_unet_opt{(self.step+self.resume_step):06d}.pt"),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

        def save_checkpoint1(rate, params):
            state_dict = self.mpdis_trainer.master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving edge discriminator model {rate}...")
                if not rate:
                    filename = f"edge_discriminator_model{(self.step+self.resume_step):06d}.pt"
                else:
                    filename = f"edge_discriminator_ema_{rate}_{(self.step+self.resume_step):06d}.pt"
                with bf.BlobFile(bf.join("", filename), "wb") as f:
                    th.save(state_dict, f)

        save_checkpoint1(0, self.mpdis_trainer.master_params)


        if dist.get_rank() == 0:
            with bf.BlobFile(
                bf.join("", f"edge_discri_opt{(self.step+self.resume_step):06d}.pt"),
                "wb",
            ) as f:
                th.save(self.dis_opt.state_dict(), f)


        def save_checkpoint2(rate, params):
            state_dict = self.mp_trainer1.master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"image saving unet model {rate}...")
                if not rate:
                    filename = f"image_unet_model{(self.step+self.resume_step):06d}.pt"
                else:
                    filename = f"image_unet_ema_{rate}_{(self.step+self.resume_step):06d}.pt"

                with bf.BlobFile(bf.join("", filename), "wb") as f:
                    th.save(state_dict, f)

        save_checkpoint2(0, self.mp_trainer1.master_params)

        for rate, params in zip(self.ema_rate, self.ema_params1):
            save_checkpoint2(rate, params)

        if dist.get_rank() == 0:
            with bf.BlobFile(
                bf.join("", f"image_unet_opt{(self.step+self.resume_step):06d}.pt"),
                "wb",
            ) as f:
                th.save(self.opt1.state_dict(), f)

        def save_checkpoint3(rate, params):
            state_dict = self.mpdis_trainer1.master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving image discriminator model {rate}...")
                if not rate:
                    filename = f"image_discriminator_model{(self.step+self.resume_step):06d}.pt"
                else:
                    filename = f"image_discriminator_ema_{rate}_{(self.step+self.resume_step):06d}.pt"
                with bf.BlobFile(bf.join("", filename), "wb") as f:
                    th.save(state_dict, f)

        save_checkpoint3(0, self.mpdis_trainer1.master_params)

        if dist.get_rank() == 0:
            with bf.BlobFile(
                bf.join("", f"image_discri_opt{(self.step+self.resume_step):06d}.pt"),
                "wb",
            ) as f:
                th.save(self.dis_opt1.state_dict(), f)


        dist.barrier()


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():

    return logger.get_dir()


def find_resume_checkpoint():

    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
