import torch
from omegaconf import OmegaConf
import numpy as np
from PIL import Image
from einops import rearrange
from torchvision.utils import make_grid

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from absl import app, flags
import warnings

from omegaconf import OmegaConf
import os
import math
import wandb
import random
import logging
import inspect
import argparse
import datetime
import subprocess
import warnings
import threading
import GPUtil
import diffusers
import argparse, os, sys, glob, yaml, math, random
from diffusers.optimization import get_scheduler
from diffusion import GaussianDiffusion_distillation_Trainer, GaussianDiffusionTrainer, GaussianDiffusionSampler, distillation_cache_Trainer, GaussianDiffusion_joint_Sampler




def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--learning_rate", type=float, default=3e-5, help="Learning rate for training")
    parser.add_argument("--scale_lr", type=bool, default=False, help="Flag to scale learning rate")
    parser.add_argument("--lr_warmup_steps", type=int, default=0, help="Number of learning rate warmup steps")
    parser.add_argument("--lr_scheduler", type=str, default="constant", help="Learning rate scheduler type")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of gradient accumulation steps")

    
    parser.add_argument("--trainable_modules", type=tuple, default=(None,), help="Tuple of trainable modules")
    parser.add_argument("--num_workers", type=int, default=32, help="Number of workers for data loading")
    parser.add_argument("--train_batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="Beta1 parameter for Adam optimizer")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="Beta2 parameter for Adam optimizer")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay for Adam optimizer")

    return parser

def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt)  # , map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    for param in model.parameters():
        param.requires_grad = False
    model.cuda()
    model.eval()
    return model


def get_model_teacher():
    config = OmegaConf.load("configs/latent-diffusion/cin256-v2.yaml")
    model = load_model_from_config(config, "models/ldm/cin256-v2/model.ckpt")
    return model

def load_model_from_config_without_ckpt(config):
    print("Initializing model without checkpoint")
    model = instantiate_from_config(config.model)
    for param in model.parameters():
        param.requires_grad = True
    model.cuda()  # 모델을 CUDA로 이동 (필요한 경우)
    model.eval()  # 평가 모드로 설정
    return model

def get_model_student():
    config = OmegaConf.load("configs/latent-diffusion/cin256-v2.yaml")
    model = load_model_from_config_without_ckpt(config)
    return model

def initialize_params(model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                if param.dim() > 1:  # Convolutional layers and Linear layers typically have more than 1 dimension
                    init.xavier_uniform_(param)
                else:
                    init.zeros_(param)


def distill_caching_random(args):
    teacher_model = get_model_teacher()
    student_model= get_model_student()
    initialize_params(student_model)

    optimizer = torch.optim.AdamW(
        trainable_params_student,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    sampler = DDIMSampler(model)
    
    trainer = distillation_cache_Trainer(
        teacher_model, student_model, FLAGS.train_is_feature, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T,
        FLAGS.mean_type, FLAGS.var_type, FLAGS.distill_features).to(device)
    joint_sampler = GaussianDiffusion_joint_Sampler(
        teacher_model, student_model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, FLAGS.img_size,
        FLAGS.mean_type, FLAGS.var_type).to(device)
    student_sampler = GaussianDiffusionSampler(
        student_model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, FLAGS.img_size,
        FLAGS.mean_type, FLAGS.var_type).to(device)
    

    
    # ###### prepare cache ######
    
    # img_cache = torch.randn(FLAGS.cache_n*1000, 3, FLAGS.img_size, FLAGS.img_size).to(device)
    # t_cache = torch.ones(FLAGS.cache_n*1000, dtype=torch.long, device=device)*(FLAGS.T-1)

    # with torch.no_grad():
    #     for i in range(FLAGS.T):
    #         start_time = time.time()
            
    #         start_idx = (i * FLAGS.cache_n)
    #         end_idx = start_idx + FLAGS.cache_n

    #         x_t = img_cache[start_idx:end_idx]
    #         t = t_cache[start_idx:end_idx]

    #         img_cache[start_idx:end_idx] = trainer.teacher_sampling(x_t, i)
    #         t_cache[start_idx:end_idx] = torch.ones(FLAGS.cache_n, dtype=torch.long, device=device)*(i)
    #         print(f"start_idx: {start_idx}, end_idx: {end_idx}")
    #         print(t_cache)

    #         elapsed_time = time.time() - start_time
    #         print(f"Iteration {i + 1}/{FLAGS.T} completed in {elapsed_time:.2f} seconds.")

    #         visualize_t_cache_distribution(t_cache)

    # ##################################
    
    img_cache = torch.load("img_cache_cache.pt")
    t_cache = torch.load("t_cache_cache.pt")
    
    with trange(FLAGS.total_steps, dynamic_ncols=True) as pbar:
        for step in pbar:
            optimizer.zero_grad()

            # Step 2: Randomly sample from img_cache and t_cache without shuffling
            indices = torch.randint(0, img_cache.size(0), (FLAGS.batch_size,), device=device)

            # Sample img_cache and t_cache using the random indices
            x_t = img_cache[indices]
            t = t_cache[indices]

            # Calculate distillation loss
            output_loss, x_t_, total_loss = trainer(x_t, t) -> 여기 부분에 아마 c_emb값 들어가야 할둣

            # Backward and optimize
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), FLAGS.grad_clip)
            optimizer.step()
            lr_scheduler.step()

            ### cache update ###
            img_cache[indices] = x_t_
            t_cache[indices] -= 1
            
            num_999 = torch.sum(t_cache == (FLAGS.T - 1)).item()

            if num_999 < FLAGS.cache_n:
                missing_999 = FLAGS.cache_n - num_999
                non_999_indices = (t_cache != (FLAGS.T - 1)).nonzero(as_tuple=True)[0]
                random_indices = torch.randperm(non_999_indices.size(0), device=device)[:missing_999]
                selected_indices = non_999_indices[random_indices]
                t_cache[selected_indices] = FLAGS.T - 1
                img_cache[selected_indices] = torch.randn(missing_999, 3, FLAGS.img_size, FLAGS.img_size, device=device)

            # t_cache에서 값이 0인 인덱스를 찾아 초기화
            zero_indices = (t_cache < 0).nonzero(as_tuple=True)[0]
            num_zero_indices = zero_indices.size(0)

            # 0인 인덱스가 있는 경우에만 초기화 수행
            if num_zero_indices > 0:
                # 0인 인덱스를 1에서 FLAGS.T-1 사이의 랜덤한 정수로 초기화
                t_cache[zero_indices] = torch.randint(0, FLAGS.T, size=(num_zero_indices,), dtype=torch.long, device=device)
                img_cache[zero_indices] = trainer.diffusion(img_cache[zero_indices],t_cache[zero_indices])



            if step % 100 == 0:  # 예를 들어, 100 스텝마다 시각화
                visualize_t_cache_distribution(t_cache)

            # Logging with WandB
            wandb.log({
                'distill_loss': total_loss.item(),
                'output_loss': output_loss.item()
                       }, step=step)
            pbar.set_postfix(distill_loss='%.3f' % total_loss.item())
             
            # Sample and save student outputs
            if FLAGS.sample_step > 0 and step % FLAGS.sample_step == 0:
                student_model.eval()
                with torch.no_grad():
                    x_T = torch.randn(FLAGS.sample_size, 3, FLAGS.img_size, FLAGS.img_size).to(device)
                    joint_samples = joint_sampler(x_T)
                    grid = (make_grid(joint_samples, nrow=16) + 1) / 2
                    
                    # Create the directory if it doesn't exist
                    sample_dir = os.path.join(FLAGS.logdir, 'sample')
                    os.makedirs(sample_dir, exist_ok=True)
                    
                    path = os.path.join(sample_dir, 'joint_%d.png' % step)
                    save_image(grid, path)
                    wandb.log({"joint_sample": wandb.Image(path)}, step=step)

                student_model.train()

            # Save student model
            if FLAGS.save_step > 0 and step % FLAGS.save_step == 0:
                ckpt = {
                    'student_model': student_model.state_dict(),
                    'sched': sched.state_dict(),
                    'optim': optim.state_dict(),
                    'step': step,
                }
                torch.save(ckpt, os.path.join(FLAGS.logdir, 'student_ckpt.pt'))

            # Evaluate student model
            ddim_pipeline = DDIMPipeline(
            unet=student_model,  # 학습된 UNet 모델
            scheduler=DDPMScheduler()  # DDIM 방식으로 샘플링하기 위한 스케줄러
            ).to("cuda")


            
            if  FLAGS.eval_step > 0 and step % FLAGS.eval_step == 0:
                student_IS, student_FID, _ = evaluate(ddim_pipeline, student_model)
                metrics = {
                    'Student_IS': student_IS[0],
                    'Student_IS_std': student_IS[1],
                    'Student_FID': student_FID,
                }
                pbar.write(
                    "%d/%d " % (step, FLAGS.total_steps) +
                    ", ".join('%s:%.3f' % (k, v) for k, v in metrics.items()))
                wandb.log(metrics, step=step)

    wandb.finish()

    
    

def main(argv):
    warnings.simplefilter(action='ignore', category=FutureWarning)
    
    # distill_caching_base()
    parser = get_parser()
    distill_args = parser.parse_args()
    seed_everything(distill_args.seed)
    distill_caching_random(distill_args)
    
    # print(torch.cuda.device_count())  # 시스템에서 사용 가능한 GPU 수를 출력
    # print(torch.cuda.current_device())  # 현재 사용 중인 GPU 장치 번호를 출력


if __name__ == '__main__':
    app.run(main)
