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


    
def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt)  # , map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    return model


def get_model():
    config = OmegaConf.load("configs/latent-diffusion/cin256-v2.yaml")
    model = load_model_from_config(config, "models/ldm/cin256-v2/model.ckpt")
    return model


def distill_caching_random():
    model = get_model()
    sampler = DDIMSampler(model)
    
    classes = [55, 1, 2, 3]  # define classes to be sampled here
    n_samples_per_class = 16
    
    ddim_steps = 20
    ddim_eta = 1.0
    scale = 1.5 # for unconditional guidance
    
    all_samples = list()
    
    with torch.no_grad():
        with model.ema_scope():
            uc = model.get_learned_conditioning(
                {model.cond_stage_key: torch.tensor(n_samples_per_class * [1000]).to(model.device)}
            )
    
            for class_label in classes:
                print(f"rendering {n_samples_per_class} examples of class '{class_label}' in {ddim_steps} steps and using s={scale:.2f}.")
                xc = torch.tensor(n_samples_per_class * [class_label])
                c = model.get_learned_conditioning({model.cond_stage_key: xc.to(model.device)})
    
                samples_ddim, _ = sampler.sample(S=ddim_steps,
                                                 conditioning=c,
                                                 batch_size=n_samples_per_class,
                                                 shape=[3, 64, 64],
                                                 verbose=False,
                                                 unconditional_guidance_scale=scale,
                                                 unconditional_conditioning=uc,
                                                 eta=ddim_eta)
            
    
                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0,
                                             min=0.0, max=1.0)
                all_samples.append(x_samples_ddim)
    
    # display as grid
    grid = torch.stack(all_samples, 0)
    grid = rearrange(grid, 'n b c h w -> (n b) c h w')
    grid = make_grid(grid, nrow=n_samples_per_class)
    
    # to image
    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
    
    # save as image
    output_image = Image.fromarray(grid.astype(np.uint8))
    output_image.save('output.png')  # 파일로 저장

def main(argv):
    warnings.simplefilter(action='ignore', category=FutureWarning)
    
    # distill_caching_base()
    distill_caching_random()
    
    # print(torch.cuda.device_count())  # 시스템에서 사용 가능한 GPU 수를 출력
    # print(torch.cuda.current_device())  # 현재 사용 중인 GPU 장치 번호를 출력


if __name__ == '__main__':
    app.run(main)
