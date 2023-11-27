import torch
from omegaconf import OmegaConf
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, LMSDiscreteScheduler, KDPM2AncestralDiscreteScheduler, DPMSolverSinglestepScheduler,\
    DDIMScheduler
from my_model import unet_2d_condition
import json
from PIL import Image

from utils import Phrase2idx, setup_logger, save_image, draw_box

from RnB import compute_rnb
import hydra
import os
from tqdm import tqdm
from diffusers import StableDiffusionPipeline
import math
import numpy as np

torch.autograd.set_detect_anomaly(True)
T = 51

def inference(device, unet, vae, tokenizer, text_encoder, prompt, bboxes, phrases, cfg, logger, p_len=77):
    

    logger.info("Inference")
    logger.info(f"Prompt: {prompt}")
    logger.info(f"Phrases: {phrases}")

    # Get Object Positions

    logger.info("Conver Phrases to Object Positions")
    object_positions = Phrase2idx(prompt, phrases)

    # Encode Classifier Embeddings
    uncond_input = tokenizer(
        ["lowres, bad anatomy, bad hands, bad faces, text, error, missing fingers, extra digit, fewer digits, \
         cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry"] * \
            cfg.inference.batch_size, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt"
    )
    uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]

    # Encode Prompt
    input_ids = tokenizer(
            [prompt] * cfg.inference.batch_size,
            padding="max_length",
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        )

    cond_embeddings = text_encoder(input_ids.input_ids.to(device))[0]
    text_embeddings = torch.cat([uncond_embeddings, cond_embeddings])
    generator = torch.manual_seed(cfg.inference.rand_seed)  # Seed generator to create the inital latent noise

    latents = torch.randn(
        (cfg.inference.batch_size, 4, 64, 64),
        generator=generator,
    ).to(device)
    

    noise_scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
    
    noise_scheduler.set_timesteps(cfg.inference.timesteps)

    latents = latents * noise_scheduler.init_noise_sigma
    
    loss = torch.tensor(10000)

    global_iter = 1
    attn_weight = None
    for index, t in enumerate(tqdm(noise_scheduler.timesteps)):
        iteration = 0
        
        while loss / cfg.inference.loss_scale > cfg.inference.loss_threshold and iteration < cfg.inference.max_iter and index < cfg.inference.max_index_step:
            
            latents = latents.requires_grad_(True)
            latent_model_input = latents
            latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t)
            noise_pred, attn_map_integrated_up, attn_map_integrated_mid, attn_map_integrated_down, attn_self_up, attn_self_down = \
                unet(latent_model_input, t, encoder_hidden_states=cond_embeddings)
            attn16 = []
            for item in attn_self_up[0]:
                attn16.append(item)
            for item in attn_self_down[-1]:
                attn16.append(item)
            
            
            res = compute_rnb(attn_map_integrated_mid, attn_map_integrated_up, attn_maps_down=attn_map_integrated_down, \
                                       attn_self=attn16, bboxes=bboxes, object_positions=object_positions, iter=index, attn_weight=attn_weight) #* cfg.inference.loss_scale 
            

            loss = res[0]
            attn_weight = res[1]
        
            if loss != 0:
                grad_cond = torch.autograd.grad(loss.requires_grad_(True), [latents])[0]
                scale = 70
                latents = latents - grad_cond * scale 
                iteration += 1
                global_iter += 1
                torch.cuda.empty_cache()
            
            elif index < 5:
                loss = 10
                iteration += 1
                global_iter += 1
                torch.cuda.empty_cache()

        with torch.no_grad():
            latent_model_input = torch.cat([latents] * 2)

            latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t)
            noise_pred, attn_map_integrated_up, attn_map_integrated_mid, attn_map_integrated_down, attn_self_up, attn_self_down = \
                unet(latent_model_input, t, encoder_hidden_states=text_embeddings)

            noise_pred = noise_pred.sample

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + cfg.inference.classifier_free_guidance * (noise_pred_text - noise_pred_uncond)

            latents = noise_scheduler.step(noise_pred, t, latents).prev_sample
            torch.cuda.empty_cache()
        

    with torch.no_grad():
        logger.info("Decode Image...")
        latents = 1 / 0.18215 * latents
        image = vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]
        return pil_images
    

@hydra.main(version_base=None, config_path="conf", config_name="base_config")
def main(cfg):

    # build and load model
    with open(cfg.general.unet_config) as f:
        unet_config = json.load(f)


    unet = unet_2d_condition.UNet2DConditionModel(**unet_config).from_pretrained(cfg.general.model_path, subfolder="unet")
    tokenizer = CLIPTokenizer.from_pretrained(cfg.general.model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(cfg.general.model_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(cfg.general.model_path, subfolder="vae")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    unet.to(device)
    text_encoder.to(device)
    vae.to(device)
    
    
    # bounding box: [x1, y1, x2, y2].
    # (x1, y1): up-left
    # (x2, y2): down-right

    examples_list = [
        {"prompt": "Four shoes on the field with a football in the background",
            "phrases": "shoes;football",
            "bboxes": [
                [[0.08, 0.7, 0.28, 0.9], [0.3, 0.65, 0.5, 0.85], [0.55, 0.6, 0.75, 0.8], [0.77, 0.55, 0.97, 0.75]],
                [[0.2, 0.2, 0.5, 0.5]],
            ]
        }
    ]
    for examples in examples_list:


        if not os.path.exists(cfg.general.save_path):
            os.makedirs(cfg.general.save_path)
        logger = setup_logger(cfg.general.save_path, __name__)

        logger.info(cfg)
        # Save cfg
        logger.info("save config to {}".format(os.path.join(cfg.general.save_path, 'config.yaml')))
        OmegaConf.save(cfg, os.path.join(cfg.general.save_path, 'config.yaml'))

        # Inference
        pil_images = inference(device, unet, vae, tokenizer, text_encoder, examples['prompt'], examples['bboxes'], examples['phrases'], cfg, logger, p_len=torch.tensor([0, 1, 3, 5]).cuda())

        # Save example images
        for index, pil_image in enumerate(pil_images):
            image_path = os.path.join(cfg.general.save_path, f"{examples['prompt'][:100]}_{cfg.inference.rand_seed}.png")

            draw_box(pil_image, examples['bboxes'], examples['phrases'], image_path)
            save_image(pil_image, examples['bboxes'], examples['phrases'], image_path)


if __name__ == "__main__":
    main()