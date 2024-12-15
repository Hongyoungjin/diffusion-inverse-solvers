import os.path
import cv2
import logging

import numpy as np
import torch
import torch.nn.functional as F
from datetime import datetime
from collections import OrderedDict
import hdf5storage

from utils import utils_model
from utils import utils_logger
from utils import utils_sisr as sr
from utils import utils_image as util
from utils.utils_resizer import Resizer
from utils.utils_deblur import MotionBlurOperator, GaussialBlurOperator
from utils.utils_inpaint import mask_generator
from scipy import ndimage

from functools import partial

import yaml
import argparse
import shutil
import random
from piq import ssim
from torch.utils.data import Dataset, DataLoader

# from guided_diffusion import dist_util
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
)

class CustomDataset(Dataset):
    def __init__(self, img_paths, config):
        self.img_paths = img_paths
        self.config = config

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]

        # --------------------------------
        # load kernel
        # --------------------------------

        if self.config.task == "sr":
            kernels = hdf5storage.loadmat(os.path.join(self.config.cwd, 'kernels', 'kernels_bicubicx234.mat'))['kernels']
            k_index = self.config.sf-2 if self.config.sf < 5 else 2
            k = kernels[0, k_index].astype(np.float64)
        elif self.config.task == 'deblur':
            if self.config.use_DIY_kernel:
                np.random.seed(seed=idx*10)  # for reproducibility of blur kernel for each image
                if self.config.blur_mode == 'Gaussian':
                    kernel_std_i = self.config.kernel_std * np.abs(np.random.rand()*2+1)
                    kernel = GaussialBlurOperator(kernel_size=self.config.kernel_size, intensity=kernel_std_i, device=self.config.device)
                elif self.config.blur_mode == 'motion':
                    kernel = MotionBlurOperator(kernel_size=self.config.kernel_size, intensity=self.config.kernel_std, device=self.config.device)
                k_tensor = kernel.get_kernel().to(self.config.device, dtype=torch.float)
                k = k_tensor.clone().detach().cpu().numpy()       #[0,1]
                k = np.squeeze(k)
                k = np.squeeze(k)
            else:
                k_index = 0
                kernels = hdf5storage.loadmat(os.path.join(self.config.cwd, 'kernels', 'Levin09.mat'))['kernels']
                k = kernels[0, k_index].astype(np.float32)
        else:
            k = torch.ones((1,1,1,1)) # dummy kernel
        
        # --------------------------------
        # get Measurement
        # --------------------------------

        img_name= os.path.basename(img_path)
        Reference = util.imread_uint(img_path, n_channels=self.config.n_channels)
        Reference = util.modcrop(Reference, self.config.sf)  # modcrop
        if self.config.task == "sr":
            Reference_tensor = np.transpose(Reference, (2, 0, 1))
            Reference_tensor = torch.from_numpy(Reference_tensor)[None,:,:,:].to(self.config.device)
            Reference_tensor = Reference_tensor / 255 
            down_sample = Resizer(Reference_tensor.shape, 1/self.config.sf).to(self.config.device)
            if self.config.sr_mode == 'blur':
                Measurement = util.imresize_np(util.uint2single(Reference), 1/self.config.sf)
            elif self.config.sr_mode == 'cubic':
                Measurement = down_sample(Reference_tensor)
                Measurement = Measurement.cpu().numpy()       #[0,1]
                Measurement = np.squeeze(Measurement)
                if Measurement.ndim == 3:
                    Measurement = np.transpose(Measurement, (1, 2, 0))
            mask = np.ones_like(Measurement)
        elif self.config.task == 'deblur':
            # mode='wrap' is important for analytical solution
            Measurement = ndimage.convolve(Reference, np.expand_dims(k, axis=2), mode='wrap')
            Measurement = util.uint2single(Measurement)
            mask = np.ones_like(Measurement)
        elif self.config.task == 'denoise':
            def shot_noise(x, c):
                x = x / 255.
                return np.clip(np.random.poisson(x * c) / c, 0, 1)
            Measurement = shot_noise(Reference, self.config.shot_noise_c)
            mask = np.ones_like(Measurement)
        elif self.config.task == 'inpaint':
            if self.config.load_mask:
                mask = util.imread_uint(self.config.mask_path, n_channels=self.config.n_channels).astype(bool)
            else:
                mask_gen = mask_generator(mask_type=self.config.mask_type, mask_len_range=self.config.mask_len_range, mask_prob_range=self.config.mask_prob_range)
                mask = mask_gen(util.uint2tensor4(Reference)).numpy()
                mask = np.squeeze(mask)
                mask = np.transpose(mask, (1, 2, 0))
            Measurement = Reference * mask  / 255.   #(256,256,3)         [0,1]

        Measurement = Measurement * 2 - 1
        Measurement += np.random.normal(0, self.config.noise_level_img * 2, Measurement.shape) # add AWGN
        Measurement = Measurement / 2 + 0.5

        # Return images names and kernels
        return Reference, Measurement, img_name, k, mask

class Config:
    def __init__(self, dictionary):
        for k, v in dictionary.items():
            if isinstance(v, dict):
                setattr(self, k, Config(v))
            else:
                setattr(self, k, v)

def parse_args_and_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--opt", type=str, help="Path to option YMAL file.")
    args = parser.parse_args()
    # Load the YAML file
    with open(args.opt, 'r') as file:
        config = yaml.safe_load(file)
    config = Config(config)
    config.world_size = torch.cuda.device_count()
    config.opt = args.opt

    config.noise_level_img = config.noise_level_img / 255. # noise level of noisy image
    # config.skip = config.num_train_timesteps // config.iter_num     # skip interval
    config.noise_level_model = config.noise_level_img   # set noise level of model, default: 0
    config.sigma = max(0.001, config.noise_level_img)  # noise level associated with condition y
    # paths
    config.model_zoo = os.path.join(config.cwd, 'model_zoo')    # fixed
    config.datasets = os.path.join(config.cwd, 'datasets')     # fixed
    config.results = os.path.join(config.cwd, 'results')      # fixed
    config.result_name = f'{config.testset_name}_{config.task}_{config.generate_mode}_{config.model_name}_sigma{config.noise_level_img}_NFE{config.iter_num}_eta{config.eta}_zeta{config.zeta}_lambda{config.lambda_}'
    if config.task == "sr":
        config.result_name += f'_{config.sr_mode}{str(config.sf)}'
    elif config.task == "deblur":
        config.kernel_std = config.kernel_std if config.blur_mode == 'Gaussian' else 1.0
        config.result_name += f'_blurmode_{config.blur_mode}kernel_std{config.kernel_std}_'
        
    elif config.task == "inpaint":
        if config.mask_type == "random":
            config.result_name += f'_mask_{config.mask_prob_range}_type_{config.mask_type}'
        elif config.mask_type == "box":
            config.result_name += f'_mask_{config.mask_len_range}_type_{config.mask_type}'
        elif config.mask_type == "both":
            config.result_name += f'_mask_{config.mask_len_range}_{config.mask_prob_range}_type_{config.mask_type}'
        else:
            config.result_name += f'_mask_type_{config.mask_type}'    
        assert config.generate_mode in ['DiffPIR', 'repaint', 'vanilla']

    config.model_path = os.path.join(config.model_zoo, config.model_name+'.pt')
    config.measurement_path = os.path.join(config.datasets, config.testset_name) # measurement_path, for Low-quality images
    config.reference_path = os.path.join(config.results, config.result_name)   # reference_path, for Estimated images
    util.mkdir(config.reference_path)

    # set random seed everywhere
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)  # for multi-GPU.
    np.random.seed(config.seed)  # Numpy module.
    random.seed(config.seed)  # Python random module.
    torch.manual_seed(config.seed)
    return config


def main():

    # ----------------------------------------
    # Preparation
    # ----------------------------------------

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = parse_args_and_config()
    config.device = device
    measurement_paths = util.get_image_paths(config.measurement_path)

    # schedule
    betas = np.linspace(config.beta_start, config.beta_end, config.num_train_timesteps, dtype=np.float32)
    betas                   = torch.from_numpy(betas).to(device)
    alphas                  = 1.0 - betas
    alphas_cumprod          = np.cumprod(alphas.cpu(), axis=0)
    sqrt_alphas_cumprod     = torch.sqrt(alphas_cumprod)
    sqrt_1m_alphas_cumprod  = torch.sqrt(1. - alphas_cumprod)
    reduced_alpha_cumprod   = torch.div(sqrt_1m_alphas_cumprod, sqrt_alphas_cumprod)        # equivalent noise sigma on image

    if config.skip_noise_model_t:
        config.noise_model_t = utils_model.find_nearest(reduced_alpha_cumprod, 2 * config.noise_level_model)
    else:
        config.noise_model_t = 0

    if config.noise_init_img == 'max':
        config.t_start = config.num_train_timesteps - 1   
    else:
        config.t_start = utils_model.find_nearest(reduced_alpha_cumprod, 2 * config.noise_init_img / 255) # start timestep of the diffusion process

    # set up logger
    logger_name = config.result_name
    utils_logger.logger_info(logger_name, log_path=os.path.join(config.reference_path, logger_name+'.log'))
    logger = logging.getLogger(logger_name)

    # ----------------------------------------
    # load datasets
    # ----------------------------------------
    # Assuming you have measurement_paths as your list of image file paths
    dataset = CustomDataset(measurement_paths, config)
    # Define batch size and create a DataLoader
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)

    # ----------------------------------------
    # load model
    # ----------------------------------------

    model_config = dict(
            model_path=config.model_path,
            num_channels=128,
            num_res_blocks=1,
            attention_resolutions="16",
        ) if config.model_name == 'ffhq_10m' \
        else dict(
            model_path=config.model_path,
            num_channels=256,
            num_res_blocks=2,
            attention_resolutions="8,16,32",
        )
    
    args = utils_model.create_argparser(model_config).parse_args([])
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys()))
    model.load_state_dict(torch.load(args.model_path, map_location="cpu"))
    model.eval()
    if config.generate_mode != 'DPS_y0':
        # for DPS_yt, we can avoid backward through the model
        for k, v in model.named_parameters():
            v.requires_grad = False
    model = model.to(device)

    # save config
    shutil.copyfile(config.opt, os.path.join(config.reference_path, os.path.basename('config.yaml')))

    # Core of DiffPIR

    def test_rho(config): 
        parameters = f'eta:{config.eta}, zeta:{config.zeta}, lambda:{config.lambda_}, guidance_scale:{config.guidance_scale}'
        parameters = parameters + f', inIter:{config.inIter}, gamma:{config.gamma}' if (config.task == "sr" and config.sr_mode == 'cubic') else parameters
        logger.info(parameters)
        test_results = OrderedDict()
        test_results['psnr'] = []
        test_results['psnr_y'] = []
        test_results['ssim'] = []
        if config.calc_LPIPS:
            test_results['lpips'] = []
        total_num = 0
        
        def shot_noise(x, c):
            lambda_ = torch.clamp(x * c, min=0)
            result = torch.clamp(torch.poisson(lambda_) / c, 0, 1)
            return result
        
        for idx, batch in enumerate(dataloader):
            model_out_type = config.model_output_type
            batch_size = batch[0].shape[0]
            C, H, W = batch[0].shape[3], batch[0].shape[1], batch[0].shape[2]
            Reference, Measurement, names, k, mask = batch
            # convert to numpy
            Reference = Reference.numpy()
            Measurement = Measurement.numpy()
            k = k.numpy()
            mask = mask.numpy()
            
            # --------------------------------
            # (2) get rhos and sigmas
            # -------------------------------- 
            
            sigmas = []
            sigma_ks = []
            rhos = []
            for i in range(config.num_train_timesteps):
                sigmas.append(reduced_alpha_cumprod[config.num_train_timesteps-1-i])
                if model_out_type == 'pred_xstart' and config.generate_mode == 'DiffPIR':
                    sigma_ks.append((sqrt_1m_alphas_cumprod[i]/sqrt_alphas_cumprod[i]))
                #elif model_out_type == 'pred_x_prev':
                else:
                    sigma_ks.append(torch.sqrt(betas[i]/alphas[i]))
                rhos.append(config.lambda_*(config.sigma**2)/(sigma_ks[i]**2))
                    
            rhos, sigmas, sigma_ks = torch.tensor(rhos).to(config.device), torch.tensor(sigmas).to(config.device), torch.tensor(sigma_ks).to(config.device)
            
            # --------------------------------
            # (3) initialize x, and pre-calculation
            # --------------------------------
            y = util.single2tensor4_batch(Measurement).to(config.device)   #(1,3,256,256) [0,1]

            if config.task == "sr":
                degrade_op = Resizer((batch_size, C, H, W), 1/config.sf).to(config.device)
                x = F.interpolate(torch.from_numpy(Measurement).permute(0, 3, 1, 2), size=(Measurement.shape[1]*config.sf, Measurement.shape[2]*config.sf), mode='bicubic', align_corners=False).to(config.device)
                if config.sr_mode == 'cubic':
                    up_sample = partial(F.interpolate, scale_factor=config.sf)
            elif config.task == "deblur":
                util.imsave_batch(k*255.*200, names, config.reference_path, 'motion_kernel_')
                #np.save(os.path.join(reference_path, 'motion_kernel.npy'), k)
                k_4d = torch.from_numpy(k).to(device)
                # print("torch.eye(3): ", torch.eye(3).shape)
                # print("k_4d: ", k_4d.shape)
                k_4d = k_4d.unsqueeze(1)

                x = y
                def degrade_op(x):
                    x = x / 2 + 0.5
                    pad_2d = torch.nn.ReflectionPad2d(k.shape[0]//2)
                    x_blurs = []
                    for i in range(x.shape[0]):
                        x_blurs.append(F.conv2d(pad_2d(x[i:i+1]), k_4d))
                    return torch.cat(x_blurs, 0)
            elif config.task == 'inpaint':
                Measurement = Measurement * mask
                mask = util.single2tensor4_batch(mask.astype(np.float32)).to(device)
                x = y * mask
            elif config.task == "denoise":
                x = shot_noise(torch.clamp(y, 0, 1), config.shot_noise_c)

            x = sqrt_alphas_cumprod[config.t_start] * (2*x-1) + sqrt_1m_alphas_cumprod[config.t_start] * torch.randn_like(x)
            

            if config.task in ['sr', 'deblur']:
                k_tensor = util.single2tensor4_batch(np.expand_dims(k, 3)).to(config.device) 
                FB, FBC, F2B, FBFy = sr.pre_calculate(y, k_tensor, config.sf)

            # create sequence of timestep for sampling
            skip = config.num_train_timesteps // config.iter_num
            if config.skip_type == 'uniform':
                seq = [i*skip for i in range(config.iter_num)]
                if skip > 1:
                    seq.append(config.num_train_timesteps-1)
            elif config.skip_type == "quad":
                seq = np.sqrt(np.linspace(0, config.num_train_timesteps**2, config.iter_num))
                seq = [int(s) for s in list(seq)]
                seq[-1] = seq[-1] - 1
            progress_seq = seq[::max(len(seq)//10,1)]
            if progress_seq[-1] != seq[-1]:
                progress_seq.append(seq[-1])
            
            # DiffPIR doesn't initialize from random noise
            # reverse diffusion for one image from random noise
            for i in range(len(seq)):
                curr_sigma = sigmas[seq[i]].cpu().numpy()
                # time step associated with the noise level sigmas[i] to find the initial time step
                t_i = utils_model.find_nearest(reduced_alpha_cumprod,curr_sigma)
                # skip iters where the time steps is smaller than t_i
                if t_i > config.t_start:
                    continue
                # repeat for semantic consistence: from repaint
                for u in range(config.iter_num_U):

                    # add noise, make the image noise level consistent in pixel level (risky)
                    if config.task == "inpaint":
                        if config.generate_mode == 'repaint':
                            x = (sqrt_alphas_cumprod[t_i] * (2*y-1) + sqrt_1m_alphas_cumprod[t_i] * torch.randn_like(x)) * mask \
                                    + (1-mask) * x

                        # solve equation 6b with one reverse diffusion step
                        if model_out_type == 'pred_xstart':
                            x0 = utils_model.model_fn(x, noise_level=curr_sigma*255, model_out_type=model_out_type, \
                                    model_diffusion=model, diffusion=diffusion, ddim_sample=config.ddim_sample, alphas_cumprod=alphas_cumprod)
                        else:
                            x = utils_model.model_fn(x, noise_level=curr_sigma*255, model_out_type=model_out_type, \
                                    model_diffusion=model, diffusion=diffusion, ddim_sample=config.ddim_sample, alphas_cumprod=alphas_cumprod)
                    else:
                        if 'DPS' in config.generate_mode:
                            x = x.requires_grad_()
                            xt, x0 = utils_model.model_fn(x, noise_level=curr_sigma*255, model_out_type='pred_x_prev_and_start', \
                                        model_diffusion=model, diffusion=diffusion, ddim_sample=config.ddim_sample, alphas_cumprod=alphas_cumprod)
                        else:
                            x0 = utils_model.model_fn(x, noise_level=curr_sigma*255, model_out_type=model_out_type, \
                                    model_diffusion=model, diffusion=diffusion, ddim_sample=config.ddim_sample, alphas_cumprod=alphas_cumprod)
                            
                    # degradation functions -> minimizing the reconstruction distance between the re-degradaed image and measurement

                    if seq[i] != seq[-1]:
                        if config.sub_1_analytic:
                            if model_out_type == 'pred_xstart':

                                tau = rhos[t_i].float().repeat(1, 1, 1, 1)
                                # when noise level less than given image noise, skip
                                if i < config.num_train_timesteps-config.noise_model_t: 
                                    if config.task == "inpaint":
                                        x0_p = (mask * (2*y-1) + tau * x0).div(mask + tau)
                                        x0 = x0 + config.guidance_scale * (x0_p-x0)
                                    elif config.task == "denoise":
                                        x0_p = shot_noise(y, config.shot_noise_c) + tau * x0
                                        x0 = x0 + config.guidance_scale * (x0_p-x0)
                                    elif config.task == "deblur" or config.sr_mode == 'blur':
                                        x0_p = x0 / 2 + 0.5
                                        x0_p = sr.data_solution(x0_p.float(), FB, FBC, F2B, FBFy, tau, config.sf)
                                        x0_p = x0_p * 2 - 1
                                        # effective x0
                                        x0 = x0 + config.guidance_scale * (x0_p-x0)
                                    elif config.sr_mode == 'cubic': 
                                        # iterative back-projection (IBP) solution
                                        for _ in range(config.inIter):
                                            x0 = x0 / 2 + 0.5
                                            x0 = x0 + config.gamma * up_sample((y - degrade_op(x0))) / (1+rhos[t_i])
                                            x0 = x0 * 2 - 1
                                else:
                                    model_out_type = 'pred_x_prev'
                                    x0 = utils_model.model_fn(x, noise_level=curr_sigma*255,model_out_type=model_out_type, \
                                            model_diffusion=model, diffusion=diffusion, ddim_sample=config.ddim_sample, alphas_cumprod=alphas_cumprod)
                                    pass
                            elif model_out_type == 'pred_x_prev' and config.task == "inpaint":
                                # when noise level less than given image noise, skip
                                if i < config.num_train_timesteps-config.noise_model_t: 
                                    x = (mask * (2*y-1) + tau * x0).div(mask + tau) # y-->yt ?
                                else:
                                    pass
                        else:
                            x0 = x0.requires_grad_()
                            if config.task == "deblur":
                                measurement = y  
                            else:
                                measurement = 2 * y - 1
                            norm_grad, norm = utils_model.grad_and_value(operator=degrade_op,x=x0, x_hat=x0, measurement=measurement)
                                                
                            x0 = x0 - norm_grad * norm / (rhos[t_i]) 
                            x0 = x0.detach_()
                            pass                          

                    # Core step that distinguish this method from the others: add noise back to t=i-1 (very risky)
                    if ((config.task == "inpaint" or config.generate_mode == 'DiffPIR') and model_out_type == 'pred_xstart') and not (seq[i] == seq[-1] and u == config.iter_num_U-1):
                        
                        t_im1 = utils_model.find_nearest(reduced_alpha_cumprod,sigmas[seq[i+1]].cpu().numpy())
                        # calculate \hat{\eposilon}
                        eps = (x - sqrt_alphas_cumprod[t_i] * x0) / sqrt_1m_alphas_cumprod[t_i]
                        eta_sigma = config.eta * sqrt_1m_alphas_cumprod[t_im1] / sqrt_1m_alphas_cumprod[t_i] * torch.sqrt(betas[t_i])
                        x = sqrt_alphas_cumprod[t_im1] * x0 + np.sqrt(1-config.zeta) * (torch.sqrt(sqrt_1m_alphas_cumprod[t_im1]**2 - eta_sigma**2) * eps \
                                    + eta_sigma * torch.randn_like(x)) + np.sqrt(config.zeta) * sqrt_1m_alphas_cumprod[t_im1] * torch.randn_like(x)
                    else:
                        # x = x0
                        pass
                        
                    # set back to x_t from x_{t-1}
                    if u < config.iter_num_U-1 and seq[i] != seq[-1]:
                        sqrt_alpha_effective = sqrt_alphas_cumprod[t_i] / sqrt_alphas_cumprod[t_im1]
                        x = sqrt_alpha_effective * x + torch.sqrt(sqrt_1m_alphas_cumprod[t_i]**2 - \
                                sqrt_alpha_effective**2 * sqrt_1m_alphas_cumprod[t_im1]**2) * torch.randn_like(x)

                # save the process
                x_0 = ( x / 2 + 0.5)

            total_num += batch_size

            # recover conditional part
            if config.task == "inpaint" and config.generate_mode in ['repaint','DiffPIR']:
                x[mask.to(torch.bool)] = (2*y-1)[mask.to(torch.bool)]

            # --------------------------------
            # (3) img_E
            # --------------------------------

            img_E = util.tensor2uint_batch(x_0)
            Reference_tensor = np.transpose(Reference, (0, 3, 1, 2))
            Reference_tensor = torch.from_numpy(Reference_tensor).to(device)
            Reference_tensor = Reference_tensor / 255 * 2 -1
            psnr = util.calculate_psnr_batch(x_0.detach()*2-1, Reference_tensor)
            test_results['psnr'].append(psnr * batch_size)
            ssim_socre = ssim(x_0.detach(), Reference_tensor / 2 + 0.5)
            test_results['ssim'].append(ssim_socre * batch_size)
            
            if config.calc_LPIPS:
                lpips_score = loss_fn_vgg(x_0.detach()*2-1, Reference_tensor)
                lpips_score = lpips_score.cpu().detach().numpy()[0][0][0][0]
                test_results['lpips'].append(lpips_score * batch_size)
                logger.info(f"batch{idx+1:->4d}--> PSNR: {psnr:.4f}dB; LPIPS: {lpips_score:.4f}; ave LPIPS: {sum(test_results['lpips']) / total_num:.4f}")
            else:
                logger.info(f'batch{idx+1:->4d}--> PSNR: {psnr:.4f}dB')

            if config.save_E:
                # util.imsave(img_E, os.path.join(config.reference_path, f"{img_name}_x{sf}_{config.model_name+ext}"))
                util.imsave_batch(img_E, names, config.reference_path, f"{config.model_name}_x{config.sf}_lambda{config.lambda_:.4f}_zeta{config.zeta:.4f}_")

            if config.n_channels == 1:
                Reference = Reference.squeeze()

            # --------------------------------
            # (4) Measurement
            # --------------------------------

            Measurement = util.single2uint(Measurement).squeeze()

            if config.save_L:
                util.imsave_batch(Measurement, names, config.reference_path, f"LR_x{config.sf}_")

            if config.n_channels == 3:
                img_E_y = util.rgb2ycbcr_batch(x_0.detach()*2-1, only_y=True)
                Reference_y = util.rgb2ycbcr_batch(Reference_tensor, only_y=True)
                psnr_y = util.calculate_psnr_batch(img_E_y, Reference_y)
                test_results['psnr_y'].append(psnr_y * batch_size)
            
        # --------------------------------
        # Average PSNR and LPIPS for all images
        # --------------------------------

        ave_psnr = sum(test_results['psnr']) / total_num
        logger.info(f'-----------> Average PSNR(RGB) of ({config.testset_name}) scale factor: ({config.sf}), sigma: ({config.noise_level_model:.3f}): {ave_psnr:.4f} dB')
        test_results_ave['psnr_sf'].append(ave_psnr)
        
        ave_ssim = sum(test_results['ssim']) / total_num
        logger.info(f'-----------> Average SSIM(RGB) of ({config.testset_name}) scale factor: ({config.sf}), sigma: ({config.noise_level_model:.3f}): {ave_ssim:.4f}')
        test_results_ave['ssim_sf'].append(ave_ssim)

        if config.n_channels == 3:  # RGB image
            ave_psnr_y = sum(test_results['psnr_y']) / total_num
            logger.info(f'-----------> Average PSNR(Y) of ({config.testset_name}) scale factor: ({config.sf}), sigma: ({config.noise_level_model:.3f}): {ave_psnr_y:.4f} dB')
            test_results_ave['psnr_y_sf'].append(ave_psnr_y)

        if config.calc_LPIPS:
            ave_lpips = sum(test_results['lpips']) / total_num
            logger.info(f'-----------> Average LPIPS of ({config.testset_name}) scale factor: ({config.sf}), sigma: ({config.noise_level_model:.3f}): {ave_lpips:.4f}')
            test_results_ave['lpips'].append(ave_lpips)    
        return test_results_ave


    test_results_ave = OrderedDict()
    test_results_ave['psnr_sf'] = []
    test_results_ave['psnr_y_sf'] = []
    test_results_ave['ssim_sf'] = []
    if config.calc_LPIPS:
        import lpips
        loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)
        test_results_ave['lpips'] = []


    if config.task == "sr":
        ### SR
        border = config.sf
        logger.info('--------- sf:{:>1d} ---------'.format(config.sf))

        # experiments
        lambdas = [config.lambda_*i for i in range(2,13)]
        for lambda_ in lambdas:
            for zeta_i in [config.zeta]:
                config.lambda_ = lambda_
                config.zeta = zeta_i
                test_results_ave = test_rho(config)
    elif config.task == "deblur":
        ### Deblur
        border = 0
        lambdas = [config.lambda_*i for i in range(7,8)]
        for lambda_ in lambdas:
            for zeta_i in [config.zeta*i for i in range(3,4)]:
                config.lambda_ = lambda_
                config.zeta = zeta_i
                test_results_ave = test_rho(config)
    elif config.task == "inpaint":
        ### Inpaint
        border = 0
        lambdas = [config.lambda_*i for i in range(1,2)]
        for lambda_ in lambdas:
            #for zeta_i in [0,0.3,0.8,0.9,1.0]:
            for zeta_i in [config.zeta*i for i in range(1,2)]:
                config.lambda_ = lambda_
                config.zeta = zeta_i
                test_results_ave = test_rho(config)
    elif config.task == "denoise":
        ### Denoise
        border = 0
        lambdas = [config.lambda_*i for i in range(1,2)]
        for lambda_ in lambdas:
            #for zeta_i in [0,0.3,0.8,0.9,1.0]:
            for zeta_i in [config.zeta*i for i in range(1,2)]:
                config.lambda_ = lambda_
                config.zeta = zeta_i
                test_results_ave = test_rho(config)



    # ---------------------------------------
    # Average PSNR and LPIPS for all sf and parameters
    # ---------------------------------------

    ave_psnr_sf = sum(test_results_ave['psnr_sf']) / len(test_results_ave['psnr_sf'])
    logger.info(f'-----------> Average PSNR of ({config.testset_name}) {ave_psnr_sf:.4f} dB')
    ave_ssim_sf = sum(test_results_ave['ssim_sf']) / len(test_results_ave['ssim_sf'])
    logger.info(f'-----------> Average SSIM of ({config.testset_name}) {ave_ssim_sf:.4f}')
    if config.n_channels == 3:
        ave_psnr_y_sf = sum(test_results_ave['psnr_y_sf']) / len(test_results_ave['psnr_y_sf'])
        logger.info(f'-----------> Average PSNR-Y of ({config.testset_name}) {ave_psnr_y_sf:.4f} dB')
    if config.calc_LPIPS:
        ave_lpips_sf = sum(test_results_ave['lpips']) / len(test_results_ave['lpips'])
        logger.info(f'-----------> Average LPIPS of ({config.testset_name}) {ave_lpips_sf:.4f}')

if __name__ == '__main__':

    main()
