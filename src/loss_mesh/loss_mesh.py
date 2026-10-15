import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import (
    chamfer_distance, 
    mesh_edge_loss, 
    mesh_laplacian_smoothing, 
    mesh_normal_consistency,
    point_mesh_face_distance
)

from src import util
from src import regularizer
import src.renderutils as ru

from .ssim import ssim_mask, ssim
from src import util

class LossMesh(nn.Module):
    def __init__(self, cfg, base_mesh, opt_mesh):
        super().__init__()
        #convert it to torch3d
        vpos = base_mesh.v_pos.unsqueeze(0)
        t_pos_idx = base_mesh.t_pos_idx.unsqueeze(0)
        self.base_mesh_3d = Meshes(verts=vpos, faces=t_pos_idx)

        self.lap_loss_fn = regularizer.laplace_regularizer_const(opt_mesh)
        self.edge_loss_fn = regularizer.edge_length_regularizer(opt_mesh)
        # normal_loss_fn = regularizer.face_normal_regularizer(opt_mesh)            
        self.image_loss_l1 =  lambda img, ref, mask: ru.image_loss(img, ref, mask, loss='l1', tonemapper="none", use_python=True)
        self.cfg = cfg
        self.opt_mesh = opt_mesh
        self._lap_fac_initialized = False

        # === move loss related ===
        self.prev_v_pos = None  # cache previous vertex positions        
        
    def forward(self, color_opt, color_base, params):
        """
        :param outputs: outputs from P2MModel
        :return: loss, loss_summary (dict)
        """
        num_points = 5000
        opt_mesh = self.opt_mesh.eval(params)
        vpos_opt = opt_mesh.v_pos.unsqueeze(0)
        t_pos_idx_opt = opt_mesh.t_pos_idx.unsqueeze(0)
        opt_mesh_3d = Meshes(verts=vpos_opt, faces=t_pos_idx_opt)

        sampled_points = sample_points_from_meshes(opt_mesh_3d, num_points)  # (1, N, 3)
        # print("sampled_points.shape", sampled_points.shape)
        # 计算距离 (注意：返回的是平方距离)
        loss_distances = point_mesh_face_distance(self.base_mesh_3d,  Pointclouds(points=[sampled_points.squeeze(0)])) # torch3D
        loss_lap_opt = mesh_laplacian_smoothing(opt_mesh_3d, method="uniform")

        # loss_normal_opt = mesh_normal_consistency(opt_mesh_3d)

        # compute the mask for valid pixels of img_base,  the pixels that are not background, i.e, both of r g b is 1
        color_base_mask = util.compute_mask(color_base)
        color_opt_mask = util.compute_mask(color_opt)
        mask = color_base_mask | color_opt_mask
        # util.save_mask(cfg.out_dir + '/' + ('mask_test2.png'), mask[0].detach().cpu().numpy()  ) 
        # return

       
        # tonemaper 
        color_base = torch.clamp(color_base, 0.0, 1.0)
        color_opt  = torch.clamp(color_opt, 0.0, 1.0)             
        color_base = util.tonemap_srgb(color_base)
        color_opt = util.tonemap_srgb(color_opt)

        whole_img_l1 = self.image_loss_l1(color_opt, color_base,None)

        #  Compute loss: image-space loss
        img_l1 = self.image_loss_l1(color_opt, color_base, mask)   

        # img_loss = ((color_opt - color_base) ** 2).mean()      
        lap_loss = self.lap_loss_fn.eval(params)
        # _, ml_s = loss_mesh_fn(opt_mesh.eval(params)) # Compute mesh loss
        edge_loss = self.edge_loss_fn.eval(params)
        # normal_loss = normal_loss_fn.eval(params)
        
        color_base = color_base.permute(0, 3, 1, 2).contiguous()
        color_opt  = color_opt.permute(0, 3, 1, 2).contiguous()
        # SSIM expects input in [0,1], shape (B,C,H,W)
        # ssim_loss = 1.0 - ssim(
        #     color_base,
        #     color_opt,
        #     data_range=1.0,
        #     size_average=True
        # )                
        # Compute SSIM loss with mask
        ssim = 1- ssim_mask(
            color_base,
            color_opt,
            mask,
            data_range=1.0,
        )  # [B, 1, H, W]


        #=== Incremental move  loss related ====
        if self.prev_v_pos is None:
            # first iteration: no penalty
            move_loss = torch.zeros((), device=vpos_opt.device)
        else:
            # L_move = mean ||v_t - v_{t-1}||^2
            move_loss = ((vpos_opt - self.prev_v_pos) ** 2).sum(dim=1).mean()


        # Compute total aggregate loss
        #total_loss = whole_img_l1  + lap_loss  + edge_loss 
        total_loss = img_l1  + lap_loss  + edge_loss 
        self.prev_v_pos = vpos_opt.detach()

        return {
            "loss_total": total_loss,
            "loss_img": img_l1 ,
            "loss_whole_img": whole_img_l1,
            "loss_lap": lap_loss ,
            "loss_edge": edge_loss ,
            "loss_laplace": loss_lap_opt,
            "loss_distances": loss_distances,
            "loss_move": move_loss,
            "loss_ssim": ssim,
        }

#########################################split line###################################################        
