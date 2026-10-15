
import os
import time
import numpy as np
import torch
from collections import deque
from src import obj
from src import util
from src import mesh
from src import texture
from src import render
from src.loss_mesh.loss_mesh import LossMesh

import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
import yaml
import nvdiffrast.torch as dr
from src import regularizer

from src.loss_mesh.ssim import ssim_mask, ssim
from pytorch3d.loss import chamfer_distance
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import (
    chamfer_distance, 
    mesh_edge_loss, 
    mesh_laplacian_smoothing, 
    mesh_normal_consistency,
    point_mesh_face_distance
)
import lpips

RADIUS = 3.5
SAVE_DIR= 'tmp_imgs_test'
NUM_VIEWS = 16
NUM_SAMPLES = 10000
lpips_fn = lpips.LPIPS(net='alex').cuda()

#----------------------------------------------------------------------------

def val(cfg, it_num, glctx, base_mesh, numpy_aabb, aabb_center, proj_mtx, view_height, 
                base_mesh_aabb, background, mesh_scale, opt_mesh,spp,render_base_mesh):
    viewpoints = util.random_viewpoint_samples_z(numpy_aabb, view_height, NUM_VIEWS)
    viewpoints_up = cfg.camera_up
    mvp = np.zeros((NUM_VIEWS, 4,4),  dtype=np.float32)
    campos   = np.zeros((NUM_VIEWS, 3), dtype=np.float32)
    lightpos = np.zeros((NUM_VIEWS, 3), dtype=np.float32)
    for b in range(NUM_VIEWS):  #  Build transform stack for minibatching
        eye = np.array(viewpoints[b])
        up  = np.array(viewpoints_up)
        if cfg.camera_view_random:
            len_vertex = base_mesh.v_pos.shape[0]
            rand_point_idx = np.random.randint(0, len_vertex)
            at = base_mesh.v_pos[rand_point_idx].cpu().numpy()
        else:
            at  = np.array(aabb_center.cpu().numpy())
        # at  = np.array(aabb_center.cpu().numpy())
        a_mv =  util.lookAt(eye, at, up)
        mvp[b]  = np.matmul(proj_mtx, a_mv).astype(np.float32)[None, ...]
        campos[b] = np.linalg.inv(a_mv)[None, :3, 3]
        # lightpos[b] = util.cosine_sample(campos[b])*RADIUS
        lightpos[b] = campos[b]

    params = {'mvp' : mvp, 'lightpos' : lightpos, 'campos' : campos, 'resolution' : cfg.train_res, 'time' : 0}

    with torch.no_grad():
        # Center meshes
        _opt_detail = mesh.center_by_reference(opt_mesh.eval(params), base_mesh_aabb, mesh_scale)
        _opt_base    = mesh.center_by_reference(render_base_mesh.eval(params), base_mesh_aabb, mesh_scale)

        img_opt = render.render_mesh(glctx, _opt_detail, mvp, campos, lightpos,cfg.light_power, cfg.train_res, 
                                        num_layers=cfg.num_layer, background=background, min_roughness=cfg.min_roughness)
        img_base = render.render_mesh(glctx, _opt_base, mvp, campos, lightpos, cfg.light_power, cfg.train_res,
                                        num_layers=1, spp=spp, background=background, min_roughness=cfg.min_roughness)
        img_out  =util.scale_img_nhwc(img_opt,  cfg.train_res)
        img_gt  = util.scale_img_nhwc(img_base,  cfg.train_res)        
        img_out = torch.clamp(img_out, 0.0, 1.0)
        img_gt  = torch.clamp(img_gt, 0.0, 1.0)      
        img_out = util.tonemap_srgb(img_out)
        img_gt  = util.tonemap_srgb(img_gt)


    psnr_val, ssim_val, lpips_val = compute_vision_metrics(img_gt, img_out)
    chamfer_distance, geographic_distance, laplacian = compute_geometry_loss(base_mesh, opt_mesh.eval())
    print("==== Evaluation Results ====")
    print("PSNR: %.4f" % psnr_val)
    print("SSIM: %.4f" % ssim_val)
    print("LPIPS: %.4f" % lpips_val)
    print("Chamfer Distance: %.6f" % chamfer_distance)
    print("Geographic Distance: %.6f" % geographic_distance)
    print("Laplacian Regularizer: %.6f" % laplacian)

    ######save images
    result_image = torch.cat([img_base, img_opt], axis=2)[0].detach().cpu().numpy()
    str_name = os.path.join(cfg.out_dir, os.path.basename(cfg.base_mesh+"_" + str(cfg.percent_rate)))   
    os.makedirs(str_name, exist_ok=True)
    img_path = os.path.join(str_name, 'img_%06d.png' % it_num)
    util.save_image(img_path, result_image)

    ###### write the mesh to obj file
    if cfg.b_saveMesh:
        obj_path = os.path.join(str_name, str(it_num)) 
        os.makedirs(obj_path, exist_ok=True)      
        obj.write_obj(obj_path, opt_mesh.eval())  


    return {'psnr': psnr_val, 'ssim': ssim_val, 'lpips': lpips_val,
            'chamfer_distance': chamfer_distance,
            'geographic_distance': geographic_distance,
            'laplacian': laplacian} 


#----------------------------------------------------------------------------
def compute_vision_metrics(imgs_gt, imgs_out):
    # ===== PSNR =====
    # 批量计算每张图的 MSE
    mse = torch.mean((imgs_gt - imgs_out) ** 2, dim=[1, 2, 3])  # (N,)
    psnrs = 20 * torch.log10(1.0 / torch.sqrt(mse + 1e-8))
    psnrs = psnrs.cpu().contiguous().numpy()

    # ===== SSIM =====
    # NHWC -> NCHW
    g_t = imgs_gt.permute(0, 3, 1, 2).cuda()
    o_t = imgs_out.permute(0, 3, 1, 2).cuda()
    # 调用支持 batch 的 SSIM 函数
    ssims = ssim(o_t, g_t, data_range=1)  # (N,)
    ssims = ssims.contiguous().cpu().numpy()

    # ===== LPIPS =====
    g_l = (g_t * 2 - 1)
    o_l = (o_t * 2 - 1)
    lpips_vals = lpips_fn(o_l, g_l).contiguous().view(-1).detach().cpu().numpy()  # (N,)

    # 返回均值
    return np.mean(psnrs), np.mean(ssims), np.mean(lpips_vals)

def psnr(x,y):
    mse = np.mean((x-y)**2)
    return 20*np.log10(1.0/np.sqrt(mse+1e-8))

def compute_geometry_loss(mesh_gt, mesh_out):
    v_gt = mesh_gt.v_pos.unsqueeze(0)      # (1,N,3)
    f_gt = mesh_gt.t_pos_idx.unsqueeze(0)
    v_out = mesh_out.v_pos.unsqueeze(0)
    f_out = mesh_out.t_pos_idx.unsqueeze(0)
    m_gt = Meshes(verts=v_gt, faces=f_gt)
    m_out = Meshes(verts=v_out, faces=f_out)
    
    # Chamfer distance on points from sampled surface
    sample_gt = sample_points_from_meshes(m_gt, NUM_SAMPLES)   # (1,N,3)
    sample_out = sample_points_from_meshes(m_out, NUM_SAMPLES)

    chamfer, _ = chamfer_distance(sample_gt, sample_out)
    # print("Raw chamfer: ", chamfer.item())
    chamfer = torch.sqrt(chamfer + 1e-8)   # convert to L2 distance
    geometric_distance = point_mesh_face_distance(m_out,  Pointclouds(points=[sample_gt.squeeze(0)])) # torch3D
    # print("Raw geo dist: ", geometric_distance.item())
    geometric_distance = torch.sqrt(geometric_distance + 1e-8)
    loss_lap_opt = mesh_laplacian_smoothing(m_out, method="uniform")

    return chamfer.item(), geometric_distance.item(), loss_lap_opt.item()