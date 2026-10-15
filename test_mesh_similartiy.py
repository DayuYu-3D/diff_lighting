
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
GT_MESH = '/workspace/root/miniconda3/nLOD/data/mesh/building_multi.obj'
RESULT_MESH = '/workspace/root/miniconda3/nLOD/test/building_multi.obj_0.1/mesh.obj'
SAVE_DIR= 'tmp_imgs_test'
NUM_VIEWS = 16
NUM_SAMPLES = 10000

def render_imgs(cfg, glctx, base_mesh, target_mesh, view_height):
    # Projection matrix
    assert view_height < 10000
    proj_mtx = util.projection(x=0.4, n = 1, f=1000.0) # x为近平面的宽度,n为近平面，f为远平面
 
    # Compute AABB of base mesh. Used for centering during rendering TODO: Use pre frame AABB?
    base_mesh_aabb = mesh.aabb(base_mesh.eval())

    ### Compute smooth vertex normals
    target_mesh = mesh.auto_normals(target_mesh)

    ### Set up tangent space
    target_mesh = mesh.compute_tangents(target_mesh)
    render_base_mesh = mesh.compute_tangents(base_mesh)

    background = torch.ones((1, cfg.train_res[0], cfg.train_res[1], 3), dtype=torch.float32, device='cuda')

    spp = 1
    mesh_scale=2.0

    aabb_center = base_mesh_aabb[0] + (base_mesh_aabb[1] - base_mesh_aabb[0]) * 0.5
    numpy_aabb = tuple(tensor.cpu().numpy() for tensor in base_mesh_aabb)

    mvp = np.zeros((NUM_VIEWS, 4,4),  dtype=np.float32)
    campos   = np.zeros((NUM_VIEWS, 3), dtype=np.float32)
    lightpos = np.zeros((NUM_VIEWS, 3), dtype=np.float32)
    
    viewpoints = util.random_viewpoint_samples(numpy_aabb, view_height, NUM_VIEWS)
    viewpoints_up = cfg.camera_up
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

    #  Evaluate all mesh ops (may change when positions are modified etc) and center/align meshes
    _opt_base  = mesh.center_by_reference(render_base_mesh.eval(params), base_mesh_aabb, mesh_scale)
    _opt_detail = mesh.center_by_reference(target_mesh.eval(params), base_mesh_aabb, mesh_scale)

    with torch.no_grad():
        color_base = render.render_mesh(glctx, _opt_base, mvp, campos, lightpos, cfg.light_power, cfg.train_res, 
            spp=spp, num_layers=1,msaa=True , background=background, min_roughness=cfg.min_roughness)
        color_opt = render.render_mesh(glctx, _opt_detail, mvp, campos, lightpos, cfg.light_power, cfg.train_res, 
            spp=spp, num_layers=1, msaa=True , background=background, min_roughness=cfg.min_roughness)

        img_out  =util.scale_img_nhwc(color_opt,  cfg.train_res)
        img_gt  = util.scale_img_nhwc(color_base,  cfg.train_res)        
        img_out = torch.clamp(img_out, 0.0, 1.0)
        img_gt  = torch.clamp(img_gt, 0.0, 1.0)      
        img_out = util.tonemap_srgb(img_out)
        img_gt  = util.tonemap_srgb(img_gt)

        print("size of img_out:", img_out.shape)
        
        result_image = torch.cat([img_out, img_gt], axis=2)
        for i in range(result_image.shape[0]):
            str_name = os.path.join(SAVE_DIR, os.path.basename(cfg.base_mesh+"_" + str(cfg.percent_rate)))   
            os.makedirs(str_name, exist_ok=True)
            img_path = os.path.join(str_name, 'img_%06d.png' % i)
            util.save_image(img_path, result_image[i].cpu().numpy())
        return img_gt, img_out

class test_mesh_simply:
    def __init__(self, cfg):
        self.cfg = cfg
        self.lpips_fn = lpips.LPIPS(net='alex').cuda()

    def do_computing(self):
        torch.cuda.empty_cache() 
 
        self.glctx = dr.RasterizeCudaContext()
        print("ssssssssssdf")
        base_mesh = self.load_mesh(GT_MESH) # base mesh
        print("Base mesh has %d triangles and %d vertices." % (base_mesh.t_pos_idx.shape[0], base_mesh.v_pos.shape[0]))
        print("Avg edge length of base mesh: %f" % regularizer.avg_edge_length(base_mesh))
        base_mesh, scale = mesh.unit_size(base_mesh) 
        print("The base mesh scale to: ", scale)
        output_mesh = self.load_mesh(RESULT_MESH) # output mesh
        print("Out mesh has %d triangles and %d vertices." % (output_mesh.t_pos_idx.shape[0], output_mesh.v_pos.shape[0]))
        print("Avg edge length of Out mesh: %f" % regularizer.avg_edge_length(output_mesh))

        # Check if the training texture resolution is acceptable
        base_texture_res = base_mesh.material['kd'].getRes()  ### kd 是 map_kd，若存在，否则为 kd
        if self.cfg.texture_res[0] < base_texture_res[0] or self.cfg.texture_res[1] < base_texture_res[1]:
            print("## WARNING: Picked a texture resolution lower than the base mesh [%d, %d] < [%d, %d]" % (self.cfg.texture_res[0],
             self.cfg.texture_res[1], base_texture_res[0], base_texture_res[1]))
        
        viewheight = 30 * scale
        imgs_gt, imgs_out = render_imgs(self.cfg, self.glctx, base_mesh, output_mesh, viewheight)     

        psnr_val, ssim_val, lpips_val = self.compute_vision_metrics(imgs_gt, imgs_out)
        chamfer_distance, geographic_distance, laplacian = self.compute_geometry_loss(base_mesh, output_mesh)
        print("==== Evaluation Results ====")
        print("PSNR: %.4f" % psnr_val)
        print("SSIM: %.4f" % ssim_val)
        print("LPIPS: %.4f" % lpips_val)
        print("Chamfer Distance: %.6f" % chamfer_distance)
        print("Geographic Distance: %.6f" % geographic_distance)
        print("Laplacian Regularizer: %.6f" % laplacian)
        return
        
    def load_mesh(self, filename, mtl_override=None): # mtl_override is a path to a .mtl file
        name, ext = os.path.splitext(filename)
        if ext == ".obj":
            return obj.load_obj(filename, clear_ks=True, mtl_override=mtl_override)
        assert False, "Invalid mesh file extension"

    def compute_geometry_loss(self, mesh_gt, mesh_out):
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

    def compute_vision_metrics(self, imgs_gt, imgs_out):
        # ===== PSNR =====
        # 批量计算每张图的 MSE
        mse = torch.mean((imgs_gt - imgs_out) ** 2, dim=[1, 2, 3])  # (N,)
        psnrs = 20 * torch.log10(1.0 / torch.sqrt(mse + 1e-8))
        psnrs = psnrs.cpu().numpy()

        # ===== SSIM =====
        # NHWC -> NCHW
        g_t = imgs_gt.permute(0, 3, 1, 2).cuda()
        o_t = imgs_out.permute(0, 3, 1, 2).cuda()
        # 调用支持 batch 的 SSIM 函数
        ssims = ssim(o_t, g_t, data_range=1)  # (N,)
        ssims = ssims.cpu().numpy()

        # ===== LPIPS =====
        g_l = (g_t * 2 - 1)
        o_l = (o_t * 2 - 1)
        lpips_vals = self.lpips_fn(o_l, g_l).view(-1).detach().cpu().numpy()  # (N,)

        # 返回均值
        return np.mean(psnrs), np.mean(ssims), np.mean(lpips_vals)


    def psnr(x,y):
        mse = np.mean((x-y)**2)
        return 20*np.log10(1.0/np.sqrt(mse+1e-8))
#----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='lod_simplification')
    parser.add_argument('--config', type=str, default='./building.yaml', help='Config file')

    cfg = parser.parse_args()

    if cfg.config is not None:
        with open(cfg.config, 'r') as f:
            data = yaml.safe_load(f)
            for key in data:
                setattr(cfg, key, data[key])
    sss = test_mesh_simply(cfg)
    sss.do_computing()


if __name__ == '__main__':
    main()
#----------------------------------------------------------------------------
