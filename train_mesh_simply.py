import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
import argparse
import yaml
import numpy as np
import torch
from collections import deque
from pathlib import Path
import copy
import gc
import traceback
import psutil
from datetime import datetime
import re

import nvdiffrast.torch as dr

from src import obj
from src import regularizer
from src import mesh
from src import util
from src import render
from src import vis_loss_decay
from mesh_sampler import MeshSampler

from mesh_simply import optimal_mesh

class train_mesh_simply:
    def __init__(self, cfg):
        self.cfg = cfg

    def do_computing(self):
        torch.cuda.empty_cache() 
        os.makedirs(self.cfg.out_dir, exist_ok=True)
        os.makedirs(os.path.join(self.cfg.out_dir, "temp/"), exist_ok=True)

        self.glctx = dr.RasterizeCudaContext()
        base_mesh_path = self.cfg.base_mesh

        # Base mesh
        base_mesh = self.load_mesh(base_mesh_path)
        print("Base mesh has %d triangles and %d vertices." % (base_mesh.t_pos_idx.shape[0], base_mesh.v_pos.shape[0]))
        print("Avg edge length of base mesh: %f" % regularizer.avg_edge_length(base_mesh))
        base_mesh, scale = mesh.unit_size(base_mesh) 
        print("The base mesh scale to: ", scale)

        # Check if the training texture resolution is acceptable
        base_texture_res = base_mesh.material['kd'].getRes()  ### kd 是 map_kd，若存在，否则为 kd
        if self.cfg.texture_res[0] < base_texture_res[0] or self.cfg.texture_res[1] < base_texture_res[1]:
            print("## WARNING: Picked a texture resolution lower than the base mesh [%d, %d] < [%d, %d]" % (self.cfg.texture_res[0],
             self.cfg.texture_res[1], base_texture_res[0], base_texture_res[1]))
        
        #the simply mesh needed to be optimized
        mesh_sampler = MeshSampler()
        base_mesh_aabb = mesh.aabb(base_mesh.eval())   
        print("base_mesh aabb: ", base_mesh_aabb)

        viewheight = 30 * scale
        percent = self.cfg.percent_rate
        # mesh_target = mesh_sampler.random_sample_QEM(base_mesh, percent)
        mesh_target = mesh_sampler.random_sample_poisson(base_mesh, percent) 
        
        os.makedirs("./OBJ_initial", exist_ok=True)
        obj.write_obj( "./OBJ_initial", mesh_target.eval())   

        print("The target mesh has %d triangles and %d vertices." % (mesh_target.t_pos_idx.shape[0], mesh_target.v_pos.shape[0]))
        print("Avg edge length: %f" % regularizer.avg_edge_length(mesh_target))   
        print("opt_mesh aabb: ", mesh.aabb(mesh_target.eval()))  

        starT2 = time.time()
        loss_10, opt_mesh = optimal_mesh(self.cfg, self.glctx, base_mesh, mesh_target, viewheight)

        print("Total time2222222: ", time.time() - starT2)
        # ===================================================================
        # 【修改点】：先解包计算，拿到真正的 Mesh 对象！
        final_mesh = opt_mesh.eval() 
        # ===================================================================

        with torch.no_grad():
            # 1. 撤销缩放 (注意这里改成用 final_mesh 来访问)
            restored_v_pos = final_mesh.v_pos / scale 
            
            # 2. 撤销居中平移 
            restored_v_pos = restored_v_pos + base_mesh.local_center
            
            # 3. 撤销真实地理偏移 (转为 float64 计算)
            offset_tensor = torch.tensor(base_mesh.real_world_offset, dtype=torch.float64, device=restored_v_pos.device)
            restored_v_pos = restored_v_pos.to(torch.float64) + offset_tensor
            
            # 4. 把正确的坐标塞回去
            final_mesh.v_pos = restored_v_pos
        # ===================================================================
        str_name = os.path.join(self.cfg.out_dir, os.path.basename(self.cfg.base_mesh+"_" + str(percent) +"_"+ self.cfg.tag))   
        os.makedirs(str_name, exist_ok=True)      
        obj.write_obj(str_name, final_mesh.eval())                
        return


    def build_loss_decay_schedule(self, cfg):
        if cfg.vis_loss_decay  == 'exp':
            return lambda percent: vis_loss_decay.vis_loss(percent, type='exp', min_loss=cfg.vis_loss_min, max_loss=cfg.vis_loss_max)
        else:
            raise ValueError("Invalid decay_type function")
        
    def load_mesh(self, filename, mtl_override=None): # mtl_override is a path to a .mtl file
        name, ext = os.path.splitext(filename)
        if ext == ".obj":
            return obj.load_obj(filename, clear_ks=True, mtl_override=mtl_override)
        assert False, "Invalid mesh file extension"


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
    lod = train_mesh_simply(cfg)
    lod.do_computing()


if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
