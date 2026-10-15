
import os

import time
import numpy as np
import torch
from collections import deque
from torch.utils.tensorboard import SummaryWriter
import psutil
from datetime import datetime

from src import obj
from src import util
from src import mesh
from src import texture
from src import render
from src.loss_mesh.loss_mesh import LossMesh
import math
from tools.logger import Logger
from src.loss_mesh.loss_val import val

RADIUS = 3.5


#内存监控
def log_memory(stage, log_file="memory_usage.log"):
    process = psutil.Process(os.getpid())

    # CPU 内存（GiB）
    rss = process.memory_info().rss / 1024**3

    info = (
        f"[{datetime.now():%Y-%m-%d %H:%M:%S}] "
        f"[{stage}] "
        f"CPU_RSS={rss:.3f}GB"
    )

    if torch.cuda.is_available():
        device = torch.cuda.current_device()

        # 当前进程由 PyTorch 管理的显存
        allocated = torch.cuda.memory_allocated(device) / 1024**3
        reserved = torch.cuda.memory_reserved(device) / 1024**3

        # 显存峰值
        peak_allocated = (
            torch.cuda.max_memory_allocated(device) / 1024**3
        )
        peak_reserved = (
            torch.cuda.max_memory_reserved(device) / 1024**3
        )

        # GPU 当前总显存使用量（包含其他进程）
        gpu_used = (
            torch.cuda.mem_get_info(device)[1]
            - torch.cuda.mem_get_info(device)[0]
        ) / 1024**3

        info += (
            f" | GPU={device}"
            f" | Allocated={allocated:.3f}GB"
            f" | Reserved={reserved:.3f}GB"
            f" | PeakAllocated={peak_allocated:.3f}GB"
            f" | PeakReserved={peak_reserved:.3f}GB"
            f" | GPU_TotalUsed={gpu_used:.3f}GB"
        )

    print(info, flush=True)

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(info + "\n")
 
def optimal_mesh(cfg, glctx, base_mesh, target_mesh, view_height):
    loss_tag = (
        f"i{cfg.lamda_img}"
        f"_s{cfg.lamda_ssim}"
        f"_l{cfg.lamda_lap}"
        f"_e{cfg.lamda_edge}"
        f"_m{cfg.lamda_move}"
        f"_g{cfg.lamda_geom}"
    )
    mesh_dir = os.path.join(
        "./logs",
        os.path.basename(cfg.base_mesh),
        cfg.tag,
        f"{loss_tag}-{time.strftime('%Y%m%d-%H%M%S')}"
    )    
    os.makedirs(mesh_dir, exist_ok=True)
    logger = Logger(total_iters=cfg.num_iter)
    writer = SummaryWriter(log_dir=mesh_dir)

    loss_img_quene = deque(maxlen=20)

    # Projection matrix
    assert view_height < 10000
    proj_mtx = util.projection(x=0.4, n = 1, f=1000.0) # x为近平面的宽度,n为近平面，f为远平面

    # Create normalized size versions of the base and reference meshes. Normalized base_mesh is important as it makes it easier to configure learning rate.
    normalized_target_mesh, _ = mesh.unit_size(target_mesh)

    #如何 mip map
    # assert not cfg.random_train_res or cfg.custom_mip

    #  Initialize weights / variables for trainable mesh
    trainable_list = [] 
    v_pos_opt = normalized_target_mesh.v_pos.clone().detach().requires_grad_(True)

    # Trainable normal map, initialize to (0,0,1) & make sure normals are always in positive hemisphere
    if cfg.random_textures:
        normal_map_opt = texture.create_trainable(np.array([0, 0, 1]), cfg.texture_res, cfg.custom_mip)
    else:
        if 'normal' not in base_mesh.material:
            normal_map_opt = texture.create_trainable(np.array([0, 0, 1]), cfg.texture_res, cfg.custom_mip)
        else:
            normal_map_opt = texture.create_trainable(base_mesh.material['normal'], cfg.texture_res, cfg.custom_mip)

    # Setup Kd, Ks albedo and specular textures
    if cfg.random_textures: 
        if cfg.num_layer > 1:
            kd_map_opt = texture.create_trainable(np.random.uniform(size=cfg.texture_res + [4], 
                                                low=0.0, high=1.0), cfg.texture_res, cfg.custom_mip)
        else:
            kd_map_opt = texture.create_trainable(np.random.uniform(size=cfg.texture_res + [3],
                                                low=0.0, high=1.0), cfg.texture_res, cfg.custom_mip)
        ksR = np.random.uniform(size=cfg.texture_res + [1], low=0.0, high=0.01)
        ksG = np.random.uniform(size=cfg.texture_res + [1], low=cfg.min_roughness, high=1.0)
        ksB = np.random.uniform(size=cfg.texture_res + [1], low=0.0, high=1.0)
        ks_map_opt = texture.create_trainable(np.concatenate((ksR, ksG, ksB), axis=2), cfg.texture_res, cfg.custom_mip)
    else:
        #kd_map_opt = target_mesh.material['kd']
        kd_map_opt = texture.create_trainable(base_mesh.material['kd'], cfg.texture_res, cfg.custom_mip)
        ks_map_opt = texture.create_trainable(base_mesh.material['ks'], cfg.texture_res, cfg.custom_mip)

    #  Setup material for optimized mesh
    if cfg.bsdf_type == 'pbr':
        opt_material = {
            'bsdf'   : 'pbr',
            'kd'     : kd_map_opt,
            'ks'     : ks_map_opt,
            'normal' : normal_map_opt
        }
        base_mesh.material['bsdf'] = 'pbr'
        trainable_list += [v_pos_opt]
        trainable_list += normal_map_opt.getMips() 
        trainable_list += kd_map_opt.getMips()
        trainable_list += ks_map_opt.getMips()
    elif cfg.bsdf_type == 'diffuse':
        opt_material = {
            'bsdf'   : 'diffuse',
            'kd'     : kd_map_opt,
            'normal' : normal_map_opt
        }
        base_mesh.material['bsdf'] = 'diffuse'
        trainable_list += [v_pos_opt]   
        trainable_list += normal_map_opt.getMips() 
        trainable_list += kd_map_opt.getMips()
    elif cfg.bsdf_type == 'kd_only':
        opt_material = {
            'bsdf'   : 'kd_only',
            'kd'     : kd_map_opt
        }
        base_mesh.material['bsdf'] = 'kd_only'
        trainable_list += [v_pos_opt]   
        trainable_list += kd_map_opt.getMips()
   
    # Compute AABB of base mesh. Used for centering during rendering TODO: Use pre frame AABB?
    base_mesh_aabb = mesh.aabb(base_mesh.eval())

    #  Setup optimized mesh operation graph, precomputes topology etc.
    ### Create optimized mesh with trainable positions 
    opt_mesh = mesh.Mesh(v_pos_opt, normalized_target_mesh.t_pos_idx, material=opt_material, target=normalized_target_mesh)
    ### Scale from [-1, 1] local coordinate space to match extents of the base mesh

    opt_mesh = mesh.align_with_reference(opt_mesh, base_mesh)
    ### Compute smooth vertex normals
    opt_mesh = mesh.auto_normals(opt_mesh)
    ### Set up tangent space
    opt_mesh = mesh.compute_tangents(opt_mesh)

    # initial the mesh loss
    loss_mesh = LossMesh(cfg, base_mesh.eval(), opt_mesh)

    # pytorch optimizer and schduler
    optimizer  = torch.optim.Adam(trainable_list, lr=cfg.learning_rate)
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: max(0.0, 10**(-x*0.0002))) 
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.num_iter, eta_min=0.000001) 

    #  Image loss
    print("cfg.loss: ", cfg.loss)

    # Background color
    if cfg.background == 'white':
        background = torch.ones((1, cfg.train_res[0], cfg.train_res[1], 3), dtype=torch.float32, device='cuda')
    else:
        raise ValueError("Invalid background color")

    # 为了批量处理的日志
    log_interval=1000
    spp = 1
    mesh_scale=2.0
    iter_dur_vec = []
    aabb_center = base_mesh_aabb[0] + (base_mesh_aabb[1] - base_mesh_aabb[0]) * 0.5
    numpy_aabb = tuple(tensor.cpu().numpy() for tensor in base_mesh_aabb)
    print("aabb_center: ", aabb_center)
    for it in range(cfg.num_iter+1): #  Training loop
        #  Setup base mesh. Compute tangentspace and animate with skinning
        render_base_mesh = mesh.compute_tangents(base_mesh)
        #  Validation
        if cfg.val_step:
            if it % cfg.val_step == 0:
                val_metrics = val(cfg, it, glctx, base_mesh, numpy_aabb, aabb_center, proj_mtx, view_height, 
                                  base_mesh_aabb, background, mesh_scale, opt_mesh,spp,render_base_mesh)
                writer.add_scalar("Val/PSNR", val_metrics['psnr'], it)
                writer.add_scalar("Val/SSIM", val_metrics['ssim'], it)
                writer.add_scalar("Val/LPIPS", val_metrics['lpips'], it)
                writer.add_scalar("Val/ChamferDistance", val_metrics['chamfer_distance'], it)
                writer.add_scalar("Val/GeographicDistance", val_metrics['geographic_distance'], it)
                writer.add_scalar("Val/Laplacian", val_metrics['laplacian'],it)

                # ============================
                # 新增内存监控
                # ============================
                process = psutil.Process(os.getpid())

                # CPU 物理内存
                cpu_rss = process.memory_info().rss / (1024 ** 3)

                writer.add_scalar("Memory/CPU_RSS_GB", cpu_rss, it)

                if torch.cuda.is_available():
                    device = torch.cuda.current_device()

                    gpu_allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
                    gpu_reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
                    gpu_peak = torch.cuda.max_memory_allocated(device) / (1024 ** 3)

                    writer.add_scalar("Memory/GPU_Allocated_GB", gpu_allocated, it)
                    writer.add_scalar("Memory/GPU_Reserved_GB", gpu_reserved, it)
                    writer.add_scalar("Memory/GPU_Peak_GB", gpu_peak, it)

                # 同时记录到日志文件
                log_memory(f"Iteration {it} | After Validation")
        #  Initialize training
        iter_start_time = time.time()
        iter_spp = spp # samples per pixel 

        mvp = np.zeros((cfg.batch, 4,4),  dtype=np.float32)
        campos   = np.zeros((cfg.batch, 3), dtype=np.float32)
        lightpos = np.zeros((cfg.batch, 3), dtype=np.float32)
        # for b in range(cfg.batch): #  Build transform stack for minibatching
        #     # Random rotation/translation matrix for optimization.
        #     r_rot      = util.random_rotation_translation(2.5) 
        #     r_mv       = np.matmul(util.translate(0, 0, -RADIUS), r_rot) # (model-view matrix) move camera back RADIUS units at z
        #     mvp[b]     = np.matmul(proj_mtx, r_mv).astype(np.float32) # model view projection matrix
        #     campos[b]  = np.linalg.inv(r_mv)[:3, 3]
        #     lightpos[b] = util.cosine_sample(campos[b])*RADIUS
        
        viewpoints = util.random_viewpoint_samples(numpy_aabb, view_height, cfg.batch)
        viewpoints_up = cfg.camera_up
        for b in range(cfg.batch):  #  Build transform stack for minibatching
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

        # Random bg color
        # randomBgColor = torch.rand(cfg.batch, iter_res, iter_res, 3, dtype=torch.float32, device='cuda')
    
        #  Evaluate all mesh ops (may change when positions are modified etc) and center/align meshes
        _opt_base  = mesh.center_by_reference(render_base_mesh.eval(params), base_mesh_aabb, mesh_scale)
        _opt_detail = mesh.center_by_reference(opt_mesh.eval(params), base_mesh_aabb, mesh_scale)

        #  Render reference mesh
        with torch.no_grad():
            color_base = render.render_mesh(glctx, _opt_base, mvp, campos, lightpos, cfg.light_power, cfg.train_res, 
                spp=iter_spp, num_layers=1, background=background, min_roughness=cfg.min_roughness)

        #  Render the trainable mesh
        color_opt = render.render_mesh(glctx, _opt_detail, mvp, campos, lightpos, cfg.light_power, cfg.train_res, 
            spp=iter_spp, num_layers=cfg.num_layer, msaa=True , background=background, min_roughness=cfg.min_roughness)

        loss_dict = loss_mesh(color_opt, color_base, params)
        loss_img_quene.append(loss_dict['loss_img'])

        if it == 0:
            loss_topo = cfg.lamda_lap* loss_dict['loss_lap']+ cfg.lamda_edge*loss_dict['loss_edge']+ cfg.lamda_move*loss_dict['loss_move']
            loss_img = cfg.lamda_img *loss_dict['loss_img'] + cfg.lamda_ssim *loss_dict['loss_ssim']
            loss_geom = loss_dict['loss_distances']*cfg.lamda_geom
            loss_scheduler = LossWeidghtScheduler(cfg, loss_img, loss_geom, loss_topo)
            geom_fac, topo_fac = loss_scheduler(it)
        else:
            geom_fac, topo_fac = loss_scheduler(it)

        loss_total =  (cfg.lamda_img*loss_dict['loss_img'] + cfg.lamda_ssim *loss_dict['loss_ssim']) \
                             + geom_fac*loss_dict['loss_distances']*cfg.lamda_geom + \
                            topo_fac* (cfg.lamda_lap*loss_dict['loss_lap']+ cfg.lamda_edge*loss_dict['loss_edge']+ cfg.lamda_move*loss_dict['loss_move'])
        
        #  Backpropagate
        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()
        scheduler.step()
        
        #  Clamp trainables to reasonable range
        normal_map_opt.clamp_(min=-1, max=1)
        kd_map_opt.clamp_(min=0, max=1)
        ks_map_opt.clamp_rgb_(minR=0, maxR=1, minG=cfg.min_roughness, maxG=1.0, minB=0.0, maxB=1.0)

        iter_dur_vec.append(time.time() - iter_start_time)
        the_last20_img_loss = sum(loss_img_quene)/len(loss_img_quene)
        #  Log & save outputs
        if log_interval and (it % log_interval == 0):
            logger.log(it, loss_dict, the_last20_img_loss, optimizer)
            writer.add_scalar("Loss/Total", loss_dict['loss_total'], it)
            writer.add_scalar("Loss/Image", loss_dict['loss_img'], it)
            writer.add_scalar("Loss/Last20Img", the_last20_img_loss, it)
            writer.add_scalar("Loss/WholeImg", loss_dict['loss_whole_img'], it)
            writer.add_scalar("Loss/Laplacian", loss_dict['loss_lap'], it)
            writer.add_scalar("Loss/Edge", loss_dict['loss_edge'], it)
            writer.add_scalar("Loss/Lap3D", loss_dict['loss_laplace'], it)
            writer.add_scalar("Loss/Dist3D", loss_dict['loss_distances'], it)
            writer.add_scalar("Loss/Move3D", loss_dict['loss_move'], it)
            writer.add_scalar("Loss/SSIM", loss_dict['loss_ssim'], it)
            writer.add_scalar("Loss/img+ssim", loss_dict['loss_ssim']+loss_dict['loss_img'], it)
            writer.add_scalar("LR", optimizer.param_groups[0]['lr'], it)
    writer.close()
    return min(loss_img_quene), opt_mesh



class LossWeidghtScheduler:
    def __init__(self, cfg, img_loss, geom_loss, topo_loss):
        self.it_max = cfg.num_iter * cfg.schedule_loss_maxIter
        geom_fac = img_loss.item() / geom_loss.item()
        self.min_geom = cfg.schedule_loss_geom_min * geom_fac
        self.max_geom = geom_fac * cfg.schedule_loss_geom_max

        topo_fac = img_loss.item() / topo_loss.item()
        self.min_topo = cfg.schedule_loss_topo_min * topo_fac
        self.max_topo = topo_fac * cfg.schedule_loss_topo_max

    def __call__(self, it):
        if it < self.it_max:
            ratio = 0.5 * (1 + math.cos(math.pi * it / self.it_max))
            geom_fac = self.min_geom + (self.max_geom - self.min_geom) * ratio
            topo_fac = self.min_topo + (self.max_topo - self.min_topo) * ratio
        else:
            geom_fac = self.min_geom
            topo_fac = self.min_topo
        return geom_fac, topo_fac


