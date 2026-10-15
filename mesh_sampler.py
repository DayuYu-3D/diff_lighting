from typing import List
import numpy as np
import math
import time 

import torch
from scipy.spatial import Delaunay
from matplotlib import pyplot as plt
import trimesh

from src.mesh import Mesh
from src import mesh
import xatlas
import random
from itertools import product

from tools.class_mesh_simplify import mesh_simplify

import fast_simplification
import pyfqmr

import open3d as o3d

class MeshSampler(object):
    def __init__(self):
        pass

    #   Tin降采样
    def random_sample(self, base_mesh: Mesh, sampling_rate: float) ->Mesh:
        v_pos_base = base_mesh.v_pos
        mask = torch.rand(v_pos_base.shape[0], device='cuda') < sampling_rate
        v_pos_sampled = v_pos_base[mask].clone().detach()

        if base_mesh.v_tex is not None:
            # assert(len(base_mesh.v_pos) == len(base_mesh.v_tex))
            if len(base_mesh.v_pos) < len(base_mesh.v_tex):
                v_tex_sampled = base_mesh.v_tex[:mask.shape[0],:][mask].clone().detach()
            elif len(base_mesh.v_pos) > len(base_mesh.v_tex):
                raise ValueError("v_pos < v_tex")
            else:
                v_tex_sampled = base_mesh.v_tex[mask].clone().detach()

        if base_mesh.v_nrm is not None:
            assert(len(base_mesh.v_pos) == len(base_mesh.v_nrm))
            v_nrm_sampled = base_mesh.v_nrm[mask].clone().detach()

        faces = self.surface_reconstruction_delaunay(v_pos_sampled)
        faces = torch.tensor(faces, dtype=torch.int64, device='cuda')

        # t_nrm_idx
        t_nrm_idx_sample = faces.clone().detach()
        t_tex_idx_sample = faces.clone().detach()

        return Mesh(v_pos=v_pos_sampled, t_pos_idx=faces, 
                    v_nrm=v_nrm_sampled,t_nrm_idx=t_nrm_idx_sample, 
                    v_tex=v_tex_sampled, t_tex_idx=t_tex_idx_sample,
                    material=base_mesh.material)

    def random_sample_poisson(self, base_mesh: Mesh, sampling_rate: float) -> Mesh:
        # 1. 随机采样顶点
        v_pos_base = base_mesh.v_pos
        print("原始顶点个数: ", v_pos_base.shape)
        mask = torch.rand(v_pos_base.shape[0], device='cuda') < sampling_rate
        v_pos_sampled = v_pos_base[mask].detach().cpu().numpy()

        
        #法线和顶点不能正常对应
        # 2. 为 Poisson 计算与 v_pos 一一对应的 vertex normal
        # 优先使用原始顶点法线
        v_nrm_base = base_mesh.v_nrm

        # 原始法线不存在，或者法线数量与顶点数量不一致时，自动重新计算
        if v_nrm_base is None or v_nrm_base.shape[0] != base_mesh.v_pos.shape[0]:
            print("[WARNING] 原始法线不存在或数量与顶点不匹配，自动重新计算顶点法线")

            normal_mesh = mesh.auto_normals(base_mesh).eval()
            v_nrm_base = normal_mesh.v_nrm

            print("重新计算后的顶点法线:", v_nrm_base.shape)

        # 检查 mask 与顶点数量是否对应
        if mask.shape[0] != base_mesh.v_pos.shape[0]:
            raise ValueError(
                f"mask 数量 {mask.shape[0]} 与顶点数量 "
                f"{base_mesh.v_pos.shape[0]} 不一致"
            )

        # 提取采样顶点对应的法线
        v_nrm_sampled = v_nrm_base[mask].detach().cpu().numpy()

        print("Poisson 重建前网格顶点数:", v_pos_sampled.shape)
        print("Poisson 重建前网格法线数:", v_nrm_sampled.shape)
        

        # 2. 构建 Open3D 点云
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(v_pos_sampled)
        if v_nrm_sampled is not None:
            pcd.normals = o3d.utility.Vector3dVector(v_nrm_sampled)
        else:
            raise ValueError("Poisson 重建需要法线信息")

        # 3. Poisson 重建
        mesh_o3d, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=7  # depth 控制八叉树深度，越大越精细，8约等于顶点数量翻倍
        )

        # 4. 可选：根据密度过滤掉低质量三角形
        mesh_o3d.remove_unreferenced_vertices()
        vertices_to_remove = densities < np.quantile(densities, 0.05)
        mesh_o3d.remove_vertices_by_mask(vertices_to_remove)

        mesh_o3d_simplied = mesh_o3d.simplify_quadric_decimation(target_number_of_triangles=v_pos_sampled.shape[0]*2)
        mesh_o3d_simplied.remove_unreferenced_vertices()
        
        out_vertices = np.asarray(mesh_o3d_simplied.vertices)
        out_faces = np.asarray(mesh_o3d_simplied.triangles)
        print("Poisson 重建后网格顶点数: ", out_vertices.shape)
        print("Poisson 重建后网格面元数: ", out_faces.shape)


        # 不同的参数化方案
        #xatlas
        vts, idxs, uvs = self.draw_uv_by_mesh_parametrized(out_vertices,out_faces)
        #tri
        # vts, idxs, uvs, uv_idxs = self.draw_uv_by_triangles_for_mesh(
        #     out_vertices, out_faces
        # )
        #sphere
        # vts = out_vertices
        # idxs = out_faces
        # uvs = self.draw_uv_by_sphere_maping_ellipsoid(out_vertices)
        
        # 5. 转回 torch.Tensor
        vts = torch.tensor(vts, dtype=torch.float32, device='cuda')
        idxs = torch.from_numpy(idxs.astype(np.int64)).to(device='cuda')
        uvs = torch.tensor(uvs, dtype=torch.float32, device='cuda')
        # 对于triangle参数化方案，UV 面片索引转换，使用独立的索引
        # uv_idxs = torch.from_numpy(uv_idxs.astype(np.int64)).to(device='cuda')

        # t_tex_sampled = torch.tensor(t_tex_sampled.astype(np.int32), dtype=torch.int64, device='cuda')
        # 不同的参数化方案
        print("参数化后 重建后网格顶点数: ", vts.shape)
        print("参数化后 重建后网格面元数: ", idxs.shape)

        # 注意：Poisson 输出不包含 v_tex / v_nrm 映射
        # exchange
        #xatlas、sphere
        return Mesh(
            v_pos=vts,
            t_pos_idx=idxs,
            v_tex=uvs,
            t_tex_idx=idxs,
            material=base_mesh.material
        )
        #triangle
        # return Mesh(
        #     v_pos=vts,
        #     t_pos_idx=idxs,
        #     v_tex=uvs,
        #     t_tex_idx=uv_idxs,
        #     material=base_mesh.material
        # )
        # exchange

    def surface_reconstruction_delaunay(self, v_pos_sampled: torch.Tensor) -> List[List[int]]:  
        # 三维点 -> 二维点
        tri = Delaunay(v_pos_sampled[:, :2].cpu().numpy())
        t_indx = tri.simplices
        return t_indx

# 是会造成简化率过低的话，无法达到指定简化比
    def random_sample_QEM(self, base_mesh: Mesh, sampling_rate: float) ->Mesh:
        points = base_mesh.v_pos.cpu().numpy()
        faces = base_mesh.t_pos_idx.cpu().numpy()
        target_f = faces.shape[0]*sampling_rate
        print("tarrrr: ",target_f)
        target_v = points.shape[0]*sampling_rate   

        points_out, faces_out = fast_simplification.simplify(points=points, triangles=faces,             
                                                    target_reduction= 1-sampling_rate, agg=8)
        print("faces_out.shape: ", faces_out.shape)
        print("points_out.shape: ", points_out.shape)
        sampling_rate = target_v/ points_out.shape[0]

        print("second faces_out.shape: ", faces_out.shape)
        print("second points_out.shape: ", points_out.shape)
        
        vts, idxs, uvs = self.draw_uv_by_mesh_parametrized(points_out,faces_out)

        # 5. 转回 torch.Tensor
        vts = torch.tensor(vts, dtype=torch.float32, device='cuda')
        idxs = torch.from_numpy(idxs.astype(np.int64)).to(device='cuda')
        uvs = torch.tensor(uvs, dtype=torch.float32, device='cuda')
        # t_tex_sampled = torch.tensor(t_tex_sampled.astype(np.int32), dtype=torch.int64, device='cuda')
        print("参数化后 重建后网格顶点数: ", vts.shape)
        print("参数化后 重建后网格面元数: ", idxs.shape)

        # 注意：Poisson 输出不包含 v_tex / v_nrm 映射
        return Mesh(
            v_pos=vts,
            t_pos_idx=idxs,
            v_tex=uvs,
            t_tex_idx=idxs,
            material=base_mesh.material
        )

    #   Delaunay降采样
    def random_sample_unclosed(self, base_mesh: Mesh, sampling_rate: float) ->Mesh:
        v_pos_base = base_mesh.v_pos
        mask = torch.rand(v_pos_base.shape[0], device='cuda') < sampling_rate
        points = v_pos_base[mask].clone().detach().cpu().numpy()

        if base_mesh.v_nrm is not None:
            assert(len(base_mesh.v_pos) == len(base_mesh.v_nrm))
            points_normal = base_mesh.v_nrm[mask].clone().detach().cpu().numpy()
        print("points.shape: ", points.shape)

        #参数化
        def map_to_2d_plane(x, y, z):
            u = (np.arctan2(y, x) + np.pi) / (2 * np.pi)
            v = 1 - np.arccos(z / np.sqrt(x**2 + y**2 + z**2)) / np.pi
            return u, v

        mapped_points = []
        for vertex in points:
            x, y, z = vertex
            u, v = map_to_2d_plane(x, y, z)
            mapped_points.append((u, v))

        # delanay
        tri = Delaunay(mapped_points, qhull_options='QJ QbB')
        print("faces.shape: ", tri.simplices.shape)
        print("vertices.shape: ", len(mapped_points))
        
        _,t_tex_sampled, v_tex_sampled = self.draw_uv_by_mesh_parametrized(points, tri.simplices)

        v_pos_sampled = torch.tensor(points, dtype=torch.float32, device='cuda')
        t_indx_sampled = torch.tensor(tri.simplices, dtype=torch.int64, device='cuda')
        
        v_tex_sampled = torch.tensor(v_tex_sampled, dtype=torch.float32, device='cuda')
        t_tex_sampled = torch.tensor(t_tex_sampled.astype(np.int32), dtype=torch.int64, device='cuda')
        print("v_pos_sampled.shape: ", v_pos_sampled.shape)
        print("t_indx_sampled.shape: ", t_indx_sampled.shape)
        print("v_tex_sampled.shape: ", v_tex_sampled.shape)
        print("t_tex_sampled.shape: ", t_tex_sampled.shape)

        return Mesh(v_pos=v_pos_sampled, t_pos_idx=t_indx_sampled, v_tex=v_tex_sampled, t_tex_idx=t_tex_sampled,
                    material=base_mesh.material)
    
    # 平面采样
    def random_sample_plane(self, base_mesh: Mesh, sampling_rate: float) ->Mesh:
        points = base_mesh.v_pos.cpu().numpy()
        faces = base_mesh.t_pos_idx.cpu().numpy()
        target_f = faces.shape[0]*sampling_rate
        print("tarrrr: ",target_f)
        target_v = points.shape[0]*sampling_rate   
        
        x = np.linspace(0, 1, int(np.sqrt(target_f)))
        y = np.linspace(0, 1, int(np.sqrt(target_f)))
        z = 0.5
        points_out = np.array(list(product(x, y, [z])))

        # Generate triangular faces
        faces_out = []
        for i in range(int(np.sqrt(target_f)) - 1):
            for j in range(int(np.sqrt(target_f)) - 1):
                v1 = i * int(np.sqrt(target_f)) + j
                v2 = v1 + 1
                v3 = v1 + int(np.sqrt(target_f))
                v4 = v3 + 1

                # Add two triangles for each square
                faces_out.append([v1, v2, v3])
                faces_out.append([v2, v4, v3])
        faces_out = np.array(faces_out)
        points_out = np.array(points_out)
        print("faces_out.shape: ", faces_out.shape)
        print("points_out.shape: ", points_out.shape)
        
        vts, idxs, uvs = self.draw_uv_by_mesh_parametrized(points_out,faces_out)

        # 5. 转回 torch.Tensor
        vts = torch.tensor(vts, dtype=torch.float32, device='cuda')
        idxs = torch.from_numpy(idxs.astype(np.int64)).to(device='cuda')
        uvs = torch.tensor(uvs, dtype=torch.float32, device='cuda')
        # t_tex_sampled = torch.tensor(t_tex_sampled.astype(np.int32), dtype=torch.int64, device='cuda')
        print("参数化后 重建后网格顶点数: ", vts.shape)
        print("参数化后 重建后网格面元数: ", idxs.shape)

        # 注意：Poisson 输出不包含 v_tex / v_nrm 映射
        return Mesh(
            v_pos=vts,
            t_pos_idx=idxs,
            v_tex=uvs,
            t_tex_idx=idxs,
            material=base_mesh.material
        )

    #  ball pivoting 降采样
    def random_sample_BPA(
        self,
        base_mesh: Mesh,
        sampling_rate: float,
        radius_factors=(1.5, 2.0, 2.5, 3.0),
        normal_knn: int = 30
    ) -> Mesh:

        # ============================================================
        # 1. Randomly sample vertices
        # ============================================================
        v_pos_base = base_mesh.v_pos

        num_vertices = v_pos_base.shape[0]
        num_sampled = max(3, int(num_vertices * sampling_rate))

        # 固定采样点数，比 rand < sampling_rate 更适合公平实验
        sampled_idx = torch.randperm(
            num_vertices,
            device=v_pos_base.device
        )[:num_sampled]

        v_pos_sampled = v_pos_base[sampled_idx].clone().detach()

        points = v_pos_sampled.cpu().numpy().astype(np.float64)

        print("BPA sampled points:", points.shape)


        # ============================================================
        # 2. Construct Open3D PointCloud
        # ============================================================
        pcd = o3d.geometry.PointCloud()

        pcd.points = o3d.utility.Vector3dVector(points)


        # ============================================================
        # 3. Normals
        # BPA requires oriented normals
        # ============================================================

        use_original_normals = False

        if base_mesh.v_nrm is not None:

            if len(base_mesh.v_nrm) == len(base_mesh.v_pos):

                normals = (
                    base_mesh.v_nrm[sampled_idx]
                    .clone()
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float64)
                )

                # 检查法向量是否合法
                normal_length = np.linalg.norm(normals, axis=1)

                if np.all(normal_length > 1e-12):

                    normals = normals / normal_length[:, None]

                    pcd.normals = o3d.utility.Vector3dVector(normals)

                    use_original_normals = True

                    print("BPA: use original mesh normals")


        # 如果原 mesh 没有可靠法向，则 Open3D 自动估计
        if not use_original_normals:

            print("BPA: estimate point normals")

            k = min(normal_knn, num_sampled - 1)

            # 至少保证 k >= 3
            k = max(k, 3)

            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamKNN(
                    knn=k
                )
            )

            # 统一法向方向
            orient_k = min(normal_knn, num_sampled - 1)
            orient_k = max(orient_k, 3)

            pcd.orient_normals_consistent_tangent_plane(
                orient_k
            )

        pcd.normalize_normals()


        # ============================================================
        # 4. Estimate point spacing
        # ============================================================
        # BPA最关键参数是 ball radius。
        # 使用最近邻距离来自适应模型尺度。

        nn_distances = np.asarray(
            pcd.compute_nearest_neighbor_distance()
        )

        # 去除 0 / inf / nan
        valid_distances = nn_distances[
            np.isfinite(nn_distances) &
            (nn_distances > 1e-12)
        ]

        if len(valid_distances) == 0:
            raise ValueError(
                "BPA failed: cannot estimate point spacing."
            )

        # median 比 mean 对异常点更加稳定
        point_spacing = np.median(valid_distances)

        print("BPA point spacing:", point_spacing)


        # ============================================================
        # 5. Ball radii
        # ============================================================
        radii = [
            point_spacing * factor
            for factor in radius_factors
        ]

        print("BPA radii:", radii)


        # ============================================================
        # 6. Ball Pivoting reconstruction
        # ============================================================
        bpa_mesh = (
            o3d.geometry.TriangleMesh
            .create_from_point_cloud_ball_pivoting(
                pcd,
                o3d.utility.DoubleVector(radii)
            )
        )


        # ============================================================
        # 7. Clean reconstructed mesh
        # ============================================================
        bpa_mesh.remove_duplicated_triangles()
        bpa_mesh.remove_degenerate_triangles()
        bpa_mesh.remove_unreferenced_vertices()


        # ============================================================
        # 8. Convert Open3D mesh -> numpy
        # ============================================================
        points_out = np.asarray(
            bpa_mesh.vertices
        ).astype(np.float32)

        faces_out = np.asarray(
            bpa_mesh.triangles
        ).astype(np.int64)

        print("BPA reconstructed vertices:", points_out.shape)
        print("BPA reconstructed faces:", faces_out.shape)


        if points_out.shape[0] == 0 or faces_out.shape[0] == 0:
            raise RuntimeError(
                "BPA reconstruction generated an empty mesh. "
                "Try increasing radius_factors."
            )


        # ============================================================
        # 9. UV parameterization
        # ============================================================
        vts, idxs, uvs = self.draw_uv_by_mesh_parametrized(
            points_out,
            faces_out
        )

        # 参数化函数可能返回 uint32 / float64，
        # 在进入 PyTorch 前统一数据类型
        vts = np.asarray(vts, dtype=np.float32)
        idxs = np.asarray(idxs, dtype=np.int64)
        uvs = np.asarray(uvs, dtype=np.float32)

        print("BPA parameterized numpy:")
        print("vts:", vts.shape, vts.dtype)
        print("idxs:", idxs.shape, idxs.dtype)
        print("uvs:", uvs.shape, uvs.dtype)


        # ============================================================
        # 10. Convert to torch.Tensor
        # ============================================================
        vts = torch.from_numpy(vts).to(
            device=v_pos_base.device,
            dtype=torch.float32
        )

        idxs = torch.from_numpy(idxs).to(
            device=v_pos_base.device,
            dtype=torch.int64
        )

        uvs = torch.from_numpy(uvs).to(
            device=v_pos_base.device,
            dtype=torch.float32
        )

        print("BPA after parameterization:")
        print("v_pos:", vts.shape, vts.dtype)
        print("faces:", idxs.shape, idxs.dtype)
        print("uv:", uvs.shape, uvs.dtype)


        # ============================================================
        # 11. Return Mesh
        # ============================================================
        return Mesh(
            v_pos=vts,
            t_pos_idx=idxs,
            v_tex=uvs,
            t_tex_idx=idxs,
            material=base_mesh.material
        )


    def random_sample_ellipsoid(self, base_mesh: Mesh, sampling_rate: float) ->Mesh:
        v_pos_base = base_mesh.v_pos
        num_sampled = int(v_pos_base.shape[0] * sampling_rate)
        # cal the center of v_pos_base
        center = torch.mean(v_pos_base, dim=0).cpu().numpy()
        x_max = torch.max(v_pos_base[:,0]).cpu().numpy()
        x_min = torch.min(v_pos_base[:,0]).cpu().numpy()
        y_max = torch.max(v_pos_base[:,1]).cpu().numpy()
        y_min = torch.min(v_pos_base[:,1]).cpu().numpy()
        z_max = torch.max(v_pos_base[:,2]).cpu().numpy()
        z_min = torch.min(v_pos_base[:,2]).cpu().numpy()
        
        v_pos_sampled, faces = self.generate_ellipsoid_mesh(num_sampled, a=(x_max-x_min)/2, 
                                                            b=(y_max-y_min)/2, c=(z_max-z_min)/2, center=center)
        
        # n = math.ceil( np.sqrt( faces.shape[0]/2 ) )
        # print("n: ", n)
        # v_tex_sampled, t_tex_idx_sampled = self.draw_triangles(n, n)
        # t_tex_idx_sampled = torch.tensor(t_tex_idx_sampled[:faces.shape[0],:], dtype=torch.int64, device='cuda')

        t_tex_sampled, v_tex_sampled = self.draw_uv_by_mesh_parametrized(v_pos_sampled, faces)

        v_pos_sampled = torch.tensor(v_pos_sampled, dtype=torch.float32, device='cuda')
        t_indx_sampled = torch.tensor(faces, dtype=torch.int64, device='cuda')
        
        v_tex_sampled = torch.tensor(v_tex_sampled, dtype=torch.float32, device='cuda')
        t_tex_sampled = torch.tensor(t_tex_sampled.astype(np.int32), dtype=torch.int64, device='cuda')
        print("v_pos_sampled.shape: ", v_pos_sampled.shape)
        print("t_indx_sampled.shape: ", t_indx_sampled.shape)
        print("v_tex_sampled.shape: ", v_tex_sampled.shape)
        print("t_tex_sampled.shape: ", t_tex_sampled.shape)
        return Mesh(v_pos=v_pos_sampled, t_pos_idx=t_indx_sampled, v_tex=v_tex_sampled, t_tex_idx=t_tex_sampled,
                    material=base_mesh.material)

    def generate_ellipsoid_mesh(self, num_vertices, a=1, b=1, c=1, center=[0,0,0]):
        #sqrt
        num_vertices_per_axis = int(np.sqrt(num_vertices))

        # Generate ellipsoid surface points
        u = np.linspace(0, 2 * np.pi, num_vertices_per_axis)
        v = np.linspace(0, np.pi, num_vertices_per_axis)
        u, v = np.meshgrid(u, v)
        x = a * np.sin(v) * np.cos(u) + center[0]
        y = b * np.sin(v) * np.sin(u) + center[1]
        z = c * np.cos(v) + center[2]
        points = np.vstack([x.flatten(), y.flatten(), z.flatten()]).T
        print("points.shape: ", points.shape)
        obj = trimesh.Trimesh(points)
        obj=trimesh.convex.convex_hull(obj)

        return obj.vertices, obj.faces

    def generate_plane_mesh(self, num_vertices, maxx,minx,maxy,miny,height):
        maxx = maxx.item()
        minx = minx.item()
        maxy = maxy.item()
        miny = miny.item()
        height = height.item()
        n = int(np.sqrt(num_vertices))
        # 生成均匀分布的n个x坐标和n个y坐标
        x_coords = np.linspace(minx, maxx, n)
        y_coords = np.linspace(miny, maxy, n)
        # 使用meshgrid创建2D坐标网格
        x_grid, y_grid = np.meshgrid(x_coords, y_coords)
        # 将高度添加到所有点
        z_coords = np.full_like(x_grid, height)
        # 将三个坐标合并成一个点云
        points = np.column_stack((x_grid.ravel(), y_grid.ravel(), z_coords.ravel()))
        u_coords = (x_grid - minx) / (maxx - minx)
        v_coords = (y_grid - miny) / (maxy - miny)
        uvs = np.column_stack((u_coords.ravel(), v_coords.ravel()))
        tri = Delaunay(points[:, :2])      
        faces = tri.simplices  
        return points, faces, uvs

    def draw_uv_by_triangles(self, m, n):
        triangles = []
        vertices = []
        for i in range(m):
            for j in range(n):
                x = i / m
                y = j / n
                triangle1 = np.array([[x, y], [x + 1 / m, y], [x, y + 1 / n]])
                vertices.extend(triangle1)
                # 获取顶点在vertices中的索引
                indices1 = [len(vertices) - 3, len(vertices) - 2, len(vertices) - 1]
                triangles.append(indices1)
                triangle2 = np.array([[x + 1 / m, y], [x + 1 / m, y + 1 / n], [x, y + 1 / n]])
                vertices.extend(triangle2)
                indices2 = [len(vertices) - 3, len(vertices) - 2, len(vertices) - 1]
                triangles.append(indices2)
        vertices = np.array(vertices)
        triangles = np.array(triangles)
        return vertices, triangles
    
    def draw_uv_by_triangles_for_mesh(self, vertices, faces):

        vertices = np.asarray(vertices, dtype=np.float32)
        faces = np.asarray(faces, dtype=np.int64)

        num_faces = len(faces)

        if num_faces == 0:
            raise ValueError("Mesh contains no faces")

        num_cells = math.ceil(num_faces / 2)

        m = math.ceil(math.sqrt(num_cells))
        n = math.ceil(num_cells / m)

        uvs, uv_faces = self.draw_uv_by_triangles(m, n)

        uv_faces = uv_faces[:num_faces]

        return (
            vertices,
            faces,
            uvs.astype(np.float32),
            uv_faces.astype(np.int64)
        )
    

    def draw_uv_by_sphere_maping_ellipsoid(self, vertices):
        def map_to_2d_plane(x, y, z):
            u = (np.arctan2(y, x) + np.pi) / (2 * np.pi)
            v = (np.pi - np.arccos(z / np.sqrt(x**2 + y**2 + z**2))) / np.pi
            return u, v
        mapped_vertices = []
        for vertex in vertices:
            x, y, z = vertex
            u, v = map_to_2d_plane(x, y, z)
            mapped_vertices.append((u, v))
        return np.array(mapped_vertices)

    def draw_uv_by_mesh_parametrized(self, vertices,faces):
        # `vmapping` contains the original vertex index for each new vertex (shape N, type uint32).
        # `indices` contains the vertex indices of the new triangles (shape Fx3, type uint32)
        # `uvs` contains texture coordinates of the new vertices (shape Nx2, type float32)
        vmapping, indices, uvs = xatlas.parametrize(vertices, faces)
        return vertices[vmapping], indices, uvs
    
    def draw_mesh(self, v_pos: List[List[float]], t_indx: List[List[int]]):
        plt.triplot(v_pos[:, 0], v_pos[:, 1], t_indx)
        plt.plot(v_pos[:, 0], v_pos[:, 1], 'o')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Triangulation_2D')
        # 保存结果为图像文件
        plt.savefig('Triangulation_2D.png')      