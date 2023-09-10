import random
import numpy as np
import torch
from typing import Callable, Optional, Tuple, List, Union

from thrid_party.nerfacc import OccupancyGrid, ContractionType, \
    ray_marching, contract_inv, render_weight_from_density, accumulate_along_rays, unpack_info
from models.basemodels import BaseModel3D
from common.ray import generate_per_pixel_rays

class OccupancyGridPlus(OccupancyGrid):
    def __init__(self, 
                 roi_aabb: Union[List[int], torch.Tensor], 
                 resolution: Union[int, List[int], torch.Tensor] = 128, 
                 contraction_type: ContractionType = ContractionType.AABB
                 ) -> None:
        super().__init__(roi_aabb, resolution, contraction_type)
        
    @torch.no_grad()
    def update_with_pcd(
        self,
        step: int,
        pcd,                    # (n, 3)
        occ_eval_fn: Callable,
        occ_thre: float = 0.01, 
        ema_decay: float = 0.95,
        warmup_steps: int = 256,
    ) -> None:
        """Update the occ field in the EMA way."""
        # sample cells
        if step < warmup_steps:
            indices = self._get_all_cells()
        else:
            N = self.num_cells // 4
            indices = self._sample_uniform_and_occupied_cells(N)

        # infer occupancy: density * step_size
        grid_coords = self.grid_coords[indices]
        x = (
            grid_coords + torch.rand_like(grid_coords, dtype=torch.float32)
        ) / self.resolution
        if self._contraction_type == ContractionType.UN_BOUNDED_SPHERE:
            # only the points inside the sphere are valid
            mask = (x - 0.5).norm(dim=1) < 0.5
            x = x[mask]
            indices = indices[mask]
        # voxel coordinates [0, 1]^3 -> world
        x = contract_inv(
            x,
            roi=self._roi_aabb,
            type=self._contraction_type,
        )
        occ, _, _ = occ_eval_fn(pcd.unsqueeze(0), x.unsqueeze(0))
        occ = occ.squeeze(0)

        # ema update
        self.occs[indices] = torch.maximum(self.occs[indices] * ema_decay, occ)
        # suppose to use scatter max but emperically it is almost the same.
        # self.occs, _ = scatter_max(
        #     occ, indices, dim=0, out=self.occs * ema_decay
        # )
        self._binary = (
            self.occs > torch.clamp(self.occs.mean(), max=occ_thre)
        ).reshape(self._binary.shape)

def per_pixel_feature_integral(
    model3d:BaseModel3D, 
    pcd:torch.Tensor,
    image:torch.Tensor,
    pose:torch.Tensor,
    k:torch.Tensor,
    image_edge_buff:int=0,
    occupancy_grid_res:int=32,
    march_step:float=0.02,
    integral_type:str='to_surface',
    occ_thre:float=0.8,
    alpha_thre:float=0.0,
    early_stop_eps:float=0.0
    ) -> Tuple [
        torch.Tensor, # integral_sal
        torch.Tensor, # integral_feat
        ]:
    '''
    Args:
        model3d:        
        pcd             (n, 3)
        image:          (h, w, 3)
        occupancy_grid_res: 
        march_step:
        integral_type:  'to_surface': ignore all sample points after the surface. 
                            integral feature of all sample points to the first sample point which occ > occ_thre
                        'surface': only take the first sample point which occ > occ_thre 
                        'all': integral all sample points which occ > occ_thre on the ray 
        occ_thre: occupancy threshold of being a surface. 

    Returns:
        integral_sal:   (h, w, 1)
        integral_feat:  (h, w, feat_dim)
    '''
    origins, dirs = generate_per_pixel_rays(
        image=image,
        pose=pose,
        K=k,
        edge_buffer=image_edge_buff)
    
    integral_sal, integral_feat, samples_pos, samples_occ, samples_sal = feature_integral(
        model3d=model3d,
        pcd=pcd,
        origins=origins,
        viewdirs=dirs,
        occupancy_grid_res=occupancy_grid_res,
        march_step=march_step,
        integral_type=integral_type,
        occ_thre=occ_thre,
        alpha_thre=alpha_thre,
        early_stop_eps=early_stop_eps)
    
    integral_sal = integral_sal.reshape((image.shape[0]-image_edge_buff*2, image.shape[1]-image_edge_buff*2, 1))
    integral_feat = integral_feat.reshape((image.shape[0]-image_edge_buff*2, image.shape[1]-image_edge_buff*2, -1))
    return integral_sal, integral_feat

def feature_integral(
    model3d:BaseModel3D, 
    pcd:torch.Tensor,
    origins:torch.Tensor, 
    viewdirs:torch.Tensor, 
    occupancy_grid_res:int=32,
    march_step:float=0.02,
    integral_type:str='to_surface',
    occ_thre:float=0.8,
    alpha_thre:float=0.0,
    early_stop_eps:float=0.0
    ) -> Tuple[
        torch.Tensor,       # integral_sal
        torch.Tensor,       # integral_feat
        torch.Tensor,       # samples_pos
        torch.Tensor,       # samples_occ
        torch.Tensor,       # samples_sal
        ]:
    '''
    Args:
        model3d:        
        pcd             (n, 3)
        origins:        (num_rays, 3)
        viewdirs:       (num_rays, 3)
        occupancy_grid_res: 
        march_step:
        integral_type: 

    Returns:
        integral_sal:   (num_rays)
        integral_feat:  (num_rays)
        samples_pos     (n_samples, 3)
        samples_occ     (n_samples, 1)
        samples_sal     (n_samples, 1)
    '''
    roi_aabb = torch.Tensor([-0.5,-0.5,-0.5,0.5,0.5,0.5]).to(device=pcd.device)
    
    occupancy_grid = OccupancyGridPlus(
        roi_aabb=roi_aabb,
        resolution=occupancy_grid_res,
        contraction_type=ContractionType.AABB,
    ).to(pcd.device)

    # update occupancy grid
    occupancy_grid.update_with_pcd(
        step=1,     # all steps count, no warmup
        pcd=pcd,
        occ_eval_fn=lambda pcd, x: model3d(pcd, x),
        occ_thre=occ_thre,
        warmup_steps=0
    )

    rays_feat, opacity, depth, rays_sal, samples_pos, samples_occ, samples_sal = _feature_integral(
        pcd=pcd,
        radiance_field=model3d,
        occupancy_grid=occupancy_grid,
        ray_origins=origins,
        ray_viewdirs=viewdirs,
        scene_aabb=roi_aabb,
        render_step_size=march_step,
        integral_type=integral_type,
        occ_thre=occ_thre,
        alpha_thre=alpha_thre,
        early_stop_eps=early_stop_eps
    )

    return rays_sal, rays_feat, samples_pos, samples_occ, samples_sal

def _feature_integral(
    # scene
    radiance_field:BaseModel3D,
    occupancy_grid:OccupancyGrid,
    scene_aabb:torch.Tensor,
    pcd:torch.Tensor,
    ray_origins:torch.Tensor,
    ray_viewdirs:torch.Tensor,
    # rendering options
    near_plane:Optional[float]=None,
    far_plane:Optional[float]=None,
    render_step_size:float=1e-3,
    render_bkgd:Optional[torch.Tensor]=None,
    cone_angle:float=0.0,
    integral_type:str='to_surface',
    occ_thre:float=0.8,
    alpha_thre:float=0.0,
    early_stop_eps:float=0.0
):
    """Integral the feature of rays
    
    Args:
        pcd             (n, 3)
        radiance_field
        occupancy_grid
        ray_origins     (n_rays, 3)     n_rays = h * w
        ray_viewdirs    (n_rays, 3)
        scene_aabb      (6,)

    Returns:
        ray_feats,      (n_rays, feat_dim)
        opacity,        (n_rays,)
        depth,          (n_rays,)
        ray_sals        (n_rays, 1)
        sample_pos      (n_samples, 3)
        sample_occ      (n_samples,)
        smaple_sal      (n_samples)
    """
    rays_shape = ray_origins.shape
    num_rays, _ = rays_shape

    def sigma_fn(t_starts, t_ends, ray_indices):
        '''
        Args:
            t_starts: (n_samples,)
            t_end: (n_samples,)
            ray_indices: (n_samples,)

        Returns: 
            density of all sample points (n_sample, 1)
        '''
        ray_indices = ray_indices.long()
        t_origins = ray_origins[ray_indices]
        t_dirs = ray_viewdirs[ray_indices]
        sample_poses = t_origins + t_dirs * (t_starts + t_ends) / 2.0 # all positions of sample points which distributed on rays.
        
        sample_occs, sample_sals, sample_feats = radiance_field(pcd.unsqueeze(0), sample_poses.unsqueeze(0))
        
        sample_occs = sample_occs.detach()
        
        sample_occs = sample_occs.squeeze() # (n_sample)
        sample_sals = sample_sals.squeeze() # (n_sample)
        sample_feats = sample_feats.squeeze() # (n_sample, feat_dim)
        '''
        if integral_type == 'to_surface':
            sample_occs = _behind_surface_removal(sample_occs, ray_indices, occ_thre)
        elif integral_type == 'surface':
            sample_occs = _non_surface_removal(sample_occs, ray_indices, occ_thre)
        else:
            assert integral_type == 'all'
        '''
        return sample_occs.squeeze().unsqueeze(-1) # (n_sample, 1)

    def feat_sigma_fn(t_starts, t_ends, ray_indices):
        '''
        Args:
            t_starts (n_samples, )
            t_ends   (n_samples, )
            ray_indices (n_samples, )
        Returns:
            sample_occs     (n_samples, ) [0, 1]
            sample_sals     (n_samples, ) [0, 1]
            sample_feats    (n_samples, feat_dim)
            sample_poses    (n_samples, 3)
        '''
        ray_indices = ray_indices.long()
        t_origins = ray_origins[ray_indices]
        t_dirs = ray_viewdirs[ray_indices]
        sample_poses = t_origins + t_dirs * (t_starts + t_ends) / 2.0

        sample_occs, sample_sals, sample_feats = radiance_field(pcd.unsqueeze(0), sample_poses.unsqueeze(0))
        
        sample_occs = sample_occs.detach()
        
        sample_occs = sample_occs.squeeze() # (n_sample)
        sample_sals = sample_sals.squeeze() # (n_sample)
        sample_feats = sample_feats.squeeze() # (n_sample, feat_dim)
        '''
        if integral_type == 'to_surface':
            sample_occs = _behind_surface_removal(sample_occs, ray_indices, occ_thre)
        elif integral_type == 'surface':
            sample_occs = _non_surface_removal(sample_occs, ray_indices, occ_thre)
        else:
            assert integral_type == 'all'
        '''
        return sample_feats, sample_occs.unsqueeze(-1), sample_sals.unsqueeze(-1), sample_poses
    
    feat_dim = radiance_field.cfg.dim_desc
    ray_feats = torch.zeros(0, num_rays, feat_dim).to(pcd.device)
    opacities = torch.zeros(0, num_rays).to(pcd.device)
    depths = torch.zeros(0, num_rays).to(pcd.device)
    sals = torch.zeros(0, num_rays).to(pcd.device)
    
    packed_info, t_starts, t_ends = ray_marching(
        ray_origins,
        ray_viewdirs,
        scene_aabb=scene_aabb,
        grid=occupancy_grid,
        sigma_fn=sigma_fn,
        near_plane=near_plane,
        far_plane=far_plane,
        render_step_size=render_step_size,
        stratified=True,
        cone_angle=cone_angle,
        alpha_thre=alpha_thre,
    )

    ray_feats, opacities, depths, ray_sals, samples_pos, samples_occ, samples_sal = _feat_rendering(
        feat_sigma_fn,
        packed_info,
        t_starts,
        t_ends,
        early_stop_eps=early_stop_eps,
        alpha_thre=alpha_thre,
    )
    
    #t_sample_n = len(t_starts)

    return (ray_feats, opacities, depths, ray_sals, samples_pos, samples_occ, samples_sal)

def _occlusion_removal(
    sample_occs:torch.Tensor,
    ray_indices:torch.Tensor,
    removal_occ_thre:int=3.0,
) -> torch.Tensor: # modified sample_occs
    '''Set occluded samples' occ to 0
    Args:
        sample_occs: (n_samples,)
        ray_indices: (n_samples,)
        accumulated_max: the threshold to stop accumulate occ value
    Returns:
        sample_occs: (n_samples,), modified sample occ. Set samples after the threshold index on the ray to 0.
    '''
    occluded_removed_occs = torch.zeros((0,)).to(sample_occs.device)
    
    for i in range(0, max(ray_indices)+1):
        ray_occs = sample_occs[ray_indices == i]
        accumulation = torch.cumsum(ray_occs, dim=0)
        idx = ray_occs.size(0) # all
        indices = (accumulation > removal_occ_thre).nonzero(as_tuple=True)[0]
        if indices.size(0) > 0:
            idx = indices[0].item() + 1
        ray_occs[idx:] = 0.0        # after intersection removal
        #ray_occs[:idx-1] = 0.0      # before intersection removal
        occluded_removed_occs = torch.cat((occluded_removed_occs, ray_occs), dim=0)
    
    return occluded_removed_occs

def _non_surface_removal(
    sample_occs:torch.Tensor,
    ray_indices:torch.Tensor,
    surface_occ_thre:int=0.9,
) -> torch.Tensor: # modified sample_occs
    '''Set occluded samples' occ to 0
    Args:
        sample_occs: (n_samples,)
        ray_indices: (n_samples,)
        surface_occ_thre: the threshold select sample as surface.
    Returns:
        sample_occs: (n_samples,), modified sample occ. Set samples after the threshold index on the ray to 0.
    '''
    non_surface_removed_occs = torch.zeros((0,)).to(sample_occs.device)
    
    for i in range(0, max(ray_indices)+1):
        ray_occs = sample_occs[ray_indices == i]
        idx = ray_occs.size(0) # all
        indices = (ray_occs > surface_occ_thre).nonzero(as_tuple=True)[0]
        if indices.size(0) > 0:
            idx = indices[0].item() + 1
            ray_occs[:idx-1] = 0.0      # before intersection removal
        else:
            idx = 0
            
        ray_occs[idx:] = 0.0        # after intersection removal
        non_surface_removed_occs = torch.cat((non_surface_removed_occs, ray_occs), dim=0)
    
    return non_surface_removed_occs

def _behind_surface_removal(
    sample_occs:torch.Tensor,
    ray_indices:torch.Tensor,
    surface_occ_thre:int=0.9,
) -> torch.Tensor: # modified sample_occs
    '''Set occluded samples' occ to 0
    Args:
        sample_occs: (n_samples,)
        ray_indices: (n_samples,)
        surface_occ_thre: the threshold select sample as surface.
    Returns:
        sample_occs: (n_samples,), modified sample occ. Set samples after the threshold index on the ray to 0.
    '''
    behind_surface_removed_occs = torch.zeros((0,)).to(sample_occs.device)
    
    for i in range(0, max(ray_indices)+1):
        ray_occs = sample_occs[ray_indices == i]
        idx = ray_occs.size(0) # all
        indices = (ray_occs > surface_occ_thre).nonzero(as_tuple=True)[0]
        if indices.size(0) > 0:
            idx = indices[0].item() + 1
        else:
            idx = 0
        ray_occs[idx:] = 0.0        # after intersection removal
        behind_surface_removed_occs = torch.cat((behind_surface_removed_occs, ray_occs), dim=0)
    
    return behind_surface_removed_occs

def _feat_rendering(
    feat_sigma_fn: Callable,
    # ray marching results
    packed_info: torch.Tensor,
    t_starts: torch.Tensor,
    t_ends: torch.Tensor,
    # rendering options
    early_stop_eps: float=0.0,
    alpha_thre: float=0.0,
#    render_bkgd: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, 
           torch.Tensor,
           torch.Tensor,
           torch.Tensor,
           torch.Tensor,
           torch.Tensor,
           torch.Tensor
           ]:
    """
    Render the rays through the radience field defined by `feat_sigma_fn`.

    This function is differentiable to the outputs of `feat_sigma_fn` so it can be used for
    gradient-based optimization.

    Warning:
        This function is not differentiable to `t_starts`, `t_ends`.
    
    Args:
        feat_sigma_fn: A function that takes in samples {t_starts (N, 1), t_ends (N, 1), \
            ray indices (N,)} and returns the post-activation feature (N, feat_dim) and density \
            values (N, 1).
        packed_info: Packed ray marching info. See :func:`ray_marching` for details.
        t_starts: Per-sample start distance. Tensor with shape (n_samples, 1).
        t_ends: Per-sample end distance. Tensor with shape (n_samples, 1).
        early_stop_eps: Early stop threshold during trasmittance accumulation. Default: 1e-4.
        alpha_thre: Alpha threshold for skipping empty space. Default: 0.0.
        render_bkgd: Optional. Background color. Tensor with shape (3,).

    Returns:
        rays_feat (n_rays, feat_dim). 
        opacities (n_rays, 1)
        depths (n_rays, 1).
        ray_sal (n_rays, 1)
        samples_pos (n_samples, 3)
        samples_occ (n_samples, 1)
        samples_sal (n_samples, 1)
    """
    n_rays = packed_info.shape[0]
    ray_indices = unpack_info(packed_info)

    # Query sigma and color with gradients
    # sigmas->occ
    samples_feat, samples_occ, samples_sal, samples_pos = feat_sigma_fn(t_starts, t_ends, ray_indices.long())

    assert (
        samples_occ.shape == t_starts.shape
    ), "sigmas must have shape of (N, 1)! Got {}".format(samples_occ.shape)

    # Rendering: compute weights and ray indices.
    weights = render_weight_from_density(
        packed_info, t_starts, t_ends, samples_occ, early_stop_eps, alpha_thre
    )

    # Rendering: accumulate sample features, opacities, and depths along the rays.
    rays_feat = accumulate_along_rays(
        weights, ray_indices, values=samples_feat, n_rays=n_rays
    )
    opacities = accumulate_along_rays(
        weights, ray_indices, values=None, n_rays=n_rays
    )
    depths = accumulate_along_rays(
        weights,
        ray_indices,
        values=(t_starts + t_ends) / 2.0,
        n_rays=n_rays,
    )
    rays_sal = accumulate_along_rays(
        weights, ray_indices, values=samples_sal, n_rays=n_rays
    )

    # Background composition.
    #if render_bkgd is not None:
    #    ray_feats = ray_feats + render_bkgd * (1.0 - opacities)

    return rays_feat, opacities, depths, rays_sal, samples_pos, samples_occ, samples_sal
