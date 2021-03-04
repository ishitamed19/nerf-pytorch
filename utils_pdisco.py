import numpy as np
import os
import torch

EPS = 1e-6
def reduce_masked_mean(x, mask, dim=None, keepdim=False):
    # x and mask are the same shape
    # returns shape-1
    # axis can be a list of axes
    # st()
    assert(x.size() == mask.size())
    prod = x*mask
    if dim is None:
        numer = torch.sum(prod)
        denom = EPS+torch.sum(mask)
    
    else:
        numer = torch.sum(prod, dim=dim, keepdim=keepdim)
        denom = EPS+torch.sum(mask, dim=dim, keepdim=keepdim)
        
    mean = numer/denom
    return mean

def depth2pointcloud(z, pix_T_cam):
    B, C, H, W = list(z.shape)
    y, x = meshgrid2D(B, H, W)
    z = torch.reshape(z, [B, H, W])
    fx, fy, x0, y0 = split_intrinsics(pix_T_cam)
    xyz = Pixels2Camera(x, y, z, fx, fy, x0, y0)
    return xyz

def normalize_grid2D(grid_y, grid_x, Y, X, clamp_extreme=True):
    # make things in [-1,1]
    grid_y = 2.0*(grid_y / float(Y-1)) - 1.0
    grid_x = 2.0*(grid_x / float(X-1)) - 1.0
    
    if clamp_extreme:
        grid_y = torch.clamp(grid_y, min=-2.0, max=2.0)
        grid_x = torch.clamp(grid_x, min=-2.0, max=2.0)
        
    return grid_y, grid_x

def meshgrid2D(B, Y, X, stack=False, norm=False):
    # returns a meshgrid sized B x Y x X
    grid_y = torch.linspace(0.0, Y-1, Y, device=torch.device('cuda'))
    grid_y = torch.reshape(grid_y, [1, Y, 1])
    grid_y = grid_y.repeat(B, 1, X)

    grid_x = torch.linspace(0.0, X-1, X, device=torch.device('cuda'))
    grid_x = torch.reshape(grid_x, [1, 1, X])
    grid_x = grid_x.repeat(B, Y, 1)

    if norm:
        grid_y, grid_x = normalize_grid2D(
            grid_y, grid_x, Y, X)

    if stack:
        # note we stack in xy order
        # (see https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.grid_sample)
        grid = torch.stack([grid_x, grid_y], dim=-1)
        return grid
    else:
        return grid_y, grid_x

def split_intrinsics(K):
    # K is B x 3 x 3 or B x 4 x 4
    fx = K[:,0,0]
    fy = K[:,1,1]
    x0 = K[:,0,2]
    y0 = K[:,1,2]
    return fx, fy, x0, y0

def Pixels2Camera(x,y,z,fx,fy,x0,y0):
    # x and y are locations in pixel coordinates, z is a depth image in meters
    # their shapes are B x H x W
    # fx, fy, x0, y0 are scalar camera intrinsics
    # returns xyz, sized [B,H*W,3]
    
    B, H, W = list(z.shape)

    fx = torch.reshape(fx, [B,1,1])
    fy = torch.reshape(fy, [B,1,1])
    x0 = torch.reshape(x0, [B,1,1])
    y0 = torch.reshape(y0, [B,1,1])
    
    # unproject
    EPS = 1e-6
    x = ((z+EPS)/fx)*(x-x0)
    y = ((z+EPS)/fy)*(y-y0)
    
    x = torch.reshape(x, [B,-1])
    y = torch.reshape(y, [B,-1])
    z = torch.reshape(z, [B,-1])
    xyz = torch.stack([x,y,z], dim=2)
    return xyz

def pack_intrinsics(fx, fy, x0, y0):
    K = torch.zeros(1, 4, 4, dtype=torch.float32, device=torch.device('cuda'))
    K[:,0,0] = fx
    K[:,1,1] = fy
    K[:,0,2] = x0
    K[:,1,2] = y0
    K[:,2,2] = 1.0
    K[:,3,3] = 1.0
    return K

def apply_4x4(RT, xyz):
    B, N, _ = list(xyz.shape)
    ones = torch.ones_like(xyz[:,:,0:1])
    xyz1 = torch.cat([xyz, ones], 2)
    xyz1_t = torch.transpose(xyz1, 1, 2)
    # this is B x 4 x N
    xyz2_t = torch.matmul(RT, xyz1_t)
    xyz2 = torch.transpose(xyz2_t, 1, 2)
    xyz2 = xyz2[:,:,:3]
    return xyz2

def eye_4x4(B):
    rt = torch.eye(4, device=torch.device('cuda')).view(1,4,4).repeat([B, 1, 1])
    return rt

def fill_ray_single(xyz):
    # xyz is N x 3, and in bird coords
    # we want to fill a voxel tensor with 1's at these inds,
    # and also at any ind along the ray before it

    # xyz = torch.reshape(xyz, (-1,s 3))
    x, y, z = xyz[:,0], xyz[:,1], xyz[:,2]
    # these are N

    x = x.unsqueeze(1)
    y = y.unsqueeze(1)
    z = z.unsqueeze(1)

    # get the hypotenuses
    u = torch.sqrt(x**2+z**2) # flat to ground
    v = torch.sqrt(x**2+y**2+z**2)
    w = torch.sqrt(x**2+y**2)

    # the ray is along the v line
    # we want to find xyz locations along this line

    # get the angles
    EPS=1e-6
    sin_theta = y/(EPS + v) # soh 
    cos_theta = u/(EPS + v) # cah
    sin_alpha = z/(EPS + u) # soh
    cos_alpha = x/(EPS + u) # cah

    samps = 100 #int(np.sqrt(Y**2 + Z**2))
    # for each proportional distance in [0.0, 1.0], generate a new hypotenuse
    dists = torch.linspace(0.0, 1.0, samps, device=torch.device('cuda'))
    dists = torch.reshape(dists, (1, samps))
    v_ = dists * v.repeat(1, samps)

    # now, for each of these v_, we want to generate the xyz
    y_ = sin_theta*v_
    u_ = torch.abs(cos_theta*v_)
    z_ = sin_alpha*u_
    x_ = cos_alpha*u_
    # these are the ref coordinates we want to fill
    x = x_.flatten()
    y = y_.flatten()
    z = z_.flatten()

    xyz = torch.stack([x,y,z], dim=1) #.unsqueeze(0)
    # xyz = Ref2Mem(xyz, Z, Y, X)
    # xyz = torch.squeeze(xyz, dim=0)
    # these are the mem coordinates we want to fill
    N_rays = int(xyz.shape[0] / samps)
    xyz = torch.reshape(xyz, (N_rays, samps, 3))

    return xyz