import numpy as np

import scipy.ndimage as spim
from scipy.spatial.distance import euclidean
import porespy as ps

import skimage as ski
from skimage import feature
from skimage import filters
from skimage import morphology
from skimage.morphology import disk, ball

from tqdm import tqdm

from numba import njit, prange

import scipy as sp
import openpnm as op

from porespy.tools import extend_slice
import openpnm.models.geometry as op_gm

import networkx as nx


def regions_to_network(im, dt=None, sigma=0.5, voxel_size=1):
    r"""
    Analyzes an image that has been partitioned into pore regions and extracts
    the pore and throat geometry as well as network connectivity.

    Parameters
    ----------
    im : ND-array
        An image of the pore space partitioned into individual pore regions.
        Note that this image must have zeros indicating the solid phase.

    dt : ND-array
        The distance transform of the pore space.  If not given it will be
        calculated, but it can save time to provide one if available.

    voxel_size : scalar
        The resolution of the image, expressed as the length of one side of a
        voxel, so the volume of a voxel would be **voxel_size**-cubed.  The
        default is 1, which is useful when overlaying the PNM on the original
        image since the scale of the image is alway 1 unit lenth per voxel.

    Returns
    -------
    A dictionary containing all the pore and throat size data, as well as the
    network topological information.  The dictionary names use the OpenPNM
    convention (i.e. 'pore.coords', 'throat.conns') so it may be converted
    directly to an OpenPNM network object using the ``update`` command.

    """
    G = nx.Graph()

    struc_elem = disk if im.ndim == 2 else ball

    if dt is None:
        dt = spim.distance_transform_edt(im > 0)
        dt = spim.gaussian_filter(input=dt, sigma=sigma)

    # Get 'slices' into im for each pore region
    slices = spim.find_objects(im)

    # Initialize arrays
    G.add_nodes_from(sp.arange(1, sp.amax(im)+1))

    # Start extracting size information for pores and throats
    for i in G:

        if slices[i - 1] is None:
            continue
        s = extend_slice(slices[i - 1], im.shape)
        sub_im = im[s]
        pore_im = sub_im == i
        sub_dt = dt[s]
        padded_mask = sp.pad(pore_im, pad_width=1, mode='constant')
        pore_dt = spim.distance_transform_edt(padded_mask)

        s_offset = sp.array([i.start for i in s])
        G.nodes[i]['p_coords'] = (
            spim.center_of_mass(pore_im) + s_offset)*voxel_size
        G.nodes[i]['p_volume'] = sp.sum(pore_im)*(voxel_size**3)
        G.nodes[i]['p_dia_local'] = (
            (2*sp.amax(pore_dt)) - sp.sqrt(3))*voxel_size
        G.nodes[i]['p_dia_global'] = 2*sp.amax(sub_dt)*voxel_size
        G.nodes[i]['p_area_surf'] = sp.sum(pore_dt == 1)*(voxel_size)**2

        G.nodes[i]['incrived_diameter'] = G.nodes[i]['p_dia_local'] * voxel_size
        G.nodes[i]['equivalent_diameter'] = 2 * \
            ((3/4*G.nodes[i]['p_volume']/sp.pi)**(1/3))

        im_w_throats = spim.binary_dilation(
            input=pore_im, structure=struc_elem(1))
        im_w_throats = im_w_throats*sub_im

        Pn = sp.unique(im_w_throats)[1:]

        for j in Pn:
            if j < i:
                G.add_edge(i, j)
                vx = sp.where(im_w_throats == j)
                G.edges[i, j]['t_diameter'] = 2*sp.amax(sub_dt[vx])*voxel_size
                G.edges[i, j]['t_perimeter'] = sp.sum(
                    sub_dt[vx] < 2)*voxel_size
                G.edges[i, j]['t_area'] = sp.size(vx[0])*(voxel_size**2)

                G.edges[i, j]['t_equiv_dia'] = (
                    G.edges[i, j]['t_area'] * (voxel_size**2))**0.5

                t_inds = tuple([i+j for i, j in zip(vx, s_offset)])
                temp = sp.where(dt[t_inds] == sp.amax(dt[t_inds]))[0][0]
                if im.ndim == 2:
                    G.edges[i, j]['t_coords'] = np.array(
                        [t_inds[0][temp], t_inds[1][temp]])*voxel_size
                else:
                    G.edges[i, j]['t_coords'] = np.array(
                        [t_inds[0][temp], t_inds[1][temp], t_inds[2][temp]])*voxel_size
                PT1 = euclidean(G.nodes[i]['p_coords'],
                                G.edges[i, j]['t_coords'])*voxel_size
                PT2 = euclidean(G.nodes[j]['p_coords'],
                                G.edges[i, j]['t_coords'])*voxel_size

                G.edges[i, j]['total_length'] = PT1 + PT2

                PT1 = PT1-G.nodes[i]['p_dia_local']/2*voxel_size
                PT2 = PT2-G.nodes[j]['p_dia_local']/2*voxel_size
                G.edges[i, j]['length'] = PT1 + PT2

                dist = (G.nodes[i]['p_coords']-G.nodes[j]
                        ['p_coords'])*voxel_size
                G.edges[i, j]['direct_length'] = euclidean(
                    G.nodes[j]['p_coords'], G.nodes[j]['p_coords'])*voxel_size

    return G

def remove_padding(regions,boundary_faces):
    if 'left' in boundary_faces:
        regions = regions[3:, :]  # x
    if 'right' in boundary_faces:
        regions = regions[:-3, :]
    if 'front' in boundary_faces and 'bottom' in boundary_faces:
        regions = regions[:, 3:]  # y
    if 'back' in boundary_faces and 'top' in boundary_faces:
        regions = regions[:, :-3]

    return regions

def watershed_seg(bin_im, size=np.ones((3, 3, 3)), sigma=0.4):

    edt = spim.distance_transform_edt(bin_im)
    edt = spim.gaussian_filter(input=edt, sigma=sigma)
    local_maxi = feature.peak_local_max(
        edt, indices=False, footprint=size, labels=bin_im)

    markers = spim.label(local_maxi)[0]
    labels = morphology.watershed(-edt, markers, mask=bin_im)

    b_num = np.amax(labels)

    boundary_faces = ['top', 'bottom', 'left', 'right', 'front', 'back']

    p_regions = ps.networks.add_boundary_regions(
        regions=labels, faces=boundary_faces)

    edt = ps.tools.pad_faces(im=edt, faces=boundary_faces)
    p_bin_im = ps.tools.pad_faces(im=bin_im, faces=boundary_faces)

    p_regions = p_regions*p_bin_im
    p_regions = ps.tools.make_contiguous(p_regions)

    net = regions_to_network(p_regions, edt)

    for node in net:
        if(node <= b_num):
            net.nodes[node]['internal'] = True

    regions = remove_padding(p_regions, boundary_faces)

    return regions, net


def merge_net(net, regions, thresh_ratio):

    for p in net:
        pore_diameter = net.nodes[p]['equivalent_diameter']

        neighbors = net.neighbors(p)

        t_max = np.NINF
        t_id = None
        for n in neighbors:          
            throat_diameter = net.edges[n, p]['t_diameter']

            if(t_max < throat_diameter):
                t_id = n
                t_max = throat_diameter

        if pore_diameter*thresh_ratio < t_max:
            net = nx.contracted_edge(net, (n, p), self_loops=False)
            regions[regions == p] = n

    return regions, net