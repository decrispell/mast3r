#!/usr/bin/env python3
# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# gradio demo functions
# --------------------------------------------------------
import pycolmap
import os
from pathlib import Path
import json
import numpy as np
import functools
import trimesh
import copy
from scipy.spatial.transform import Rotation
import tempfile
import shutil
import PIL.Image
import torch

from kapture.converter.colmap.database_extra import kapture_to_colmap
from kapture.converter.colmap.database import COLMAPDatabase

from mast3r.model import AsymmetricMASt3R

from mast3r.colmap.mapping import kapture_import_image_folder_or_list, run_mast3r_matching, glomap_run_mapper
from mast3r.demo import set_scenegraph_options
from mast3r.retrieval.processor import Retriever
from mast3r.image_pairs import make_pairs

import mast3r.utils.path_to_dust3r  # noqa
from dust3r.utils.image import load_images
from dust3r.viz import add_scene_cam, CAM_COLORS, OPENGL
from dust3r.demo import get_args_parser as dust3r_get_args_parser

import matplotlib.pyplot as pl

torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12


def save_recon(img_names, recon, outdir):
    if not outdir.is_dir():
        raise ValueError(f"Directory {outdir} does not exist.")

    camera_info = {}
    extr_keys = sorted(recon.world_to_cam.keys())
    intr_keys = sorted(recon.intrinsics.keys())
    if extr_keys != intr_keys:
        raise ValueError(f"extrinsics keys {extr_keys} != intrinsics keys {intr_keys}")

    for key in extr_keys:
        img_idx = key - 1  # keys are 1-based indices
        img_name = img_names[img_idx]
        world_to_cam = recon.world_to_cam[key]
        camera_info[img_name] = {
            "K": recon.intrinsics[key].tolist(),
            "R": world_to_cam[:3, :3].tolist(),
            "T": world_to_cam[:3, 3].tolist(),
        }
    camera_filename = outdir / "cameras.json"
    print(f"Saving {camera_filename}")
    with camera_filename.open("w") as fd:
        json.dump(camera_info, fd, indent=4)
    points_and_colors = np.array(recon.points3d)
    points = points_and_colors[:, 0, :].astype(np.float32)
    colors = points_and_colors[:, 1, :].astype(np.uint8)
    pcd = trimesh.PointCloud(points, colors=colors)
    points_filename = outdir / "points.ply"
    print(f"Saving {points_filename}")
    pcd.export(points_filename)


class GlomapRecon:
    def __init__(self, world_to_cam, intrinsics, points3d, imgs):
        self.world_to_cam = world_to_cam
        self.intrinsics = intrinsics
        self.points3d = points3d
        self.imgs = imgs


class GlomapReconState:
    def __init__(self, glomap_recon, should_delete=False, cache_dir=None, outfile_name=None):
        self.glomap_recon = glomap_recon
        self.cache_dir = cache_dir
        self.outfile_name = outfile_name
        self.should_delete = should_delete

    def __del__(self):
        if not self.should_delete:
            return
        if self.cache_dir is not None and os.path.isdir(self.cache_dir):
            shutil.rmtree(self.cache_dir)
        self.cache_dir = None
        if self.outfile_name is not None and os.path.isfile(self.outfile_name):
            os.remove(self.outfile_name)
        self.outfile_name = None


def get_reconstructed_scene(glomap_bin, outdir, model, retrieval_model, device, image_size,
                            filelist, transparent_cams, cam_size, scenegraph_type, winsize,
                            win_cyclic, refid, shared_intrinsics, **kw):
    """
    from a list of images, run mast3r inference, sparse global aligner.
    then run get_3D_model_from_scene
    """
    imgs = load_images(filelist, size=image_size, verbose=True)
    if len(imgs) == 1:
        imgs = [imgs[0], copy.deepcopy(imgs[0])]
        imgs[1]['idx'] = 1
        filelist = [filelist[0], filelist[0]]

    scene_graph_params = [scenegraph_type]
    if scenegraph_type in ["swin", "logwin"]:
        scene_graph_params.append(str(winsize))
    elif scenegraph_type == "oneref":
        scene_graph_params.append(str(refid))
    elif scenegraph_type == "retrieval":
        scene_graph_params.append(str(winsize))  # Na
        scene_graph_params.append(str(refid))  # k

    if scenegraph_type in ["swin", "logwin"] and not win_cyclic:
        scene_graph_params.append('noncyclic')
    scene_graph = '-'.join(scene_graph_params)

    sim_matrix = None
    if 'retrieval' in scenegraph_type:
        assert retrieval_model is not None
        retriever = Retriever(retrieval_model, backbone=model, device=device)
        with torch.no_grad():
            sim_matrix = retriever(filelist)

        # Cleanup
        del retriever
        torch.cuda.empty_cache()

    print("making pairs")
    pairs = make_pairs(imgs, scene_graph=scene_graph, prefilter=None, symmetrize=True, sim_mat=sim_matrix)

    cache_dir = os.path.join(outdir, 'cache')

    root_path = os.path.commonpath(filelist)
    filelist_relpath = [
        os.path.relpath(filename, root_path).replace('\\', '/')
        for filename in filelist
    ]
    kdata = kapture_import_image_folder_or_list((root_path, filelist_relpath), shared_intrinsics)
    image_pairs = [
        (filelist_relpath[img1['idx']], filelist_relpath[img2['idx']])
        for img1, img2 in pairs
    ]

    colmap_db_path = os.path.join(cache_dir, 'colmap.db')
    if os.path.isfile(colmap_db_path):
        os.remove(colmap_db_path)

    os.makedirs(os.path.dirname(colmap_db_path), exist_ok=True)
    colmap_db = COLMAPDatabase.connect(colmap_db_path)
    try:
        kapture_to_colmap(kdata, root_path, tar_handler=None, database=colmap_db,
                          keypoints_type=None, descriptors_type=None, export_two_view_geometry=False)
        colmap_image_pairs = run_mast3r_matching(model, image_size, 16, device,
                                                 kdata, root_path, image_pairs, colmap_db,
                                                 False, 5, 1.001,
                                                 False, 3)
        colmap_db.close()
    except Exception as e:
        print(f'Error {e}')
        colmap_db.close()
        exit(1)

    if len(colmap_image_pairs) == 0:
        raise Exception("no matches were kept")

    # colmap db is now full, run colmap
    colmap_world_to_cam = {}
    print("verify_matches")
    f = open(cache_dir + '/pairs.txt', "w")
    for image_path1, image_path2 in colmap_image_pairs:
        f.write("{} {}\n".format(image_path1, image_path2))
    f.close()
    pycolmap.verify_matches(colmap_db_path, cache_dir + '/pairs.txt')

    reconstruction_path = os.path.join(cache_dir, "reconstruction")
    if os.path.isdir(reconstruction_path):
        shutil.rmtree(reconstruction_path)
    os.makedirs(reconstruction_path, exist_ok=True)
    glomap_run_mapper(glomap_bin, colmap_db_path, reconstruction_path, root_path)

    outfile_name = tempfile.mktemp(suffix='_scene.glb', dir=outdir)

    ouput_recon = pycolmap.Reconstruction(os.path.join(reconstruction_path, '0'))
    print(ouput_recon.summary())

    colmap_world_to_cam = {}
    colmap_intrinsics = {}
    colmap_image_id_to_name = {}
    images = {}
    num_reg_images = ouput_recon.num_reg_images()
    for idx, (colmap_imgid, colmap_image) in enumerate(ouput_recon.images.items()):
        colmap_image_id_to_name[colmap_imgid] = colmap_image.name
        if callable(colmap_image.cam_from_world.matrix):
            colmap_world_to_cam[colmap_imgid] = colmap_image.cam_from_world.matrix(
            )
        else:
            colmap_world_to_cam[colmap_imgid] = colmap_image.cam_from_world.matrix
        camera = ouput_recon.cameras[colmap_image.camera_id]
        K = np.eye(3)
        K[0, 0] = camera.focal_length_x
        K[1, 1] = camera.focal_length_y
        K[0, 2] = camera.principal_point_x
        K[1, 2] = camera.principal_point_y
        colmap_intrinsics[colmap_imgid] = K

        with PIL.Image.open(os.path.join(root_path, colmap_image.name)) as im:
            images[colmap_imgid] = np.asarray(im)

        if idx + 1 == num_reg_images:
            break  # bug with the iterable ?
    points3D = []
    num_points3D = ouput_recon.num_points3D()
    for idx, (pt3d_id, pts3d) in enumerate(ouput_recon.points3D.items()):
        points3D.append((pts3d.xyz, pts3d.color))
        if idx + 1 == num_points3D:
            break  # bug with the iterable ?
    scene = GlomapRecon(colmap_world_to_cam, colmap_intrinsics, points3D, images)
    scene_state = GlomapReconState(scene, False, cache_dir, outfile_name)
    outfile = get_3D_model_from_scene( scene_state, transparent_cams, cam_size)
    return scene_state, outfile


def get_3D_model_from_scene(scene_state, transparent_cams=False, cam_size=0.05):
    """
    extract 3D_model (glb file) from a reconstructed scene
    """
    if scene_state is None:
        return None
    outfile = scene_state.outfile_name
    if outfile is None:
        return None

    recon = scene_state.glomap_recon

    scene = trimesh.Scene()
    pts = np.stack([p[0] for p in recon.points3d], axis=0)
    col = np.stack([p[1] for p in recon.points3d], axis=0)
    pct = trimesh.PointCloud(pts, colors=col)
    scene.add_geometry(pct)

    # add each camera
    cams2world = []
    for i, (id, pose_w2c_3x4) in enumerate(recon.world_to_cam.items()):
        intrinsics = recon.intrinsics[id]
        focal = (intrinsics[0, 0] + intrinsics[1, 1]) / 2.0
        camera_edge_color = CAM_COLORS[i % len(CAM_COLORS)]
        pose_w2c = np.eye(4)
        pose_w2c[:3, :] = pose_w2c_3x4
        pose_c2w = np.linalg.inv(pose_w2c)
        cams2world.append(pose_c2w)
        add_scene_cam(scene, pose_c2w, camera_edge_color,
                      None if transparent_cams else recon.imgs[id], focal,
                      imsize=recon.imgs[id].shape[1::-1], screen_width=cam_size)

    rot = np.eye(4)
    rot[:3, :3] = Rotation.from_euler('y', np.deg2rad(180)).as_matrix()
    scene.apply_transform(np.linalg.inv(cams2world[0] @ OPENGL @ rot))
    print('(exporting 3D scene to', outfile, ')')
    scene.export(file_obj=outfile)

    return outfile

def main_demo(filelist, glomap_bin, tmpdirname, model, retrieval_model, device, image_size):
    print('Outputing stuff in', tmpdirname)

    available_scenegraph_type = [("complete: all possible image pairs", "complete"),
                                 ("swin: sliding window", "swin"),
                                 ("logwin: sliding window with long range", "logwin"),
                                 ("oneref: match one image with all", "oneref")]
    if retrieval_model is not None:
        available_scenegraph_type.insert(1, ("retrieval: connect views based on similarity", "retrieval"))

    inputfiles = []
    shared_intrinsics = False
    scenegraph_type = "retrieval"
    #scenegraph_type = "complete"

    # adjust the camera size in the output pointcloud
    cam_size = 0.01
    transparent_cams = False

    winsize = 50
    win_cyclic = False
    refid = 0

    scene, outmodel = get_reconstructed_scene(glomap_bin, tmpdirname, model, retrieval_model, device, image_size,
                            filelist, transparent_cams, cam_size, scenegraph_type, winsize,
                            win_cyclic, refid, shared_intrinsics)
    print(f"saved glb to {outmodel}")
    return scene.glomap_recon


if __name__ == "__main__":

    parser = dust3r_get_args_parser()
    parser.add_argument('--glomap_bin', default='glomap', type=str, help='glomap bin')
    parser.add_argument('--retrieval_model', default=None, type=str, help="retrieval_model to be loaded")

    actions = parser._actions
    for action in actions:
        if action.dest == 'model_name':
            action.choices = ["MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"]
    # change defaults
    parser.prog = 'mast3r demo'

    parser.add_argument("--img_dir", type=Path, help="folder containing input images")
    parser.add_argument("--output_dir", type=Path, help="output folder")
    args = parser.parse_args()

    if args.weights is not None:
        weights_path = args.weights
    else:
        weights_path = "naver/" + args.model_name

    model = AsymmetricMASt3R.from_pretrained(weights_path).to(args.device)

    filelist = sorted(args.img_dir.glob("*.*"))
    img_names = [str(f.relative_to(args.img_dir)) for f in filelist]

    filelist_str = [str(f) for f in filelist]

    print(f"Found {len(filelist)} input images in {args.img_dir}.")

    recon = main_demo(filelist_str, args.glomap_bin, args.output_dir, model, args.retrieval_model, args.device, args.image_size)

    save_recon(img_names, recon, args.output_dir)
