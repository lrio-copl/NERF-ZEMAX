from yaml import load, CLoader
from pathlib import Path
from os import name as osname, chdir, getcwd
import numpy as np


def extend_nerf_rays(rays, file):
    extended_rays_location = np.loadtxt(file)
    if rays.ndim == 1:
        rays.shape = (1, rays.shape[0])
    output_rays = np.empty(
        (extended_rays_location.shape[0] * rays.shape[0], rays.shape[1])
    )
    for locationi, location in enumerate(extended_rays_location):
        output_rays[locationi * rays.shape[0] : (locationi + 1) * rays.shape[0]] = (
            rays + location
        )
    return output_rays


def load_model(config):
    from nerfstudio.utils.eval_utils import eval_setup

    yaml_file = load(open(config), Loader=CLoader)

    default = getcwd()
    chdir(yaml_file["Working directory"])
    eval_output = eval_setup(
        Path(yaml_file["Nerf file"]), eval_num_rays_per_chunk=None, test_mode="test"
    )
    chdir(default)
    return eval_output


def tranform_nerf_rays(ray_data, trans_x, trans_y, trans_z, c2w, rot_z):
    ray_data[:, 0] *= -1
    ray_data[:, 3] *= -1
    ray_data[:, 1] += trans_x
    ray_data[:, 0] += trans_y
    ray_data[:, 2] += trans_z
    ray_data[:, :3] = np.dot(
        np.concatenate([ray_data[:, :3], np.ones((ray_data.shape[0], 1))], axis=1),
        c2w.T,
    )[:, :3]
    ray_data[:, 3:] = np.dot(
        np.concatenate([ray_data[:, 3:], np.ones((ray_data.shape[0], 1))], axis=1),
        c2w.T,
    )[:, :3]
    rot = np.array(
        [
            [
                np.cos(rot_z),
                -np.sin(rot_z),
                0,
                0,
            ],
            [
                np.sin(rot_z),
                np.cos(rot_z),
                0,
                0,
            ],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )

    ray_data[:, :3] = np.dot(
        np.concatenate([ray_data[:, :3], np.ones((ray_data.shape[0], 1))], axis=1),
        rot.T,
    )[:, :3]
    ray_data[:, 3:] = np.dot(
        np.concatenate([ray_data[:, 3:], np.ones((ray_data.shape[0], 1))], axis=1),
        rot.T,
    )[:, :3]
    ray_data[:, 3:] /= np.sqrt(np.sum(ray_data[:, 3:] ** 2, axis=1))[..., None]
    return ray_data


def extract_nerf_rays(
    config, pipeline, offset_trans=np.array([0, 0, 0]), offset_rot=np.array([0, 0, 0])
):
    yaml_file = load(open(config), Loader=CLoader)
    ray_data_load = np.loadtxt(
        yaml_file["Ray file"], skiprows=2, encoding="UTF-16", delimiter=","
    )
    if ray_data_load.ndim == 1:
        ray_data_load.shape = (1, ray_data_load.shape[0])

    ray_data = ray_data_load.copy()

    if "Extended ray file" in yaml_file.keys():
        ray_data = extend_nerf_rays(ray_data, yaml_file["Extended ray file"])
    ray_data_source = ray_data.copy()

    if "Scale" in yaml_file.keys():
        ray_data[:, :3] *= yaml_file["Scale"]

    if osname == "nt":
        from torch import from_numpy, ones, zeros
        from nerfstudio.cameras.rays import RayBundle
        from nerfstudio.exporter.exporter_utils import collect_camera_poses

        if "Reference pose" in yaml_file.keys():
            c2w = np.concatenate(
                [
                    np.array(
                        [
                            poses["transform"]
                            for poses in sum(collect_camera_poses(pipeline), [])
                            if str(yaml_file["Reference pose"]) in poses["file_path"]
                        ][0]
                    ),
                    [
                        [0, 0, 0, 1],
                    ],
                ],
                axis=0,
            )
            c2w = c2w[np.array([1, 0, 2, 3]), :]
            c2w[2, :] *= -1
            theta = (-8.5 + 2.5 + 90 + 45 + 30) * np.pi / 180
            c2w = np.dot(
                c2w,
                np.array(
                    [
                        [np.cos(theta), -np.sin(theta), 0, 0],
                        [np.sin(theta), np.cos(theta), 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1],
                    ]
                ).T,
            )
        ray_data = tranform_nerf_rays(
            ray_data,
            (yaml_file["X translation"] + offset_trans[0]) * yaml_file["Scale"],
            (yaml_file["Y translation"] + offset_trans[1]) * yaml_file["Scale"],
            (yaml_file["Z translation"] + offset_trans[2]) * yaml_file["Scale"],
            c2w,
            yaml_file["Z rotation"],
        )

        write_rgb = np.empty_like(ray_data[:, :3])
        write_acc = np.empty_like(ray_data[:, :1])
        factor_local = int(ray_data_load.shape[0] * yaml_file["Factor"])
        for ray_data_i in range(ray_data.shape[0] // factor_local):
            ray_data_chunk = ray_data[
                ray_data_i * factor_local : (ray_data_i + 1) * factor_local
            ]
            bundle = RayBundle(
                origins=from_numpy(ray_data_chunk[:, :3]).cuda(),
                directions=from_numpy(ray_data_chunk[:, 3:]).cuda(),
                pixel_area=ones((ray_data_chunk.shape[0], 1)).cuda() * 0.01,
                nears=zeros((ray_data_chunk.shape[0], 1)).cuda(),
                fars=ones((ray_data_chunk.shape[0], 1)).cuda() * 10,
                camera_indices=zeros((ray_data_chunk.shape[0], 1)).cuda(),
            )
            ray_outputs = pipeline.model.get_outputs(bundle)
            write_rgb[ray_data_i * factor_local : (ray_data_i + 1) * factor_local] = (
                ray_outputs["rgb"].cpu().detach().numpy()
            )
            write_acc[ray_data_i * factor_local : (ray_data_i + 1) * factor_local] = (
                ray_outputs["accumulation"].cpu().detach().numpy()
            )

    else:
        from numpy import ones

        write_rgb = np.empty_like(ray_data[:, :3])
        write_acc = np.empty_like(ray_data[:, :1])
        for ray_data_i in range(ray_data.shape[0] // ray_data_load.shape[0]):
            ray_data_chunk = ray_data[
                ray_data_i
                * ray_data_load.shape[0] : (ray_data_i + 1)
                * ray_data_load.shape[0]
            ]
            ray_outputs = {
                "rgb": ones((ray_data_chunk.shape[0], 3)),
                "accumulation": ones((ray_data_chunk.shape[0], 1)),
            }
            write_rgb[
                ray_data_i
                * ray_data_load.shape[0] : (ray_data_i + 1)
                * ray_data_load.shape[0]
            ] = ray_outputs["rgb"]
            write_acc[
                ray_data_i
                * ray_data_load.shape[0] : (ray_data_i + 1)
                * ray_data_load.shape[0]
            ] = ray_outputs["accumulation"]

    ray_data_source[:, 3:] *= -1
    write_rgb = np.clip(write_rgb, 0, 1)

    return (
        yaml_file["Output file"],
        ray_data_source,
        np.concatenate(
            [write_rgb, write_acc],
            axis=1,
        ),
    )


if __name__ == "__main__":
    extract_nerf_rays("data/config_test.yml")
