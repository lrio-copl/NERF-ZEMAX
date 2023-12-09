from yaml import load, CLoader, dump
from pathlib import Path
from os import chdir, getcwd
import numpy as np
import torch as th
from nerfstudio.process_data.realitycapture_utils import _get_rotation_matrix as rot_mat
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.exporter.exporter_utils import collect_camera_poses
import nerfstudio.utils.poses as pose_utils
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def dc2rot(lmn, rays, show = False):
    x = rays[:,0]
    y = rays[:,1]
    z = (-lmn[:,0]*x - lmn[:,1]*y)/lmn[:,2] #coordonée z pour orthogonal (porduit scalaire = 0) 0 = x1x2+y1y2+z1z2 -> z2 = -(x1x2+y1y2)/z1
    v2 = np.concatenate([x.reshape(-1, 1),y.reshape(-1, 1), z.reshape(-1, 1)], axis=1) #premier vecteur orthogonal
    v2 /= np.linalg.norm(v2, axis=1, keepdims=True) #normalisation
    v3 = np.cross(lmn, v2, axis=1) #troisième vecteur orthogonal
    v3 /= np.linalg.norm(v3, axis=1, keepdims=True) #normalisation
    matrices = np.dstack([v2, v3, lmn])
    matrices = R.from_matrix(matrices)

    if show:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        origin = np.zeros(3)
        colors = ['r', 'g', 'b']
        for i, color in enumerate(colors):
            column = matrices.as_matrix()[0][:, i]
            ax.quiver(*origin, *column, color=color, arrow_length_ratio=0.1, label=f'Vector {i+1}')

        ax.quiver(*origin, 0, 0, 1, color='k', arrow_length_ratio=0.1)
        ax.quiver(*origin, 0, 1, 0, color='k', arrow_length_ratio=0.1)
        ax.quiver(*origin, 1, 0, 0, color='k', arrow_length_ratio=0.1)

        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_zlim([0, 1])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        plt.show()

    return matrices


def square_plane(nb_rays, w=1, h=1, z= 0):
    z *= np.ones((nb_rays, 1)) #place le plan au z voulu
    xy = (np.random.rand(nb_rays, 2) -0.5)*2 #x et y de -1 à 1
    xy[:,0] *= w #ajustement à la largeur voulue
    xy[:,1] *= h #ajustement à la hauteur voulue
    xyz = np.concatenate([xy, z], axis=1)

    return xyz

def circle_plane(nb_rays, rayon = 1, z = 1):
    z *= np.ones((nb_rays, 1)) #place le plan au z voulu
    rt = np.random.rand(nb_rays, 2) #génère rayon et theta
    #polaire à cartésien
    x = rayon * rt[:, 0] * np.cos(2*np.pi * rt[:, 1])
    y = rayon * rt[:, 0] * np.sin(2*np.pi * rt[:, 1])
    xyz = np.column_stack([x, y, z])
    return xyz

def half_sphere(nb_rays, theta=180, phi=180, ax='z', show=False):
    axis_val = {'x': 0, 'y': 1, 'z': 2} #orientation de la sphère
    #nombre de degrés 
    theta = np.deg2rad(theta) 
    phi = np.deg2rad(phi)
    #nombre de degrés centrés autour de 90 degrés
    theta = np.random.rand(nb_rays) * theta + (np.pi - theta) / 2
    phi = np.random.rand(nb_rays) * phi + (np.pi - phi) / 2
    r = np.ones(nb_rays)

    #sphérique à cartésien
    xyz = np.zeros((nb_rays, 3))
    if ax == 'x':
        xyz[:, 1] = r * np.sin(theta) * np.cos(phi)
        xyz[:, 0] = r * np.sin(theta) * np.sin(phi)
        xyz[:, 2] = r * np.cos(theta)
        
    elif ax == 'y':
        xyz[:, 2] = r * np.sin(theta) * np.cos(phi)
        xyz[:, 1] = r * np.sin(theta) * np.sin(phi)
        xyz[:, 0] = r * np.cos(theta)

    elif ax == 'z':
        xyz[:, 1] = r * np.sin(theta) * np.cos(phi)
        xyz[:, 2] = r * np.sin(theta) * np.sin(phi)
        xyz[:, 0] = r * np.cos(theta)

    if show:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=5)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()

    return xyz

def generate_diffuse_plane(nbrays, plane1 = square_plane, plane2 = circle_plane):
    xyz = plane1(nbrays, z= 0) #origine à 0
    lmn = plane2(nbrays, z = 1) - xyz #arrivée à 1
    lmn /= np.linalg.norm(lmn, axis=1,keepdims=True) #normalisation lmn
    rays = np.concatenate([xyz, lmn], axis=1)

    return rays

def generate_diffuse_sphere(nbrays, rayon1 = 1,direction_theta = 90, direction_phi = 90, convexe = False):
    if convexe: #sommet à 0
        xyz = half_sphere(nbrays, ax='z', show=False)
        lmn = xyz.copy()
        xyz[:, 2] -= 1
        xyz *= -rayon1
    else: #centre à 0
        xyz = half_sphere(nbrays, ax='z', show=False)
        lmn = xyz.copy()
        xyz *= rayon1

    lmn /= np.linalg.norm(lmn, axis=1,keepdims=True) #normalisation
    directions = half_sphere(nbrays, theta = direction_theta, phi=direction_phi, ax='z', show=False) #génération de directions aléatoires sur une demi-sphère
    rot = dc2rot(lmn,directions) #rotation pour que les rayons soient bien orientés par rapport à la sphère
    directions = rot.apply(directions)
    rays = np.concatenate([xyz, directions], axis=1)

    return rays
   
class NerfRays:
    def __init__(self, config):
        self.config_file = config
        self.reload_yaml()
        self.config, self.pipeline, _, self.step = self.load_model(self.yaml_file)
        self.ray_data_load = np.loadtxt(
            self.yaml_file["Ray file"], skiprows=2, encoding="UTF-16", delimiter=","
        )
        self.num_diffuse_rays = self.yaml_file.get("num_diffuse_rays", None)
        self.ray_data_load_0 = np.array([[0, 0, 0, 0, 0, 1]]).astype(float)

        if self.num_diffuse_rays is not None and shape == 'sphere':
            self.ray_data = generate_diffuse_sphere(self.num_diffuse_rays)

            self.ray_data_0 = np.tile(self.ray_data_load_0, (len(self.ray_data), 1))
        elif self.num_diffuse_rays is not None and shape == 'plane':
            self.ray_data = generate_diffuse_plane(self.num_diffuse_rays)

            self.ray_data_0 = np.tile(self.ray_data_load_0, (len(self.ray_data), 1))
        else:
            self.ray_data_0 = self.extend_nerf_rays(
                self.ray_data_load_0, self.yaml_file["Extended ray file"]
            )
            self.ray_data = self.extend_nerf_rays(
                self.ray_data_load, self.yaml_file["Extended ray file"]
            )


    def reload_yaml(self):
        self.yaml_file = load(open(self.config_file), Loader=CLoader)

    def extend_nerf_rays(self, rays, file):
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

    def incident(self, rays, lmn = None):
        if lmn is None:
            theta = np.linspace(-np.pi/2, np.pi/2, rays.shape[0])
            phi = np.linspace(-np.pi/2, np.pi/2, rays.shape[0])
            l = np.sin(phi) * np.cos(theta)
            m = np.sin(phi) * np.sin(theta)
            n = np.cos(phi)
            lmn = np.concatenate([l.reshape(-1, 1), m.reshape(-1, 1), n.reshape(-1, 1)], axis=1)

        matrices = dc2rot(lmn, rays)
        rays[:, 3:] = matrices.apply(rays[:, 3:])
        
        return rays


    def aspherique(self, rays, rayon, coef=np.zeros(1), k=0):
        if rayon != 0:
            i = rays[:,:2]
            theta = 1 / rayon
            angles = i*theta #position de chaque microlentille sur la sphère
            trans = i - rayon * np.sin(angles) #translations pour que la distance entre les microlentilles soit la même sur la sphère
            a =np.arange(4, 4+(2*len(coef)), 2) #exposant pour termes asphériques
            r = np.sqrt(np.sum(np.square(trans),axis=1)) #coordonées polaires
            c = np.sum(coef * np.power(np.tile(r, (len(coef), 1)).T, a), axis=1)#application des puissances sur r pour chaque coefficient

            #calcul de la translation en z
            argument = 1 - (1 + k) * (np.square(r) / rayon**2)
            argument = np.maximum(argument, 0)  #évite les chiffres négatifs dans la racine
            trans_z = np.square(r) / (rayon * (1 + np.sqrt(argument))) + c
            trans = np.concatenate((trans, trans_z[:, np.newaxis]), axis=1)

            rays[:, :3] -= trans

            matrix = R.from_euler('yz', angles)
            rays[:, 3:] = matrix.apply(rays[:,3:])

        return rays

    def load_model(self, yaml_file):
        from nerfstudio.utils.eval_utils import eval_setup

        default = getcwd()
        chdir(yaml_file["Working directory"])
        eval_output = eval_setup(
            Path(yaml_file["Nerf file"]), eval_num_rays_per_chunk=None, test_mode="test"
        )
        chdir(default)
        return eval_output

    def apply_c2w(self, c2w, ray_data):
        ray_data_cop = ray_data.copy()
        ray_data_cop[:, :3] = pose_utils.multiply(
            c2w, th.from_numpy(ray_data[:, :3].T)
        ).T
        rotation = c2w[..., :3, :3]
        ray_data_cop[:, 3:] = th.sum(
            th.from_numpy(ray_data[:, 3:])[..., None, :] * rotation, dim=-1
        )
        return ray_data_cop

    def export_to_yaml(
        self,
        offset_trans=np.array([0, 0, 0]).astype(float),
        offset_rot=np.array([0, 0, 0]).astype(float),
        offset_microlens=0,
        offset_scale=1,
        reference_pose=None,
        output_file=None,
    ):
        offset_scale *= self.yaml_file["Scale"]
        offset_trans = np.asarray(offset_trans).astype(float)
        offset_trans += [
            self.yaml_file["Y translation"],
            self.yaml_file["X translation"],
            self.yaml_file["Z translation"],
        ]
        offset_trans[:2] *= offset_scale
        offset_trans[2] *= self.yaml_file["Scale"]
        offset_microlens += self.yaml_file["Z distance microlens"]
        offset_microlens *= self.yaml_file["Scale"]
        offset_rot = np.asarray(offset_rot).astype(float)
        offset_rot += [
            self.yaml_file["X rotation"],
            self.yaml_file["Y rotation"],
            self.yaml_file["Z rotation"],
        ]

        self.yaml_file["Y translation"] = float(offset_trans[0] / offset_scale)
        self.yaml_file["X translation"] = float(offset_trans[1] / offset_scale)
        self.yaml_file["Z translation"] = float(offset_trans[2] / (offset_scale))
        self.yaml_file["X rotation"] = float(offset_rot[0])
        self.yaml_file["Y rotation"] = float(offset_rot[1])
        self.yaml_file["Z rotation"] = float(offset_rot[2])
        self.yaml_file["Z distance microlens"] = float(
            offset_microlens / (offset_scale)
        )
        self.yaml_file["Scale"] = float(offset_scale)
        if output_file is not None:
            self.yaml_file["Output file"] = output_file
        if reference_pose is not None:
            self.yaml_file["Reference pose"] = reference_pose
        dump(self.yaml_file, open(self.config_file, "w"))
        self.reload_yaml()

    def reset_yaml(self, reference_pose=None, output_file=None):
        self.yaml_file["Y translation"] = 0
        self.yaml_file["X translation"] = 0
        self.yaml_file["Z translation"] = 0
        self.yaml_file["X rotation"] = 0
        self.yaml_file["Y rotation"] = 0
        self.yaml_file["Z rotation"] = 0
        self.yaml_file["Z distance microlens"] = 0
        self.yaml_file["Scale"] = self.yaml_file["Scale"]
        if reference_pose is not None:
            self.yaml_file["Reference pose"] = reference_pose
        if output_file is not None:
            self.yaml_file["Output file"] = output_file
        dump(self.yaml_file, open(self.config_file, "w"))
        self.reload_yaml()

    def update_output_file(self, output_file):
        self.yaml_file["Output file"] = output_file

    def tranform_nerf_rays(
        self, ray_data, trans_x, trans_y, trans_z, micro_z, c2w, rot_x, rot_y, rot_z
    ):   
        ray_data[:, 1] += trans_x
        ray_data[:, 0] += trans_y
        ray_data = self.apply_c2w(
            pose_utils.to4x4(
                th.from_numpy(
                    np.concatenate(
                        (
                            rot_mat(rot_x, rot_y, rot_z),
                            np.array([0, 0, 0]).reshape(-1, 1),
                        ),
                        axis=1,
                    )
                )
            ),
            ray_data,
        )
        ray_data[:, :3] += micro_z * ray_data[:, 3:] / ray_data[:, -1].reshape(-1, 1)
        ray_data[:, 2] -= trans_z + micro_z
        ray_data = self.apply_c2w(c2w, ray_data)
        ray_data[:, 3:] *= -1
        return ray_data

    def extract(
        self,
        offset_trans=np.array([0, 0, 0]).astype(float),
        offset_rot=np.array([0, 0, 0]).astype(float),
        offset_microlens=0,
        offset_scale=1,
        telecentric=False,
        red_plane=False,
    ):
        offset_scale *= self.yaml_file["Scale"]
        offset_trans = np.asarray(offset_trans).astype(float)
        offset_trans += [
            self.yaml_file["Y translation"],
            self.yaml_file["X translation"],
            self.yaml_file["Z translation"],
        ]
        offset_trans = offset_trans.astype(float)
        offset_trans[:2] *= offset_scale
        offset_trans[2] *= self.yaml_file["Scale"]
        offset_microlens += self.yaml_file["Z distance microlens"]
        offset_microlens *= self.yaml_file["Scale"]
        offset_rot = np.asarray(offset_rot).astype(float)
        offset_rot += [
            self.yaml_file["X rotation"],
            self.yaml_file["Y rotation"],
            self.yaml_file["Z rotation"],
        ]
        if telecentric:
            ray_data_load = self.ray_data_load_0
            ray_data_source = self.ray_data_0.copy()
        else:
            ray_data_load = self.ray_data_load
            ray_data_source = self.ray_data.copy()

        ray_data = ray_data_source.copy()

        ray_data[:, :2] *= offset_scale

        c2w = np.diag(np.ones(4))
        if "Reference pose" in self.yaml_file.keys():
            c2w_pose = pose_utils.to4x4(
                th.from_numpy(
                    np.array(
                        [
                            poses["transform"]
                            for poses in sum(collect_camera_poses(self.pipeline), [])
                            if str(self.yaml_file["Reference pose"])
                            in poses["file_path"]
                        ][0]
                    )
                )
            )
            c2w = c2w_pose
  
        #ray_data = self.incident(ray_data)
        ray_data = self.aspherique(ray_data, offset_microlens)
        ray_data = self.tranform_nerf_rays(
            ray_data,
            offset_trans[0],
            offset_trans[1],
            offset_trans[2],
            offset_microlens,
            c2w,
            offset_rot[0],
            offset_rot[1],
            offset_rot[2],
        )
        write_rgb = np.empty_like(ray_data[:, :3])
        write_depth = np.empty_like(ray_data[:, :1])
        write_acc = np.empty_like(ray_data[:, :1])
        if telecentric:
            factor_local = int(ray_data_load.shape[0] * 9945)
        else:
            factor_local = int(ray_data_load.shape[0] * self.yaml_file["Factor"])

        pixel_area = th.ones((factor_local, 1)).cuda()
        nears = th.zeros((factor_local, 1)).cuda()
        fars = th.ones((factor_local, 1)).cuda() * 10
        camera_indices = th.zeros((factor_local, 1)).cuda()
        ray_data_torch = th.from_numpy(ray_data)
        # red = th.from_numpy(np.array([1, 0, 0]).astype(float)).cuda()
        for ray_data_i in range(ray_data.shape[0] // factor_local):
            if not telecentric:
                print(f"Done: {100*(ray_data_i/(ray_data.shape[0] // factor_local))} %")
            bundle = RayBundle(
                origins=ray_data_torch[
                    ray_data_i * factor_local : (ray_data_i + 1) * factor_local, :3
                ].cuda(),
                directions=ray_data_torch[
                    ray_data_i * factor_local : (ray_data_i + 1) * factor_local, 3:
                ].cuda(),
                pixel_area=pixel_area,
                nears=nears,
                fars=fars,
                camera_indices=camera_indices,
            )
            ray_outputs = self.pipeline.model.get_outputs(bundle)
            write_depth[ray_data_i * factor_local : (ray_data_i + 1) * factor_local] = (
                ray_outputs["depth"].cpu().detach().numpy()
            )
            write_rgb[ray_data_i * factor_local : (ray_data_i + 1) * factor_local] = (
                ray_outputs["rgb"].cpu().detach().numpy()
            )
            write_acc[ray_data_i * factor_local : (ray_data_i + 1) * factor_local] = (
                ray_outputs["accumulation"].cpu().detach().numpy()
            )

        if red_plane:
            write_rgb[
                np.isclose(
                    write_depth,
                    offset_microlens,
                    atol=0.05,
                )[:, 0],
                :,
            ] = [1, 0, 0]

        ray_data_source[:, 3:] *= -1
        write_rgb = np.clip(write_rgb, 0, 1)
        return (
            self.yaml_file["Output file"],
            ray_data_source,
            np.concatenate(
                [write_rgb, write_acc],
                axis=1,
            ),
        )
