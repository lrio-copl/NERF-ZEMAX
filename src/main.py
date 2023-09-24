from numpy import loadtxt, concatenate
from pathlib import Path
import os
from write_sdf import write_sdf

test = loadtxt(
    "data/new_mapping_xyz_lmn_camera_position_2023-09-_z5(0.2)_precis.txt",
    skiprows=2,
    encoding="UTF-16",
    delimiter=",",
)

# 0.9875 153x65
# test[:, 3:] /= sqrt(sum(test[:, 3:] ** 2, axis=1)).reshape(-1, 1)
# test[:, :3] *= 25.4
test = 
test[:, 3:] *= -1

# print(test)
# import matplotlib.pyplot as plt
# plt.plot(test[:,0], test[:,1], 'o')
# plt.show()

file = Path(r"C:\outputs\poster\nerfacto\2023-09-13_154955\config.yml")

if os.name == "nt":
    from torch import from_numpy, ones, zeros
    from nerfstudio.utils.eval_utils import eval_setup
    from nerfstudio.cameras.rays import RayBundle

    default = os.getcwd()
    os.chdir(r"C:\\")
    config, pipeline, _, step = eval_setup(
        file, eval_num_rays_per_chunk=None, test_mode="test"
    )
    os.chdir(default)

    bundle = RayBundle(
        origins=from_numpy(test[:, :3]).cuda(),
        directions=from_numpy(test[:, 3:]).cuda(),
        pixel_area=ones((test.shape[0], 1)).cuda() * 0.001,
        nears=zeros((test.shape[0], 1)).cuda(),
        fars=ones((test.shape[0], 1)).cuda() * 3000,
        camera_indices=zeros((test.shape[0], 1)).cuda(),
    )

    outputs = pipeline.model.get_outputs(bundle)

    write_sdf(
        "test.sdf",
        test,
        concatenate(
            [
                outputs["rgb"].cpu().detach().numpy(),
                outputs["accumulation"].cpu().detach().numpy(),
            ],
            axis=1,
        ),
    )
