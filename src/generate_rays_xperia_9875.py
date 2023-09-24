import numpy as np

pinhole_locations = np.indices((153, 65)).reshape(2, -1).T
pinhole_locations = pinhole_locations.astype(float)
pinhole_locations -= np.array([152 // 2, 64 // 2])
pinhole_locations *= 0.9875
np.savetxt(
    "../data/microlensxperia9875.dat",
    np.concatenate(
        [
            pinhole_locations,
            np.zeros_like(pinhole_locations),
            np.zeros_like(pinhole_locations),
        ],
        axis=1,
    ),
)
# 0.9875 153x65
