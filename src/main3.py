from extract_nerf_rays import extract_nerf_rays, load_model
from write_sdf import write_sdf
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.widgets import Slider


if __name__ == "__main__":
    config, pipeline, _, step = load_model("data/config_test_simple.yml")
    _, _, write_rgb = extract_nerf_rays(
        "data/config_test_simple.yml", pipeline, offset_trans=np.array([0, 0, 0])
    )

    fig, axs = plt.subplots(1, 1, figsize=(10, 8))
    fig.subplots_adjust(bottom=0.5)

    im = axs.imshow(
        np.swapaxes(write_rgb[:, :3].reshape(153, 65, 3), 0, 1), origin="lower"
    )

    sliderx_ax = fig.add_axes([0.20, 0.3, 0.60, 0.03])
    sliderx = Slider(
        ax=sliderx_ax, label="Offset X", valmin=-200, valmax=200, valinit=0
    )

    slidery_ax = fig.add_axes([0.20, 0.2, 0.60, 0.03])
    slidery = Slider(
        ax=slidery_ax, label="Offset Y", valmin=-200, valmax=200, valinit=0
    )

    sliderz_ax = fig.add_axes([0.20, 0.1, 0.60, 0.03])
    sliderz = Slider(
        ax=sliderz_ax, label="Offset Z", valmin=-200, valmax=200, valinit=0
    )

    def update(val):
        # Update the image's colormap
        _, _, write_rgb = extract_nerf_rays(
            "data/config_test_simple.yml",
            pipeline,
            offset_trans=np.array([slidery.val, sliderx.val, sliderz.val]),
        )
        im.set_data(np.swapaxes(write_rgb[:, :3].reshape(153, 65, 3), 0, 1))
        im.autoscale()

        # Redraw the figure to ensure it updates
        fig.canvas.draw_idle()

    sliderz.on_changed(update)
    slidery.on_changed(update)
    sliderx.on_changed(update)

    plt.show()
    # plt.show()
