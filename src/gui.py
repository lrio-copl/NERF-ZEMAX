from extract_nerf_rays import NerfRays
from write_sdf import write_sdf
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.widgets import Slider, Button, TextBox


if __name__ == "__main__":
    nerf_rays = NerfRays("data/config_test.yml")
    _, _, write_rgb = nerf_rays.extract(
        offset_trans=np.array([0, 0, 0]), telecentric=True, red_plane=True
    )

    fig, axs = plt.subplots(1, 1, figsize=(10, 8))
    fig.subplots_adjust(bottom=0.6)
    fig.subplots_adjust(left=0.2)

    im = axs.imshow(
        np.swapaxes(write_rgb[:, :3].reshape(153, 65, 3), 0, 1), origin="lower"
    )
    textbox_ax = fig.add_axes([0.20, 0.55, 0.6, 0.05])
    textbox = TextBox(
        textbox_ax,
        "Reference pose",
        initial=nerf_rays.yaml_file["Reference pose"],
    )
    textbox_output_ax = fig.add_axes([0.20, 0.45, 0.6, 0.05])
    textbox_output = TextBox(
        textbox_output_ax,
        "Output file",
        initial=nerf_rays.yaml_file["Output file"],
    )

    sliderx_ax = fig.add_axes([0.20, 0.4, 0.60, 0.03])
    sliderx = Slider(
        ax=sliderx_ax, label="Offset X", valmin=-200, valmax=200, valinit=0
    )

    slidery_ax = fig.add_axes([0.20, 0.35, 0.60, 0.03])
    slidery = Slider(
        ax=slidery_ax, label="Offset Y", valmin=-200, valmax=200, valinit=0
    )

    sliderz_ax = fig.add_axes([0.20, 0.3, 0.60, 0.03])
    sliderz = Slider(
        ax=sliderz_ax, label="Offset Z", valmin=-200, valmax=200, valinit=0
    )
    slider_rotx_ax = fig.add_axes([0.20, 0.25, 0.60, 0.03])
    slider_rotx = Slider(
        ax=slider_rotx_ax, label="Rotation X", valmin=-180, valmax=180, valinit=0
    )

    slider_roty_ax = fig.add_axes([0.20, 0.2, 0.60, 0.03])
    slider_roty = Slider(
        ax=slider_roty_ax, label="Rotation Y", valmin=-180, valmax=180, valinit=0
    )

    slider_rotz_ax = fig.add_axes([0.20, 0.15, 0.60, 0.03])
    slider_rotz = Slider(
        ax=slider_rotz_ax, label="Rotation Z", valmin=-180, valmax=180, valinit=0
    )

    sliderzm_ax = fig.add_axes([0.20, 0.1, 0.60, 0.03])
    sliderzm = Slider(
        ax=sliderzm_ax, label="Distance microlens", valmin=-200, valmax=200, valinit=0
    )
    slider_scale_ax = fig.add_axes([0.20, 0.05, 0.60, 0.03])
    slider_scale = Slider(
        ax=slider_scale_ax, label="Scale", valmin=-2, valmax=2, valinit=0
    )

    button_ax = fig.add_axes([0.075, 0.8, 0.15, 0.05])
    button = Button(button_ax, "Export to yaml")
    buttonsdf_ax = fig.add_axes([0.075, 0.7, 0.15, 0.05])
    buttonsdf = Button(buttonsdf_ax, "Write SDF")

    def update(val):
        # Update the image's colormap
        _, _, write_rgb = nerf_rays.extract(
            offset_trans=np.array([slidery.val, sliderx.val, sliderz.val]),
            offset_rot=np.array(
                [
                    slider_rotx.val,
                    slider_roty.val,
                    slider_rotz.val,
                ]
            ),
            offset_microlens=sliderzm.val,
            offset_scale=10 ** (slider_scale.val),
            telecentric=True,
            red_plane=True,
        )
        im.set_data(np.swapaxes(write_rgb[:, :3].reshape(153, 65, 3), 0, 1))
        im.autoscale()

        # Redraw the figure to ensure it updates
        fig.canvas.draw_idle()

    def update_botton(event):
        nerf_rays.export_to_yaml(
            offset_trans=np.array([slidery.val, sliderx.val, sliderz.val]),
            offset_rot=np.array(
                [
                    slider_rotx.val,
                    slider_roty.val,
                    slider_rotz.val,
                ]
            ),
            offset_microlens=sliderzm.val,
            offset_scale=10 ** (slider_scale.val),
            output_file=textbox_output.text,
        )
        sliderx.reset()
        slidery.reset()
        sliderz.reset()
        slider_rotx.reset()
        slider_roty.reset()
        slider_rotz.reset()
        sliderzm.reset()
        slider_scale.reset()
        update(10)

    def update_textbox(event):
        nerf_rays.reset_yaml(
            reference_pose=textbox.text,
            output_file=textbox_output.text,
        )
        sliderx.reset()
        slidery.reset()
        sliderz.reset()
        slider_rotx.reset()
        slider_roty.reset()
        slider_rotz.reset()
        sliderzm.reset()
        slider_scale.reset()
        update(10)

    def compute_button(event):
        output = nerf_rays.extract(
            offset_trans=np.array([slidery.val, sliderx.val, sliderz.val]),
            offset_rot=np.array(
                [
                    slider_rotx.val,
                    slider_roty.val,
                    slider_rotz.val,
                ]
            ),
            offset_microlens=sliderzm.val,
            offset_scale=10 ** (slider_scale.val),
        )

        print(f"Writing to {output[0]}")
        write_sdf(*output)
        print("Done")

    def update_output(event):
        nerf_rays.update_output_file(
            output_file=textbox_output.text,
        )

    sliderz.on_changed(update)
    slidery.on_changed(update)
    sliderx.on_changed(update)
    slider_rotx.on_changed(update)
    slider_roty.on_changed(update)
    slider_rotz.on_changed(update)
    sliderzm.on_changed(update)
    slider_scale.on_changed(update)
    button.on_clicked(update_botton)
    textbox.on_submit(update_textbox)
    textbox_output.on_submit(update_output)
    buttonsdf.on_clicked(compute_button)

    plt.show()
