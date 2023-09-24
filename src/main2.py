from extract_nerf_rays import extract_nerf_rays, load_model
from write_sdf import write_sdf

if __name__ == "__main__":
    config, pipeline, _, step = load_model("data/config_test.yml")
    write_sdf(*extract_nerf_rays("data/config_test.yml", pipeline))
