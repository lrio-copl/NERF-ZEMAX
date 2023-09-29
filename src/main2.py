from extract_nerf_rays import NerfRays
from write_sdf import write_sdf

if __name__ == "__main__":
    nerf_rays = NerfRays("data/config_test.yml")
    write_sdf(*nerf_rays.extract())
