from yaml import load, dump, CLoader
from pathlib import Path
import struct
from numpy import concatenate, repeat, tile

YAML_FILE_DEFAULT = Path(__file__).parent / "source_config.yml"

WAVELENGTHS = [0.656, 0.566, 0.471]


def write_sdf(output_file, rays, rgbaData, config=YAML_FILE_DEFAULT):
    yaml_file = load(open(config), Loader=CLoader)
    assert rays.shape[1] == 6
    assert rgbaData.shape[1] == 4
    assert rays.shape[0] == rgbaData.shape[0]
    with open(output_file, "wb") as file:
        file.write(struct.pack("i", yaml_file["format version"]))
        file.write(struct.pack("i", rays.shape[0] * 3))
        if len(yaml_file["description"]) < 100:
            file.write(
                struct.pack("100s", yaml_file["description"].ljust(100, "\0").encode())
            )
        else:
            file.write(struct.pack("100s", yaml_file["description"][:99].encode()))
        file.write(struct.pack("f", yaml_file["source flux"]))
        file.write(struct.pack("f", yaml_file["ray set flux"]))
        file.write(struct.pack("f", yaml_file["wavelength"]))
        file.write(struct.pack("2f", *yaml_file["inclination"]))
        file.write(struct.pack("2f", *yaml_file["azimuth"]))
        file.write(struct.pack("l", yaml_file["dimension units"]))
        file.write(struct.pack("3f", *yaml_file["loc"]))
        file.write(struct.pack("3f", *yaml_file["rot"]))
        file.write(struct.pack("3f", *yaml_file["scale"]))
        file.write(struct.pack("4f", *[0, 0, 0, 0]))
        file.write(struct.pack("2i", yaml_file["ray format"], yaml_file["flux type"]))
        file.write(struct.pack("2i", *[0, 0]))

        lines = concatenate(
            [
                repeat(rays, 3, axis=0),
                (
                    rgbaData[:, :3].reshape(-1) * repeat(rgbaData[:, -1], 3, axis=0)
                ).reshape(-1, 1),
                tile(WAVELENGTHS, rays.shape[0]).reshape(-1, 1),
            ],
            axis=1,
        )
        file.writelines((struct.pack("8f", *rayinfo) for rayinfo in lines))


if __name__ == "__main__":

    def main():
        import numpy as np

        write_sdf(
            "../../output/test.sdf",
            np.eye(7, 6),
            np.cumsum(np.cumsum(np.ones((7, 4)), axis=0), axis=1),
        )

    main()
