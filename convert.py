import gmsh
import sys
import os


def step_to_geo(step_path, geo_path):
    """
    Convert a STEP file to a Gmsh .geo file using the Gmsh Python API.

    :param step_path: Path to the input STEP file (.step or .stp)
    :param geo_path: Path to the output GEO file (.geo)
    """
    # Validate input file
    if not os.path.isfile(step_path):
        raise FileNotFoundError(f"STEP file not found: {step_path}")
    if not step_path.lower().endswith((".step", ".stp")):
        raise ValueError("Input file must have .step or .stp extension")

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)  # Enable console messages

    try:
        # Import STEP geometry
        print("Import STEP geometry...")
        gmsh.model.add("converted_model")
        gmsh.model.occ.importShapes(step_path)
        gmsh.model.occ.synchronize()

        # Save as GEO script
        print("Save as GEO script...")
        gmsh.write(geo_path)
        print(f"✅ Conversion successful: {geo_path}")

    except Exception as e:
        print(f"❌ Error during conversion: {e}")
    finally:
        gmsh.finalize()


if __name__ == "__main__":
    # if len(sys.argv) != 3:
    #     print("Usage: python step_to_geo.py input.step output.geo")
    #     sys.exit(1)

    # step_file = sys.argv[1]
    # geo_file = sys.argv[2]

    step_file = "..\\box_sub_STEP.STEP"
    geo_file = "model_py.geo"

    try:
        step_to_geo(step_file, geo_file)
    except Exception as err:
        print(f"Error: {err}")
        sys.exit(1)
