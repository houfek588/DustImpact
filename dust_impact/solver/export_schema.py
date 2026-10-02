# -*- coding: utf-8 -*-
"""
Export JSON Schema for config.json from Pydantic models.
"""
import os
import json
from dust_impact.solver.schema import SimulationConfigSchema


def export_schema(target_path: str = "schema/config.schema.json") -> str:
    """Generate and write JSON Schema file."""
    os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)
    schema_dict = SimulationConfigSchema.model_json_schema()
    schema_dict["title"] = "DustImpact3DConfig"
    schema_dict["description"] = "JSON Schema for DustImpact 3D PIC Simulation Configuration"

    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(schema_dict, f, indent=4)
    print(f"[OK] JSON Schema exported to: {target_path}")
    return target_path


if __name__ == "__main__":
    export_schema()
