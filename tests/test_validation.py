# -*- coding: utf-8 -*-
"""
Unit tests for declarative Pydantic v2 configuration validation.
Tests strict type checking, range validation, scalar-to-vector expansion,
forbidden extra fields, and diagnostic error formatting.
"""

import copy
import json
import os
import tempfile
import unittest

from pydantic import ValidationError

from dust_impact.sim3d.schema import SimulationConfigSchema, format_validation_error
from dust_impact.sim3d.config_loader import load_and_validate_config, setup_simulation_parameters_3d


class TestConfigValidation(unittest.TestCase):
    def setUp(self):
        with open("config.json", "r", encoding="utf-8") as f:
            self.base_config = json.load(f)

    def test_default_config_validates(self):
        """Standard config.json must validate cleanly without error."""
        config_schema, base_dir = load_and_validate_config("config.json")
        self.assertIsInstance(config_schema, SimulationConfigSchema)
        self.assertEqual(config_schema.geometry.source, "spis")
        self.assertTrue(os.path.isdir(base_dir))

    def test_all_sample_configs_validate(self):
        """All existing sample configs must validate cleanly."""
        configs_to_test = [
            "config.json",
            os.path.join("inputs", "config_template.json"),
            os.path.join("inputs", "config_analytical_sphere.json"),
            os.path.join("inputs", "config_analytical_box.json"),
        ]
        for cfg_path in configs_to_test:
            with self.subTest(config=cfg_path):
                self.assertTrue(os.path.exists(cfg_path), f"File {cfg_path} does not exist.")
                schema, base_dir = load_and_validate_config(cfg_path)
                self.assertIsInstance(schema, SimulationConfigSchema)

    def test_forbidden_extra_fields(self):
        """Unknown keys (e.g. typos in toggles or physics) must fail validation immediately."""
        cfg = copy.deepcopy(self.base_config)
        cfg["toggles"]["enabel_collisions"] = True  # typo for enable_collisions

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as tmp:
            json.dump(cfg, tmp)
            tmp_path = tmp.name

        try:
            with self.assertRaises(ValueError) as ctx:
                load_and_validate_config(tmp_path)
            err_msg = str(ctx.exception)
            self.assertIn("enabel_collisions", err_msg)
            self.assertIn("Extra inputs are not permitted", err_msg)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_negative_ion_mass_rejected(self):
        """Physical parameters requiring > 0 must fail if non-positive."""
        cfg = copy.deepcopy(self.base_config)
        cfg["physics"]["ion_mass_amu"] = -1.0

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as tmp:
            json.dump(cfg, tmp)
            tmp_path = tmp.name

        try:
            with self.assertRaises(ValueError) as ctx:
                load_and_validate_config(tmp_path)
            err_msg = str(ctx.exception)
            self.assertIn("ion_mass_amu", err_msg)
            self.assertIn("greater than 0", err_msg)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_invalid_vector_dimension_rejected(self):
        """3D vectors must strictly have 3 elements."""
        cfg = copy.deepcopy(self.base_config)
        cfg["numeric"]["grid_nodes"] = [100, 100]  # Only 2 elements

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as tmp:
            json.dump(cfg, tmp)
            tmp_path = tmp.name

        try:
            with self.assertRaises(ValueError) as ctx:
                load_and_validate_config(tmp_path)
            self.assertIn("grid_nodes", str(ctx.exception))
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_grid_nodes_minimum_two(self):
        """Grid nodes in any axis must be at least 2."""
        cfg = copy.deepcopy(self.base_config)
        cfg["numeric"]["grid_nodes"] = [1, 50, 50]

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as tmp:
            json.dump(cfg, tmp)
            tmp_path = tmp.name

        try:
            with self.assertRaises(ValueError) as ctx:
                load_and_validate_config(tmp_path)
            self.assertIn("grid_nodes", str(ctx.exception))
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_weighting_threshold_range(self):
        """weighting_threshold must be strictly within (0.0, 1.0]."""
        cfg = copy.deepcopy(self.base_config)
        cfg["geometry"]["spis"]["weighting_threshold"] = 0.0

        with self.assertRaises(ValidationError):
            SimulationConfigSchema.model_validate(cfg)

        cfg["geometry"]["spis"]["weighting_threshold"] = 1.2
        with self.assertRaises(ValidationError):
            SimulationConfigSchema.model_validate(cfg)

        cfg["geometry"]["spis"]["weighting_threshold"] = 0.5
        valid_schema = SimulationConfigSchema.model_validate(cfg)
        self.assertEqual(valid_schema.geometry.spis.weighting_threshold, 0.5)

    def test_scalar_expansion_to_3d_vector(self):
        """Scalars specified for grid_nodes or domain_half_length_m expand to [val, val, val]."""
        cfg = copy.deepcopy(self.base_config)
        cfg["numeric"]["grid_nodes"] = 64
        cfg["numeric"]["domain_half_length_m"] = 8.5

        schema = SimulationConfigSchema.model_validate(cfg)
        self.assertEqual(schema.numeric.grid_nodes, [64, 64, 64])
        self.assertEqual(schema.numeric.domain_half_length_m, [8.5, 8.5, 8.5])

    def test_invalid_geometry_source_rejected(self):
        """Only 'spis' and 'analytical' geometry sources are permitted."""
        cfg = copy.deepcopy(self.base_config)
        cfg["geometry"]["source"] = "cad_step"

        with self.assertRaises(ValidationError):
            SimulationConfigSchema.model_validate(cfg)

    def test_antenna_identical_endpoints_rejected(self):
        """Analytical antennas cannot have zero length (p_start == p_end)."""
        analytical_path = os.path.join("inputs", "config_analytical_sphere.json")
        with open(analytical_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        cfg["geometry"]["analytical"]["antennas"][0]["p_start"] = [1.0, 0.0, 0.0]
        cfg["geometry"]["analytical"]["antennas"][0]["p_end"] = [1.0, 0.0, 0.0]

        with self.assertRaises(ValidationError) as ctx:
            SimulationConfigSchema.model_validate(cfg)
        self.assertIn("cannot be identical", str(ctx.exception))

    def test_human_readable_diagnostic_error_formatter(self):
        """format_validation_error must clearly format errors with key path and reason."""
        invalid_data = {
            "geometry": {"source": "invalid_mode"},
            "toggles": {"unknown_flag": True},
        }
        try:
            SimulationConfigSchema.model_validate(invalid_data)
        except ValidationError as e:
            msg = format_validation_error(e)
            self.assertIn("[CHYBA KONFIGURACE]", msg)
            self.assertIn("geometry -> source", msg)
            self.assertIn("toggles -> unknown_flag", msg)

    def test_setup_simulation_parameters_integration(self):
        """setup_simulation_parameters_3d should return expected dataclasses when loading config."""
        params, toggles, plot_cfg = setup_simulation_parameters_3d(
            Vf=-25.0, Vf_antenne=10.0, config_file="config.json"
        )
        self.assertEqual(params.grid_nodes_x, 150)
        self.assertEqual(params.grid_nodes_y, 150)
        self.assertEqual(params.grid_nodes_z, 150)
        self.assertTrue(toggles.enable_spis_background_field)


if __name__ == "__main__":
    unittest.main()
