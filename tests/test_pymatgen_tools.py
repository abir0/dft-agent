"""
Tests for Pymatgen Tools - Materials Project search with alias fields.

Tests the field alias normalization functionality in search_materials_project.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Import the underlying function directly to test it without LangChain wrapper
import backend.agents.dft_tools.pymatgen_tools as pm_tools


class MockMPDoc:
    """Mock Materials Project document for testing."""

    def __init__(self, material_id: str, formula_pretty: str, **kwargs):
        self.material_id = material_id
        self.formula_pretty = formula_pretty

        # Add optional attributes
        for key, value in kwargs.items():
            setattr(self, key, value)


class MockSymmetry:
    """Mock symmetry object."""

    def __init__(self, symbol: str = "P1", number: int = 1):
        self.symbol = symbol
        self.number = number


class TestMaterialsProjectAliasFields:
    """Test Materials Project search with alias fields."""

    def setup_method(self):
        """Set up test fixtures."""
        self.test_api_key = "test_api_key_12345"
        self.test_formula = "TiO2"

        # Mock documents with various properties
        self.mock_docs = [
            MockMPDoc(
                material_id="mp-1234",
                formula_pretty="TiO2",
                formation_energy_per_atom=-5.5,
                band_gap=3.2,
                density=4.23,
                energy_above_hull=0.0,
                symmetry=MockSymmetry("P42/mnm", 136)
            ),
            MockMPDoc(
                material_id="mp-5678",
                formula_pretty="TiO2",
                formation_energy_per_atom=-5.3,
                band_gap=0.0,
                density=4.15,
                energy_above_hull=0.05,
                symmetry=MockSymmetry("Pbnm", 62)
            )
        ]

    @patch('backend.agents.dft_tools.pymatgen_tools.MPRester')
    def test_pretty_formula_alias(self, mock_mprester):
        """Test that pretty_formula alias is correctly mapped to formula_pretty."""
        # Setup mock
        mock_mpr = MagicMock()
        mock_mpr.materials.summary.search.return_value = self.mock_docs
        mock_mprester.return_value.__enter__.return_value = mock_mpr

        with tempfile.TemporaryDirectory() as temp_dir:
            os.chdir(temp_dir)

            # Test with pretty_formula alias - call the function directly
            result = pm_tools.search_materials_project.func(
                formula=self.test_formula,
                properties=["material_id", "pretty_formula"],
                api_key=self.test_api_key
            )

            # Verify the search was called with normalized field
            mock_mpr.materials.summary.search.assert_called_once_with(
                formula=self.test_formula,
                fields=sorted(["material_id", "formula_pretty"])
            )

            # Check result contains expected materials
            assert "mp-1234" in result
            assert "TiO2" in result

    @patch('backend.agents.dft_tools.pymatgen_tools.MPRester')
    def test_energy_above_hull_aliases(self, mock_mprester):
        """Test e_above_hull and eAboveHull aliases map to energy_above_hull."""
        mock_mpr = MagicMock()
        mock_mpr.materials.summary.search.return_value = self.mock_docs
        mock_mprester.return_value.__enter__.return_value = mock_mpr

        with tempfile.TemporaryDirectory() as temp_dir:
            os.chdir(temp_dir)

            # Test both alias variations
            for alias in ["e_above_hull", "eAboveHull"]:
                mock_mpr.materials.summary.search.reset_mock()

                _ = pm_tools.search_materials_project.func(
                    formula=self.test_formula,
                    properties=["material_id", alias],
                    api_key=self.test_api_key
                )

                # Verify normalized field is used
                mock_mpr.materials.summary.search.assert_called_once_with(
                    formula=self.test_formula,
                    fields=sorted(["material_id", "energy_above_hull"])
                )

    @patch('backend.agents.dft_tools.pymatgen_tools.MPRester')
    def test_spacegroup_aliases(self, mock_mprester):
        """Test various spacegroup aliases all map to symmetry."""
        mock_mpr = MagicMock()
        mock_mpr.materials.summary.search.return_value = self.mock_docs
        mock_mprester.return_value.__enter__.return_value = mock_mpr

        with tempfile.TemporaryDirectory() as temp_dir:
            os.chdir(temp_dir)

            spacegroup_aliases = [
                "spacegroup", "space_group", "spacegroup_symbol",
                "spacegroup_number", "crystal_system"
            ]

            for alias in spacegroup_aliases:
                mock_mpr.materials.summary.search.reset_mock()

                _ = pm_tools.search_materials_project.func(
                    formula=self.test_formula,
                    properties=["material_id", alias],
                    api_key=self.test_api_key
                )

                # All should map to symmetry field
                mock_mpr.materials.summary.search.assert_called_once_with(
                    formula=self.test_formula,
                    fields=sorted(["material_id", "symmetry"])
                )

    @patch('backend.agents.dft_tools.pymatgen_tools.MPRester')
    def test_multiple_aliases_combined(self, mock_mprester):
        """Test multiple aliases in one query are correctly normalized."""
        mock_mpr = MagicMock()
        mock_mpr.materials.summary.search.return_value = self.mock_docs
        mock_mprester.return_value.__enter__.return_value = mock_mpr

        with tempfile.TemporaryDirectory() as temp_dir:
            os.chdir(temp_dir)

            # Use multiple aliases
            properties = [
                "material_id",
                "pretty_formula",  # -> formula_pretty
                "e_above_hull",    # -> energy_above_hull
                "spacegroup",      # -> symmetry
                "crystal_system",  # -> symmetry (duplicate)
                "band_gap"         # no alias needed
            ]

            _ = pm_tools.search_materials_project.func(
                formula=self.test_formula,
                properties=properties,
                api_key=self.test_api_key
            )

            # Should normalize and deduplicate
            expected_fields = sorted([
                "material_id", "formula_pretty", "energy_above_hull",
                "symmetry", "band_gap"
            ])

            mock_mpr.materials.summary.search.assert_called_once_with(
                formula=self.test_formula,
                fields=expected_fields
            )

    @patch('backend.agents.dft_tools.pymatgen_tools.MPRester')
    def test_symmetry_alias_ensures_symmetry_field(self, mock_mprester):
        """Test that any symmetry alias ensures symmetry field is requested."""
        mock_mpr = MagicMock()
        mock_mpr.materials.summary.search.return_value = self.mock_docs
        mock_mprester.return_value.__enter__.return_value = mock_mpr

        with tempfile.TemporaryDirectory() as temp_dir:
            os.chdir(temp_dir)

            # Request only spacegroup alias
            _ = pm_tools.search_materials_project.func(
                formula=self.test_formula,
                properties=["material_id", "spacegroup"],
                api_key=self.test_api_key
            )

            # Should include symmetry field
            expected_fields = sorted(["material_id", "symmetry"])
            mock_mpr.materials.summary.search.assert_called_once_with(
                formula=self.test_formula,
                fields=expected_fields
            )

    @patch('backend.agents.dft_tools.pymatgen_tools.MPRester')
    def test_non_alias_fields_unchanged(self, mock_mprester):
        """Test that non-alias fields are passed through unchanged."""
        mock_mpr = MagicMock()
        mock_mpr.materials.summary.search.return_value = self.mock_docs
        mock_mprester.return_value.__enter__.return_value = mock_mpr

        with tempfile.TemporaryDirectory() as temp_dir:
            os.chdir(temp_dir)

            # Use fields that don't have aliases
            properties = ["material_id", "band_gap", "density", "formation_energy_per_atom"]

            _ = pm_tools.search_materials_project.func(
                formula=self.test_formula,
                properties=properties,
                api_key=self.test_api_key
            )

            # Should pass through unchanged
            mock_mpr.materials.summary.search.assert_called_once_with(
                formula=self.test_formula,
                fields=sorted(properties)
            )

    @patch('backend.agents.dft_tools.pymatgen_tools.MPRester')
    def test_result_contains_symmetry_info(self, mock_mprester):
        """Test that results properly extract symmetry information."""
        mock_mpr = MagicMock()
        mock_mpr.materials.summary.search.return_value = self.mock_docs
        mock_mprester.return_value.__enter__.return_value = mock_mpr

        with tempfile.TemporaryDirectory() as temp_dir:
            os.chdir(temp_dir)

            pm_tools.search_materials_project.func(
                formula=self.test_formula,
                properties=["material_id", "spacegroup", "spacegroup_number"],
                api_key=self.test_api_key
            )

            # Check that results file contains symmetry info
            results_dir = Path("materials_project_data")
            results_file = results_dir / "search_TiO2_results.json"

            assert results_file.exists()

            with open(results_file) as f:
                results_data = json.load(f)

            # Should contain spacegroup info for each material
            assert len(results_data) == 2
            assert "spacegroup" in results_data[0]
            assert "spacegroup_number" in results_data[0]
            assert results_data[0]["spacegroup"] == "P42/mnm"
            assert results_data[0]["spacegroup_number"] == 136

    @patch('backend.agents.dft_tools.pymatgen_tools.MPRester')
    def test_default_properties_work(self, mock_mprester):
        """Test that default properties work without aliases."""
        mock_mpr = MagicMock()
        mock_mpr.materials.summary.search.return_value = self.mock_docs
        mock_mprester.return_value.__enter__.return_value = mock_mpr

        with tempfile.TemporaryDirectory() as temp_dir:
            os.chdir(temp_dir)

            # Use default properties (no aliases)
            _ = pm_tools.search_materials_project.func(
                formula=self.test_formula,
                api_key=self.test_api_key
            )

            expected_fields = sorted([
                "material_id", "formula_pretty", "structure",
                "formation_energy_per_atom", "band_gap", "density"
            ])

            mock_mpr.materials.summary.search.assert_called_once_with(
                formula=self.test_formula,
                fields=expected_fields
            )

    @patch('backend.agents.dft_tools.pymatgen_tools.MPRester')
    def test_empty_results_handling(self, mock_mprester):
        """Test proper handling when no materials are found."""
        mock_mpr = MagicMock()
        mock_mpr.materials.summary.search.return_value = []  # Empty results
        mock_mprester.return_value.__enter__.return_value = mock_mpr

        with tempfile.TemporaryDirectory() as temp_dir:
            os.chdir(temp_dir)

            result = pm_tools.search_materials_project.func(
                formula="NonExistentCompound",
                properties=["material_id", "pretty_formula"],
                api_key=self.test_api_key
            )

            assert "No materials found" in result
            assert "NonExistentCompound" in result

    @patch('backend.agents.dft_tools.pymatgen_tools.MPRester')
    def test_field_sorting_consistency(self, mock_mprester):
        """Test that fields are consistently sorted for API calls."""
        mock_mpr = MagicMock()
        mock_mpr.materials.summary.search.return_value = self.mock_docs
        mock_mprester.return_value.__enter__.return_value = mock_mpr

        with tempfile.TemporaryDirectory() as temp_dir:
            os.chdir(temp_dir)

            # Provide fields in different order
            properties1 = ["pretty_formula", "material_id", "e_above_hull"]
            properties2 = ["e_above_hull", "material_id", "pretty_formula"]

            for props in [properties1, properties2]:
                mock_mpr.materials.summary.search.reset_mock()

                _ = pm_tools.search_materials_project.func(
                    formula=self.test_formula,
                    properties=props,
                    api_key=self.test_api_key
                )

                # Should always use sorted fields
                expected_fields = sorted(["formula_pretty", "material_id", "energy_above_hull"])
                mock_mpr.materials.summary.search.assert_called_once_with(
                    formula=self.test_formula,
                    fields=expected_fields
                )


if __name__ == "__main__":
    pytest.main([__file__])

