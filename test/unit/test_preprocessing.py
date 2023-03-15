"""This module contains unit tests for preprocessings. Note that this module contains test for demonstration and does not achieve a sufficient test coverage."""
import pytest
from text_classifier.machine_learning.preprocessing import Preprocessor

class TestPreprocessing:
    
    @pytest.mark.parametrize(
        "raw_input_string, expected_processed_string",
        [
            ("1String2WithNumbers3", " String WithNumbers "),
            ("@String_WithNumbers(Characters)..", " String WithNumbers Characters   "),
            ("StringWithoutNonalphabeticSymbols", "StringWithoutNonalphabeticSymbols"),
        ]
    )
    def test_removing_non_alphabetic_symbols(self, raw_input_string, expected_processed_string):
        """Test check the removing of non alphabetic symbols."""
        preprocessor = Preprocessor()
        processed_string = preprocessor.remove_non_alphabetic_symbols(raw_input_string)
        assert processed_string == expected_processed_string


    @pytest.mark.parametrize(
    "raw_input_string, expected_processed_string",
        [
            ("string_without_whitespaces", "string_without_whitespaces"),
            ("string with one whitespace", "string with one whitespace"),
            ("string  with  two  whitespaces", "string with two whitespaces"),
            ("string   with   three   whitespaces", "string with three whitespaces"),
        ]
    )
    def test_removing_multiple_spaces(self, raw_input_string, expected_processed_string):
        """Test check the removing of multiple white spaces."""
        preprocessor = Preprocessor()
        processed_string = preprocessor.remove_multiple_spaces(raw_input_string)
        assert processed_string == expected_processed_string
