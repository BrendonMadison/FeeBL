"""Boundary and failure checks for the portable CLIC regression comparator."""
from decimal import Decimal
import unittest

from compare_numeric import compare_text, field_equal


class NumericComparisonTests(unittest.TestCase):
    def test_relative_boundary_and_symmetry(self):
        for left, right in [("1", "1.000001"), ("-1", "-1.000001")]:
            self.assertTrue(field_equal(left, right, Decimal("1e-6")))
            self.assertTrue(field_equal(right, left, Decimal("1e-6")))
        self.assertFalse(field_equal("1", "1.000001000001", Decimal("1e-6")))

    def test_minimum_denominator_is_not_relaxed_to_maximum(self):
        self.assertFalse(field_equal("1", "1.10001", Decimal(".1")))
        self.assertFalse(field_equal("1.10001", "1", Decimal(".1")))

    def test_zero_has_no_absolute_tolerance(self):
        self.assertTrue(field_equal("0", "-0.000e12", Decimal("1e-6")))
        self.assertFalse(field_equal("0", "1e-100", Decimal("1e-6")))

    def test_decimal_formats(self):
        self.assertTrue(field_equal("+1,000.000", "1e3", Decimal(0)))
        self.assertTrue(field_equal(".5", "5.0e-1", Decimal(0)))
        self.assertFalse(field_equal("1,00", "100", Decimal("1e-6")))

    def test_nonnumeric_and_embedded_assignments_are_literal(self):
        self.assertFalse(field_equal("de1=1;", "de1=1.0000001;", Decimal("1e-6")))
        self.assertFalse(field_equal("beam1", "beam2", Decimal(1)))

    def test_nonfinite_results_never_pass(self):
        for value in ("NaN", "nan(payload)", "inf", "-Infinity"):
            self.assertFalse(field_equal(value, value, Decimal("1e-6")))

    def test_line_count_and_field_count_are_checked(self):
        self.assertTrue(compare_text("a\n", "a\n\n", Decimal(0)))
        self.assertTrue(compare_text("a 1\n", "a\n", Decimal(0)))

    def test_horizontal_whitespace_only_is_ignored(self):
        self.assertEqual(compare_text(" a\t1 \n", "a 1\n", Decimal(0)), [])
        self.assertTrue(compare_text("a\nb\n", "b\na\n", Decimal(0)))

    def test_banner_and_new_settings_are_not_filtered(self):
        self.assertTrue(compare_text("version 1\nx = 0\n", "version 2\nx = 0\n", Decimal("1e-6")))
        self.assertTrue(compare_text("x = 0\nnew_flag = 0\n", "x = 0\n", Decimal("1e-6")))


if __name__ == "__main__":
    unittest.main()
