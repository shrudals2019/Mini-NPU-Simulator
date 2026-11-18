"""표준 라이브러리 검증: python verify.py"""
import contextlib
import io
import json
import unittest
from unittest.mock import patch
import main


class SimulatorChecks(unittest.TestCase):
    def test_mac_and_flatten(self):
        for size in (3, 5, 13, 25):
            cross, x = main.generate_patterns(size)
            self.assertEqual(main.mac_2d(cross, cross), 2 * size - 1)
            self.assertEqual(main.mac_2d(cross, x), 1)
            self.assertEqual(main.mac_2d(cross, x), main.mac_1d(main.flatten(cross), main.flatten(x)))

    def test_labels_and_epsilon(self):
        for label in ("+", "cross", "Cross"):
            self.assertEqual(main.normalize_label(label), "Cross")
        self.assertEqual(main.normalize_label("x"), "X")
        self.assertEqual(main.decide(0.9, 0.8999999999999999), "UNDECIDED")
        self.assertEqual(main.decide(2e-9, 0), "Cross")

    def test_bad_matrices(self):
        for matrix in ([[1]], [[1, 2, 3]] * 2, [[True] * 3] * 3, [[float("nan")] * 3] * 3):
            with self.assertRaises(ValueError):
                main.validate_matrix(matrix, 3)

    def test_row_retry(self):
        with patch("builtins.input", side_effect=["1 2", "a b c", "0 1 0", "1 1 1", "0 1 0"]):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                result = main.read_matrix("test")
        self.assertEqual(len(result), 3)
        self.assertEqual(output.getvalue().count("입력 형식 오류"), 2)

    def test_official_results(self):
        with (main.ROOT / "data.json").open(encoding="utf-8-sig") as file:
            data = json.load(file)
        with contextlib.redirect_stdout(io.StringIO()):
            filters, errors = main.load_filters(data)
        self.assertFalse(errors)
        actual = {}
        for key, case in data["patterns"].items():
            size = int(key.split("_")[1])
            actual[key] = main.decide(main.mac_2d(case["input"], filters[size]["Cross"]),
                                      main.mac_2d(case["input"], filters[size]["X"]))
        self.assertEqual(actual, {"size_5_1": "UNDECIDED", "size_5_2": "Cross",
                                 "size_13_1": "X", "size_13_2": "UNDECIDED",
                                 "size_25_1": "UNDECIDED", "size_25_2": "Cross"})


if __name__ == "__main__":
    unittest.main()
