"""CPU argument checks for the consecutive AMP training smoke script."""

import unittest

from scripts.smoke_amp_training import run_amp_training_smoke


class TestAmpTrainingSmokeArguments(unittest.TestCase):
    def test_invalid_environment_count_fails_before_construction(self) -> None:
        with self.assertRaises(ValueError):
            run_amp_training_smoke(num_envs=0)

    def test_invalid_update_count_fails_before_construction(self) -> None:
        with self.assertRaises(ValueError):
            run_amp_training_smoke(updates=0)


if __name__ == "__main__":
    unittest.main()
