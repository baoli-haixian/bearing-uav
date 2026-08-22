import unittest
from unittest.mock import patch

import numpy as np

from cvphr.models.posaglreg.models import RSBlockDatasetPA_v3q


class DatasetReadRetryTest(unittest.TestCase):
    def test_transient_failures_are_retried_without_skipping(self):
        bgr = np.zeros((4, 5, 3), dtype=np.uint8)
        with patch(
            'cvphr.models.posaglreg.models.cv2.imread',
            side_effect=[None, None, bgr],
        ) as imread, patch('cvphr.models.posaglreg.models.time.sleep') as sleep:
            image = RSBlockDatasetPA_v3q.load_cvimg_to_rgb_pil('sample.jpg')

        self.assertEqual(imread.call_count, 3)
        self.assertEqual(sleep.call_count, 2)
        self.assertEqual(image.mode, 'RGB')
        self.assertEqual(image.size, (5, 4))

    def test_persistent_failure_raises_after_all_attempts(self):
        with patch(
            'cvphr.models.posaglreg.models.cv2.imread', return_value=None
        ) as imread, patch('cvphr.models.posaglreg.models.time.sleep'):
            with self.assertRaisesRegex(FileNotFoundError, 'after 8 attempts'):
                RSBlockDatasetPA_v3q.load_cvimg_to_rgb_pil('sample.jpg')

        self.assertEqual(imread.call_count, 8)


if __name__ == '__main__':
    unittest.main()
