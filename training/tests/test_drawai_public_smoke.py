"""A rolling deploy must wait for the new revision before sending AI requests."""
import unittest
from unittest.mock import patch
from tools.smoke_public_api import wait_ready


class ReadinessTests(unittest.TestCase):
    @patch('tools.smoke_public_api.time.sleep')
    @patch('tools.smoke_public_api.request')
    def test_old_ready_revision_is_not_accepted(self,request,sleep):
        request.side_effect=[dict(status='ready',code_revision='old'),dict(status='ready',code_revision='new')]
        self.assertEqual(wait_ready('https://test.invalid','new')['code_revision'],'new')
        self.assertEqual(request.call_count,2)

    @patch('tools.smoke_public_api.time.sleep')
    @patch('tools.smoke_public_api.request')
    def test_sleeping_host_retries_health(self,request,sleep):
        request.side_effect=[OSError('sleep'),dict(status='starting'),dict(status='ready',code_revision='new')]
        self.assertEqual(wait_ready('https://test.invalid','new')['status'],'ready')
        self.assertEqual(request.call_count,3)

    @patch('tools.smoke_public_api.time.perf_counter',side_effect=[0,2])
    @patch('tools.smoke_public_api.request')
    def test_bounded_deadline(self,request,clock):
        with self.assertRaisesRegex(RuntimeError,'timed out'):wait_ready('https://test.invalid','new',timeout=1)
        request.assert_not_called()
