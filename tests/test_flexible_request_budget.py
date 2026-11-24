import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.flexible_execution.request_budget import RequestBudget
from experiments.task_attack.model import ModelAttemptLimitExceeded, ModelClient, ModelDeadlineExceeded
from uav_bridge.responses_transport import TransportError


class LocalStore:
    def is_model_available(self, model):
        return True

    def get_access_token(self):
        return 'local-test-token'


class RequestBudgetTest(unittest.TestCase):
    def test_retry_and_restart_share_actual_dispatch_cap(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger = Path(directory) / 'requests.jsonl'
            with patch('uav_bridge.responses_transport._perform',
                       side_effect=TransportError('network_error', 'fixture')) as dispatch:
                with patch('experiments.task_attack.model.time.sleep'):
                    model = ModelClient(credential_store=LocalStore(), request_budget=RequestBudget(ledger, 3))
                    with self.assertRaises(TransportError):
                        model.request('Read', {'task': 'read gauge'}, attempt_limit=9)
                    self.assertEqual(dispatch.call_count, 2)
                    restarted = ModelClient(credential_store=LocalStore(), request_budget=RequestBudget(ledger, 3))
                    with self.assertRaises(ModelAttemptLimitExceeded):
                        restarted.request('Read', {'task': 'read gauge'})
                    self.assertEqual(dispatch.call_count, 3)
                    self.assertEqual(restarted.request_budget.summary()['remaining'], 0)

    def test_rejected_input_and_elapsed_deadline_do_not_dispatch(self):
        with tempfile.TemporaryDirectory() as directory:
            budget = RequestBudget(Path(directory) / 'requests.jsonl', 6)
            model = ModelClient(credential_store=LocalStore(), request_budget=budget)
            with patch('uav_bridge.responses_transport._perform') as dispatch:
                with self.assertRaises(TransportError):
                    model.request('Read', {}, [{'label': 'invalid photo', 'data': b'invalid'}])
                model.deadline_monotonic = time.monotonic() - 1
                with self.assertRaises(ModelDeadlineExceeded):
                    model.request('Read', {'task': 'read gauge'})
                dispatch.assert_not_called()
            self.assertEqual(budget.summary()['reserved'], 0)
