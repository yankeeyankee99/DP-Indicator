import inspect
import os
import tempfile
import unittest

from dp_indicator.agents.core_agents import ReasonerAgent, RankerAgent
from dp_indicator.core.orchestrator import Orchestrator


class PublicApiTests(unittest.TestCase):
    def test_runtime_api_has_no_unused_config_parameter(self):
        original_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                os.chdir(temp_dir)
                orchestrator = Orchestrator(api_key="test-key")
            finally:
                os.chdir(original_cwd)

        self.assertIsNotNone(orchestrator.model_router)
        self.assertNotIn(
            "config",
            inspect.signature(ReasonerAgent.__init__).parameters,
        )
        self.assertNotIn(
            "config",
            inspect.signature(RankerAgent.__init__).parameters,
        )


if __name__ == "__main__":
    unittest.main()
