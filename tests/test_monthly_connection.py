"""Connection precedence tests without actual network connections."""

import os
import unittest
from unittest.mock import patch

from database.monthly_connection import get_monthly_mongo_uri


class MonthlyConnectionTests(unittest.TestCase):
    """Monthly workflows should reuse existing connection configuration."""

    def test_connection_precedence(self):
        """Explicit monthly setting wins; existing app URI beats localhost."""
        cases = [
            ({}, "mongodb://localhost:27017"),
            ({"MONGO_URL": "mongodb://app-host"}, "mongodb://app-host"),
            ({"MONGO_URL": "mongodb://app-host", "MONTHLY_MONGO_URL": "mongodb://monthly-host"}, "mongodb://monthly-host"),
            ({"MONGO_URL": "mongodb://app-host", "MONTHLY_MONGO_URL": ""}, "mongodb://app-host"),
        ]
        for environment, expected in cases:
            with self.subTest(environment=environment), patch.dict(os.environ, environment, clear=True), patch("database.monthly_connection.load_dotenv"):
                self.assertEqual(get_monthly_mongo_uri(), expected)
