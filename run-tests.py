"""
run-tests.py
Master test runner for the IBM AI Enterprise Workflow Capstone.
Executes all unit tests (API, Model, Logger) via a single script and verifies
that 100% of test cases pass with full test isolation.
"""

import unittest
import sys
import os
import time

# Ensure project root is in Python path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from unittests.ApiTests import ApiTests
from unittests.ModelTests import ModelTests
from unittests.LoggerTests import LoggerTests

def run_all_tests():
    """
    Assembles and executes all unit test suites.
    """
    print("=" * 75)
    print("   AI ENTERPRISE WORKFLOW CAPSTONE - MASTER TEST SUITE RUNNER")
    print("=" * 75)
    print(f"[*] Discovering test cases from 'unittests/'...")

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(LoggerTests))
    suite.addTests(loader.loadTestsFromTestCase(ModelTests))
    suite.addTests(loader.loadTestsFromTestCase(ApiTests))

    test_count = suite.countTestCases()
    print(f"[+] Loaded {test_count} unit tests across LoggerTests, ModelTests, and ApiTests.")
    print("-" * 75)

    start_time = time.time()
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    duration = time.time() - start_time

    print("=" * 75)
    print(f"Test Run Completed in {duration:.2f} seconds")
    print(f"Total Tests Run: {result.testsRun}")
    print(f"Passed: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("=" * 75)

    if result.wasSuccessful():
        print("[SUCCESS] ALL UNIT TESTS PASSED WITH 100% SUCCESS RATE.")
        return 0
    else:
        print("[FAILURE] ONE OR MORE UNIT TESTS FAILED.")
        return 1

if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
