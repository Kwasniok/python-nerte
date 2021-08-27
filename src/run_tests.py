import unittest

if __name__ == "__main__":
    verbosity = 1
    # find all tests and execute them
    loader = unittest.defaultTestLoader
    testSuite = loader.discover(start_dir=".", pattern="*_unittest.py")
    testRunner = unittest.runner.TextTestRunner(verbosity=verbosity)
    testRunner.run(testSuite)
