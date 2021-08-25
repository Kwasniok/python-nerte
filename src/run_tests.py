import unittest

if __name__ == "__main__":
    # find all tests and execute them
    loader = unittest.defaultTestLoader
    testSuite = loader.discover(start_dir=".", pattern="*_unittest.py")
    testRunnder = unittest.runner.TextTestRunner()
    testRunnder.run(testSuite)
