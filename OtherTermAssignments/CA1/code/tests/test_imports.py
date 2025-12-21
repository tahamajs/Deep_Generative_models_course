import sys
import os

# Ensure local package path is available when running tests from repository root
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)


def test_imports():
    import ca1
    import ca1.config
    import ca1.data
    import ca1.models
    import ca1.train

    # simple sanity: VAE class exists
    assert hasattr(ca1.models, "VAE")


if __name__ == "__main__":
    test_imports()
    print("imports ok")
