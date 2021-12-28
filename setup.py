from setuptools import setup

def find_version() -> str:
    return "0.0.1"


if __name__ == "__main__":
    try:
        setup(
            version=find_version(),
        )
    except:  # noqa
        print(
            "\n\nAn error occurred while building the project, "
            "please ensure you have the most updated version of setuptools, "
            "setuptools_scm and wheel with:\n"
            "   pip install -U setuptools setuptools_scm wheel\n\n"
        )
        raise
