# Ships the tool's C++/CUDA sources inside the package (syncsan/csrc), where `syncsan build` compiles them.
import pathlib
import shutil

from setuptools import setup
from setuptools.command.build_py import build_py

HERE = pathlib.Path(__file__).resolve().parent
SOURCES = ["CMakeLists.txt", "syncsan.cu", "inject_funcs.cu", "common.h"]


class with_sources(build_py):
    def run(self):
        super().run()
        out = pathlib.Path(self.build_lib) / "syncsan" / "csrc"
        out.mkdir(parents=True, exist_ok=True)
        for f in SOURCES:
            shutil.copy(HERE / "src" / f, out / f)


setup(cmdclass={"build_py": with_sources})
