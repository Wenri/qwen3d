# Helper for the manual (conda + pip) setup described in README.md.
# With pixi this is not needed: `pixi install` compiles libs/pointops2 and the
# environment activation sets PYTHONPATH to the repository root.

# add root folder to python path
export PYTHONPATH="${PYTHONPATH}:${PWD}"

# compile custom operators
cd libs/pointops2
rm -rf build
python setup.py install
cd -
