# install test dependencies

```bash
conda create -n test_scformer python=3.7 rpy2 -y
conda activate test_scformer
pip install -r examples/test_requirements.txt
```

# run test script

```bash
cd examples
python3 test.py --data-source pbmc_dataset --model-dir path/to/modeldir --save-dir path/to/savedir
```
