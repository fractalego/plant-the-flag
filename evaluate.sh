python3 sample_io_with_name.py data/val.json 100 5 1 12345 input > input.json
python3 sample_io_with_name.py data/val.json 100 5 1 12345 output > output.json
python3 test_codalab.py input.json > predict.json
python3 evaluate.py predict.json output.json
