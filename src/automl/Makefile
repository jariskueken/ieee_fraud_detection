all: compile checkstyle

compile:
	python3 -m py_compile *.py

checkstyle:
	flake8 *.py
	mypy --allow-redefinition --disallow-untyped-defs *.py

clean:
	rm -f *.pyc
	rm -rf .mypy_cache
	find . -type d -name __pycache__ -exec rm -r {} \+

test:
	python3 -m doctest *.py

run:
	python3 automl.py --traindata ../../data/train_data_prepared.csv --testdata ../../data/test_data_prepared.csv \
	-st ../../data/sample_submission.csv -sd ../../submission --datasetidentifier base-999-filler --timelimit 1200.0