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

testset:
	python3 -W"ignore" main.py --traindata ../data/train.csv

prediction:
	python3 -W"ignore" main.py --predict --traindata ../data/train.csv --testdata ../data/test.csv -st ../data/sample_submission.csv -sd ../submission

ensemble:
	python3 -W"ignore" main.py --predict --ensemble --traindata ../data/train.csv --testdata ../data/test.csv -st ../data/sample_submission.csv -sd ../submission

debug:
	python3 -W"ignore" main.py --ensemble --traindata ../data/train.csv --testdata ../data/test.csv -st ../data/sample_submission.csv -sd ../submission -n 5 --verbose --debug
