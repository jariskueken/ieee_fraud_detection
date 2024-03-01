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

evaluate-base:
	python3 -W"ignore" main.py --traindata ../data/train_data_prepared.csv \
--verbose --evaluate -md ../models --dataset-description base-999-filler

evaluate-scaledown:
	python3 -W"ignore" main.py --traindata ../data/train_data_downscaled.csv \
--verbose --evaluate -md ../models --dataset-description base-999-filler-scaledown

evaluate-openfe:
	python3 -W"ignore" main.py --traindata ../data/train_data_openfe.csv \
--verbose --evaluate -md ../models --dataset-description openfe-features

evaluate-pca:
	python3 -W"ignore" main.py --traindata ../data/train_data_pca.csv \
--verbose --evaluate -md ../models --dataset-description pca_2_comp

hpo:
	python3 -W"ignore" main.py --traindata ../data/train_data_pca.csv \
--verbose -hpo -md ../models --dataset-description pca_2_comp

prediction:
	python3 -W"ignore" main.py --predict --traindata \
../data/train_data_resampled.csv --testdata ../data/test_data_prepared.csv -st \
../data/sample_submission.csv -sd ../submission --verbose -md ../models --dataset-description smote-resampled

ensemble:
	python3 -W"ignore" main.py --predict --ensemble --traindata \
../data/train_data_prepared.csv --testdata ../data/test_data_prepared.csv -st \
../data/sample_submission.csv -sd ../submission -md ../models --dataset-description base-999-filler

debug:
	python3 -W"ignore" main.py --ensemble --traindata \
../data/train_data_prepared.csv --testdata ../data/test_data_prepared.csv -st ../data/sample_submission.csv -sd ../submission -n 5 --verbose --debug
