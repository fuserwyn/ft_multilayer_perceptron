# Multilayer Perceptron
# Targets rebuild only what is out of date: changing the dataset re-splits,
# changing the split or the model code retrains.

PYTHON  ?= python3
DATASET ?= data.csv
SEED    ?= 42
ARGS    ?=

TRAIN_CSV = data_train.csv
VALID_CSV = data_valid.csv
MODEL     = saved_model.npz

.PHONY: all install explore split train predict test clean fclean re help

all: predict

install:
	$(PYTHON) -m pip install numpy pandas matplotlib

explore:
	$(PYTHON) prepare_data.py

$(TRAIN_CSV): $(DATASET) split_data.py mlp.py
	$(PYTHON) split_data.py --dataset $(DATASET) --seed $(SEED)

$(VALID_CSV): $(TRAIN_CSV)

split: $(TRAIN_CSV)

$(MODEL): $(TRAIN_CSV) $(VALID_CSV) train_mlp.py mlp.py
	$(PYTHON) train_mlp.py --train $(TRAIN_CSV) --valid $(VALID_CSV) --seed $(SEED) --model $(MODEL) $(ARGS)

# `make train` always retrains (handy to try other hyper-parameters through ARGS)
train: $(TRAIN_CSV)
	$(PYTHON) train_mlp.py --train $(TRAIN_CSV) --valid $(VALID_CSV) --seed $(SEED) --model $(MODEL) $(ARGS)

predict: $(MODEL)
	$(PYTHON) predict.py --model $(MODEL) --dataset $(VALID_CSV)

test:
	$(PYTHON) test_mlp.py

clean:
	rm -rf __pycache__ learning_curves.png

fclean: clean
	rm -f $(TRAIN_CSV) $(VALID_CSV) $(MODEL)

re: fclean all

help:
	@echo "make install   install numpy, pandas, matplotlib"
	@echo "make explore   print a quick analysis of the dataset"
	@echo "make split     data.csv -> data_train.csv + data_valid.csv"
	@echo "make train     train and save saved_model.npz   (ARGS='--layer 24 24 24 --epochs 84')"
	@echo "make predict   evaluate the model on data_valid.csv"
	@echo "make test      unit tests + gradient check"
	@echo "make fclean    remove every generated file"
