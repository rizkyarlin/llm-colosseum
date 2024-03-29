run:
	DIAMBRA_ENVS=localhost:50051 python ./script.py

demo:
	DIAMBRA_ENVS=localhost:50051 python ./simple.py

install:
	pip3 install -r requirements.txt

go:
	while true; do make run; done