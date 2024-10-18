export AWS_PROFILE=genaitour
run:
	DIAMBRA_ENVS=localhost:50051 python ./script.py

.PHONY: run_parallel spawn_containers

NUM_ENVIRONMENT := 1
BASE_PORT := 50051

define PORT_LIST
$(shell seq -f "$(BASE_PORT) + %g" 0 $$(($(NUM_ENVIRONMENT) - 1)) | bc)
endef

PORTS := $(shell echo $(call PORT_LIST))

spawn_containers:
	@echo "Running scripts in parallel..."
	@for port in $(PORTS); do \
		(docker run -d -v $$HOME/.diambra/credentials:/tmp/.diambra/credentials \
		-v /Users/$$USER/.diambra/roms:/opt/diambraArena/roms \
		-p $$port:50051 docker.io/diambra/engine:latest) & \
	done
	@wait

run_parallel:
	@echo "Running scripts in parallel..."
	@for port in $(PORTS); do \
		(DIAMBRA_ENVS=localhost:$$port python ./script.py) & \
	done
	@wait

clean:
	docker ps -a --filter "ancestor=diambra/engine" --format "{{.ID}}" | xargs -r docker stop
	docker ps -a --filter "ancestor=diambra/engine" --format "{{.ID}}" | xargs -r docker rm


demo:
	DIAMBRA_ENVS=localhost:50051 python ./simple.py

install:
	pip3 install -r requirements.txt

go:
	while true; do make run; done