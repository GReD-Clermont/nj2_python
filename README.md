# NucleusJ 2.0 in Python

## Compute parameters

	python compute_params.py -p "D:\\path\\masks" -s 0.1032 0.1032 0.2

## Docker run

	docker run --gpus all -it --rm -v $PWD:/home/nj2_python -w /home -p 8888:8888 java
