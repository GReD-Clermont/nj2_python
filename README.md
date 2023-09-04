# NucleusJ 2.0 in Python

## Compute parameters

	python compute_params.py -p "D:\path\masks" -cc "D:\path\masks_chromocenters" -s 0.1032 0.1032 0.2

The `-cc` and the `-s` options are optional. The paths can be either a path to an image or a path to a folder of images.

## Docker run

	docker run --gpus all -it --rm -v $PWD:/home/nj2_python -w /home -p 8888:8888 java
