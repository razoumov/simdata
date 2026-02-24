## Initial setup

```sh
git clone https://github.com/razoumov/simdata
cd simdata
module load chapel-multicore/2.4.0
salloc --time=2:00:0 --mem-per-cpu=3600
chpl --fast acoustic2D.chpl
```

Models 1-7 in the code reflect the solutions shown in the plots on [this
page](https://folio.vastcloud.org/meetup20250910.html).

Models 8-9 are production models to create synthetic data to train a generative AI model. To run e.g. model 8
just one time and store every frame to create a movie, run the command

```sh
./acoustic2D --model=8 --nout=1
```

to produce 251 frames that can be stitched into a movie.

A production run will include both models 8 and 9:

```sh
./acoustic2D --model=8 --nruns=100
./acoustic2D --model=9 --nruns=100
```

This will run 200 models, with two frames (output images) per model.

Naming scheme:

`frame800010079.png` file name includes the following pieces:

- 8 is the model number
- 0001 is the run number
- 0079 is the frame number for this run

In production runs, you will see only files `*0000.png` (initial conditions) and `*0001.png` (result).

## Create training and test data

```sh
mkdir data/{training,testing}
./acoustic2D --model=8 --nruns=1000   # two files per run
./acoustic2D --model=9 --nruns=1000   # two files per run
mv frame{8,9}????000{0,1}.png data/training
./acoustic2D --model=8 --nt=0 --nruns=5   # just the initial conditions
mv frame8000{1..5}0000.png data/testing
```

## Create Python virtual env

```sh
module load python/3.12.4
python -m venv ~/env-jax
source ~/env-jax/bin/activate
python -m pip install --upgrade pip --no-index
python -m pip install --no-index jax flax pillow matplotlib
...
deactivate
```

## Training

```sh
source ~/env-jax/bin/activate
sbatch submit.sh
```

## Inference

```sh
source ~/env-jax/bin/activate
python infer.py
```

Check the image `prediction.png`.
