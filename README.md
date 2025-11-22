Initial setup:

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
