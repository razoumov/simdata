use Image, Math, IO, sciplot, Time;
use Random;
config const n = 501, nt = 250, nout = 250; // resolution, max time steps, plotting frequency
const a = 0.1;   // 0.1 - thick front, no post-wave oscillations; 0.5 - narrow front, large oscillations
config const animation = true;
config const model = 1, nruns = 1;

var h = 1.0 / (n-1), coef = -(a/h)**2;

var colour: [1..n, 1..n] 3*int;
var cmap = readColourmap('viridis.csv');   // cmap.domain is {1..256, 1..3}

const mesh = {1..n, 1..n};
const largerMesh = {0..n+1, 0..n+1};
var Vx, Vy, P, tmp: [largerMesh] real;

var watch: stopwatch;
watch.start();
for run in 1..nruns {
  P = 0;
  Vx = 0;
  Vy = 0;
  select model {
    // solutions 1-7 from https://folio.vastcloud.org/meetup20250910.html
    when 1 do
      P = [(i,j) in mesh] exp(coef*(((i-1)*h-0.5)**2+((j-1)*h-0.5)**2));
    when 2 do
      P = [(i,j) in mesh] exp(coef*(((i-1)*h-0.25)**2+((j-1)*h-0.5)**2)) +
      exp(coef*(((i-1)*h-0.75)**2+((j-1)*h-0.5)**2));
    when 3 {
      var sources = [(0.25,0.25), (0.25,0.75), (0.75,0.25), (0.75,0.75)];
      for (i,j) in mesh {
        P[i,j] = 0;
        for src in sources do
          P[i,j] += exp(coef*( ((i-1)*h-src[0])**2 + ((j-1)*h-src[1])**2 ));
      }
    }
    when 4 do
      P = [(i,j) in mesh] exp(coef*(((j-1)*h-0.5)**2));
    when 5 do
      P = [(i,j) in mesh] exp(coef*(((j-1)*h-0.5)**2+0.03*((i-1)*h-0.5)**2));
    when 6 {
      const hw = 50, smooth = 15;
      tmp[n/2-hw..n/2+hw,n/2-hw..n/2+hw] = 1.0;
      forall (i,j) in mesh[4..n-smooth,4..n-smooth] do
        P[i,j] = + reduce tmp[i-smooth..i+smooth,j-smooth..j+smooth];
    }
    when 7 {
      const hw = 80, smooth = 15;
      tmp[n/2-hw..n/2+hw,n/2-10..n/2+10] = 1.0;
      forall (i,j) in mesh[4..n-smooth,4..n-smooth] do
        P[i,j] = + reduce tmp[i-smooth..i+smooth,j-smooth..j+smooth];
    }
    when 8 do {
      // 1-5 randomly placed points
      var rint = new randomStream(int);
      var np = rint.next(min=1, max=5);
      var rreal = new randomStream(real);
      var sources: [1..np] (real, real);
      for i in 1..np do
        sources[i] = (0.2 + 0.6*rreal.next(), 0.2 + 0.6*rreal.next());
      //writeln(np, " sources: ", sources);
      for (i,j) in mesh {
        P[i,j] = 0;
        for src in sources do
          P[i,j] += exp(coef*( ((i-1)*h-src[0])**2 + ((j-1)*h-src[1])**2 ));
      }
    }
    when 9 do {
      // two randomly placed points with a line between them
      var rreal = new randomStream(real);
      var points = [(0.2+0.6*rreal.next(), 0.2+0.6*rreal.next()),
                    (0.2+0.6*rreal.next(), 0.2+0.6*rreal.next())];
      var ax = points[1][0] - points[0][0], ay = points[1][1] - points[0][1];
      for (i,j) in mesh {
        var wx = (i-1)*h - points[0][0], wy = (j-1)*h - points[0][1];
        var alength = sqrt(ax*ax + ay*ay), wlength = sqrt(wx*wx + wy*wy);
        var distanceFromTheLine = abs(ax*wy - ay*wx) / alength;
        var dimensionlessCoordinateAlongTheLine = (ax*wx + ay*wy) / (alength*alength);
        // writeln(distanceFromTheLine, "  ", dimensionlessCoordinateAlongTheLine);
        if dimensionlessCoordinateAlongTheLine > 1 then
          P[i,j] = exp(coef*(((i-1)*h-points[1][0])**2+((j-1)*h-points[1][1])**2));
        else if dimensionlessCoordinateAlongTheLine < 0 then
          P[i,j] = exp(coef*(((i-1)*h-points[0][0])**2+((j-1)*h-points[0][1])**2));
        else
          P[i,j] = exp(coef*distanceFromTheLine**2);
      }
    }
  }

  if animation then plotPressure(0,run);

  var dt = h / 1.6;
  for step in 1..nt {
    periodic(P);
    periodic(Vx);
    periodic(Vy);
    forall (i,j) in mesh {
      Vx[i,j] -= dt * (P[i,j]-P[i-1,j]) / h;
      Vy[i,j] -= dt * (P[i,j]-P[i,j-1]) / h;
    }
    forall (i,j) in mesh do
      P[i,j] -= dt * (Vx[i+1,j] - Vx[i,j] + Vy[i,j+1] - Vy[i,j]) / h;
    if step%nout == 0 && animation then plotPressure(step/nout,run);
  }
}
watch.stop();
writeln('Simulation took ', watch.elapsed(), ' seconds');

proc periodic(A) {
  A[0,1..n] = A[n,1..n]; A[n+1,1..n] = A[1,1..n];
  A[1..n,0] = A[1..n,n]; A[1..n,n+1] = A[1..n,1];
}

proc fourDigits(n: int) {
  var digits: string;
  if n >= 1000 then digits = n:string;
  else if n >= 100 then digits = "0"+n:string;
  else if n >= 10 then digits = "00"+n:string;
  else digits = "000"+n:string;
  return digits;
}

proc plotPressure(step,run) {
  var smin = min reduce(P);
  var smax = max reduce(P);
  for i in 1..n {
    for j in 1..n {
      var idx = ((P[j,n-i+1]-smin)/(smax-smin)*255):int + 1; //scale to 1..256
      colour[i,j] = ((cmap[idx,1]*255):int, (cmap[idx,2]*255):int, (cmap[idx,3]*255):int);
    }
  }
  var pixels = colorToPixel(colour);   // array of pixels
  var filename = "frame" + model:string + fourDigits(run) + fourDigits(step)+".png";
  writeln("writing ", filename);
  writeImage(filename, imageType.png, pixels);
}
