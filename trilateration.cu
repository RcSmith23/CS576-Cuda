#include <iostream>
#include <random>
#include <assert.h>
#include <math.h>

#ifndef DIM
#define DIM 2
#endif

#define FIX_PTS 3
#define STEP_MAX 0.1 

__global__
void trilateration(size_t n, float ** disp, float ** ref, float ** outp) {
  size_t index = threadIdx.x, stride = blockDim.x;
  const size_t outc = n / 4;  // Size of the outp array

  // Get special coordinate values from reference points
  float d = ref[0][1], x =  ref[0][2], y = ref[1][2]; 

  // Iterate over given set of output array
  for (int i = index; i < outc; i += stride) {
    float avg[DIM] = { 0 };
    // Iterate over corresponding input coords
    for (int j = 4 * index; j < 4 * (index + 1); ++j) {
      float pos[DIM] = { 0 };
      pos[0] = ( disp[0][j] - disp[1][j] + pow(d, 2.0) ) /  (2 * d);  // Compute x-coordinate
      pos[1] = (( disp[0][j] - disp[2][j] + pow(x, 2.0) + pow(y, 2.0) )\
          /  (2 * y) )  - ( x * pos[0] / y);  // Compute y-coordinate
      if (DIM == 3)
        pos[2] = sqrt(disp[0][j] - pow(pos[0], 2.0) - pow(pos[1], 2.0));
      for (int s = 0; s < DIM; ++s) avg[s] += pos[s];
    }
    for (int t = 0; t < DIM; ++t) outp[t][i] = avg[t] / 4;
  }
}

float sqdisplacement(float ** p1, int lp1, float ** p2, int lp2) {
  float dist = 0.0;
  for (int i = 0; i < DIM; ++i)
    dist += pow((p1[lp1][i] - p2[lp2][i]), 2.0);
  return dist;
}

// Requires:
// - argv[1] - Size of input data set 
// - argv[2] - Number of blocks
// - argv[3] - Number of threads
int main(int argc, char * argv[]) {

  // Parse input args
  assert(argc == 4);
  // Check dimension
  assert(DIM > 1 && DIM < 4);

  // Create random number generator
  std::default_random_engine eng;
  std::uniform_real_distribution<float> dist(0.0, STEP_MAX);
  std::uniform_real_distribution<float> coords(-50.0, 50.0);

  // Set the error threshold
  const size_t error = 0.5;
  size_t N = 1 << 12;   // 2^12
  float **inp, **outp, **disp, **ref;
  inp   = (float **)new float*[DIM];
  cudaMallocManaged(&outp, DIM * sizeof(float*)); 
  cudaMallocManaged(&disp, FIX_PTS * sizeof(float*)); 
  cudaMallocManaged(&ref, DIM * sizeof(float*)); 

  // Allocate space for input set and output averages
  for (int i = 0; i < DIM; ++i) {
    inp[i] = (float*)new float[N];
    cudaMallocManaged(ref + i, FIX_PTS * sizeof(float)); 
    cudaMallocManaged(outp + i, (N / 4) * sizeof(float)); 
  }

  // Allocate space for the displacement values
  for (int i = 0; i < FIX_PTS; ++i)
    cudaMallocManaged(disp + i, N * sizeof(float));

  // Generate the reference points
  // Just setting fixed values for now
  if (DIM == 3) { // All on the z = 0 plane
    ref[0][0] = 0.0; ref[1][0] = 0.0; ref[2][0] = 0.0;    // Origin
    ref[0][1] = 10.0; ref[1][0] = 0.0; ref[2][0] = 0.0;   // On x-axis
    ref[0][0] = 10.0; ref[1][0] = 20.0; ref[2][0] = 0.0;  // Floating around
  } else {
    ref[0][0] = 0.0; ref[1][0] = 0.0;   // Origin
    ref[0][1] = 10.0; ref[1][0] = 0.0;  // On x-axis
    ref[0][0] = 10.0; ref[1][0] = 20.0; // Floating around
  }

  // Generate the point sequence and compute displacements
  for (int i = 0; i < DIM; ++i) inp[0][i] = 0.0;
  for (int i = 1; i < N; ++i) {
    for (int d = 0; d < DIM; ++d)
      inp[i][d] = inp[i-1][d] + dist(eng);
    for (int p = 0; p < FIX_PTS; ++p)
      disp[i][p] = sqdisplacement(inp, i, ref, p);
  }

  // Run the trilateration
  // TODO adapt this call to CL args
  trilateration<<<1,1>>>(N, disp, ref, outp);

  // Wait for the GPU to finish
  cudaDeviceSynchronize();

  // TODO Do something here with the results
  // Like check how they stack up
  for (int i = 0; i < N / 4; ++i) {
    size_t pos = i * 4;
    assert(sqdisplacement(inp, i, outp, pos) < error);
  }

  // Free the memory
  for (int i = 0; i < DIM; ++i) {
    cudaFree(outp[i]); 
    cudaFree(ref[i]); 
    delete [] inp[i];
  }
  for (int i = 0; i < FIX_PTS; ++i)
    cudaFree(disp[i]); 

  delete [] inp;
  cudaFree(outp);
  cudaFree(disp);
  cudaFree(ref);
}
