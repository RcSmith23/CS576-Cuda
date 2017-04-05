#include <iostream>
#include <fstream>
#include <string>
#include <random>
#include <iomanip>
#include <assert.h>
#include <math.h>

#ifndef DIM
#define DIM 2
#endif

#define FIX_PTS 3
#define STEP_MAX 0.5
#define STEP_MIN -0.5

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
    for (int j = 4 * i; j < (4 * (i + 1)); ++j) {
      float pos[DIM] = { 0 };
      pos[0] = ( disp[0][j] - disp[1][j] + pow(d, 2.0) ) /  (2 * d);  // Compute x-coordinate
      pos[1] = (( disp[0][j] - disp[2][j] + pow(x, 2.0) + pow(y, 2.0) )\
          /  (2 * y) )  - ( x * pos[0] / y);  // Compute y-coordinate
#if DIM == 3
      pos[2] = sqrt(abs(disp[0][j] - pow(pos[0], 2.0) - pow(pos[1], 2.0)));
#endif
      for (int s = 0; s < DIM; ++s) avg[s] += pos[s];
    }
    for (int t = 0; t < DIM; ++t) outp[t][i] = (avg[t] / 4.0);
  }
}

float sqdisplacement(float ** p1, int lp1, float ** p2, int lp2) {
  float dist = 0.0;
  for (int i = 0; i < DIM; ++i)
    dist += pow((p1[i][lp1] - p2[i][lp2]), 2.0);
  return dist;
}

float d_check(float *p, float ** p2, int ip2) {
  float dist = 0.0;
  for (int i = 0; i < DIM; ++i)
    dist += pow((p[i] - p2[i][ip2]), 2.0);
  return sqrt(dist);
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
  size_t size = atoi(argv[1]), blks = atoi(argv[2]), thrds = atoi(argv[3]);

  // Create random number generator
  std::default_random_engine eng; eng.seed(1);
  std::uniform_real_distribution<float> dist(STEP_MIN, STEP_MAX);

  size_t N = 1 << size;   // 2^size
  float **inp, **outp, **disp, **ref;

  // Start allocating space
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

  // Set the coordinates of the fixed points
  ref[0][0] = 0.0; ref[0][1] = 100.0;  ref[0][2] = 150.0; 
  ref[1][0] = 0.0; ref[1][1] = 0.0;   ref[1][2] = 200.0;
#if DIM == 3
  ref[2][0] = 0.0; ref[2][1] = 0.0;   ref[2][2] = 0.0;
#endif


  // Generate the point sequence and compute displacements
  for (int i = 0; i < DIM; ++i) inp[i][0] = 50.0;
  for (int i = 0; i < FIX_PTS; ++i)
    disp[i][0] = sqdisplacement(inp, 0, ref, i);
  for (int i = 1; i < N; ++i) {
    for (int d = 0; d < DIM; ++d)
      inp[d][i] = inp[d][i-1] + dist(eng);
    for (int p = 0; p < FIX_PTS; ++p)
      disp[p][i] = sqdisplacement(inp, i, ref, p);
  }

  // Run the trilateration
  trilateration<<<blks, thrds>>>(N, disp, ref, outp);

  // Wait for the GPU to finish
  cudaDeviceSynchronize();

  // Output the point set and trilateration set to a file
  std::ofstream outFile;
#if DIM == 3
  outFile.open("output/3dresults-" + std::to_string(size) + "-" +\
      std::to_string(blks) + "-" + std::to_string(thrds) + ".txt");
#else
  outFile.open("output/2dresults-" + std::to_string(size) + "-" +\
      std::to_string(blks) + "-" + std::to_string(thrds) + ".txt");
#endif
  outFile << std::setw(10 * DIM + (DIM - 1)) << "Input point" << " | Trilateration Result" << std::endl;
  for (int i = 0; i < N; ++i) {
    for (int d = 0; d < DIM; ++d)
      outFile << std::setw(10) << inp[d][i] << " ";
    outFile << "| ";
    if ( ! (i % 4) ) { // Also print out the trilateration value
      for (int d = 0; d < DIM; ++d)
        outFile << std::setw(10) << outp[d][i / 4] << " ";
    }
    outFile << std::endl;
  }

  outFile.close();

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
