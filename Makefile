
all:
	nvcc -Wno-deprecated-gpu-targets -DDIM=3 -std=c++11 trilateration.cu

two:
	nvcc -Wno-deprecated-gpu-targets -DDIM=2 -std=c++11 trilateration.cu
