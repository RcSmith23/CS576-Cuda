
all:
	nvcc -Wno-deprecated-gpu-targets -DDIM=3 -std=c++11 trilateration.cu

two:
	nvcc -Wno-deprecated-gpu-targets -DDIM=2 -std=c++11 trilateration.cu

.PHONY: clean

clean:
	rm -f *.out 2>/dev/null

