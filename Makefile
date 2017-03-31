
all:
	nvcc -g -Wno-deprecated-gpu-targets -DDIM=3 -std=c++11 trilateration.cu -o 3dtri

two:
	nvcc -g -Wno-deprecated-gpu-targets -DDIM=2 -std=c++11 trilateration.cu -o 2dtri

.PHONY: clean

clean:
	rm -f *.out 2dtri 3dtri 2>/dev/null

