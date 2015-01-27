# jetsonSIFT

This is a CUDA-accelerated SIFT keypoint extraction implementation. Note it currently only perform extraction on the first octave. Enter the following commands to compile:

```bash
cd jetsonSIFT
mkdir build
cd build
cmake ../src
```

The program is used as follows:

```bash
./jetsonSIFT yourimage.jpg
```

Sample:

```bash
./jetsonSIFT ../images/lenna.jpg
```

If you receive errors regarding an unsupported CUDA architecture specification, edit the `arch=compute\_32,code=sm\_32` line to match the latest architecture supported by your (Nvidia) card.
