# Readme for new scripts n' stuff

## build.sh
This basically just runs `cmake ..` with all the args we need.

## experiment.sh \<num_procs>
This compiles and runs (`mpiexec`) `root/test/orthotree/OrthoTreeTest.cpp`. Requires number of processors as an arg to run.

## unit_tests.sh
Compiles and runs our unit tests in `root/unit_tests/OrthoTree/*`. Will run all tests with 1 processor, except for the ones that require more. Which tests need more processors to pass can be specified in the script.

## visualise.sh \<num_procs>
This calls `experiment.sh \<num_procs>` if this wors it will run the python script afterwards which visualises the output. 

The python script has two functions it can run. You have to toggle them manually in the file.

```python
data_folder = "output"

"""
This will generate one plot, where each processor will be colored differently. (not that usefull)
"""
plot_combined_processors(data_folder)

"""
This generates one plot per processor.
"""
plot_all_processors_in_subplots(data_folder)
```