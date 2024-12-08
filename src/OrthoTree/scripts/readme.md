# Readme for new scripts n' stuff

## build.sh
This basically just runs `cmake ..` with all the args we need.

## experiment.sh \<num_procs>
This compiles and runs (`mpiexec`) `root/test/orthotree/OrthoTreeTest.cpp`. Requires number of processors as an arg to run.

## unit_tests.sh
Compiles and runs our unit tests in `root/unit_tests/OrthoTree/*`. Will run all tests with 1 processor, except for the ones that require more. Which tests need more processors to pass can be specified in the script.

**IMPORTNAT NOTE**
If you add a new unit test that requires a certain number of ranks to pass do the following two things:

1. Assert for number of procs in your testfile like this:
    ```cpp
    // sanity check, tests are designed to run (and pass) with {N} ranks
    TEST(MyTest, AssertWorldSize) {
        const size_t world_size = Comm->size();
        const size_t expected_world_size = N;
        ASSERT_EQ(world_size, expected_world_size) 
            << "Tests for MyTest algo can not pass with: " 
            << world_size << " ranks, needs: " << expected_world_size;
    }
    ```
2. Add your test name and required number of ranks to the top of `unit_tests`sh`


## visualise.sh \<num_procs>
This calls `experiment.sh \<num_procs>` if this works it will run the python script afterwards which visualises the output. 

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