# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.15

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/liork/repo_ncnn/ncnn

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/liork/repo_ncnn/ncnn/build

# Include any dependencies generated for this target.
include tests/CMakeFiles/test_reshape.dir/depend.make

# Include the progress variables for this target.
include tests/CMakeFiles/test_reshape.dir/progress.make

# Include the compile flags for this target's objects.
include tests/CMakeFiles/test_reshape.dir/flags.make

tests/CMakeFiles/test_reshape.dir/test_reshape.cpp.o: tests/CMakeFiles/test_reshape.dir/flags.make
tests/CMakeFiles/test_reshape.dir/test_reshape.cpp.o: ../tests/test_reshape.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/liork/repo_ncnn/ncnn/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object tests/CMakeFiles/test_reshape.dir/test_reshape.cpp.o"
	cd /home/liork/repo_ncnn/ncnn/build/tests && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/test_reshape.dir/test_reshape.cpp.o -c /home/liork/repo_ncnn/ncnn/tests/test_reshape.cpp

tests/CMakeFiles/test_reshape.dir/test_reshape.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_reshape.dir/test_reshape.cpp.i"
	cd /home/liork/repo_ncnn/ncnn/build/tests && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/liork/repo_ncnn/ncnn/tests/test_reshape.cpp > CMakeFiles/test_reshape.dir/test_reshape.cpp.i

tests/CMakeFiles/test_reshape.dir/test_reshape.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_reshape.dir/test_reshape.cpp.s"
	cd /home/liork/repo_ncnn/ncnn/build/tests && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/liork/repo_ncnn/ncnn/tests/test_reshape.cpp -o CMakeFiles/test_reshape.dir/test_reshape.cpp.s

# Object files for target test_reshape
test_reshape_OBJECTS = \
"CMakeFiles/test_reshape.dir/test_reshape.cpp.o"

# External object files for target test_reshape
test_reshape_EXTERNAL_OBJECTS =

tests/test_reshape: tests/CMakeFiles/test_reshape.dir/test_reshape.cpp.o
tests/test_reshape: tests/CMakeFiles/test_reshape.dir/build.make
tests/test_reshape: src/libncnn.a
tests/test_reshape: /usr/lib/gcc/x86_64-linux-gnu/9/libgomp.so
tests/test_reshape: /usr/lib/x86_64-linux-gnu/libpthread.so
tests/test_reshape: tests/CMakeFiles/test_reshape.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/liork/repo_ncnn/ncnn/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable test_reshape"
	cd /home/liork/repo_ncnn/ncnn/build/tests && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test_reshape.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
tests/CMakeFiles/test_reshape.dir/build: tests/test_reshape

.PHONY : tests/CMakeFiles/test_reshape.dir/build

tests/CMakeFiles/test_reshape.dir/clean:
	cd /home/liork/repo_ncnn/ncnn/build/tests && $(CMAKE_COMMAND) -P CMakeFiles/test_reshape.dir/cmake_clean.cmake
.PHONY : tests/CMakeFiles/test_reshape.dir/clean

tests/CMakeFiles/test_reshape.dir/depend:
	cd /home/liork/repo_ncnn/ncnn/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/liork/repo_ncnn/ncnn /home/liork/repo_ncnn/ncnn/tests /home/liork/repo_ncnn/ncnn/build /home/liork/repo_ncnn/ncnn/build/tests /home/liork/repo_ncnn/ncnn/build/tests/CMakeFiles/test_reshape.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : tests/CMakeFiles/test_reshape.dir/depend

