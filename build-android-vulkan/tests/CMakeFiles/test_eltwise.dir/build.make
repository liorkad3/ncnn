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
CMAKE_BINARY_DIR = /home/liork/repo_ncnn/ncnn/build-android-vulkan

# Include any dependencies generated for this target.
include tests/CMakeFiles/test_eltwise.dir/depend.make

# Include the progress variables for this target.
include tests/CMakeFiles/test_eltwise.dir/progress.make

# Include the compile flags for this target's objects.
include tests/CMakeFiles/test_eltwise.dir/flags.make

tests/CMakeFiles/test_eltwise.dir/test_eltwise.cpp.o: tests/CMakeFiles/test_eltwise.dir/flags.make
tests/CMakeFiles/test_eltwise.dir/test_eltwise.cpp.o: ../tests/test_eltwise.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/liork/repo_ncnn/ncnn/build-android-vulkan/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object tests/CMakeFiles/test_eltwise.dir/test_eltwise.cpp.o"
	cd /home/liork/repo_ncnn/ncnn/build-android-vulkan/tests && /home/liork/android-ndk-r21d/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++ --target=aarch64-none-linux-android24 --gcc-toolchain=/home/liork/android-ndk-r21d/toolchains/llvm/prebuilt/linux-x86_64 --sysroot=/home/liork/android-ndk-r21d/toolchains/llvm/prebuilt/linux-x86_64/sysroot  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/test_eltwise.dir/test_eltwise.cpp.o -c /home/liork/repo_ncnn/ncnn/tests/test_eltwise.cpp

tests/CMakeFiles/test_eltwise.dir/test_eltwise.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_eltwise.dir/test_eltwise.cpp.i"
	cd /home/liork/repo_ncnn/ncnn/build-android-vulkan/tests && /home/liork/android-ndk-r21d/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++ --target=aarch64-none-linux-android24 --gcc-toolchain=/home/liork/android-ndk-r21d/toolchains/llvm/prebuilt/linux-x86_64 --sysroot=/home/liork/android-ndk-r21d/toolchains/llvm/prebuilt/linux-x86_64/sysroot $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/liork/repo_ncnn/ncnn/tests/test_eltwise.cpp > CMakeFiles/test_eltwise.dir/test_eltwise.cpp.i

tests/CMakeFiles/test_eltwise.dir/test_eltwise.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_eltwise.dir/test_eltwise.cpp.s"
	cd /home/liork/repo_ncnn/ncnn/build-android-vulkan/tests && /home/liork/android-ndk-r21d/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++ --target=aarch64-none-linux-android24 --gcc-toolchain=/home/liork/android-ndk-r21d/toolchains/llvm/prebuilt/linux-x86_64 --sysroot=/home/liork/android-ndk-r21d/toolchains/llvm/prebuilt/linux-x86_64/sysroot $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/liork/repo_ncnn/ncnn/tests/test_eltwise.cpp -o CMakeFiles/test_eltwise.dir/test_eltwise.cpp.s

# Object files for target test_eltwise
test_eltwise_OBJECTS = \
"CMakeFiles/test_eltwise.dir/test_eltwise.cpp.o"

# External object files for target test_eltwise
test_eltwise_EXTERNAL_OBJECTS =

tests/test_eltwise: tests/CMakeFiles/test_eltwise.dir/test_eltwise.cpp.o
tests/test_eltwise: tests/CMakeFiles/test_eltwise.dir/build.make
tests/test_eltwise: src/libncnn.a
tests/test_eltwise: /home/liork/android-ndk-r21d/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/24/libvulkan.so
tests/test_eltwise: glslang/SPIRV/libSPIRV.a
tests/test_eltwise: glslang/glslang/libglslang.a
tests/test_eltwise: glslang/OGLCompilersDLL/libOGLCompiler.a
tests/test_eltwise: glslang/glslang/OSDependent/Unix/libOSDependent.a
tests/test_eltwise: tests/CMakeFiles/test_eltwise.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/liork/repo_ncnn/ncnn/build-android-vulkan/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable test_eltwise"
	cd /home/liork/repo_ncnn/ncnn/build-android-vulkan/tests && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test_eltwise.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
tests/CMakeFiles/test_eltwise.dir/build: tests/test_eltwise

.PHONY : tests/CMakeFiles/test_eltwise.dir/build

tests/CMakeFiles/test_eltwise.dir/clean:
	cd /home/liork/repo_ncnn/ncnn/build-android-vulkan/tests && $(CMAKE_COMMAND) -P CMakeFiles/test_eltwise.dir/cmake_clean.cmake
.PHONY : tests/CMakeFiles/test_eltwise.dir/clean

tests/CMakeFiles/test_eltwise.dir/depend:
	cd /home/liork/repo_ncnn/ncnn/build-android-vulkan && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/liork/repo_ncnn/ncnn /home/liork/repo_ncnn/ncnn/tests /home/liork/repo_ncnn/ncnn/build-android-vulkan /home/liork/repo_ncnn/ncnn/build-android-vulkan/tests /home/liork/repo_ncnn/ncnn/build-android-vulkan/tests/CMakeFiles/test_eltwise.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : tests/CMakeFiles/test_eltwise.dir/depend

