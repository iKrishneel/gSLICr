# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

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
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/krishneel/Documents/programs/gSLICr

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/krishneel/Documents/programs/gSLICr/build

# Include any dependencies generated for this target.
include CMakeFiles/demo.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/demo.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/demo.dir/flags.make

CMakeFiles/demo.dir/demo.cpp.o: CMakeFiles/demo.dir/flags.make
CMakeFiles/demo.dir/demo.cpp.o: ../demo.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/krishneel/Documents/programs/gSLICr/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/demo.dir/demo.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/demo.dir/demo.cpp.o -c /home/krishneel/Documents/programs/gSLICr/demo.cpp

CMakeFiles/demo.dir/demo.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/demo.dir/demo.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/krishneel/Documents/programs/gSLICr/demo.cpp > CMakeFiles/demo.dir/demo.cpp.i

CMakeFiles/demo.dir/demo.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/demo.dir/demo.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/krishneel/Documents/programs/gSLICr/demo.cpp -o CMakeFiles/demo.dir/demo.cpp.s

CMakeFiles/demo.dir/demo.cpp.o.requires:
.PHONY : CMakeFiles/demo.dir/demo.cpp.o.requires

CMakeFiles/demo.dir/demo.cpp.o.provides: CMakeFiles/demo.dir/demo.cpp.o.requires
	$(MAKE) -f CMakeFiles/demo.dir/build.make CMakeFiles/demo.dir/demo.cpp.o.provides.build
.PHONY : CMakeFiles/demo.dir/demo.cpp.o.provides

CMakeFiles/demo.dir/demo.cpp.o.provides.build: CMakeFiles/demo.dir/demo.cpp.o

# Object files for target demo
demo_OBJECTS = \
"CMakeFiles/demo.dir/demo.cpp.o"

# External object files for target demo
demo_EXTERNAL_OBJECTS =

demo: CMakeFiles/demo.dir/demo.cpp.o
demo: CMakeFiles/demo.dir/build.make
demo: libgSLICr_lib.a
demo: /usr/local/lib/libopencv_xphoto.so.3.1.0
demo: /usr/local/lib/libopencv_xobjdetect.so.3.1.0
demo: /usr/local/lib/libopencv_ximgproc.so.3.1.0
demo: /usr/local/lib/libopencv_xfeatures2d.so.3.1.0
demo: /usr/local/lib/libopencv_tracking.so.3.1.0
demo: /usr/local/lib/libopencv_text.so.3.1.0
demo: /usr/local/lib/libopencv_surface_matching.so.3.1.0
demo: /usr/local/lib/libopencv_structured_light.so.3.1.0
demo: /usr/local/lib/libopencv_stereo.so.3.1.0
demo: /usr/local/lib/libopencv_saliency.so.3.1.0
demo: /usr/local/lib/libopencv_rgbd.so.3.1.0
demo: /usr/local/lib/libopencv_reg.so.3.1.0
demo: /usr/local/lib/libopencv_plot.so.3.1.0
demo: /usr/local/lib/libopencv_optflow.so.3.1.0
demo: /usr/local/lib/libopencv_line_descriptor.so.3.1.0
demo: /usr/local/lib/libopencv_hdf.so.3.1.0
demo: /usr/local/lib/libopencv_fuzzy.so.3.1.0
demo: /usr/local/lib/libopencv_face.so.3.1.0
demo: /usr/local/lib/libopencv_dpm.so.3.1.0
demo: /usr/local/lib/libopencv_dnn.so.3.1.0
demo: /usr/local/lib/libopencv_datasets.so.3.1.0
demo: /usr/local/lib/libopencv_ccalib.so.3.1.0
demo: /usr/local/lib/libopencv_bioinspired.so.3.1.0
demo: /usr/local/lib/libopencv_bgsegm.so.3.1.0
demo: /usr/local/lib/libopencv_aruco.so.3.1.0
demo: /usr/local/lib/libopencv_videostab.so.3.1.0
demo: /usr/local/lib/libopencv_videoio.so.3.1.0
demo: /usr/local/lib/libopencv_video.so.3.1.0
demo: /usr/local/lib/libopencv_superres.so.3.1.0
demo: /usr/local/lib/libopencv_stitching.so.3.1.0
demo: /usr/local/lib/libopencv_shape.so.3.1.0
demo: /usr/local/lib/libopencv_photo.so.3.1.0
demo: /usr/local/lib/libopencv_objdetect.so.3.1.0
demo: /usr/local/lib/libopencv_ml.so.3.1.0
demo: /usr/local/lib/libopencv_imgproc.so.3.1.0
demo: /usr/local/lib/libopencv_imgcodecs.so.3.1.0
demo: /usr/local/lib/libopencv_highgui.so.3.1.0
demo: /usr/local/lib/libopencv_flann.so.3.1.0
demo: /usr/local/lib/libopencv_features2d.so.3.1.0
demo: /usr/local/lib/libopencv_cudev.so.3.1.0
demo: /usr/local/lib/libopencv_cudawarping.so.3.1.0
demo: /usr/local/lib/libopencv_cudastereo.so.3.1.0
demo: /usr/local/lib/libopencv_cudaoptflow.so.3.1.0
demo: /usr/local/lib/libopencv_cudaobjdetect.so.3.1.0
demo: /usr/local/lib/libopencv_cudalegacy.so.3.1.0
demo: /usr/local/lib/libopencv_cudaimgproc.so.3.1.0
demo: /usr/local/lib/libopencv_cudafilters.so.3.1.0
demo: /usr/local/lib/libopencv_cudafeatures2d.so.3.1.0
demo: /usr/local/lib/libopencv_cudacodec.so.3.1.0
demo: /usr/local/lib/libopencv_cudabgsegm.so.3.1.0
demo: /usr/local/lib/libopencv_cudaarithm.so.3.1.0
demo: /usr/local/lib/libopencv_core.so.3.1.0
demo: /usr/local/lib/libopencv_calib3d.so.3.1.0
demo: /usr/local/cuda/lib64/libcudart.so
demo: /usr/local/lib/libopencv_text.so.3.1.0
demo: /usr/local/lib/libopencv_face.so.3.1.0
demo: /usr/local/lib/libopencv_ximgproc.so.3.1.0
demo: /usr/local/lib/libopencv_xfeatures2d.so.3.1.0
demo: /usr/local/lib/libopencv_shape.so.3.1.0
demo: /usr/local/lib/libopencv_cudawarping.so.3.1.0
demo: /usr/local/lib/libopencv_objdetect.so.3.1.0
demo: /usr/local/lib/libopencv_cudafilters.so.3.1.0
demo: /usr/local/lib/libopencv_cudaarithm.so.3.1.0
demo: /usr/local/lib/libopencv_calib3d.so.3.1.0
demo: /usr/local/lib/libopencv_features2d.so.3.1.0
demo: /usr/local/lib/libopencv_ml.so.3.1.0
demo: /usr/local/lib/libopencv_highgui.so.3.1.0
demo: /usr/local/lib/libopencv_videoio.so.3.1.0
demo: /usr/local/lib/libopencv_imgcodecs.so.3.1.0
demo: /usr/local/lib/libopencv_flann.so.3.1.0
demo: /usr/local/lib/libopencv_video.so.3.1.0
demo: /usr/local/lib/libopencv_imgproc.so.3.1.0
demo: /usr/local/lib/libopencv_core.so.3.1.0
demo: /usr/local/lib/libopencv_cudev.so.3.1.0
demo: CMakeFiles/demo.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable demo"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/demo.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/demo.dir/build: demo
.PHONY : CMakeFiles/demo.dir/build

CMakeFiles/demo.dir/requires: CMakeFiles/demo.dir/demo.cpp.o.requires
.PHONY : CMakeFiles/demo.dir/requires

CMakeFiles/demo.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/demo.dir/cmake_clean.cmake
.PHONY : CMakeFiles/demo.dir/clean

CMakeFiles/demo.dir/depend:
	cd /home/krishneel/Documents/programs/gSLICr/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/krishneel/Documents/programs/gSLICr /home/krishneel/Documents/programs/gSLICr /home/krishneel/Documents/programs/gSLICr/build /home/krishneel/Documents/programs/gSLICr/build /home/krishneel/Documents/programs/gSLICr/build/CMakeFiles/demo.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/demo.dir/depend

