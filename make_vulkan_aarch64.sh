export ANDROID_NDK=/home/liork/android-ndk-r21d
echo $ANDROID_NDK
export VULKAN_SDK=/home/liork/1.1.114.0/x86_64
echo $VULKAN_SDK
mkdir build-android-vulkan
cd build-android-vulkan
cmake -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake -DANDROID_ABI="arm64-v8a" -DANDROID_PLATFORM=android-24 -DNCNN_VULKAN=ON ..
make -j4
make install
