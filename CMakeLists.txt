# The following lines of boilerplate have to be in your project's CMakeLists
# in this exact order for cmake to work correctly
cmake_minimum_required(VERSION 3.16)

# Microphone for main/app_audio_stream_int8.c; cached per build directory:
#   idf.py -B build_i2s build                   (default: standard I2S mic)
#   idf.py -B build_pdm -DAUDIO_MIC=pdm build   (PDM mic)
set(AUDIO_MIC "i2s" CACHE STRING "Microphone interface: i2s or pdm")

include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(TTGO_I2S_Record_Test)
