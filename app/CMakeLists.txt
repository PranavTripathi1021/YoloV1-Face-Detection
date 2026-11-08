cmake_minimum_required(VERSION 3.15.0)
project(FaceRecognitionAppApp VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/thirdparty/libtorch")


find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS
    core
    imgproc
    imgcodecs
    highgui
    videoio
    dnn
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")


add_library(HelperLib STATIC
    helper/functions.cpp
)
target_include_directories(HelperLib
    PUBLIC
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/helper
)

target_link_libraries(HelperLib
    PUBLIC
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBS}
)
add_executable(FaceRecognitionApp main.cpp)

target_link_libraries(FaceRecognitionApp
    PRIVATE
    HelperLib
)

