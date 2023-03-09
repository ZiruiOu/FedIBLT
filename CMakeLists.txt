cmake_minimum_required(VERSION 3.14)
project(FedIBLT CXX)
set(CMAKE_CXX_STANDARD 17)

# Set default CMAKE_BUILD_TYPE = Release
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif(NOT CMAKE_BUILD_TYPE)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

#find_package(Torch REQUIRED)
#if(NOT Torch_FOUND)
#    message(FATAL_ERROR "Pytorch not found!")
#endif(NOT Torch_FOUND)

find_package(pybind11 REQUIRED)

include_directories(include)

pybind11_add_module(fedIBLT ./bind.cc)
target_link_libraries(fedIBLT PUBLIC ${TORCH_LIBRARIES})
