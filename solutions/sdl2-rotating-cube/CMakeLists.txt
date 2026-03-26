cmake_minimum_required(VERSION 3.16)
project(sdl2_cube_steps C)

set(CMAKE_C_STANDARD 11)

find_package(SDL2 REQUIRED)

foreach(step step1_triangle step2_square step3_rotating_square step4_cube
              challenge1_touch_rotation challenge2_two_cubes)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${step}.c")
        add_executable(${step} ${step}.c)
        target_include_directories(${step} PRIVATE ${SDL2_INCLUDE_DIRS})
        target_link_libraries(${step} PRIVATE ${SDL2_LIBRARIES} GLESv2 m)
    endif()
endforeach()
