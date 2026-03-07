# 设置库变量
cmake_policy(SET CMP0104 NEW)
set(CMAKE_CUDA_ARCHITECTURES native)

aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} TerrainSrc)

# 添加库
add_library(Terrain STATIC ${TerrainSrc})

add_library(viewshed_cuda
    viewshed_cuda.cu
)

target_include_directories(Terrain
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${UtilsDir}
    ${PlanDir}
    ${BuildingDir}
    ${CMAKE_SOURCE_DIR}/thirdparty/include
)

target_link_libraries(Terrain
    PUBLIC
    Utils
    viewshed_cuda
)