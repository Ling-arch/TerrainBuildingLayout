cmake_minimum_required(VERSION 3.12.0)
project(TerrainBuildingLayout
    VERSION 0.0.1
    DESCRIPTION "Generate building layout automatically matched with terrain"
    HOMEPAGE_URL "https://github.com/Ling-arch/TerrainBuildingLayout"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#设置导出lib
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

#设置动态库输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

#设置目录变量
set(UtilsDir ${CMAKE_SOURCE_DIR}/src/utils)
set(PlanDir ${CMAKE_SOURCE_DIR}/src/differentiablePlan)
set(BuildingDir ${CMAKE_SOURCE_DIR}/src/building)
set(TerrainDir ${CMAKE_SOURCE_DIR}/src/terrain)
set(LayoutDir ${CMAKE_SOURCE_DIR}/src/Layout)

#添加子目录
add_subdirectory(src)
add_subdirectory(tests)




