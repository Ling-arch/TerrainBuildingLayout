cmake_minimum_required(VERSION 3.12.0)
project(TerrainBuildingLayout
    VERSION 0.0.1
    DESCRIPTION "Generate building layout automatically matched with terrain"
    HOMEPAGE_URL "https://github.com/Ling-arch/TerrainBuildingLayout"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#set lib output for windows dll
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

#set output directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)

#set subdirectory path variables
set(UtilsDir ${CMAKE_SOURCE_DIR}/src/utils)
set(PlanDir ${CMAKE_SOURCE_DIR}/src/differentiablePlan)
set(BuildingDir ${CMAKE_SOURCE_DIR}/src/building)
set(TerrainDir ${CMAKE_SOURCE_DIR}/src/terrain)
set(LayoutDir ${CMAKE_SOURCE_DIR}/src/Layout)

# Use Qt6
find_package(Qt6 REQUIRED COMPONENTS Core Gui Widgets OpenGL OpenGLWidgets)
if(TARGET Qt6::Core)
    message(STATUS "Qt6 found.")
endif()

# Use CGAL
find_package(CGAL REQUIRED OPTIONAL_COMPONENTS Qt6)
if(TARGET CGAL::CGAL_Qt6)
    message(STATUS "CGAL found with Qt6 support.")
endif()

# Use Eigen
find_package(Eigen3 3.1.0 QUIET)

if(TARGET Eigen3::Eigen)
    message(STATUS "Eigen3 found.")
endif()
include(CGAL_Eigen3_support)

set(THIRD_PARTY_LIBS
    Eigen3::Eigen
    CGAL::CGAL
    CGAL::CGAL_Qt6
    Qt6::Core
    Qt6::Gui
    Qt6::Widgets
    Qt6::OpenGL
    Qt6::OpenGLWidgets
)


# Add subdirectories
add_subdirectory(src)
add_subdirectory(tests)




