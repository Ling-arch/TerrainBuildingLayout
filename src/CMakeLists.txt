project(total)

add_subdirectory(utils)
add_subdirectory(differentiablePlan)
add_subdirectory(building)
add_subdirectory(terrain)
add_subdirectory(layout)

add_executable(${PROJECT_NAME} main.cpp)

target_include_directories(${PROJECT_NAME} 
PUBLIC 
${UtilsDir} 
${PlanDir} 
${BuildingDir} 
${TerrainDir}
${LayoutDir}
)

target_link_libraries(${PROJECT_NAME}
PUBLIC 
Layout
)