project(app)

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
if(MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET ${PROJECT_NAME}
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${TORCH_DLLS}
        $<TARGET_FILE_DIR:${PROJECT_NAME}>)
endif(MSVC)

target_link_libraries(${PROJECT_NAME}
    PRIVATE
    Utils
    ${TORCH_LIBS}
)

