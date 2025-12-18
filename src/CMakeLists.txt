project(app)

add_subdirectory(utils)
add_subdirectory(differentiablePlan)
add_subdirectory(building)
add_subdirectory(terrain)
add_subdirectory(layout)

add_executable(${PROJECT_NAME} main.cpp)

# Ensure that the CUDA is linked properly on Windows
set_property(TARGET ${PROJECT_NAME} PROPERTY
    LINK_FLAGS "/INCLUDE:?ignore_this_library_placeholder@@YAHXZ")

target_include_directories(${PROJECT_NAME} 
PUBLIC 
${UtilsDir} 
${PlanDir} 
${BuildingDir} 
${TerrainDir}
${LayoutDir}
)


target_link_libraries(${PROJECT_NAME}
    PRIVATE
    Utils
    ${TORCH_LIBS}
)

#   if(MSVC)
#      file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
#       add_custom_command(TARGET ${PROJECT_NAME}
#           POST_BUILD
#           COMMAND ${CMAKE_COMMAND} -E copy_if_different
#           ${TORCH_DLLS}
#           $<TARGET_FILE_DIR:${PROJECT_NAME}>)

#       file(GLOB ThirdParty_DLLS "${CMAKE_SOURCE_DIR}/thirdparty/lib/*.dll")
#       add_custom_command(TARGET ${PROJECT_NAME}
#           POST_BUILD
#           COMMAND ${CMAKE_COMMAND} -E copy_if_different
#           ${ThirdParty_DLLS}
#           $<TARGET_FILE_DIR:${PROJECT_NAME}>)
#   endif(MSVC)

