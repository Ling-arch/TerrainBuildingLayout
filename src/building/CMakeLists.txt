# 设置库变量
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} BuildingSrc)

# 添加库
add_library(Building STATIC ${BuildingSrc})

target_include_directories(Building
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${UtilsDir}
    ${PlanDir}
)

target_link_libraries(Building
    PUBLIC
    Plan)