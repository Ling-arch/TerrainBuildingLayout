
# 设置库变量
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} LayoutSrc)

# 添加库
add_library(Layout STATIC ${LayoutSrc})

target_include_directories(Layout
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${UtilsDir}
    ${PlanDir}
    ${BuildingDir}
    ${TerrainDir}
)

target_link_libraries(Layout
    PUBLIC
    Building
    Terrain)