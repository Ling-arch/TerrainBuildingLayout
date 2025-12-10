# 设置库变量
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} PlanSrc)

# 添加库
add_library(Plan STATIC ${PlanSrc})

target_include_directories(Plan
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${UtilsDir}
)

target_link_libraries(Plan PUBLIC ${THIRD_PARTY_LIBS} Utils ${TORCH_LIBS})