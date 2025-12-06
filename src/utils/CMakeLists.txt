# 设置库变量
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} UtilsSrc)

# 添加库
add_library(Utils STATIC ${UtilsSrc})

target_include_directories(Utils
PUBLIC
${CMAKE_CURRENT_SOURCE_DIR}
)