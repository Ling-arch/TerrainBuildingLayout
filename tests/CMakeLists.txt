# 收集当前目录下所有 .cpp 文件
file(GLOB TEST_SRCS "*.cpp")

# 遍历每一个 .cpp 文件
foreach(test_src ${TEST_SRCS})
    # 提取不带后缀的文件名 (比如 testA)
    get_filename_component(test_name ${test_src} NAME_WE)

    # 创建可执行文件
    add_executable(${test_name} ${test_src})

    target_include_directories(${test_name}
        PUBLIC
        ${UtilsDir}
        ${PlanDir}
        ${BuildingDir}
        ${TerrainDir}
        ${LayoutDir}
    )

    # 链接项目的库（如 Terrain、Building、Layout...按需修改）
    target_link_libraries(${test_name}
        PUBLIC
        Terrain
        Building
        Layout
        Utils
        ${THIRD_PARTY_LIBS}
    )
endforeach()


