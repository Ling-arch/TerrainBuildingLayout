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
        ${TerrainDir}
        ${LayoutDir}
        ${CMAKE_SOURCE_DIR}/thirdparty/include
    )


    target_link_libraries(${test_name}
        PUBLIC
        Terrain
        Layout
        Utils
        Plan
        ${THIRD_PARTY_LIBS}
        ${Raylib_LIBS}
    )
      add_custom_command(TARGET ${test_name} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_RUNTIME_DLLS:${test_name}>
            $<TARGET_FILE_DIR:${test_name}>
        COMMAND_EXPAND_LISTS
    )

    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()


