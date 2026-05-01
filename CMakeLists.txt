# CMakeLists.txt for HALCON extension package 'halconuser' example
# Copyright (c) 2019-2023 by MVTec Software GmbH

# Option for using HALCON XL.
option(HALCON_XL "Use HALCON XL" OFF)
# Option for enabling .NET support. This will require a more recent version
# of CMake to work. .NET / CSharp 仅在 Windows + Visual Studio 生成器上可用，
# 因此在非 Windows 平台默认关闭，避免在 Linux/macOS 上误触发 CSharp 检测。
if(WIN32)
  option(HALCON_DOTNET "Build .NET support" ON)
else()
  option(HALCON_DOTNET "Build .NET support" OFF)
endif()
# If the HALCON_DOTNET option is selected, check if this version of CMake
# has the required CSharp support. Note that the Visual Studio 2010 or later
# generator has to be used.
if(HALCON_DOTNET)
  # .NET / CSharp 支持需要较新的 CMake，且仅 Windows 的 Visual Studio 生成器有效
  cmake_minimum_required(VERSION 4.1.1)
  set(_CSHARP CSharp)
else()
  # 跨平台基线：3.16 足以覆盖 HALCON UseHALCON.cmake 与本工程所需特性
  cmake_minimum_required(VERSION 3.16)
endif()
# When using CMake 3.9 or newer, do not let RPATH affect the install_name on
# macOS.
if(POLICY CMP0068)
  cmake_policy(SET CMP0068 NEW)
endif()
# If policy CMP0091 is available, set it to NEW to enable use of the
# CMAKE_MSVC_RUNTIME_LIBRARY variable.
if(POLICY CMP0091)
  cmake_policy(SET CMP0091 NEW)
endif()
# Start the project here





project(Halcon_YouloBe C CXX )
set(ProjectName Halcon_YouloBe)
# 注意：Visual Studio 为多配置生成器，CMAKE_BUILD_TYPE 在 VS 下不生效；
# 请在 IDE 里切换 Debug/Release，下方通过生成器表达式自动选择对应依赖库。
# 如果使用单配置生成器（Ninja/Makefiles）且未指定，则默认 Debug。
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()
# 禁止生成 Debug/Release 子目录，直接输出到 bin
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_SOURCE_DIR}/bin")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG "${CMAKE_SOURCE_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG "${CMAKE_SOURCE_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}/bin")
# Append HALCON's ikcmake directory to the module path. This requires the
# HALCONEXAMPLES environment variable to be set.
file(TO_CMAKE_PATH "$ENV{HALCONEXAMPLES}" HALCONEXAMPLES)
list(APPEND CMAKE_MODULE_PATH "${HALCONEXAMPLES}/cmake")
# Include common settings for HALCON projects
include(UseHALCON)
# Use the HALCON_AddExtensionPackage function defined in UseHalcon to create
# a HALCON extension package. See the description of HALCON_AddExtensionPackage
# <HALCONEXAMPLES>/cmake/UseHALCON.cmake for details on the allowed parameters.
##这以上都能不能动


HALCON_AddExtensionPackage(Halcon_YouloBe
  DEF_FILES
    def/Halcon_YouloBe.def
  SOURCES
    source/Halcon_OpenVino.cpp
    source/Halcon_YouloBe.c

  CHAPTERS
    userextensions
  CLASSES
    
  OPERATORS
   
)

##三方库包含
set(HALCON_ROOT $ENV{HALCONROOT})   # 转成 CMake 变量，后面用着方便
# Halcon 头文件目录
set(HALCON_INCLUDE_DIR ${HALCON_ROOT}/include/halconcpp)#HALCON_INCLUDE_DIR变量是${HALCON_ROOT}/include的路径

# Halcon 的 lib 目录：按平台选择架构子目录
if(WIN32)
    set(HALCON_LIB_DIR ${HALCON_ROOT}/lib/x64-win64)
elseif(APPLE)
    set(HALCON_LIB_DIR ${HALCON_ROOT}/lib/x64-macosx)
else()
    # Linux：依赖 HALCONARCH 环境变量（典型值 x64-linux / aarch64-linux）
    if(DEFINED ENV{HALCONARCH})
        set(HALCON_LIB_DIR ${HALCON_ROOT}/lib/$ENV{HALCONARCH})
    else()
        set(HALCON_LIB_DIR ${HALCON_ROOT}/lib/x64-linux)
    endif()
endif()

file(GLOB_RECURSE PROJECT_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h")
target_sources(${ProjectName} PRIVATE ${PROJECT_HEADERS})
# 1. 收集 .def 文件
file(GLOB DEF_FILES "${CMAKE_CURRENT_SOURCE_DIR}/def/*.def")
# 2. 添加到目标，但不参与编译（通过 HEADER_FILE_ONLY 属性）
target_sources(${ProjectName} PRIVATE ${DEF_FILES})
# 3. 设置属性：只显示，不编译、不链接
set_source_files_properties(${DEF_FILES} PROPERTIES
    HEADER_FILE_ONLY TRUE   # 关键：不让它参与构建
)


# ==========================================================================
# 三方库（OpenCV / OpenVINO / exiv2）配置
# Windows：使用仓库内 3rd/ 目录下提供的预编译 .lib（保持原有行为不变）
# Linux  ：优先在 3rd/<lib>/lib/linux 下查找；找不到则回退到系统 find_package
# ==========================================================================
if(WIN32)
    target_include_directories(${ProjectName}
        PUBLIC
            $<BUILD_INTERFACE:${HALCON_INCLUDE_DIR}>
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/3rd/opencv/include>
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/3rd/openvino/include>
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/3rd/exiv/include>
    )

    # 三方库按构建配置自动切换（Debug / Release 随 IDE 配置切换，无需手改）
    file(GLOB OPENCV_DEBUG_LIBS     "${CMAKE_CURRENT_SOURCE_DIR}/3rd/opencv/lib/debug/*.lib")
    file(GLOB OPENCV_RELEASE_LIBS   "${CMAKE_CURRENT_SOURCE_DIR}/3rd/opencv/lib/release/*.lib")
    file(GLOB OPENVINO_DEBUG_LIBS   "${CMAKE_CURRENT_SOURCE_DIR}/3rd/openvino/lib/debug/*.lib")
    file(GLOB OPENVINO_RELEASE_LIBS "${CMAKE_CURRENT_SOURCE_DIR}/3rd/openvino/lib/release/*.lib")

    target_link_libraries(${ProjectName}
        ${HALCON_LIB_DIR}/halconcpp.lib
        # exiv2 只提供一份 lib，Debug/Release 共用
        ${CMAKE_CURRENT_SOURCE_DIR}/3rd/exiv/lib/exiv2.lib
        # Debug 配置：链接 debug 目录下的库
        "$<$<CONFIG:Debug>:${OPENCV_DEBUG_LIBS}>"
        "$<$<CONFIG:Debug>:${OPENVINO_DEBUG_LIBS}>"
        # 非 Debug 配置：链接 release 目录下的库
        "$<$<NOT:$<CONFIG:Debug>>:${OPENCV_RELEASE_LIBS}>"
        "$<$<NOT:$<CONFIG:Debug>>:${OPENVINO_RELEASE_LIBS}>"
    )
else()
    # ---------- Linux / macOS ----------
    # 始终包含项目自己的 include 目录与 HALCON 头
    target_include_directories(${ProjectName}
        PUBLIC
            $<BUILD_INTERFACE:${HALCON_INCLUDE_DIR}>
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    )

    # 头文件目录：Linux 优先用 3rd/<lib>/include-linux（仓库内已提供），
    # 不存在时回退到 3rd/<lib>/include（与 Windows 共用一份的情况）。
    function(_pick_include_dir _out _base)
        if(EXISTS "${_base}/include-linux")
            set(${_out} "${_base}/include-linux" PARENT_SCOPE)
        else()
            set(${_out} "${_base}/include" PARENT_SCOPE)
        endif()
    endfunction()

    # 1) OpenCV：先尝试 3rd/opencv/lib/linux，找不到则用系统包
    set(_OPENCV_LOCAL_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rd/opencv/lib/linux")
    file(GLOB _OPENCV_LOCAL_LIBS
        "${_OPENCV_LOCAL_LIB_DIR}/libopencv*.so*"
        "${_OPENCV_LOCAL_LIB_DIR}/libopencv*.a"
    )
    if(_OPENCV_LOCAL_LIBS)
        _pick_include_dir(_OPENCV_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rd/opencv")
        message(STATUS "Using bundled OpenCV from ${_OPENCV_LOCAL_LIB_DIR} (headers: ${_OPENCV_INCLUDE_DIR})")
        target_include_directories(${ProjectName} PUBLIC
            $<BUILD_INTERFACE:${_OPENCV_INCLUDE_DIR}>)
        set(_OPENCV_LIBS ${_OPENCV_LOCAL_LIBS})
    else()
        find_package(OpenCV REQUIRED)
        message(STATUS "Using system OpenCV ${OpenCV_VERSION} (${OpenCV_DIR})")
        target_include_directories(${ProjectName} PUBLIC ${OpenCV_INCLUDE_DIRS})
        set(_OPENCV_LIBS ${OpenCV_LIBS})
    endif()

    # 2) OpenVINO：先尝试 3rd/openvino/lib/linux，找不到则用系统包
    set(_OPENVINO_LOCAL_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rd/openvino/lib/linux")
    file(GLOB _OPENVINO_LOCAL_LIBS
        "${_OPENVINO_LOCAL_LIB_DIR}/libopenvino*.so*"
        "${_OPENVINO_LOCAL_LIB_DIR}/libopenvino*.a"
    )
    if(_OPENVINO_LOCAL_LIBS)
        _pick_include_dir(_OPENVINO_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rd/openvino")
        message(STATUS "Using bundled OpenVINO from ${_OPENVINO_LOCAL_LIB_DIR} (headers: ${_OPENVINO_INCLUDE_DIR})")
        target_include_directories(${ProjectName} PUBLIC
            $<BUILD_INTERFACE:${_OPENVINO_INCLUDE_DIR}>)
        set(_OPENVINO_LIBS ${_OPENVINO_LOCAL_LIBS})
    else()
        find_package(OpenVINO REQUIRED)
        message(STATUS "Using system OpenVINO")
        set(_OPENVINO_LIBS openvino::runtime)
    endif()

    # 3) exiv2：先尝试 3rd/exiv/lib/linux，找不到则用系统包
    set(_EXIV_LOCAL_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rd/exiv/lib/linux")
    file(GLOB _EXIV_LOCAL_LIBS
        "${_EXIV_LOCAL_LIB_DIR}/libexiv2*.so*"
        "${_EXIV_LOCAL_LIB_DIR}/libexiv2*.a"
    )
    if(_EXIV_LOCAL_LIBS)
        _pick_include_dir(_EXIV_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rd/exiv")
        message(STATUS "Using bundled exiv2 from ${_EXIV_LOCAL_LIB_DIR} (headers: ${_EXIV_INCLUDE_DIR})")
        target_include_directories(${ProjectName} PUBLIC
            $<BUILD_INTERFACE:${_EXIV_INCLUDE_DIR}>)
        set(_EXIV_LIBS ${_EXIV_LOCAL_LIBS})
    else()
        find_package(exiv2 QUIET)
        if(exiv2_FOUND)
            message(STATUS "Using system exiv2 ${exiv2_VERSION}")
            set(_EXIV_LIBS exiv2lib)
        else()
            # Fallback: 通过 pkg-config 或裸链接（Ubuntu 上 libexiv2-dev 提供 -lexiv2）
            find_library(EXIV2_LIB NAMES exiv2 REQUIRED)
            message(STATUS "Using exiv2 via -l: ${EXIV2_LIB}")
            set(_EXIV_LIBS ${EXIV2_LIB})
        endif()
    endif()

    # HALCON 共享库：libhalconcpp.so（位于 ${HALCON_LIB_DIR}）
    find_library(HALCONCPP_LIB
        NAMES halconcpp
        PATHS ${HALCON_LIB_DIR}
        NO_DEFAULT_PATH
        REQUIRED
    )

    target_link_libraries(${ProjectName}
        ${HALCONCPP_LIB}
        ${_OPENCV_LIBS}
        ${_OPENVINO_LIBS}
        ${_EXIV_LIBS}
    )

    # 在 Linux 上让生成的 .so 默认隐藏符号，仅导出标记了 visibility("default") 的接口
    set_target_properties(${ProjectName} PROPERTIES
        CXX_VISIBILITY_PRESET hidden
        C_VISIBILITY_PRESET   hidden
        VISIBILITY_INLINES_HIDDEN ON
    )

    # exiv2 0.27.x 仍使用 std::auto_ptr，在 C++17 下只是 deprecated 警告，不致失败；
    # 这里显式以 C++14 编译，对未来切换到 C++17/20 的开发机也保持一致行为。
    target_compile_features(${ProjectName} PRIVATE cxx_std_14)

    # 把 bundled .so 目录烧进 RPATH，HDevelop 加载本扩展时即使没设置
    # LD_LIBRARY_PATH 也能找到 OpenCV / OpenVINO / exiv2。
    # $ORIGIN 是 .so 自身所在目录；同时考虑 bin/ 与 lib/$HALCONARCH/ 两种部署位置。
    set(_RPATH_FLAGS
        "LINKER:-rpath,\$ORIGIN"
        "LINKER:-rpath,\$ORIGIN/../../bin"
        "LINKER:-rpath,\$ORIGIN/../../3rd/opencv/lib/linux"
        "LINKER:-rpath,\$ORIGIN/../../3rd/openvino/lib/linux"
        "LINKER:-rpath,\$ORIGIN/../../3rd/exiv/lib/linux"
        "LINKER:-rpath,\$ORIGIN/../3rd/opencv/lib/linux"
        "LINKER:-rpath,\$ORIGIN/../3rd/openvino/lib/linux"
        "LINKER:-rpath,\$ORIGIN/../3rd/exiv/lib/linux"
        "LINKER:--disable-new-dtags"
    )
    target_link_options(${ProjectName} PRIVATE ${_RPATH_FLAGS})
    foreach(_iface_tgt ${ProjectName}cint ${ProjectName}cppint)
        if(TARGET ${_iface_tgt})
            target_link_options(${_iface_tgt} PRIVATE ${_RPATH_FLAGS})
        endif()
    endforeach()

    # HALCON 期望按 lib/$HALCONARCH/libHalcon_YouloBe*.so 加载扩展包；
    # 构建产物在 bin/，构建后把对应 .so 软链到 lib/${HALCON_ARCHITECTURE}/。
    # 三个目标各自挂 POST_BUILD，避免在主目标完成时引用到尚未生成的 .so。
    set(_HEXT_LIB_DIR "${CMAKE_SOURCE_DIR}/lib/${HALCON_ARCHITECTURE}")
    foreach(_tgt ${ProjectName} ${ProjectName}cint ${ProjectName}cppint)
        if(TARGET ${_tgt})
            add_custom_command(TARGET ${_tgt} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E make_directory "${_HEXT_LIB_DIR}"
                COMMAND ${CMAKE_COMMAND} -E create_symlink
                    "../../bin/$<TARGET_FILE_NAME:${_tgt}>"
                    "${_HEXT_LIB_DIR}/$<TARGET_FILE_NAME:${_tgt}>"
                COMMENT "Linking $<TARGET_FILE_NAME:${_tgt}> into lib/${HALCON_ARCHITECTURE}"
            )
        endif()
    endforeach()
endif()


