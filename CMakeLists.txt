# CMakeLists.txt for HALCON extension package 'halconuser' example
# Copyright (c) 2019-2023 by MVTec Software GmbH

# vcpkg 工具链：未显式指定时，若环境变量 VCPKG_ROOT 存在则自动启用（manifest 模式，见 vcpkg.json）
if(NOT DEFINED CMAKE_TOOLCHAIN_FILE AND DEFINED ENV{VCPKG_ROOT} AND
   EXISTS "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake")
  set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
      CACHE STRING "vcpkg toolchain file" FORCE)
endif()

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
# 禁止生成 Debug/Release 子目录。
#   Windows：保持原有行为，统一输出到 bin/
#   Linux  ：直接输出到 lib/<HALCONARCH>（默认 x64-linux），HALCON 直接从此处加载扩展
if(WIN32)
  set(_OUTPUT_DIR "${CMAKE_SOURCE_DIR}/bin")
else()
  if(DEFINED ENV{HALCONARCH})
    set(_OUTPUT_DIR "${CMAKE_SOURCE_DIR}/lib/$ENV{HALCONARCH}")
  else()
    set(_OUTPUT_DIR "${CMAKE_SOURCE_DIR}/lib/x64-linux")
  endif()
endif()
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY         "${_OUTPUT_DIR}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG   "${_OUTPUT_DIR}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${_OUTPUT_DIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY         "${_OUTPUT_DIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG   "${_OUTPUT_DIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE "${_OUTPUT_DIR}")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY         "${_OUTPUT_DIR}")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG   "${_OUTPUT_DIR}")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE "${_OUTPUT_DIR}")
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

## =====================================================================
## 依赖管理：vcpkg（manifest 模式，依赖清单见 vcpkg.json）
##   OpenCV 4.12 / OpenVINO 2026.3 / exiv2 0.28 均由 vcpkg 安装定位，
##   不再使用仓库内 3rd/ 预编译库。
## =====================================================================

# Halcon 头文件目录
set(HALCON_ROOT $ENV{HALCONROOT})
set(HALCON_INCLUDE_DIR ${HALCON_ROOT}/include/halconcpp)

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

# ---- vcpkg 依赖（find_package 目标） ----
find_package(OpenCV REQUIRED)
find_package(OpenVINO REQUIRED)
find_package(exiv2 CONFIG REQUIRED)

# 项目头文件 + .def 文件（仅展示不参与编译）
file(GLOB_RECURSE PROJECT_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h")
target_sources(${ProjectName} PRIVATE ${PROJECT_HEADERS})
file(GLOB DEF_FILES "${CMAKE_CURRENT_SOURCE_DIR}/def/*.def")
target_sources(${ProjectName} PRIVATE ${DEF_FILES})
set_source_files_properties(${DEF_FILES} PROPERTIES
    HEADER_FILE_ONLY TRUE
)

target_include_directories(${ProjectName}
    PUBLIC
        $<BUILD_INTERFACE:${HALCON_INCLUDE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        ${OpenCV_INCLUDE_DIRS}
)

# OpenVINO 2025+ / exiv2 0.28+ 头文件需要 C++17
target_compile_features(${ProjectName} PRIVATE cxx_std_17)

if(WIN32)
    # 注意：UseHALCON.cmake 的 HALCON_AddExtensionPackage 内部使用 target_link_libraries
    # 无关键字（plain）签名，此处必须保持 plain 签名，混用会报 CMP0023 错误
    target_link_libraries(${ProjectName}
            ${HALCON_LIB_DIR}/halconcpp.lib
            ${OpenCV_LIBS}
            openvino::runtime
            Exiv2::exiv2lib
    )

    # 把 vcpkg 运行时 DLL 拷到扩展包输出目录（bin/），保证 HDevelop 加载本扩展
    # 时能解析依赖（Windows DLL 搜索优先目标文件所在目录）：
    #   1) 链接期依赖（opencv_world / exiv2 / tbb / openvino 等）——由 vcpkg 的
    #      z-applocal 后置步骤自动处理，无需手动拷贝
    #   2) OpenVINO 插件/前端 DLL（openvino*.dll，运行时按目录加载，非链接期依赖）
    add_custom_command(TARGET ${ProjectName} POST_BUILD
        COMMAND ${CMAKE_COMMAND}
            -DCONFIG=$<CONFIG>
            "-DVCPKG_BIN_RELEASE=${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/bin"
            "-DVCPKG_BIN_DEBUG=${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/debug/bin"
            "-DOUT_DIR=${_OUTPUT_DIR}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/copy_openvino_plugins.cmake"
    )
else()
    # ---------- Linux / macOS ----------
    # HALCON 共享库：libhalconcpp.so（位于 ${HALCON_LIB_DIR}）
    find_library(HALCONCPP_LIB
        NAMES halconcpp
        PATHS ${HALCON_LIB_DIR}
        NO_DEFAULT_PATH
        REQUIRED
    )

    target_link_libraries(${ProjectName}
            ${HALCONCPP_LIB}
            ${OpenCV_LIBS}
            openvino::runtime
            Exiv2::exiv2lib
    )

    # 在 Linux 上让生成的 .so 默认隐藏符号，仅导出标记了 visibility("default") 的接口
    set_target_properties(${ProjectName} PROPERTIES
        CXX_VISIBILITY_PRESET hidden
        C_VISIBILITY_PRESET   hidden
        VISIBILITY_INLINES_HIDDEN ON
    )

    # 把 vcpkg 的 lib 目录烧进 RPATH，HDevelop 加载本扩展时即使没设置
    # LD_LIBRARY_PATH 也能找到 OpenCV / OpenVINO / exiv2。
    set(_RPATH_FLAGS
        "LINKER:-rpath,${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/lib"
        "LINKER:-rpath,\$ORIGIN"
        "LINKER:--disable-new-dtags"
    )
    target_link_options(${ProjectName} PRIVATE ${_RPATH_FLAGS})
    foreach(_iface_tgt ${ProjectName}cint ${ProjectName}cppint)
        if(TARGET ${_iface_tgt})
            target_link_options(${_iface_tgt} PRIVATE ${_RPATH_FLAGS})
        endif()
    endforeach()
endif()
