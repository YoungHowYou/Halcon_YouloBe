#=============================================================================
#  CMakeLists.txt — HALCON Extension Package "Halcon_YouloBe"
#=============================================================================
#  依赖管理：全部通过 vcpkg
#    OpenCV / OpenVINO / exiv2  →  vcpkg（manifest 模式，见 vcpkg.json）
#    HALCON                     →  HALCONROOT 环境变量
#
#  用法:
#    cmake -B build             # 工具链从 VCPKG_ROOT 环境变量自动拾取
#    cmake --build build --config Release
#
#  架构：YOLO26-seg 推理核心在 yolo_core/ 静态库（yolo::Model，HALCON-free，
#        可 install 后给任意 C++ 工程复用），扩展包 supply 层只做参数/图像转换。
#=============================================================================

cmake_minimum_required(VERSION 3.21)

#=============================================================================
#  Options  选项
#=============================================================================
option(HALCON_XL "Use HALCON XL" OFF)

# .NET 支持：Windows 默认 ON，Linux/macOS 默认 OFF
if(WIN32)
  option(HALCON_DOTNET "Build .NET support" ON)
else()
  option(HALCON_DOTNET "Build .NET support" OFF)
endif()

# ---- vcpkg toolchain 自动拾取 ----
if(DEFINED ENV{VCPKG_ROOT} AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
  set(_vcpkg_tc "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake")
  if(EXISTS "${_vcpkg_tc}")
    set(CMAKE_TOOLCHAIN_FILE "${_vcpkg_tc}" CACHE STRING "vcpkg toolchain")
  endif()
  unset(_vcpkg_tc)
endif()

#=============================================================================
#  Policies  策略
#=============================================================================
if(POLICY CMP0068)
  cmake_policy(SET CMP0068 NEW)   # macOS install_name 不受 RPATH 影响
endif()
if(POLICY CMP0091)
  cmake_policy(SET CMP0091 NEW)   # 启用 CMAKE_MSVC_RUNTIME_LIBRARY
endif()

#=============================================================================
#  Project  项目
#=============================================================================
project(Halcon_YouloBe
  VERSION     2.0.0
  DESCRIPTION "HALCON Extension — OpenVINO YOLO26 inference / OpenCV / EXIF"
  LANGUAGES   C CXX
)
set(ProjectName Halcon_YouloBe)

# .NET / CSharp 支持（可选；无 CSharp 编译器时自动禁用，不阻断构建）
if(HALCON_DOTNET)
  enable_language(CSharp OPTIONAL)
  if(NOT CMAKE_CSharp_COMPILER)
    message(WARNING "CSharp compiler not found — .NET extension disabled. "
                    "Install .NET SDK or pass -DHALCON_DOTNET=OFF")
    set(HALCON_DOTNET OFF)
  endif()
endif()

# 单配置生成器（Ninja/Makefiles）默认 Release；VS 多配置在 IDE/--config 里切
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

#=============================================================================
#  Platform detection & output directories  平台检测 & 输出目录
#=============================================================================
if(WIN32)
  set(HALCON_PLATFORM     "x64-win64")
  set(_PKG_OUTPUT_DIR     "${CMAKE_SOURCE_DIR}/bin")
elseif(APPLE)
  set(HALCON_PLATFORM     "x64-macos")
  set(_PKG_OUTPUT_DIR     "${CMAKE_SOURCE_DIR}/lib/${HALCON_PLATFORM}")
else()
  # Linux：依赖 HALCONARCH 环境变量（典型值 x64-linux / aarch64-linux）
  if(DEFINED ENV{HALCONARCH})
    set(HALCON_PLATFORM   "$ENV{HALCONARCH}")
  else()
    set(HALCON_PLATFORM   "x64-linux")
  endif()
  set(_PKG_OUTPUT_DIR     "${CMAKE_SOURCE_DIR}/lib/${HALCON_PLATFORM}")
endif()

message(STATUS "HALCON platform : ${HALCON_PLATFORM}")
message(STATUS "Output directory: ${_PKG_OUTPUT_DIR}")

# 所有构建产物统一输出到 _PKG_OUTPUT_DIR（不分 Debug/Release 子目录）
foreach(_cfg "" _DEBUG _RELEASE _MINSIZEREL _RELWITHDEBINFO)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY${_cfg} "${_PKG_OUTPUT_DIR}")
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY${_cfg} "${_PKG_OUTPUT_DIR}")
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY${_cfg} "${_PKG_OUTPUT_DIR}")
endforeach()
unset(_cfg)

#=============================================================================
#  HALCON setup
#=============================================================================
file(TO_CMAKE_PATH "$ENV{HALCONEXAMPLES}" HALCONEXAMPLES)
list(APPEND CMAKE_MODULE_PATH "${HALCONEXAMPLES}/cmake")
include(UseHALCON)

set(HALCON_ROOT        "$ENV{HALCONROOT}")
set(HALCON_INCLUDE_DIR "${HALCON_ROOT}/include/halconcpp")
set(HALCON_LIB_DIR     "${HALCON_ROOT}/lib/${HALCON_PLATFORM}")

#=============================================================================
#  yolo_core：YOLO26-seg 推理核心静态库（HALCON-free，可独立安装复用）
#=============================================================================
add_subdirectory(yolo_core)

#=============================================================================
#  扩展包主目标
#=============================================================================
HALCON_AddExtensionPackage(${ProjectName}
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

#=============================================================================
#  Dependencies — 全部通过 vcpkg
#=============================================================================
find_package(OpenCV REQUIRED)
find_package(OpenVINO REQUIRED)
find_package(exiv2 CONFIG REQUIRED)

#=============================================================================
#  Include directories & 编译选项
#=============================================================================
target_include_directories(${ProjectName}
  PUBLIC
    $<BUILD_INTERFACE:${HALCON_INCLUDE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

# OpenVINO 2025+ / exiv2 0.28+ 头文件需要 C++17
target_compile_features(${ProjectName} PRIVATE cxx_std_17)

# IDE 辅助：头文件 & .def 文件在 IDE 中可见
file(GLOB_RECURSE PROJECT_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h")
target_sources(${ProjectName} PRIVATE ${PROJECT_HEADERS})

file(GLOB DEF_FILES "${CMAKE_CURRENT_SOURCE_DIR}/def/*.def")
target_sources(${ProjectName} PRIVATE ${DEF_FILES})
set_source_files_properties(${DEF_FILES} PROPERTIES HEADER_FILE_ONLY TRUE)

#=============================================================================
#  Link libraries  链接
#  注意：UseHALCON.cmake 的 HALCON_AddExtensionPackage 内部使用 target_link_libraries
#  无关键字（plain）签名，此处必须保持 plain 签名，混用会报 CMP0023 错误
#=============================================================================
if(WIN32)
  target_link_libraries(${ProjectName}
    ${HALCON_LIB_DIR}/halconcpp.lib
    yolo_core
    Exiv2::exiv2lib
  )
else()
  find_package(Threads REQUIRED)
  find_library(HALCONCPP_LIB
    NAMES halconcpp
    PATHS ${HALCON_LIB_DIR}
    NO_DEFAULT_PATH
    REQUIRED
  )
  target_link_libraries(${ProjectName}
    ${HALCONCPP_LIB}
    yolo_core
    Exiv2::exiv2lib
    Threads::Threads
    ${CMAKE_DL_LIBS}
  )
endif()

#=============================================================================
#  Post-build  构建后步骤（Windows）
#=============================================================================
if(WIN32)
  # 1) 同步生成的算子签名数据库 → 包 help/。
  #    HALCON 从 help/operators_en_US.num 读取各算子签名做调用校验；
  #    若不同步，改了参数个数的算子会按旧签名校验失败而被静默跳过（不派发）。
  add_custom_command(TARGET ${ProjectName} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
      "${CMAKE_BINARY_DIR}/help"
      "${CMAKE_CURRENT_SOURCE_DIR}/help"
    COMMENT "Syncing operator signature database → help/"
  )

  # 2) OpenVINO 插件/前端 DLL（openvino*.dll，运行时按目录加载，非链接期依赖，
  #    z-applocal 不会拷贝；缺了会报 "Available frontends:" 为空）
  add_custom_command(TARGET ${ProjectName} POST_BUILD
    COMMAND ${CMAKE_COMMAND}
      -DCONFIG=$<CONFIG>
      "-DVCPKG_BIN_RELEASE=${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/bin"
      "-DVCPKG_BIN_DEBUG=${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/debug/bin"
      "-DOUT_DIR=${_PKG_OUTPUT_DIR}"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/copy_openvino_plugins.cmake"
  )
else()
  # ---- Linux / macOS ----
  # 隐藏符号，仅导出标记 visibility("default") 的接口
  set_target_properties(${ProjectName} PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    C_VISIBILITY_PRESET   hidden
    VISIBILITY_INLINES_HIDDEN ON
  )

  # vcpkg 的 lib 目录烧进 RPATH，HDevelop 加载本扩展时无需 LD_LIBRARY_PATH
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
