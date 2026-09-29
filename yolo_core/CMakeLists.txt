#=============================================================================
#  yolo_core/CMakeLists.txt — YOLO26-seg OpenVINO 推理核心库
#=============================================================================
# 形态：静态库 yolo_core，直接链入 Halcon_YouloBe 扩展包，supply 层直接调 C++ API；
# 也可 install 后给任意 C++ 工程复用（无需 HALCON）。
#   公开头：include/yolo/yolo.hpp
#   依赖：OpenCV + OpenVINO（PUBLIC 链出）
#
# Build options:
#    YOLO_BUILD_TESTS  ON/OFF  (default: ON 仅独立构建时；作为子项目默认 OFF)
#=============================================================================

cmake_minimum_required(VERSION 3.21)
project(yolo_core
  VERSION     1.0.0
  DESCRIPTION "YOLO26 instance segmentation inference core (OpenVINO, HALCON-free)"
  LANGUAGES   CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(PROJECT_IS_TOP_LEVEL)
  set(_yolo_default ON)
else()
  set(_yolo_default OFF)
endif()
option(YOLO_BUILD_TESTS "Build yolo_core tests" ${_yolo_default})
unset(_yolo_default)

find_package(OpenCV REQUIRED)
find_package(OpenVINO REQUIRED)

add_library(yolo_core STATIC
  src/yolo.cpp
)

target_include_directories(yolo_core
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(yolo_core
  PUBLIC
    ${OpenCV_LIBS}
    openvino::runtime
)
# OpenCV 的导入目标自带接口头路径；仅构建期需要显式 include（安装消费方经由
# find_dependency(OpenCV) 获得）。不加 BUILD_INTERFACE 会导致 install(EXPORT) 报错。
target_include_directories(yolo_core PUBLIC
  $<BUILD_INTERFACE:${OpenCV_INCLUDE_DIRS}>)

#=============================================================================
#  Install & export — 外部 CMake 工程免 HALCON 复用：
#    find_package(yolo_core CONFIG REQUIRED)  # 自动 find_dependency OpenCV/OpenVINO
#    target_link_libraries(app PRIVATE yolo::yolo_core)
#=============================================================================
add_library(yolo::yolo_core ALIAS yolo_core)
include(GNUInstallDirs)
install(TARGETS yolo_core
  EXPORT yolo_core-targets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/yolo_core-config.cmake"
[=[include(CMakeFindDependencyMacro)
find_dependency(OpenCV)
find_dependency(OpenVINO)
include("${CMAKE_CURRENT_LIST_DIR}/yolo_core-targets.cmake")
]=])
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/yolo_core-config.cmake"
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/yolo_core")
install(EXPORT yolo_core-targets
  FILE yolo_core-targets.cmake
  NAMESPACE yolo::
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/yolo_core")
