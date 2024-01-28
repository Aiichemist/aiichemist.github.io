# CMake简介
CMake是一个跨平台的开源构建工具，它用于管理和自动生成软件项目的构建过程。CMake的主要目标是简化和规范化项目的构建流程，使开发人员能够更容易地跨不同操作系统和编译器平台构建其软件。
1. 跨平台：CMake支持多种操作系统，包括Windows、Linux、macOS等，以及多种编译器，如GCC、Clang、Visual Studio等。这使得开发人员可以在不同平台上轻松地构建他们的项目。
2. 自动生成构建文件：CMake使用CMakeLists.txt文件来描述项目的结构和构建过程。通过这些文件，CMake可以生成适用于不同编译器和操作系统的构建文件，如Makefile、Visual Studio项目文件等。
3. 模块化和可扩展：CMake允许开发人员使用模块化的方式组织项目，以便更容易管理项目的不同部分。此外，CMake支持自定义的CMake模块，可以轻松扩展其功能。
4. 支持多种构建系统：CMake支持多种构建系统，包括本地编译、交叉编译和交叉构建。这使得开发人员可以针对不同目标和环境进行构建。
5. 集成测试支持：CMake还具有用于构建和运行测试的内置支持，这有助于确保项目的质量和稳定性。
6. 社区支持：CMake具有广泛的社区支持，有大量文档、教程和社区资源可供开发人员参考和获取帮助。
CMake已经成为许多开源项目和商业软件项目的首选构建工具之一，因为它可以简化跨平台构建的复杂性，提高项目的可维护性，并提供一致的构建流程。
## CMake构建项目流程：
- 项目源码
- 编写CMakeLists.txt
- 执行`cmake`：生成Makefile文件(包含构建项目所需指令
- 执行`make`：批处理命令
- 生成目标文件(可执行文件/静态库/动态库)

# CMake使用
预设示例项目目录结构如下：
```shell
$ tree
.
├── func1.c
├── func2.c
├── func3.c
├── head.h
└── main.c
```
## 注释
### 注释行
```Cmake
# 注释行
```
## 最低版本
```cmake
# 指定使用的最低版本
cmake_minimum_required(VERSION 3.0)
```
- `cmake_minimum_required`：指定使用的 CMake 的最低版本
	- 可选，非必须，如果不加可能会有警告
## 项目名称
```cmake
# 定义项目名称，并可指定工程的版本、工程描述、web主页地址、支持的语言（默认情况支持所有语言）
project(Demo)
# 完整版
project(<PROJECT-NAME>
       [VERSION <major>[.<minor>[.<patch>[.<tweak>]]]]
       [DESCRIPTION <project-description-string>]
       [HOMEPAGE_URL <url-string>]
       [LANGUAGES <language-name>...])
```
## 生成可执行程序
```cmake
add_executable(可执行程序名 源文件名)
# example
add_executable(app func1.c func2.c func3.c main.c)
```

## SET
存储文件名对应的字符串，方便重复使用
```cmake
# SET 指令的语法是：
# [] 中的参数为可选项, 如不需要可以不写
SET(VAR [VALUE] [CACHE TYPE DOCSTRING [FORCE]])
# example
SET(SRC_LIST func1.c func2.c func3.c main.c)
add_executable(app ${SRC_LIST})
```
- `VAR`：变量名
- `VALUE`：变量值
## build目录
```shell
mkdir build
cd build # 移动到makefile文件的生成目录
cmake .. # CmakeLists.txt位于build的上一级目录
```
运行结果：
```shell
$ tree build -L 1
build
├── CMakeCache.txt
├── CMakeFiles
├── cmake_install.cmake
└── Makefile

1 directory, 3 files
```
## 指定使用的C++标准
在编译命令中指定使用标准
```shell
$ g++ *.cpp -std=c++11 -o app
```
在CMkae中指定C++标准
```cmake
# 通过set命令指定为11/14/17
set(CMAKE_CXX_STANDARD 17)
```
在执行cmake命令时指定
```shell
# path_to_CMakrLists.txt为对应文件路径
cmake path_to_CMakeLists.txt -DCMAKE_CXX_STANDARD=14)
```
## 指定输出路径
```cmake
# 定义绝对路径路径(或不存在会自动创建)
# 若使用相对路径，则基于makefile文件所在目录
set(PATH /Home/Documents)
# 指定可执行文件的输出路径
set(EXECUTABLE_OUTPUT_PATH ${PATH}/bin)
```
## 搜索文件
### aux_source_directory()
```cmake
aux_source_directory(< dir >< variable >)
# example
aux_source_directory(./src SRC_LIST)# SRC_LIST为变量而非路径
add_executable(app ${SRC_LIST})
```
- `dir`：要搜索的目录
- `variable`：将从dir目录下搜索到的源文件列表存储到该**变量**中
### file()
```cmake
# GLOB/GLOB_RECURSE 二选一
file(GLOB/GLOB_RECURSE 变量名 要搜索的文件路径和文件类型)
# example
file(GLOB SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)

```
- `GLOB`: 将指定目录下搜索到的满足条件的所有文件名生成一个列表，并将其存储到变量中。
- `GLOB_RECURSE`：递归搜索指定目录，将搜索到的满足条件的文件名生成一个列表，并将其存储到变量中。
---
更新项目目录结构如下：
```shell
$ tree
$ tree
.
├── build
├── CMakeLists.txt
├── include
│   └── head.h
└── src
    ├── add.cpp
    ├── div.cpp
    ├── main.cpp
    ├── mult.cpp
    └── sub.cpp
```
## 包含头文件
在编译项目源文件的时候，很多时候都需要将源文件对应的头文件路径指定出来，这样才能保证在编译过程中编译器能够找到这些头文件，并顺利通过编译。
```cmake
# 设置要包含的目录
include_directories(HeadPath)
# example
cmake_minimum_required(VERSION 3.0)
project(DEMO)
set(CMAKE_CXX_STANDARD 17)
set(PATH /Home/Documents)
set(EXECUTABLE_OUTPUT_PATH ${PATH}/out/)
include_directories(${PROJECT_SOURCE_DIR}/include)
file(GLOB SRC_LIST ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
add_executable(app  ${SRC_LIST})
```
- `PROJECT_SOURCE_DIR`：一般是工程的根目录
## 生成静态库/动态库
### 生成静态库
```cmake
add_library(库名称 STATIC 源文件1 [源文件2] ...) 
# example
cmake_minimum_required(VERSION 3.0)
project(DEMO)
include_directories(${PROJECT_SOURCE_DIR}/include)
file(GLOB SRC_LIST "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
add_library(demo STATIC ${SRC_LIST})
```
最终生成对应的静态库文件`libdemo.a`(Linux)、`libdemo.lib`(Windows)
### 生成动态库
```cmake
add_library(库名称 SHARED 源文件1 [源文件2] ...) 
# example
cmake_minimum_required(VERSION 3.0)
project(DEMO)
include_directories(${PROJECT_SOURCE_DIR}/include)
file(GLOB SRC_LIST "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
add_library(demo SHARED ${SRC_LIST})
```
最终生成对应的动态库文件`libdemo.so`(Linux)、`libdemo.dll`(Windows)




# 参考文章
[CMake 保姆级教程（上）](https://subingwen.cn/cmake/CMake-primer/)
[CMake 保姆级教程（下）](https://subingwen.cn/cmake/CMake-advanced/?highlight=cmake)