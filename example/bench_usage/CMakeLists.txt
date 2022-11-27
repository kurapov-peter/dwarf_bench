add_executable(bench_usage main.cpp)
target_link_libraries(bench_usage dbench)
target_include_directories(bench_usage PRIVATE ${PROJECT_SOURCE_DIR})