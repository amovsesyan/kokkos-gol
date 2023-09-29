project(kokkos-game-of-life)

find_package(Kokkos REQUIRED)
set(Kokkos_ROOT /home/amovsesy/spack/opt/spack/linux-rhel8-zen3/gcc-11.3.0/kokkos-3.7.01-75nmoutfjze35jd7frgtzbswxq63y2v6/lib64/cmake/Kokkos)
# set(CMAKE_CXX_COMPILER /home/amovsesy/spack/opt/spack/linux-rhel8-zen3/gcc-11.3.0/kokkos-nvcc-wrapper-3.2.00-i5v6unjdblj2y45pfo7tskjjk5f6m66u/bin/nvcc_wrapper)
set(Kokkos_ENABLE_CUDA OFF)
add_executable(GOL src/game_of_life.cpp)

target_link_libraries(GOL Kokkos::kokkos)

