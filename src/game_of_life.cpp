#include <stdio.h>
#include <iostream>

#include <Kokkos_Core.hpp>

using grid_type = Kokkos::View<int**>;

struct Iterate_GOL {
    grid_type initial_grid, new_grid;
    int num_rows, num_cols, grid_size;

    Iterate_GOL(grid_type grid_old, grid_type grid_new): initial_grid(grid_old), new_grid(grid_new) {
        num_rows = initial_grid.extent(0);
        num_cols = initial_grid.extent(1);
        grid_size = num_rows * num_cols;
        printf("finished iterate constructor\n");
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const int i ) const {
        // printf("in iterate () operator at id %d\n", i);
        // printf("num_rows: %d, num_cols: %d, grid_size: %d", num_rows, num_cols, grid_size);

        // return;
        for(int j = i; j < grid_size; j += (i + 1)) {
            // printf("in partition %d", j);
            //get row and column
            int row = j / num_cols;
            int col = j % num_cols;
            // printf(" -- working on row %d, col %d\n", row, col);
            //calculate num alive neighbours (8 total neighbours)
            int num_alive = 0;
            num_alive += initial_grid(((row - 1) % num_rows), ((col - 1) % num_cols));
            num_alive += initial_grid(((row - 1) % num_rows), col);
            num_alive += initial_grid(((row - 1) % num_rows), ((col + 1) % num_cols));

            num_alive += initial_grid(((row + 1) % num_rows), ((col - 1) % num_cols));
            num_alive += initial_grid(((row + 1) % num_rows), col);
            num_alive += initial_grid(((row + 1) % num_rows), ((col + 1) % num_cols));
            
            num_alive += initial_grid(row, ((col - 1) % num_cols));
            num_alive += initial_grid(row, ((col + 1) % num_cols));

            // check if cell is alive
            int alive = initial_grid(row, col);
            
            //calculate if cell is dead or alive in next iteration
            if(alive && (num_alive == 2 || num_alive == 3)) { // alive cell survives
                new_grid(row, col) = 1;
            } else if(!alive && num_alive == 3) { // dead cell becomes alive
                new_grid(row, col) = 1;
            } else { // cell die
                new_grid(row, col) = 0;
            }
        }
    }
};

struct Reset_Grid {
    grid_type old_grid, new_grid;
    int num_rows, num_cols, grid_size;
    Reset_Grid(grid_type grid_old, grid_type grid_new): old_grid(grid_old), new_grid(grid_new) {
        num_rows = old_grid.extent(0);
        num_cols = old_grid.extent(1);
        grid_size = num_rows * num_cols;
        printf("num_rows: %d, num_cols: %d, grid_size: %d", num_rows, num_cols, grid_size);
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const int i) const {
        // std::cout << "Hello from i = " << i << std::endl;
        
        for(int j = i; j < grid_size; j+= (i + 1)) {
            int row = j / num_cols, col = j % num_cols;
            old_grid(row, col) = new_grid(row, col);
        }
    }
};

void print_grid(grid_type grid, int iter) {
    printf("Grid at iteration %d\n", iter);
    for(int i = 0; i < grid.extent(0); i++) {
        for(int j = 0; j < grid.extent(1); j++) {
            printf(" %d ", grid(i, j));
        }
        printf("\n");
    }
    printf("\n");
}

void set_up_grid(grid_type grid) {
    for(int i = 0; i < grid.extent(0); i++) {
        for(int j = 0; j < grid.extent(1); j++) {
            grid(i, j) = rand() % 2;
        }
    }
}

int main(int argc, char** argv) {
    printf("Game of Life\n");
    std::cout << "(... using Kokkos)" << std::endl;
    int num_rows = 16, num_cols = 16;
    int num_iterations = 3;
    int num_workers = 8;
    Kokkos::initialize(argc, argv); {
        grid_type init_grid ("initial grid", num_rows, num_cols);
        grid_type update_grid ("updated grid", num_rows, num_cols);

        set_up_grid(init_grid);

        for(int i = 0; i < num_iterations; i++) {
            print_grid(init_grid, i);
            printf("starting update\n");
            Kokkos::parallel_for(num_workers, Iterate_GOL(init_grid, update_grid));
            Kokkos::fence();
            printf("starting reset\n");
            Kokkos::parallel_for(num_workers, Reset_Grid(init_grid, update_grid));
            Kokkos::fence();
        }
    }
    Kokkos::finalize();
    return 0;
}