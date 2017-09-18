#include "lin_alg_classes.h"
#include <chrono>

using basetype = float;

int main() {
    const size_t matrix_size = 4;
    Matrix<basetype, matrix_size, matrix_size> test3(1.0, true), test4;
    Matrix<basetype, 1, matrix_size> vec1(1.0);
    Matrix<basetype, matrix_size, 1> vec2(1.0);

    test4 = test3;

    test4(0, 1) = 2;
    test4(1, 0) = 2;
//
//    std::cout << test5 << std::endl;
//    std::cout << test6 << std::endl;

    long num_repeats = 10000;

    {
        Matrix<basetype, 1, matrix_size> result;
        auto t1 = std::chrono::high_resolution_clock::now();
        {
            for (auto i = 0; i < num_repeats; ++i) {
                result = vec1 * test4;
            }
        }
        auto t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> ms = t2 - t1;
        std::cout << "determinant result "
                  << result.square_norm() << " took " << ms.count()/num_repeats << " ms" << std::endl;
        std::cout << "MFLOPS " << (matrix_size * matrix_size * 3) * 1e-3 * num_repeats / ms.count() << std::endl;
    }

    {
        Matrix<basetype, matrix_size, 1> result;
        auto t1 = std::chrono::high_resolution_clock::now();
        {
            for (auto i = 0; i < num_repeats; ++i) {
                result = test4 * vec2;
            }
        }
        auto t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> ms = t2 - t1;
        std::cout << "determinant result "
                  << result.transpose().square_norm() << " took " << ms.count()/num_repeats << " ms" << std::endl;
        std::cout << "MFLOPS " << (matrix_size * matrix_size * 3) * 1e-3 * num_repeats / ms.count() << std::endl;
    }

    return 0;
}