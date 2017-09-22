#include "lin_alg_classes.h"
#include <chrono>
#include "emmintrin.h"

using basetype = float;

class Vector2d {
private:
    __m128d a;
public:

    Vector2d() {
        a = _mm_set_pd(0.0, 0.0);
    }

    Vector2d(double x, double y) {
        a = _mm_set_pd(x, y);
    }

    explicit Vector2d(__m128d&& x) {
        a = x;
    };

    inline friend Vector2d operator+(const Vector2d& lhs, const Vector2d& rhs) {
        return Vector2d(_mm_add_pd(lhs.a, rhs.a));
    }

    inline friend Vector2d operator-(const Vector2d& lhs, const Vector2d& rhs) {
        return Vector2d(_mm_sub_pd(lhs.a, rhs.a));
    }

    inline friend double operator*(const Vector2d& lhs, const Vector2d& rhs) {
        double intermediate_result[2];
        _mm_store_pd(intermediate_result, _mm_mul_pd(lhs.a, rhs.a));
        return intermediate_result[0] + intermediate_result[1];
    }

    inline friend Vector2d operator/(const Vector2d& lhs, const Vector2d& rhs) = delete;

    inline friend Vector2d operator/(const Vector2d& lhs, const double& rhs) {
        return Vector2d(_mm_div_pd(lhs.a, _mm_set_pd(rhs, rhs)));
    }

    inline friend Vector2d operator*(const Vector2d& lhs, const double& rhs) {
        return Vector2d(_mm_mul_pd(lhs.a, _mm_set_pd(rhs, rhs)));
    }

    inline friend Vector2d operator*(const double& lhs, const Vector2d& rhs) {
        return Vector2d(_mm_mul_pd(_mm_set_pd(lhs, lhs), rhs.a));
    }

    Vector2d& operator+=(const double& rhs) {
        this->a = _mm_add_pd(this->a, _mm_set_pd(rhs, rhs));
        return *this;
    }

    Vector2d& operator-=(const double& rhs) {
        this->a = _mm_sub_pd(this->a, _mm_set_pd(rhs, rhs));
        return *this;
    }

    Vector2d& operator+=(const Vector2d& rhs) {
        this->a = _mm_add_pd(this->a, rhs.a);
        return *this;
    }

    Vector2d& operator-=(const Vector2d& rhs) {
        this->a = _mm_sub_pd(this->a, rhs.a);
        return *this;
    }

    Vector2d& operator*=(const double& rhs) {
        this->a = _mm_mul_pd(this->a, _mm_set_pd(rhs, rhs));
        return *this;
    }

    Vector2d& operator/=(const double& rhs) {
        this->a = _mm_div_pd(this->a, _mm_set_pd(rhs, rhs));
        return *this;
    }

    double operator()(std::size_t index) const {
        double intermediate[2];
        _mm_store_pd(intermediate, this->a);
        return intermediate[index];
    }

    double operator[](std::size_t index) const {
        double intermediate[2];
        _mm_store_pd(intermediate, this->a);
        return intermediate[index];
    }

    friend std::ostream& operator<<(std::ostream& os, const Vector2d& out) {
        os << "{" << out[0] << ", " << out[1] << "}";
    };
};

int main() {
    Vector2d a{1.0, 1.0}, base_a{1.0, 1.0};
    Matrix<double, 2, 1> b, base_b;
    b(0, 0) = 1.0;
    b(1, 0) = 1.0;
    base_b = b;

    uint64_t repeats = 1000000;

    {
        auto t1 = std::chrono::high_resolution_clock::now();
        for (auto i = 0; i < repeats; ++i) {
            a += base_a;
        }
        auto t2 = std::chrono::high_resolution_clock::now();
        auto time_taken = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1);
        std::cout << "Time taken per rep: " << double(time_taken.count()) / repeats << "ns; Result: " << a << std::endl;
    }

    {
        auto t1 = std::chrono::high_resolution_clock::now();
        for (auto i = 0; i < repeats; ++i) {
            b += base_b;
        }
        auto t2 = std::chrono::high_resolution_clock::now();
        auto time_taken = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1);
        std::cout << "Time taken per rep: " << double(time_taken.count()) / repeats << "ns; Result: " << b << std::endl;
    }

    return 0;
}