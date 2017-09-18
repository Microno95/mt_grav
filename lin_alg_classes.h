//
// Created by ekin4 on 9/16/2017.
//

#ifndef MT_GRAV_LIN_ALG_CLASSES_H
#define MT_GRAV_LIN_ALG_CLASSES_H

#include <iostream>
#include <cmath>
#include <array>
#include <algorithm>

template<class dtype, size_t N, size_t M>
class Matrix {
public:
    std::vector<dtype> data;

    explicit Matrix() {
        this->data.resize(N * M);
        std::fill(this->data.begin(), this->data.end(), dtype());
    }

    explicit Matrix(dtype val) {
        this->data.resize(N * M);
        std::fill(this->data.begin(), this->data.end(), val);
    }

    Matrix(dtype val, bool diag) {
        this->data.resize(N * M);
        if (!diag) std::fill(this->data.begin(), this->data.end(), val);
        else {
            std::fill(this->data.begin(), this->data.end(), dtype());
            for (size_t i = 0; i < (N < M ? N : M); ++i) {
                this->set(i, i, val);
            }
        }
    }

    template<class odtype, size_t oN, size_t oM>
    explicit Matrix(const Matrix<odtype, oN, oM>& other) {
        this->data.resize(N * M);
        std::fill(this->data.begin(), this->data.end(), dtype());
#pragma omp parallel for
        for (size_t i = 0; i < (oN < N ? oN : N); ++i) {
#pragma omp parallel for
            for (size_t j = 0; j < (oM < M ? oM : M); ++j) {
                this->set(i, j, other.get(i, j));
            }
        }
    }

    template<class odtype, size_t other_arr_size>
    explicit Matrix(const std::array<odtype, other_arr_size>& other) {
        this->data.resize(N * M);
        std::fill(this->data.begin(), this->data.end(), dtype());
        std::copy(other.begin(), other.begin() + (N * M < other_arr_size ? N * M : other_arr_size), this->data.begin());
    }

    template<class odtype>
    explicit Matrix(const std::vector<odtype>& other) {
        this->data.resize(N * M);
        std::fill(this->data.begin(), this->data.end(), dtype());
        std::copy(other.begin(), other.begin() + (N * M < other.size() ? N * M : other.size()), this->data.begin());
    }

    Matrix(const Matrix<dtype, N, M>& other) {
        this->data.resize(N * M);
        std::copy(other.data.begin(), other.data.end(), this->data.begin());
    }

    Matrix(Matrix<dtype, N, M>& other) {
        this->data.resize(N * M);
        std::move(other.data.begin(), other.data.end(), this->data.begin());
    }

    template<size_t other_size>
    explicit Matrix(std::array<dtype, other_size> initialization_vals) {
        this->data.resize(N * M);
        std::fill(this->data.begin(), this->data.end(), dtype());
        std::copy_n(initialization_vals.begin(), (other_size <= N * M ? other_size : N * M), this->data.begin());
    }

    explicit Matrix(std::array<dtype, N*M> initialization_vals) {
        this->data.resize(N * M);
        std::copy(initialization_vals.begin(), initialization_vals.end(), this->data.begin());
    }

    void set(size_t i, size_t j, dtype val) {
        this->data[i * M + j] = val;
    }

    void set(size_t idx, dtype val) {
        this->data[idx] = val;
    }

    template<typename ...Args>
    void set(size_t idx, dtype val, Args... vals) {
        if (idx > N*M - 1) {
            return;
        };
        this->data[idx] = val;
        this->set(idx+1, vals...);
    }

    template<class odtype, size_t N_, size_t M_>
    Matrix<dtype, N, M> set_block(size_t offset_row, size_t offset_col, Matrix<odtype, N_, M_> other) {
#pragma omp parallel for
        for (size_t i = 0; i < (N_ < N - offset_row ? N_ : N - offset_row); ++i) {
#pragma omp parallel for
            for (size_t j = 0; j < (M_ < M - offset_col ? M_ : M - offset_col); ++j) {
                this->set(i + offset_row, j + offset_col, other.get(i, j));
            }
        }
        return *this;
    };

    dtype get(size_t i, size_t j) const{
        return this->data[i * M + j];
    }

    dtype& get(size_t i, size_t j){
        return this->data[i * M + j];
    }

    Matrix<dtype, N, M> copy() const {
        return Matrix<dtype, N, M>(this->data);
    };

    Matrix<dtype, N - 1, M - 1> drop_cross(size_t drop_row, size_t drop_col) const{
        Matrix<dtype, N - 1, M - 1> ret_matrix;
        for (size_t i = 0; i < N; ++i) {
            if (i < drop_row) {
                for (size_t j = 0; j < M; ++j) {
                    if (j < drop_col) {
                        ret_matrix.set(i, j, this->get(i, j));
                    } else if (j > drop_col){
                        ret_matrix.set(i, j - 1, this->get(i, j));
                    }
                }
            } else if (i > drop_row) {
                for (size_t j = 0; j < M; ++j) {
                    if (j < drop_col) {
                        ret_matrix.set(i - 1, j, this->get(i, j));
                    } else if (j > drop_col){
                        ret_matrix.set(i - 1, j - 1, this->get(i, j));
                    }
                }
            }
        }
        return ret_matrix;
    };

    Matrix<dtype, N, M> swap_rows(size_t first_row, size_t second_row) const {
        Matrix<dtype, N, M> matrix_copy = this->copy();
        auto iter_begin_first = matrix_copy.data.begin();
        auto iter_end_first = matrix_copy.data.begin();
        auto iter_begin_second = matrix_copy.data.begin();
        std::advance(iter_begin_first, first_row * N);
        std::advance(iter_end_first, (first_row + 1) * N);
        std::advance(iter_begin_second, second_row * N);
        std::swap_ranges(iter_begin_first, iter_end_first, iter_begin_second);
        return matrix_copy;
    };

    inline Matrix<dtype, N, M> swap_columns(size_t first_column, size_t second_column) const {
        return this->copy().transpose().swap_rows(first_column, second_column);
    };

    template <std::size_t A = N, std::size_t B = M>
    std::enable_if_t<(A == 1 && A == B), Matrix<dtype, 1, 1>> det() const
    { return this->copy(); }

    template <std::size_t A = N, std::size_t B = M>
    std::enable_if_t<(A == 2 && A == B), Matrix<dtype, 1, 1>> det() const
    { return Matrix<dtype, 1, 1>(this->get(0, 0) * this->get(1, 1) - this->get(0, 1) * this->get(1, 0)); }

    template <std::size_t A = N, std::size_t B = M>
    std::enable_if_t<(A == 3 && A == B), Matrix<dtype, 1, 1>> det() const
    { return Matrix<dtype, 1, 1>(
                this->get(0, 0) * (this->get(1, 1) * this->get(2, 2) - this->get(1, 2) * this->get(2, 1)) -
                this->get(0, 1) * (this->get(1, 0) * this->get(2, 2) - this->get(1, 2) * this->get(2, 0)) +
                this->get(0, 2) * (this->get(1, 0) * this->get(2, 1) - this->get(1, 1) * this->get(2, 0))); }

    template <std::size_t A = N, std::size_t B = M>
    std::enable_if_t<(A > 3 && A < 12 && A == B), Matrix<dtype, 1, 1>> det() const {
        std::array<Matrix<dtype, 1, 1>, N> cofactors;
#pragma omp parallel for
        for (size_t i = 0; i < N; ++i) {
            cofactors[i] = (i % 2 == 0 ? 1 : -1) * (this->get(0, i) * this->drop_cross(0, i).det());
        }
        return std::accumulate(cofactors.begin(), cofactors.end(), Matrix<dtype, 1, 1>());
    }

    Matrix<dtype, 1, (N < M ? N : M)> diag() const {
        Matrix<dtype, 1, (N < M ? N : M)> ret_matrix;
#pragma omp parallel for
        for (size_t i = 0; i < (N < M ? N : M); ++i) {
            ret_matrix.set(0, i, this->get(i, i));
        }
        return ret_matrix;
    };

    dtype trace() const {
        dtype ret_value = dtype();
        for (size_t i = 0; i < (N < M ? N : M); ++i) {
            ret_value += this->get(i, i);
        }
        return ret_value;
    };

    template<class odtype, size_t Z>
    inline Matrix<dtype, N, Z> vdot(Matrix<odtype, Z, M> other) {
        return (*this) * other.transpose();
    };

    inline dtype square_norm() const {
        return std::inner_product(this->data.begin(), this->data.end(), this->data.begin(), dtype());
    }

    inline dtype norm() const {
        return std::sqrt(this->square_norm());
    }

    inline dtype frob_norm() const {
        return this->norm();
    };

    dtype& operator()(size_t i, size_t j) {
        return this->get(i, j);
    };

    dtype& operator[](size_t) = delete;

    Matrix<dtype, M, N> transpose() const{
        Matrix<dtype, M, N> ret_matrix;
#pragma omp parallel for
        for (size_t i = 0; i < N; ++i) {
#pragma omp parallel for
            for (size_t j = 0; j < M; ++j) {
                ret_matrix.set(j, i, this->get(i, j));
            }
        }
        return ret_matrix;
    };

    template<class ldtype, class rdtype, size_t iN, size_t iM>
    friend Matrix<ldtype, iN, iM> operator+(Matrix<ldtype, iN, iM> lhs, Matrix<rdtype, iN, iM> rhs);

    template<class ldtype, class rdtype, size_t iN, size_t iM>
    friend Matrix<ldtype, iN, iM> operator-(Matrix<ldtype, iN, iM> lhs, Matrix<rdtype, iN, iM> rhs);

    template<class ldtype, class rdtype, size_t iN, size_t iM, size_t iZ>
    friend Matrix<ldtype, iN, iZ> operator*(Matrix<ldtype, iN, iM> lhs, Matrix<rdtype, iM, iZ> rhs);

    template<class ldtype, class rdtype, size_t iN, size_t iM>
    friend Matrix<ldtype, iN, iM> operator*(Matrix<ldtype, iN, iM> lhs, Matrix<rdtype, 1, 1> rhs);

    template<class ldtype, class rdtype, size_t iN, size_t iM>
    friend Matrix<ldtype, iN, iM> operator/(Matrix<ldtype, iN, iM> lhs, rdtype rhs);;

    template<class ldtype, class rdtype, size_t iN, size_t iM>
    friend Matrix<ldtype, iN, iM> operator/(Matrix<ldtype, iN, iM> lhs, Matrix<rdtype, 1, 1> rhs);

    template<class odtype, size_t oN, size_t oM>
    friend std::ostream& operator<<(std::ostream& os, Matrix<odtype, oN, oM> other);

    static Matrix<dtype, N, M> eye(dtype val) {
        return Matrix<dtype, N, M>(val, true);
    };

    static Matrix<dtype, N, M> tri(dtype val_lower, dtype val_diag, dtype val_upper) {
        Matrix<dtype, N, M> ret_matrix(Matrix<dtype, N, M>(val_diag, true));
        for (size_t i = 0; i < N - 1; ++i) {
            ret_matrix(i, i + 1) = val_upper;
            ret_matrix(i + 1, i) = val_lower;
        }
        return ret_matrix;
    }
};

template<class ldtype, class rdtype, size_t iN, size_t iM>
Matrix<ldtype, iN, iM> operator+(Matrix<ldtype, iN, iM> lhs, Matrix<rdtype, iN, iM> rhs) {
    std::array<ldtype, iN*iM> res_array;
#if defined(_OPENMP)
#pragma omp parallel for
    for (size_t i = 0; i < iN*iM; ++i) {
        res_array[i] = lhs.data[i] + rhs.data[i];
    }
#else
    std::transform(lhs.data.begin(), lhs.data.end(), rhs.data.begin(), res_array.begin(), std::plus<ldtype>());
#endif
    return Matrix<ldtype, iN, iM>(res_array);
};

template<class ldtype, class rdtype, size_t iN, size_t iM>
Matrix<ldtype, iN, iM> operator-(Matrix<ldtype, iN, iM> lhs, Matrix<rdtype, iN, iM> rhs) {
    std::array<ldtype, iN*iM> res_array;
#if defined(_OPENMP)
#pragma omp parallel for
    for (size_t i = 0; i < iN*iM; ++i) {
        res_array[i] = lhs.data[i] - rhs.data[i];
    }
#else
    std::transform(lhs.data.begin(), lhs.data.end(), rhs.data.begin(), res_array.begin(), std::minus<ldtype>());
#endif
    return Matrix<ldtype, iN, iM>(res_array);
}

template<class ldtype, class rdtype, size_t iN, size_t iM, size_t iZ>
Matrix<ldtype, iN, iZ> operator*(Matrix<ldtype, iN, iM> lhs, Matrix<rdtype, iM, iZ> rhs) {
    std::array<ldtype, iN*iZ> res_array;
    std::fill(res_array.begin(), res_array.end(), ldtype());
    for (size_t i = 0; i < iN; ++i) {
#pragma omp parallel for
        for (size_t j = 0; j < iZ; ++j) {
            for (size_t k = 0; k < iM; ++k) {
                res_array[i * iZ + j] += lhs.get(i, k) * rhs.get(k, j);
            }
        }
    }
    return Matrix<ldtype, iN, iZ>(res_array);
}

template<class ldtype, class rdtype, size_t iM, size_t iZ>
Matrix<ldtype, iM, 1> operator*(Matrix<ldtype, iM, iZ> lhs, Matrix<rdtype, iZ, 1> rhs) {
    std::vector<ldtype> res_array(iM);
    std::fill(res_array.begin(), res_array.end(), ldtype());
    for (size_t j = 0; j < iM; ++j) {
        for (size_t k = 0; k < iZ; ++k) {
            res_array[j] += lhs.get(j, k) * rhs.get(k, 0);
        }
    }
    return Matrix<ldtype, iM, 1>(res_array);
}

template<class ldtype, class rdtype, size_t iM, size_t iZ>
Matrix<ldtype, 1, iZ> operator*(Matrix<ldtype, 1, iM> lhs, Matrix<rdtype, iM, iZ> rhs) {
    std::vector<ldtype> res_array(iZ);
    std::fill(res_array.begin(), res_array.end(), ldtype());
    for (size_t k = 0; k < iM; ++k) {
        for (size_t j = 0; j < iZ; ++j) {
            res_array[j] += lhs.get(0, k) * rhs.get(k, j);
        }
    }
    return Matrix<ldtype, 1, iZ>(res_array);
}

template<class ldtype, class rdtype, size_t iN, size_t iM>
Matrix<ldtype, iN, iM> operator*(Matrix<ldtype, iN, iM> lhs, Matrix<rdtype, 1, 1> rhs) {
    Matrix<ldtype, iN, iM> ret_matrix;
#pragma omp parallel for
    for (size_t i = 0; i < lhs.data.size(); ++i) ret_matrix.data[i] = lhs.data[i] * rhs.data[0];
    return ret_matrix;
};

template<class ldtype, class rdtype, size_t iN, size_t iM>
Matrix<ldtype, iN, iN> operator*(Matrix<ldtype, iN, iM> lhs, Matrix<rdtype, iM, iN> rhs) {
    Matrix<ldtype, iN, iN> res_matrix;
#pragma omp parallel for
    for (size_t i = 0; i < iN; ++i) {
#pragma omp parallel for
        for (size_t j = 0; j < iN; ++j) {
            ldtype accum = ldtype();
            for (size_t k = 0; k < iM; ++k) {
                accum += lhs.get(i, k) * rhs.get(k, j);
            }
            res_matrix.set(i, j, accum);
        }
    }
    return res_matrix;
}

template<class ldtype, class rdtype, size_t iN, size_t iM>
Matrix<ldtype, iN, iM> operator/(Matrix<ldtype, iN, iM> lhs, rdtype rhs) {
    Matrix<ldtype, iN, iM> ret_matrix;
#pragma omp parallel for
    for (size_t i = 0; i < lhs.data.size(); ++i) ret_matrix.data[i] = lhs.data[i] / rhs;
    return ret_matrix;
};

template<class ldtype, class rdtype, size_t iN, size_t iM>
Matrix<ldtype, iN, iM> operator/(Matrix<ldtype, iN, iM> lhs, Matrix<rdtype, 1, 1> rhs) {
    Matrix<ldtype, iN, iM> ret_matrix;
#pragma omp parallel for
    for (size_t i = 0; i < lhs.data.size(); ++i) ret_matrix.data[i] = lhs.data[i] / rhs.data[0];
    return ret_matrix;
};

template<class odtype, size_t oN, size_t oM>
std::ostream& operator<<(std::ostream& os, Matrix<odtype, oN, oM> other) {
    os << "{";
    for (size_t i = 0; i < oN - 1; ++i) {
        if (i > 0) os << " ";
        os << "{" << other.get(i, 0);
        for (size_t j = 1; j < oM; ++j) {
            os << ", " << other.get(i, j);
        }
        os << "}," << std::endl;
    }
    os << " {" << other.get(oN - 1, 0);
    for (size_t j = 1; j < oM; ++j) {
        os << ", " << other.get(oN - 1, j);
    }
    os << "}}";
    return os;
}

#endif //MT_GRAV_LIN_ALG_CLASSES_H
