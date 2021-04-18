#ifndef MATRIX_H
#define MATRIX_H

#include "utils.h"

#include <algorithm>
#include <exception>
#include <iomanip>
#include <iostream>
#include <vector>


struct Data
{
    std::vector<float> numbers;

    Data(uint32_t size = 0) : numbers(size, 0.0) {}

    void generate(float low, float high)
    {
        using namespace Utils;
        std::generate(numbers.begin(), numbers.end(), randfloat(low, high));
    }
    void set(std::vector<float> xs) { std::copy(xs.begin(), xs.end(), numbers.begin()); }

    float* data() { return numbers.data(); }
    const float* data() const { return numbers.data(); }

    size_t size() const { return numbers.size(); }

    friend bool operator==(const Data& lhs, const Data& rhs)
    {
        if (lhs.numbers.size() != rhs.numbers.size())
            return false;

        for (int i = 0; i < lhs.numbers.size(); i++)
            if (fabs(lhs.numbers[i] - rhs.numbers[i]) > 1e-3)
                return false;
        return true;
    }

    virtual void print() const = 0;

    friend std::ostream& operator<<(std::ostream& os, const Data& data)
    {
        data.print();
        return os;
    }
};


/**
 * @brief   A vector specifially designed for this assignment.
 */
struct Vector : Data
{
    using Data::Data;

    float& operator[](size_t i) { return at(i); }
    const float& operator[](size_t i) const { return at(i); }
    float& at(size_t i) { return numbers[i]; }
    const float& at(size_t i) const { return numbers[i]; }

    void assign(uint32_t n, float value = 0.0) { numbers.assign(n, value); }

    void resize(size_t n) { numbers.resize(n); }

    void print() const
    {
        std::cout << "[";
        if (numbers.empty())
            std::cout << "(empty)\n";
        for (int i = 0; i < numbers.size(); i++)
            std::cout << " " << numbers[i];
        std::cout << " ]";
    }
};


/**
 * @brief   A matrix specifially designed for this assignment.
 */
struct Matrix : Data
{
    uint32_t row;
    uint32_t col;

    Matrix() : Matrix(0, 0) {}
    Matrix(uint32_t row, uint32_t col) : Data(row * col), row{row}, col{col} {}

    float& at(size_t r, size_t c) { return numbers[r * col + c]; }
    const float& at(size_t r, size_t c) const { return numbers[r * col + c]; }

    void assign(uint32_t row, uint32_t col, float value = 0.0)
    {
        this->row = row;
        this->col = col;
        numbers.assign(row * col, value);
    }

    void resize(uint32_t row, uint32_t col)
    {
        this->row = row;
        this->col = col;
        numbers.resize(row * col);
    }

    Vector apply(const Vector& v) const
    {
        if (v.size() != col)
            throw std::runtime_error("Matrix::apply: vector size does not equal matrix col");

        Vector out(row);
        for (int i = 0; i < row; i++)
            for (int j = 0; j < col; j++)
                out[i] += numbers[i * col + j] * v[j];

        return out;
    }

    Matrix transposed() const
    {
        Matrix m{col, row};
        for (uint32_t i = 0; i < row; i++)
            for (uint32_t j = 0; j < col; j++)
                m.numbers[j * row + i] = numbers[i * col + j];
        return m;
    }

    void print() const
    {
        if (numbers.empty() || row * col == 0)
            std::cout << "(empty)\n";

        std::cout << std::setprecision(3) << std::fixed;
        for (int i = 0; i < row; i++)
            for (int j = 0; j < col; j++)
                std::cout << std::setw(10) << numbers[i * col + j] << (j != col - 1 ? " " : "\n");
    }

    friend bool operator==(const Matrix& lhs, const Matrix& rhs)
    {
        return lhs.row == rhs.row && lhs.col == rhs.col
               && static_cast<const Data&>(lhs) == static_cast<const Data&>(rhs);
    }

    friend Matrix operator*(const Matrix& lhs, const Matrix& rhs)
    {
        if (lhs.col != rhs.row)
            throw std::runtime_error("Matrix::operator* (Matrix, Matrix): lhs.col should equal rhs.row");

        Matrix result{lhs.row, rhs.col};
        for (int r = 0; r < lhs.row; r++)
        {
            for (int c = 0; c < rhs.col; c++)
            {
                float sum = 0.0;
                // Iterate across the lhs row and rhs column, and sum it up.
                for (int i = 0; i < lhs.col; i++)
                {
                    sum += lhs.at(r, i) * rhs.at(i, c);
                }
                result.at(r, c) = sum;
            }
        }
        return result;
    }
};


#endif