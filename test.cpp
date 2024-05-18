#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <algorithm>
#include <random>

// Определение константы PI
const double PI = 3.141592653589793238460;

class FFT {
public:
    using Complex = std::complex<double>;
    using ComplexVector = std::vector<Complex>;

    // Итеративное быстрое преобразование Фурье (FFT)
    static void fft(ComplexVector& a) {
        int n = static_cast<int>(a.size());
        int log_n = 0;
        while ((1 << log_n) < n) ++log_n;

        // Перестановка Битереверса
        for (int i = 0, j = 0; i < n; ++i) {
            if (i < j) {
                std::swap(a[i], a[j]);
            }
            for (int l = n >> 1; (j ^= l) < l; l >>= 1);
        }

        // Итеративный процесс FFT
        for (int len = 2; len <= n; len <<= 1) {
            double angle = -2 * PI / len;
            Complex wlen(cos(angle), sin(angle));
            for (int i = 0; i < n; i += len) {
                Complex w(1);
                for (int j = 0; j < len / 2; ++j) {
                    int i_j = i + j;
                    int i_j_len_half = i + j + len / 2;
                    if (i_j_len_half < n) {
                        Complex u = a[i_j];
                        Complex v = a[i_j_len_half] * w;
                        a[i_j] = u + v;
                        a[i_j_len_half] = u - v;
                        w *= wlen;
                    }
                }
            }
        }
    }

    // Итеративное обратное быстрое преобразование Фурье (IFFT)
    static void ifft(ComplexVector& a) {
        for (auto& x : a) {
            x = std::conj(x);
        }
        fft(a);
        for (auto& x : a) {
            x = std::conj(x);
        }
        for (auto& x : a) {
            x /= static_cast<double>(a.size());
        }
    }
};

// Генерация случайных комплексных данных
FFT::ComplexVector generate_random_complex_vector(int n, double min_real, double max_real, double min_imag, double max_imag) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis_real(min_real, max_real);
    std::uniform_real_distribution<> dis_imag(min_imag, max_imag);

    FFT::ComplexVector data(n);
    for (int i = 0; i < n; ++i) {
        data[i] = { dis_real(gen), dis_imag(gen) };
    }
    return data;
}

// Вычисление ошибки между двумя векторами
double calculate_error(const FFT::ComplexVector& original, const FFT::ComplexVector& transformed) {
    double error = 0.0;
    for (size_t i = 0; i < original.size(); ++i) {
        error += std::abs(original[i] - transformed[i]);
    }
    return error / original.size();
}

int main() {
    using Complex = std::complex<double>;
    using ComplexVector = std::vector<Complex>;

    // Параметры генерации случайных данных
    int n = 60; // Длина данных, кратная 2, 3 и 5 (например, 2^2 * 3^1 * 5^1 = 60)
    double min_real = -10.0;
    double max_real = 10.0;
    double min_imag = -10.0;
    double max_imag = 10.0;

    // Генерация случайных данных
    ComplexVector data = generate_random_complex_vector(n, min_real, max_real, min_imag, max_imag);
    ComplexVector original_data = data; // Копия для сравнения

    std::cout << "Input:" << std::endl;
    for (const auto& x : data) {
        std::cout << x << " ";
    }
    std::cout << std::endl;

    // Прямое преобразование Фурье
    FFT::fft(data);
    std::cout << "After FFT:" << std::endl;
    for (const auto& x : data) {
        std::cout << x << " ";
    }
    std::cout << std::endl;

    // Обратное преобразование Фурье
    FFT::ifft(data);
    std::cout << "After IFFT:" << std::endl;
    for (const auto& x : data) {
        std::cout << x << " ";
    }
    std::cout << std::endl;

    // Вычисление и вывод ошибки
    double error = calculate_error(original_data, data);
    std::cout << "Error between the original and restored data: " << error << std::endl;

    return 0;
}
