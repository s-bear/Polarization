/**********
 *   Copyright 2014 Samuel Bear Powell
 *
 *   Licensed under the Apache License, Version 2.0 (the "License");
 *   you may not use this file except in compliance with the License.
 *   You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 *   Unless required by applicable law or agreed to in writing, software
 *   distributed under the License is distributed on an "AS IS" BASIS,
 *   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *   See the License for the specific language governing permissions and
 *   limitations under the License.
\**********/
#ifndef POLARIZATION_HPP
#define POLARIZATION_HPP

#include <xmmintrin.h> //SSE1,2
#include <smmintrin.h> //SSE4
#include <cstdint>

//pattern encoding for 2x2 super-pixels
//PIXn(r,c) specifies that pixel n (0,1,2,3) belongs in (row, column) = (r,c) of the super-pixel. r, c must be either 0 or 1
#define PIX0(r,c) (((r)&1) | (((c)&1) << 1))
#define PIXN(r,c,n) (PIX0(r,c) << (2*(n)))
#define PIX1(r,c) PIXN(r,c,1)
#define PIX2(r,c) PIXN(r,c,2)
#define PIX3(r,c) PIXN(r,c,3)

//pattern decoding for 2x2 super-pixels
//PATR(pattern,n) yields 0 or 1 for which row pixel n belongs to as specified by the pattern
//PATC(pattern,n) yields the column.
#define PATR(p,n) (((p) >> (2*(n))) & 1)
#define PATC(p,n) (((p) >> (2*(n)+1)) & 1)

//compute an index for a row-major storage matrix
#define IDX(r,c,cols) ((r)*(cols)+(c))
//wrap an index 'x' from 0 to max
#define WRAP(x,max) ((x) < 0 ? ((max) - ((max) % -(x))) : ((max) % (x)))
//reflect an index x between 0 to max
#define REFLECT(x,max) (WRAP((x),2*(max)) > (max) ? 2*(max)-WRAP((x),2*(max)) : WRAP((x),2*(max)))


//#define DEFAULT_PATTERN (PIX0(0, 0) | PIX1(1, 1) | PIX2(0, 1) | PIX3(1, 0))

namespace Polarization {

    extern const uint8_t default_pattern;

    //transpose a 4x4 matrix
    void trans(__m128 mat[4]);

    //invert a 4x4 matrix
    void inv(const __m128 mat[4], __m128 out[4]);

    //perform a dot product of a row-vector and a matrix
    //the matrix is in column-major order!!
    __m128 dot(const __m128 vec, const __m128 mat[4]);

    //perform a dot product of a matrix with a column vector
    //the matrix is in row-major order!!
    __m128 dot(const __m128 mat[4], const __m128 vec);

    float dot(const __m128 vec0, const __m128 vec1);

    //packs superpixels into SSE registers according to pattern
    //the pattern is specified by P0(r,c) to P3(r,c) macros
    //P0(r,c) means that pixel 0 of the SSE register appears in row,col = r,c of the 2x2 super pixel.
    template<typename raw_type>
    void pack_superpixels(const size_t rows, const size_t cols, const raw_type* raw, __m128* out, uint8_t pattern = default_pattern) {
        size_t rows_2 = rows / 2, cols_2 = cols / 2;
        int r,c,rx2,cx2;
        #pragma omp parallel for private(r,c,rx2,cx2)
        for (r = 0; r < rows_2; ++r) {
            for (c = 0; c < cols_2; ++c) {
                rx2 = r << 1, cx2 = c << 1;
                out[IDX(r,c,cols_2)] =
                    _mm_setr_ps(
                        float(raw[IDX(rx2 | PATR(pattern, 0), cx2 | PATC(pattern, 0), cols)]),
                        float(raw[IDX(rx2 | PATR(pattern, 1), cx2 | PATC(pattern, 1), cols)]),
                        float(raw[IDX(rx2 | PATR(pattern, 2), cx2 | PATC(pattern, 2), cols)]),
                        float(raw[IDX(rx2 | PATR(pattern, 3), cx2 | PATC(pattern, 2), cols)])
                    );
            }
        }
    }

    void unpack_superpixels(const size_t rows, const size_t cols, const __m128* packed, __m128* out, uint8_t pattern = default_pattern);

    void mask_low_high(const size_t n, const __m128* raw, float low, float high, __m128* out);
    void mask_low(const size_t n, const __m128* raw, float low, __m128* out);
    void mask_high(const size_t n, const __m128* raw, float high, __m128* out);

    //packs super-pixels and does dot(gains[i], raw[i] - darks[i]) where gains[i] is row-major
    template<typename raw_type>
    void calibrate_matrix(const size_t rows, const size_t cols, const raw_type* raw, const __m128* darks, const __m128* gains, __m128* out, uint8_t pattern = default_pattern) {
        size_t rows_2 = rows / 2, cols_2 = cols / 2;
        int r, c, rx2, cx2;
        __m128 packed;
        #pragma omp parallel for private(r,c,rx2,cx2,packed)
        for (r = 0; r < rows_2; ++r) {
            for (c = 0; c < cols_2; ++c) {
                rx2 = r << 1, cx2 = c << 1;
                packed =
                    _mm_setr_ps(
                    float(raw[IDX(rx2 | PATR(pattern, 0), cx2 | PATC(pattern, 0), cols)]),
                    float(raw[IDX(rx2 | PATR(pattern, 1), cx2 | PATC(pattern, 1), cols)]),
                    float(raw[IDX(rx2 | PATR(pattern, 2), cx2 | PATC(pattern, 2), cols)]),
                    float(raw[IDX(rx2 | PATR(pattern, 3), cx2 | PATC(pattern, 2), cols)])
                    );
                out[IDX(r, c, cols_2)] = dot(&gains[IDX(r, c, cols_2)], _mm_sub_ps(packed, darks[IDX(r, c, cols_2)]));
            }
        }
    }

    //packs super-pixels does dot(raw[i] - darks[i], gains[i]) where gains[i] is column-major
    template<typename raw_type>
    void calibrate_matrix2(const size_t rows, const size_t cols, const raw_type* raw, const __m128* darks, const __m128* gains, __m128* out, uint8_t pattern = default_pattern) {
        size_t rows_2 = rows / 2, cols_2 = cols / 2;
        int r, c, rx2, cx2;
        __m128 packed;
        #pragma omp parallel for private(r,c,rx2,cx2,packed)
        for (r = 0; r < rows_2; ++r) {
            for (c = 0; c < cols_2; ++c) {
                rx2 = r << 1, cx2 = c << 1;
                packed =
                    _mm_setr_ps(
                    float(raw[IDX(rx2 | PATR(pattern, 0), cx2 | PATC(pattern, 0), cols)]),
                    float(raw[IDX(rx2 | PATR(pattern, 1), cx2 | PATC(pattern, 1), cols)]),
                    float(raw[IDX(rx2 | PATR(pattern, 2), cx2 | PATC(pattern, 2), cols)]),
                    float(raw[IDX(rx2 | PATR(pattern, 3), cx2 | PATC(pattern, 2), cols)])
                    );
                out[IDX(r, c, cols_2)] = dot(_mm_sub_ps(packed, darks[IDX(r, c, cols_2)]), gains[IDX(r, c, cols_2)]);
            }
        }
    }

    //does dot(gains[i], raw[i] - darks[i]) where gains[i] is row-major
    void calibrate_matrix(const size_t n, const __m128* raw, const __m128* darks, const __m128* gains, __m128* out);

    //does dot(raw[i] - darks[i], gains[i]) where gains[i] is column-major
    void calibrate_matrix2(const size_t n, const __m128* raw, const __m128* darks, const __m128* gains, __m128* out);

    struct edge_mode {
        enum t {
            ZERO,
            WRAP,
            REFLECT
        };
    };

    //2D filtering (with 2D kernel)
    void filter(const size_t rows, const size_t cols, const __m128* in, __m128* out, const size_t filt_rows, const size_t filt_cols, const __m128* filt, const edge_mode::t mode);
    void filter_zero(const size_t rows, const size_t cols, const __m128* in, __m128* out, const size_t filt_rows, const size_t filt_cols, const __m128* filt);
    void filter_wrap(const size_t rows, const size_t cols, const __m128* in, __m128* out, const size_t filt_rows, const size_t filt_cols, const __m128* filt);
    void filter_reflect(const size_t rows, const size_t cols, const __m128* in, __m128* out, const size_t filt_rows, const size_t filt_cols, const __m128* filt);

    //2D filtering (with 1D kernel applied on both axes)
    void filter(const size_t rows, const size_t cols, const __m128 *in, __m128 *out, const size_t filt_size, const __m128 *filt, const edge_mode::t mode);
    void filter_zero(const size_t rows, const size_t cols, const __m128 *in, __m128 *out, const size_t filt_size, const __m128 *filt);
    void filter_wrap(const size_t rows, const size_t cols, const __m128 *in, __m128 *out, const size_t filt_size, const __m128 *filt);
    void filter_reflect(const size_t rows, const size_t cols, const __m128 *in, __m128 *out, const size_t filt_size, const __m128 *filt);

    //does dot(R[i], img[i]) where R[i] is row-major
    void stokes(const size_t n, const __m128* img, const __m128* R, __m128* out);

    //does dot(R, img[i]) where R is row-major
    void stokesR(const size_t n, const __m128* img, const __m128 R[4], __m128* out);

    //does dot(img[i], R[i]) where R[i] is column-major
    void stokes2(const size_t n, const __m128* img, const __m128*  R, __m128* out);

    //does dot(img[i], R) where R is column-major
    void stokes2R(const size_t n, const __m128* img, const __m128  R[4], __m128* out);

    //assumes 0,90,45,135 pixel pattern:
    void stokes(const size_t n, const __m128* img, __m128* out);

    //extract a single element from the __m128
    void element(const size_t param, const size_t n, const __m128* simg, float* out);

    //degree of polarization: hypot(s1,s2,s3)/s0
    void dop(const size_t n, const __m128* simg, float* out);

    //degree of linear polarization: hypot(s1,s2)/s0
    void dolp(const size_t n, const __m128* simg, float* out);

    //degree of circular polarization: abs(s3)/s0
    void docp(const size_t n, const __m128* simg, float* out);

    //angle of polarization: 0.5*atan(s2/s1)
    void aop(const size_t n, const __m128* simg, float* out);

    //2x angle of polarization: atan(s2/s1)
    void aopx2(const size_t n, const __m128* simg, float* out);

    //ellipticity angle: 0.5*asin(s3/s0)
    void ella(const size_t n, const __m128* simg, float* out);

    //TODO:
    //decompose stokes vector?

}


#endif // POLARIZATION_HPP
