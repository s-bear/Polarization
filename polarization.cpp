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
#include "polarization.hpp"
#include <omp.h>
#include <cmath>
#include <limits>

const uint8_t Polarization::default_pattern = (PIX0(0, 0) | PIX1(1, 1) | PIX2(0, 1) | PIX3(1, 0));

void Polarization::trans(__m128 mat[4]) {
    _MM_TRANSPOSE4_PS(mat[0], mat[1], mat[2], mat[3]);
}

void Polarization::inv(const __m128 mat[4], __m128 out[4]) {
    //adapted from Intel AP-928 "Streaming SIMD Extensions - Inverse of a 4x4 Matrix"
    __m128 minor0, minor1, minor2, minor3;
    __m128 row0, row1, row2, row3;
    __m128 det, tmp1;

    row0 = mat[0];
    row1 = mat[1];
    row2 = mat[2];
    row3 = mat[3];
    //transpose:
    _MM_TRANSPOSE4_PS(row0, row1, row2, row3);
    //Compute cofactors:
    tmp1 = _mm_mul_ps(row2, row3);
    tmp1 = _mm_shuffle_ps(tmp1, tmp1, 0xB1);
    minor0 = _mm_mul_ps(row1, tmp1);
    minor1 = _mm_mul_ps(row0, tmp1);
    tmp1 = _mm_shuffle_ps(tmp1, tmp1, 0x4E);
    minor0 = _mm_sub_ps(_mm_mul_ps(row1, tmp1), minor0);
    minor1 = _mm_sub_ps(_mm_mul_ps(row0, tmp1), minor1);
    minor1 = _mm_shuffle_ps(minor1, minor1, 0x4E);

    tmp1 = _mm_mul_ps(row1, row2);
    tmp1 = _mm_shuffle_ps(tmp1, tmp1, 0xB1);
    minor0 = _mm_add_ps(_mm_mul_ps(row3, tmp1), minor0);
    minor3 = _mm_mul_ps(row0, tmp1);
    tmp1 = _mm_shuffle_ps(tmp1, tmp1, 0x4E);
    minor0 = _mm_sub_ps(minor0, _mm_mul_ps(row3, tmp1));
    minor3 = _mm_sub_ps(_mm_mul_ps(row0, tmp1), minor3);
    minor3 = _mm_shuffle_ps(minor3, minor3, 0x4E);

    tmp1 = _mm_mul_ps(_mm_shuffle_ps(row1, row1, 0x4E), row3);
    tmp1 = _mm_shuffle_ps(tmp1, tmp1, 0xB1);
    row2 = _mm_shuffle_ps(row2, row2, 0x4E);
    minor0 = _mm_add_ps(_mm_mul_ps(row2, tmp1), minor0);
    minor2 = _mm_mul_ps(row0, tmp1);
    tmp1 = _mm_shuffle_ps(tmp1, tmp1, 0x4E);
    minor0 = _mm_sub_ps(minor0, _mm_mul_ps(row2, tmp1));
    minor2 = _mm_sub_ps(_mm_mul_ps(row0, tmp1), minor2);
    minor2 = _mm_shuffle_ps(minor2, minor2, 0x4E);

    tmp1 = _mm_mul_ps(row0, row1);
    tmp1 = _mm_shuffle_ps(tmp1, tmp1, 0xB1);
    minor2 = _mm_add_ps(_mm_mul_ps(row3, tmp1), minor2);
    minor3 = _mm_sub_ps(_mm_mul_ps(row2, tmp1), minor3);
    tmp1 = _mm_shuffle_ps(tmp1, tmp1, 0x4E);
    minor2 = _mm_sub_ps(_mm_mul_ps(row3, tmp1), minor2);
    minor3 = _mm_sub_ps(minor3, _mm_mul_ps(row2, tmp1));

    tmp1 = _mm_mul_ps(row0, row3);
    tmp1 = _mm_shuffle_ps(tmp1, tmp1, 0xB1);
    minor1 = _mm_sub_ps(minor1, _mm_mul_ps(row2, tmp1));
    minor2 = _mm_add_ps(_mm_mul_ps(row1, tmp1), minor2);
    tmp1 = _mm_shuffle_ps(tmp1, tmp1, 0x4E);
    minor1 = _mm_add_ps(_mm_mul_ps(row2, tmp1), minor1);
    minor2 = _mm_sub_ps(minor2, _mm_mul_ps(row1, tmp1));

    tmp1 = _mm_mul_ps(row0, row2);
    tmp1 = _mm_shuffle_ps(tmp1, tmp1, 0xB1);
    minor1 = _mm_add_ps(_mm_mul_ps(row3, tmp1), minor1);
    minor3 = _mm_sub_ps(minor3, _mm_mul_ps(row1, tmp1));
    tmp1 = _mm_shuffle_ps(tmp1, tmp1, 0x4E);
    minor1 = _mm_sub_ps(minor1, _mm_mul_ps(row3, tmp1));
    minor3 = _mm_add_ps(_mm_mul_ps(row1, tmp1), minor3);
    //compute determinant:
    det = _mm_mul_ps(row0, minor0);
    det = _mm_add_ps(_mm_shuffle_ps(det, det, 0x4E), det);
    det = _mm_add_ss(_mm_shuffle_ps(det, det, 0xB1), det);
    tmp1 = _mm_rcp_ss(det);
    det = _mm_sub_ss(_mm_add_ss(tmp1, tmp1), _mm_mul_ss(det, _mm_mul_ss(tmp1, tmp1)));
    det = _mm_shuffle_ps(det, det, 0x00);
    //store:
    out[0] = _mm_mul_ps(det, minor0);
    out[1] = _mm_mul_ps(det, minor1);
    out[2] = _mm_mul_ps(det, minor2);
    out[3] = _mm_mul_ps(det, minor3);
}

//perform a dot product of a row-vector and a matrix
//the matrix is in column-major order!!
__m128 Polarization::dot(const __m128 vec, const __m128 mat[4]) {
    //expand vec to columns
    __m128 vec0 = _mm_shuffle_ps(vec, vec, _MM_SHUFFLE(0, 0, 0, 0));
    __m128 vec1 = _mm_shuffle_ps(vec, vec, _MM_SHUFFLE(1, 1, 1, 1));
    __m128 vec2 = _mm_shuffle_ps(vec, vec, _MM_SHUFFLE(2, 2, 2, 2));
    __m128 vec3 = _mm_shuffle_ps(vec, vec, _MM_SHUFFLE(3, 3, 3, 3));
    //element-wise multiply each column of vec with each column of mat
    __m128 prod0 = _mm_mul_ps(vec0, mat[0]);
    __m128 prod1 = _mm_mul_ps(vec1, mat[1]);
    __m128 prod2 = _mm_mul_ps(vec2, mat[2]);
    __m128 prod3 = _mm_mul_ps(vec3, mat[3]);
    //sum down each column
    return _mm_add_ps(_mm_add_ps(prod0, prod1), _mm_add_ps(prod2, prod3));
}
//perform a dot product of a matrix with a column vector
//the matrix is in row-major order!!
__m128 Polarization::dot(const __m128 mat[4], const __m128 vec) {
    //element-wise product of vector with each row
    __m128 prod0 = _mm_mul_ps(vec, mat[0]);
    __m128 prod1 = _mm_mul_ps(vec, mat[1]);
    __m128 prod2 = _mm_mul_ps(vec, mat[2]);
    __m128 prod3 = _mm_mul_ps(vec, mat[3]);
    //sum each product using horizontal-add
    return _mm_hadd_ps(_mm_hadd_ps(prod0, prod1), _mm_hadd_ps(prod2, prod3));
}

float Polarization::dot(const __m128 vec0, const __m128 vec1) {
    __m128 prod = _mm_mul_ps(vec0, vec1);
    return prod.m128_f32[0] + prod.m128_f32[1] + prod.m128_f32[2] + prod.m128_f32[3];
}

void Polarization::unpack_superpixels(const size_t rows, const size_t cols, const __m128* packed, __m128* out, uint8_t pattern) {
    size_t rows_2 = rows / 2, cols_2 = cols / 2;
    int r, c, rx2, cx2;
    #pragma omp parallel for private(r,c,rx2,cx2)
    for (r = 0; r < rows_2; ++r) {
        for (c = 0; c < cols_2; ++c) {
            rx2 = r << 1; cx2 = c << 1;
            __m128 p = packed[IDX(r, c, cols_2)];
            out[IDX(rx2 | PATR(pattern, 0), cx2 | PATC(pattern, 0), cols)] = _mm_setr_ps(p.m128_f32[0], 0, 0, 0);
            out[IDX(rx2 | PATR(pattern, 1), cx2 | PATC(pattern, 1), cols)] = _mm_setr_ps(0, p.m128_f32[1], 0, 0);
            out[IDX(rx2 | PATR(pattern, 2), cx2 | PATC(pattern, 2), cols)] = _mm_setr_ps(0, 0, p.m128_f32[2], 0);
            out[IDX(rx2 | PATR(pattern, 3), cx2 | PATC(pattern, 3), cols)] = _mm_setr_ps(0, 0, 0, p.m128_f32[3]);
        }
    }
}

void Polarization::mask_low_high(const size_t n, const __m128* raw, float low, float high, __m128* out) {
    __m128 low_vec = _mm_set1_ps(low);
    __m128 high_vec = _mm_set1_ps(high);
    __m128 nan_vec = _mm_set1_ps(std::numeric_limits<float>::quiet_NaN());
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        __m128 val = *raw++;
        //mask = (val < low) | (val > high)
        __m128 mask = _mm_or_ps(_mm_cmplt_ps(val, low_vec), _mm_cmpgt_ps(val, high_vec));
        //val = (~mask & val) | (mask & NAN)
        val = _mm_or_ps(_mm_andnot_ps(mask, val), _mm_and_ps(mask, nan_vec));
        *out++ = val;
    }
}
void Polarization::mask_low(const size_t n, const __m128* raw, float low, __m128* out) {
    __m128 low_vec = _mm_set1_ps(low);
    __m128 nan_vec = _mm_set1_ps(std::numeric_limits<float>::quiet_NaN());
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        __m128 val = *raw++;
        //mask = (val < low)
        __m128 mask = _mm_cmplt_ps(val, low_vec);
        //val = (~mask & val) | (mask & NAN)
        val = _mm_or_ps(_mm_andnot_ps(mask, val), _mm_and_ps(mask, nan_vec));
        *out++ = val;
    }
}
void Polarization::mask_high(const size_t n, const __m128* raw, float high, __m128* out) {
    __m128 high_vec = _mm_set1_ps(high);
    __m128 nan_vec = _mm_set1_ps(std::numeric_limits<float>::quiet_NaN());
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        __m128 val = *raw++;
        //mask = (val > high)
        __m128 mask = _mm_cmpgt_ps(val, high_vec);
        //val = (~mask & val) | (mask & NAN)
        val = _mm_or_ps(_mm_andnot_ps(mask, val), _mm_and_ps(mask, nan_vec));
        *out++ = val;
    }
}

void Polarization::calibrate_matrix(const size_t n, const __m128* raw, const __m128* darks, const __m128* gains, __m128* out) {
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        out[i] = dot(&gains[i], _mm_sub_ps(raw[i], darks[i]));
    }
}

void Polarization::calibrate_matrix2(const size_t n, const __m128* raw, const __m128* darks, const __m128* gains, __m128* out) {
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        out[i] = dot(_mm_sub_ps(raw[i], darks[i]), &gains[i]);
    }
}

void Polarization::filter(const size_t rows, const size_t cols, const __m128* in, __m128* out, const size_t filt_rows, const size_t filt_cols, const __m128* filt, const edge_mode::t mode) {
    switch (mode) {
    case edge_mode::ZERO:
        return filter_zero(rows, cols, in, out, filt_rows, filt_cols, filt);
    case edge_mode::WRAP:
        return filter_wrap(rows, cols, in, out, filt_rows, filt_cols, filt);
    case edge_mode::REFLECT:
        return filter_reflect(rows, cols, in, out, filt_rows, filt_cols, filt);
    }
}

void Polarization::filter(const size_t rows, const size_t cols, const __m128* in, __m128* out, const size_t filt_size, const __m128* filt, const edge_mode::t mode) {
    switch (mode) {
    case edge_mode::ZERO:
        return filter_zero(rows, cols, in, out, filt_size, filt);
    case edge_mode::WRAP:
        return filter_wrap(rows, cols, in, out, filt_size, filt);
    case edge_mode::REFLECT:
        return filter_reflect(rows, cols, in, out, filt_size, filt);
    }
}

void Polarization::filter_zero(const size_t rows, const size_t cols, const __m128* in, __m128* out, const size_t filt_rows, const size_t filt_cols, const __m128* filt)
{
    int roff = filt_rows / 2;
    int coff = filt_cols / 2;
    int r, c, fr, fc, rx, cx;
    __m128 sum;
    #pragma omp parallel for private(r,c,sum,fr,fc,rx,cx)
    for (r = 0; r < rows; ++r) {
        for (c = 0; c < cols; ++c) {
            sum = _mm_set1_ps(0);
            for (fr = 0; fr < filt_rows; ++fr) {
                for (fc = 0; fc < filt_cols; ++fc) {
                    rx = r + fr - roff;
                    cx = c + fc - coff;
                    if (rx < 0 || cx < 0 || rx >= rows || cx >= cols)
                        continue; //sum += 0
                    else //sum += in[rx,cx]*filt[fr,fc];
                        sum = _mm_add_ps(sum, _mm_mul_ps(in[IDX(rx,cx,cols)],filt[IDX(fr,fc,filt_cols)]));
                }
            }
            out[IDX(r, c, cols)] = sum;
        }
    }
}

void Polarization::filter_zero(const size_t rows, const size_t cols, const __m128* in, __m128* out, const size_t filt_size, const __m128* filt)
{
    int off = filt_size / 2;
    int r, c, fi, rx, cx;
    __m128 sum;
    //apply to rows
    #pragma omp parallel for private(r,c,sum,fi,rx)
    for (r = 0; r < rows; ++r) {
        for (c = 0; c < cols; ++c) {
            sum = _mm_set1_ps(0);
            for (fi = 0; fi < filt_size; ++fi) {
                rx = r + fi - off;
                if(rx < 0 || rx >= rows)
                    continue;
                else
                    sum = _mm_add_ps(sum, _mm_mul_ps(in[IDX(rx,c,cols)],filt[fi]));
            }
            out[IDX(r,c,cols)] = sum;
        }
    }
    //apply to columns:
    #pragma omp parallel for private(r,c,sum,fi,cx)
    for (r = 0; r < rows; ++r) {
        for (c = 0; c < cols; ++c) {
            sum = out[IDX(r,c,cols)];
            for (fi = 0; fi < filt_size; ++fi) {
                cx = c + fi - off;
                if(cx < 0 || cx >= cols)
                    continue;
                else
                    sum = _mm_add_ps(sum, _mm_mul_ps(in[IDX(r,cx,cols)],filt[fi]));
            }
            out[IDX(r,c,cols)] = sum;
        }
    }
}

void Polarization::filter_wrap(const size_t rows, const size_t cols, const __m128* in, __m128* out, const size_t filt_rows, const size_t filt_cols, const __m128* filt)
{
    int roff = filt_rows / 2;
    int coff = filt_cols / 2;
    int r, c, fr, fc, rx, cx;
    __m128 sum;
    #pragma omp parallel for private(r,c,sum,fr,fc,rx,cx)
    for (r = 0; r < rows; ++r) {
        for (c = 0; c < cols; ++c) {
            sum = _mm_set1_ps(0);
            for (fr = 0; fr < filt_rows; ++fr) {
                for (fc = 0; fc < filt_cols; ++fc) {
                    rx = WRAP(r + fr - roff, rows);
                    cx = WRAP(c + fc - coff, cols);
                    sum = _mm_add_ps(sum, _mm_mul_ps(in[IDX(rx, cx, cols)], filt[IDX(fr, fc, filt_cols)]));
                }
            }
            out[IDX(r, c, cols)] = sum;
        }
    }
}

void Polarization::filter_wrap(const size_t rows, const size_t cols, const __m128* in, __m128* out, const size_t filt_size, const __m128* filt)
{
    int off = filt_size / 2;
    int r, c, fi, rx, cx;
    __m128 sum;
    //apply to rows
    #pragma omp parallel for private(r,c,sum,fi,rx)
    for (r = 0; r < rows; ++r) {
        for (c = 0; c < cols; ++c) {
            sum = _mm_set1_ps(0);
            for (fi = 0; fi < filt_size; ++fi) {
                rx = WRAP(r + fi - off, rows);
                sum = _mm_add_ps(sum, _mm_mul_ps(in[IDX(rx,c,cols)],filt[fi]));
            }
            out[IDX(r,c,cols)] = sum;
        }
    }
    //apply to columns:
    #pragma omp parallel for private(r,c,sum,fi,cx)
    for (r = 0; r < rows; ++r) {
        for (c = 0; c < cols; ++c) {
            sum = out[IDX(r,c,cols)];
            for (fi = 0; fi < filt_size; ++fi) {
                cx = WRAP(c + fi - off, cols);
                sum = _mm_add_ps(sum, _mm_mul_ps(in[IDX(r,cx,cols)],filt[fi]));
            }
            out[IDX(r,c,cols)] = sum;
        }
    }
}

void Polarization::filter_reflect(const size_t rows, const size_t cols, const __m128* in, __m128* out, const size_t filt_rows, const size_t filt_cols, const __m128* filt)
{
    int roff = filt_rows / 2;
    int coff = filt_cols / 2;
    int r, c, fr, fc, rx, cx;
    __m128 sum;
    #pragma omp parallel for private(r,c,sum,fr,fc,rx,cx)
    for (r = 0; r < rows; ++r) {
        for (c = 0; c < cols; ++c) {
            sum = _mm_set1_ps(0);
            for (fr = 0; fr < filt_rows; ++fr) {
                for (fc = 0; fc < filt_cols; ++fc) {
                    rx = REFLECT(r + fr - roff, rows);
                    cx = REFLECT(c + fc - coff, cols);
                    sum = _mm_add_ps(sum, _mm_mul_ps(in[IDX(rx, cx, cols)], filt[IDX(fr, fc, filt_cols)]));
                }
            }
            out[IDX(r, c, cols)] = sum;
        }
    }
}

void Polarization::filter_reflect(const size_t rows, const size_t cols, const __m128* in, __m128* out, const size_t filt_size, const __m128* filt)
{
    int off = filt_size / 2;
    int r, c, fi, rx, cx;
    __m128 sum;
    //apply to rows
    #pragma omp parallel for private(r,c,sum,fi,rx)
    for (r = 0; r < rows; ++r) {
        for (c = 0; c < cols; ++c) {
            sum = _mm_set1_ps(0);
            for (fi = 0; fi < filt_size; ++fi) {
                rx = REFLECT(r + fi - off, rows);
                sum = _mm_add_ps(sum, _mm_mul_ps(in[IDX(rx,c,cols)],filt[fi]));
            }
            out[IDX(r,c,cols)] = sum;
        }
    }
    //apply to columns:
    #pragma omp parallel for private(r,c,sum,fi,cx)
    for (r = 0; r < rows; ++r) {
        for (c = 0; c < cols; ++c) {
            sum = out[IDX(r,c,cols)];
            for (fi = 0; fi < filt_size; ++fi) {
                cx = REFLECT(c + fi - off, cols);
                sum = _mm_add_ps(sum, _mm_mul_ps(in[IDX(r,cx,cols)],filt[fi]));
            }
            out[IDX(r,c,cols)] = sum;
        }
    }
}

//does dot(R[i], img[i]) where R[i] is row-major
void Polarization::stokes(const size_t n, const __m128* img, const __m128* R, __m128* out) {
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        out[i] = dot(&R[i], img[i]);
    }
}
//does dot(R, img[i]) where R is row-major
void Polarization::stokesR(const size_t n, const __m128* img, const __m128 R[4], __m128* out) {
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        out[i] = dot(R, img[i]);
    }
}
//does dot(img[i], R[i]) where R[i] is column-major
void Polarization::stokes2(const size_t n, const __m128* img, const __m128* R, __m128* out) {
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        out[i] = dot(img[i], &R[i]);
    }
}
//does dot(img[i], R) where R is column-major
void Polarization::stokes2R(const size_t n, const __m128* img, const __m128 R[4], __m128* out) {
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        out[i] = dot(img[i], R);
    }
}
//assumes 0,90,45,135 pixel pattern:
void Polarization::stokes(const size_t n, const __m128* img, __m128* out) {
    __m128 R[] = {
        _mm_setr_ps(0.5,  1.0,  0.0, 0.0),
        _mm_setr_ps(0.5, -1.0,  0.0, 0.0),
        _mm_setr_ps(0.5,  0.0,  1.0, 0.0),
        _mm_setr_ps(0.5,  0.0, -1.0, 0.0)
    };
    stokes2R(n, img, R, out);
}

void Polarization::element(const size_t param, const size_t n, const __m128* simg, float* out) {
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        out[i] = simg[i].m128_f32[param];
    }
}

//degree of polarization: hypot(s1,s2,s3)/s0
void Polarization::dop(const size_t n, const __m128* simg, float* out) {
    __m128 tmp;
    float sum;
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        //square each element
        tmp = _mm_mul_ps(simg[i], simg[i]);
        //(s1^2+s2^2+s3^2)/s0^2
        tmp.m128_f32[0] = (tmp.m128_f32[1] + tmp.m128_f32[2] + tmp.m128_f32[3])/tmp.m128_f32[0];
        out[i] = _mm_sqrt_ss(tmp).m128_f32[0];
    }
}

//degree of linear polarization: hypot(s1,s2)/s0
void Polarization::dolp(const size_t n, const __m128* simg, float* out) {
    __m128 tmp;
    float sum;
    #pragma omp parallel for private(tmp, sum)
    for (int i = 0; i < n; ++i) {
        //square each element
        tmp = _mm_mul_ps(simg[i], simg[i]);
        //(s1^2 + s2^2)/s0^2
        tmp.m128_f32[0] = (tmp.m128_f32[1] + tmp.m128_f32[2]) / tmp.m128_f32[0];
        out[i] = _mm_sqrt_ss(tmp).m128_f32[0];
    }
}

//degree of circular polarization: abs(s3)/s0
void Polarization::docp(const size_t n, const __m128* simg, float* out){
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        out[i] = fabs(simg[i].m128_f32[3] / simg[i].m128_f32[0]);
    }
}

//angle of polarization: 0.5*atan(s2/s1)
void Polarization::aop(const size_t n, const __m128* simg, float* out) {
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        out[i] = 0.5*atan2f(simg[i].m128_f32[2], simg[i].m128_f32[1]);
    }
}

//2x angle of polarization: atan(s2/s1)
void Polarization::aopx2(const size_t n, const __m128* simg, float* out) {
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        out[i] = atan2f(simg[i].m128_f32[2], simg[i].m128_f32[1]);
    }
}

//ellipticity angle: 0.5*asin(s3/s0)
void Polarization::ella(const size_t n, const __m128* simg, float* out) {
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        out[i] = 0.5*asinf(simg[i].m128_f32[3] / simg[i].m128_f32[0]);
    }
}
