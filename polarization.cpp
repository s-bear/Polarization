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
#include <algorithm>

#ifdef POL_STATIC
const uint8_t Polarization::default_pattern = (PIX0(0, 0) | PIX1(1, 1) | PIX2(0, 1) | PIX3(1, 0));
#endif

Polarization::Mat16f() {}
Polarization::Mat16f(const float* m) {
    load(m);
}
Polarization::Mat16f(const Vec4f& r0, const Vec4f& r1, const Vec4f& r2, const Vec4f& r3) : r0(r0), r1(r1), r2(r2), r3(r3) {}
Polarization::Mat16f(const Mat16f& m) : r0(m.r0), r1(m.r1), r2(m.r2), r3(m.r3) {}

void Polarization::Mat16f::load(const float* m) {
    r0.load(mat);
    r1.load(mat+4);
    r2.load(mat+8);
    r3.load(mat+12);
}

void Polarization::Mat16f::store(float * m) const {
    r0.store(mat);
    r1.store(mat+4);
    r2.store(mat+8);
    r3.store(mat+12);
}

void Polarization::Mat16f::trans() {
    Vec8f r01(r0, r1), r23(r2, r3);
    Vec8f c01 = blend8f<0, 4, 8, 12, 1, 5, 9, 13>(r01, r23);
    Vec8f c23 = blend8f<2, 6, 10, 14, 3, 7, 11, 15>(r01, r23);
    r0 = c01.get_low();
    r1 = c01.get_high();
    r2 = c23.get_low();
    r3 = c23.get_high();
}

Polarization::Mat16f Polarization::Mat16f::cofactors() const {
    // 4x4 Matrix:
    // r0 = (a b c d)
    // r1 = (e f g h)
    // r2 = (i j k l)
    // r3 = (m n o p)

    //Compute cofactors:
    // C(i,j) = (-1)^(i+j+1)*M(i,j)    (i,j = 0,1,2,3)
    // M(i,j) = det(B) where B = A with i,j row, col eliminated

    //det(a b c; d e f; g h i) = a(ei-fh)-b(di-fg)+c(dh-eg)
    //  = aei + bfg + cdh - ceg - bdi - afh

    // r0 = (a b c d); r0[2 3 0 1] = (c d a b)

    //row0 (abcd), row1 (efgh) products
    // p0 = af, be, ch, dg = (a b c d)*(f e h g) = r0[0 1 2 3]*r1[1 0 3 2]
    // p1 = cf, de, ah, bg = (c d a b)*(f e h g) = r0[2 3 0 1]*r1[1 0 3 2]
    // p2 = ce, df, ag, bh = (c d a b)*(e f g h) = r0[2 3 0 1]*r1[0 1 2 3]

    //row2 (ijkl), row3 (mnop) products
    // p3 = in, jm, kp, lo = r2*r3[1 0 3 2]
    // p4 = kn, lm, ip, jo = r2[2 3 0 1]*r3[1 0 3 2]
    // p5 = km, ln, io, jp = r2[2 3 0 1]*r3

    //c3[3 2 1 0] = sum of
    // af.k, be.l, ch.i, dg.j = +p0         *r2[2 3 0 1]
    // be.k, af.l, dg.i, ch.j = -p0[1 0 3 2]*r2[2 3 0 1]
    // bg.i, ah.j, de.k, cf.l = +p1[3 2 1 0]*r2         
    // cf.i, de.j, ah.k, bg.l = -p1         *r2         
    // ce.j, df.i, ag.l, bh.k = +p2         *r2[1 0 3 2]
    // ag.j, bh.i, ce.l, df.k = -p2[2 3 0 1]*r2[1 0 3 2]

    //c2[3 2 1 0] = sum of
    // be.o, af.p, dg.m, ch.n = +p0[1 0 3 2]*r3[2 3 0 1]
    // af.o, be.p, ch.m, dg.n = -p0         *r3[2 3 0 1]
    // cf.m, de.n, ah.o, bg.p = +p1         *r3         
    // bg.m, ah.n, de.o, cf.p = -p1[3 2 1 0]*r3         
    // ag.n, bh.m, ce.p, df.o = +p2[2 3 0 1]*r3[1 0 3 2]
    // ce.n, df.m, ag.p, bh.o = -p2         *r3[1 0 3 2]

    //c1[3 2 1 0] = sum of
    // c.in, d.jm, a.kp, b.lo = +r0[2 3 0 1]*p3         
    // c.jm, d.in, a.lo, b.kp = -r0[2 3 0 1]*p3[1 0 3 2]
    // a.jo, b.ip, c.lm, d.kn = +r0         *p4[3 2 1 0]
    // a.kn, b.lm, c.ip, d.jo = -r0         *p4         
    // b.km, a.ln, d.io, c.jp = +r0[1 0 3 2]*p5         
    // b.io, a.jp, d.km, c.ln = -r0[1 0 3 2]*p5[2 3 0 1]

    //c0[3 2 1 0] = sum of
    // g.jm, h.in, e.lo, f.kp = +r1[2 3 0 1]*p3[1 0 3 2]
    // g.in, h.jm, e.kp, f.lo = -r1[2 3 0 1]*p3         
    // e.kn, f.lm, g.ip, h.jo = +r1         *p4         
    // e.jo, f.ip, g.lm, h.kn = -r1         *p4[3 2 1 0]
    // f.io, e.jp, h.km, g.ln = +r1[1 0 3 2]*p5[2 3 0 1]
    // f.km, e.ln, h.io, g.jp = -r1[1 0 3 2]*p5         

    Vec4f r0_1032 = permute4f<1, 0, 3, 2>(r0);
    Vec4f r0_2301 = permute4f<2, 3, 0, 1>(r0);

    Vec4f r1_1032 = permute4f<1, 0, 3, 2>(r1);
    Vec4f r1_2301 = permute4f<2, 3, 0, 1>(r1);

    Vec4f r2_1032 = permute4f<1, 0, 3, 2>(r2);
    Vec4f r2_2301 = permute4f<2, 3, 0, 1>(r2);

    Vec4f r3_1032 = permute4f<1, 0, 3, 2>(r3);
    Vec4f r3_2301 = permute4f<2, 3, 0, 1>(r3);

    Vec4f p0 = r0*r1_1032;
    Vec4f p1 = r0_2301*r1_1032;
    Vec4f p2 = r0_2301*r1;

    Vec4f p3 = r2*r3_1032;
    Vec4f p4 = r2_2301*r3_1032;
    Vec4f p5 = r2_2301*r3;

    Vec4f p0_1032 = permute4f<1, 0, 3, 2>(p0);
    Vec4f p1_3210 = permute4f<3, 2, 1, 0>(p1);
    Vec4f p2_2301 = permute4f<2, 3, 0, 1>(p2);

    Vec4f p3_1032 = permute4f<1, 0, 3, 2>(p3);
    Vec4f p4_3210 = permute4f<3, 2, 1, 0>(p4);
    Vec4f p5_2301 = permute4f<2, 3, 0, 1>(p5);

    Mat16f cf;

    cf.r0 = permute4f<3, 2, 1, 0>((p3_1032 - p3)*r1_2301 + (p4 - p4_3210)*r1 + (p5_2301 - p5)*r1_1032);
    cf.r1 = permute4f<3, 2, 1, 0>((p3 - p3_1032)*r0_2301 + (p4_3210 - p4)*r0 + (p5 - p5_2301)*r0_1032);
    cf.r2 = permute4f<3, 2, 1, 0>((p0_1032 - p0)*r3_2301 + (p1 - p1_3210)*r3 + (p2_2301 - p2)*r3_1032);
    cf.r3 = permute4f<3, 2, 1, 0>((p0 - p0_1032)*r2_2301 + (p1_3210 - p1)*r2 + (p2 - p2_2301)*r2_1032);

    return cf;
}

float Polarization::Mat16f::det() const {
    Vec4f r1_1032 = permute4f<1, 0, 3, 2>(r1);
    Vec4f r1_2301 = permute4f<2, 3, 0, 1>(r1);
    Vec4f r2_2301 = permute4f<2, 3, 0, 1>(r2);
    Vec4f r3_1032 = permute4f<1, 0, 3, 2>(r3);
    Vec4f p3 = r2*r3_1032;
    Vec4f p4 = r2_2301*r3_1032;
    Vec4f p5 = r2_2301*r3;
    Vec4f p3_1032 = permute4f<1, 0, 3, 2>(p3);
    Vec4f p4_3210 = permute4f<3, 2, 1, 0>(p4);
    Vec4f p5_2301 = permute4f<2, 3, 0, 1>(p5);
    Vec4f c0 = permute4f<3, 2, 1, 0>((p3_1032 - p3)*r1_2301 + (p4 - p4_3210)*r1 + (p5_2301 - p5)*r1_1032);
    return horizontal_add(r0*c0);
}

Polarization::Mat16f Polarization::Mat16f::inv(float& det) const {
        //load matrix
    Mat16f c = cofactors();
    det = horizontal_add(r0*c.r0);
    c.trans();
    c.r0 /= det;
    c.r1 /= det;
    c.r2 /= det;
    c.r3 /= det;
    return c;
}

Polarization::Mat16f Polarization::Mat16f::inv_left(float& det) const {
    //inv(A'.A).A'
    Mat16f t(*this);
    t.trans();
    return dot(dot(t,*this).inv(det), t);
}

Polarization::Mat16f Polarization::Mat16f::inv_right(float& det) const {
    //A'.inv(A.A')
    Mat16f t(*this);
    t.trans();
    return dot(t, dot(*this,t).inv(det));
}

Mat16f Polarization::dot(const Mat16f& m1, const Mat16f& m2) {
    Mat16f c;
    c.r0 = m1.r0[0] * m2.r0 + m1.r0[1] * m2.r1 + m1.r0[2] * m2.r2 + m1.r0[3] * m2.r2;
    c.r1 = m1.r1[0] * m2.r0 + m1.r1[1] * m2.r1 + m1.r1[2] * m2.r2 + m1.r1[3] * m2.r2;
    c.r2 = m1.r2[0] * m2.r0 + m1.r2[1] * m2.r1 + m1.r2[2] * m2.r2 + m1.r2[3] * m2.r2;
    c.r3 = m1.r3[0] * m2.r0 + m1.r3[1] * m2.r1 + m1.r3[2] * m2.r2 + m1.r3[3] * m2.r2;
    return c;
}

Vec4f Polarization::dot(const Mat16f& m, const Vec4f& v) {
    Mat16f t(m);
    t.trans();
    return t.r0*v[0] + t.r1*v[1] + t.r2*v[2] + t.r3*v[3];
}

Vec4f Polarization::dot(const Vec4f& v, const Mat16f& m) {
    return v[0]*m.r0 + v[1]*m.r1 + v[2]*m.r2 + v[3]*m.r3;
}

float Polarization::dot(const Vec4f& v1, const Vec4f& v2) {
    return horizontal_add(v1*v2);
}

uint8_t Polarization::encode_pattern(int p00, int p01, int p10, int p11) {
    return (uint8_t)(PIXN(0,0,p00) | PIXN(0,1,p01) | PIXN(1,0,p10) | PIXN(1,1,p11));
}

void Polarization::decode_pattern(uint8_t pattern, int &p00, int &p01, int &p10, int &p11) {
    for(int n = 0; n < 4; ++n) {
        switch(PATRC(pattern,n)) {
        case 0: p00 = n; break;
        case 1: p01 = n; break;
        case 2: p10 = n; break;
        case 3: p11 = n; break;
        }
    }
}

void Polarization::unpack_superpixels(const size_t rows, const size_t cols, const float* packed, float* out, uint8_t pattern) {
    size_t rows_2 = rows / 2, cols_2 = cols / 2;
    int r, c, rx2, cx2;
#pragma omp parallel for private(r,c,rx2,cx2)
    for (r = 0; r < rows_2; ++r) {
        for (c = 0; c < cols_2; ++c) {
            rx2 = r * 2; cx2 = c * 2;
            Vec4f p;
            p.load(&packed[4 * IDX(r, c, cols_2)]);
            //__m128 p = packed[IDX(r, c, cols_2)];
            out[IDX(rx2 | PATR(pattern, 0), cx2 | PATC(pattern, 0), cols)] = p.extract(0); //_mm_setr_ps(p.m128_f32[0], 0, 0, 0);
            out[IDX(rx2 | PATR(pattern, 1), cx2 | PATC(pattern, 1), cols)] = p.extract(1); //_mm_setr_ps(0, p.m128_f32[1], 0, 0);
            out[IDX(rx2 | PATR(pattern, 2), cx2 | PATC(pattern, 2), cols)] = p.extract(2); //_mm_setr_ps(0, 0, p.m128_f32[2], 0);
            out[IDX(rx2 | PATR(pattern, 3), cx2 | PATC(pattern, 3), cols)] = p.extract(3); //_mm_setr_ps(0, 0, 0, p.m128_f32[3]);
        }
    }
}

void Polarization::expand_superpixels(const size_t rows, const size_t cols, const float* packed, float* out, uint8_t pattern) {
    size_t rows_2 = rows / 2, cols_2 = cols / 2;
    int r, c, rx2, cx2;
#pragma omp parallel for private(r,c,rx2,cx2)
    for (r = 0; r < rows_2; ++r) {
        for (c = 0; c < cols_2; ++c) {
            rx2 = r * 2; cx2 = c * 2;
            Vec4f p;
            p.load(&packed[4 * IDX(r, c, cols_2)]);
            for (int i = 0; i < 16; ++i) {
                out[IDX(rx2, cx2, cols) + i] = 0.0f;
            }
            out[IDX(rx2 | PATR(pattern, 0), cx2 | PATC(pattern, 0), cols) + 0] = p.extract(0);
            out[IDX(rx2 | PATR(pattern, 1), cx2 | PATC(pattern, 1), cols) + 4] = p.extract(1);
            out[IDX(rx2 | PATR(pattern, 2), cx2 | PATC(pattern, 2), cols) + 8] = p.extract(2);
            out[IDX(rx2 | PATR(pattern, 3), cx2 | PATC(pattern, 3), cols) + 12] = p.extract(3);
        }
    }
}

void repack_superpixels(const size_t n, const float* packed, float* repacked, uint8_t old_pattern, uint8_t new_pattern) {
    if (new_pattern == old_pattern && repacked != packed) {
        //just copy
        std::copy(packed,packed+n,repacked);
    }
    else {
        int old[4], order[4];
        //for each index in new_pattern, where was it in old_pattern?
        decode_pattern(old_pattern,old[0],old[1],old[2],old[3]);
        for(i = 0; i < 4; ++i)
            order[i] = old[PATRC(new_pattern,i)];
        //do the shuffle
        for(size_t i = 0; i < n; i += 4) {
            repacked[i]   = packed[i + order[0]];
            repacked[i+1] = packed[i + order[1]]; 
            repacked[i+2] = packed[i + order[2]];
            repacked[i+3] = packed[i + order[3]];
        }
    }
}

void Polarization::mask_low_high(const size_t n, const float* raw, float low, float high, float* out) {
    Vec4f low_vec(low);
    Vec4f high_vec(high);
    Vec4f nan_vec = nan4f(0);
    Vec4f val, mask;
    #pragma omp parallel for private(val, mask)
    for (size_t i = 0; i < n; i += 4) {
        val.load(raw+i);
        mask = (val < low_vec) | (val > high_vec);
        val = (~mask & val) | (mask & nan_vec);
        val.store(out+i);
    }
}
void Polarization::mask_low(const size_t n, const float* raw, float low, float* out) {
    Vec4f low_vec(low);
    Vec4f high_vec(high);
    Vec4f nan_vec = nan4f(0);
    Vec4f val, mask;
    #pragma omp parallel for private(val, mask)
    for (size_t i = 0; i < n; i += 4) {
        val.load(raw+i);
        mask = (val < low_vec);
        val = (~mask & val) | (mask & nan_vec);
        val.store(out+i);
    }
}
void Polarization::mask_high(const size_t n, const float* raw, float high, float* out) {
    Vec4f low_vec(low);
    Vec4f high_vec(high);
    Vec4f nan_vec = nan4f(0);
    Vec4f val, mask;
    #pragma omp parallel for private(val, mask)
    for (size_t i = 0; i < n; i += 4) {
        val.load(raw+i);
        mask = (val > high_vec);
        val = (~mask & val) | (mask & nan_vec);
        val.store(out+i);
    }
}

//does dot(gain, raw-dark)
void Polarization::calibrate_matrix(const size_t n, const float* raw, const float* darks, const float* gains, float* out) {
    Vec4f r, d;
    Mat16f g;
    #pragma omp parallel for private(r,d,g)
    for (int i = 0; i < n; i += 4) {
        // gains * (raw - darks)
        r.load(raw + i);
        d.load(darks + i);
        g.load(gains + i*4);
        dot(g, r-d).store(out + i);
    }
}

//doead dot(raw - dark, gain)  (slightly faster)
void Polarization::calibrate_matrix_t(const size_t n, const float* raw, const float* darks, const float* gains, float* out) {
    Vec4f r, d;
    Mat16f g;
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        out[i] = dot(_mm_sub_ps(raw[i], darks[i]), &gains[i]);
    }
}

void Polarization::filter(const size_t rows, const size_t cols, const float* in, float* out, const size_t filt_rows, const size_t filt_cols, const float* filt, const edge_mode::t mode) {
    switch (mode) {
    case edge_mode::ZERO:
        return filter_zero(rows, cols, in, out, filt_rows, filt_cols, filt);
    case edge_mode::WRAP:
        return filter_wrap(rows, cols, in, out, filt_rows, filt_cols, filt);
    case edge_mode::REFLECT:
        return filter_reflect(rows, cols, in, out, filt_rows, filt_cols, filt);
    }
}

void Polarization::filter(const size_t rows, const size_t cols, const float* in, float* out, const size_t filt_size, const float* filt, const edge_mode::t mode) {
    switch (mode) {
    case edge_mode::ZERO:
        return filter_zero(rows, cols, in, out, filt_size, filt);
    case edge_mode::WRAP:
        return filter_wrap(rows, cols, in, out, filt_size, filt);
    case edge_mode::REFLECT:
        return filter_reflect(rows, cols, in, out, filt_size, filt);
    }
}

void Polarization::filter_zero(const size_t rows, const size_t cols, const float* in, float* out, const size_t filt_rows, const size_t filt_cols, const float* filt)
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

void Polarization::filter_zero(const size_t rows, const size_t cols, const float* in, float* out, const size_t filt_size, const float* filt)
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

void Polarization::filter_wrap(const size_t rows, const size_t cols, const float* in, float* out, const size_t filt_rows, const size_t filt_cols, const float* filt)
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

void Polarization::filter_wrap(const size_t rows, const size_t cols, const float* in, float* out, const size_t filt_size, const float* filt)
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

void Polarization::filter_reflect(const size_t rows, const size_t cols, const float* in, float* out, const size_t filt_rows, const size_t filt_cols, const float* filt)
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

void Polarization::filter_reflect(const size_t rows, const size_t cols, const float* in, float* out, const size_t filt_size, const float* filt)
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

void Polarization::stokes(const size_t n, const float* img, float* out) {
    float R[] = {0.5, 0.5, 0.5, 0.5,
               1.0,-1.0, 0.0, 0.0,
               0.0, 0.0, 1.0,-1.0,
               0.0, 0.0, 0.0, 0.0};
   stokes(n,img,R,out);
}

void Polarization::stokes(const size_t n, const float* img, const float* R, float* out) {
    Mat16f r(R);
    Vec4f v;
    #pragma omp parallel for private(v)
    for(size_t i = 0; i < n; i += 4) {
        v.load(img+i);
        dot(r,v).store(out+i);
    }
}

void Polarization::stokes_r(const size_t n, const float* img, const float* R, float* out) {
    Mat16f r;
    Vec4f v;
    #pragma omp parallel for private(r,v)
    for(size_t i = 0; i < n; i += 4) {
        v.load(img + i);
        r.load(R + i*4);
        dot(r,v).store(out+i);
    }
}

void Polarization::element(const size_t param, const size_t n, const float* simg, float* out) {
    #pragma omp parallel for
    for (size_t i = 0; i < n; i += 4) {
        out[i/4] = simg[i+param];
    }
}

//degree of polarization: hypot(s1,s2,s3)/s0
void Polarization::dop(const size_t n, const float* simg, float* out) {
    Vec4f v,s;
    #pragma omp parallel for private(v,s,d)
    for (size_t i = 0; i < n; i += 4) {
        v.load(simg+i);
        s = square(v);
        out[i/4] = sqrt(s[1]+s[2]+s[3])/v[0];
    }
}

//degree of linear polarization: hypot(s1,s2)/s0
void Polarization::dolp(const size_t n, const float* simg, float* out) {
    Vec4f v,s;
    #pragma omp parallel for private(v,s,d)
    for (size_t i = 0; i < n; i += 4) {
        v.load(simg+i);
        s = square(v);
        out[i/4] = sqrt(s[1]+s[2])/v[0];
    }
}

//degree of circular polarization: abs(s3)/s0
void Polarization::docp(const size_t n, const float* simg, float* out){
    #pragma omp parallel for
    for (size_t i = 0; i < n; i += 4) {
        out[i/4] = fabs(simg[i+3]) / simg[i]; //TODO: loop unrolling w/ SSE
    }
}

//angle of polarization: 0.5*atan(s2/s1)
void Polarization::aop(const size_t n, const float* simg, float* out) {
    #pragma omp parallel for
    for (size_t i = 0; i < n; i += 4) {
        out[i/4] = 0.5*atan2f(simg[i+2], simg[i+1]); //TODO: loop unrolling w/ SSE
    }
}

//2x angle of polarization: atan(s2/s1)
void Polarization::aopx2(const size_t n, const float* simg, float* out) {
    #pragma omp parallel for
    for (size_t i = 0; i < n; i += 4) {
        out[i/4] = atan2f(simg[i+2], simg[i+1]); //TODO: loop unrolling w/ SSE
    }
}

//ellipticity angle: 0.5*asin(s3/s0)
void Polarization::ella(const size_t n, const float* simg, float* out) {
    #pragma omp parallel for
    for (size_t i = 0; i < n; i += 4) {
        out[i] = 0.5*asinf(simg[i+3] / simg[i]);
    }
}
