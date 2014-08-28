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

#include <vectorclass.h>
#include <cstdint>
#include <array>

#ifndef POL_STATIC
  #ifdef POL_DLL_EXPORT
    #define DLL_PUBLIC __declspec(dllexport) //gcc supports this syntax
  #else
    #define DLL_PUBLIC __declspec(dllimport) //gcc supports this syntax
  #endif
  #if __GNUC__ >= 4
    #define DLL_PRIVATE __attribute__((visibility("hidden")))
  #else
    #define DLL_PRIVATE
  #endif 
#else 
  #define DLL_PUBLIC
  #define DLL_PRIVATE
#endif

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
#define PATRC(p,n) ((PATR(p,n) << 1) | PATC(p,n))

//compute an index for a row-major storage matrix
#define IDX(r,c,cols) ((r)*(cols)+(c))
//wrap an index 'x' from 0 to max
#define WRAP(x,max) ((x) < 0 ? ((max) - ((max) % -(x))) : ((max) % (x)))
//reflect an index x between 0 to max
#define REFLECT(x,max) (WRAP((x),2*(max)) > (max) ? 2*(max)-WRAP((x),2*(max)) : WRAP((x),2*(max)))

#ifndef POL_STATIC
#define default_pattern (PIX0(0, 0) | PIX1(1, 1) | PIX2(0, 1) | PIX3(1, 0))
#endif

namespace Polarization {

    #ifdef POL_STATIC
    extern const uint8_t default_pattern;
    #endif

    struct DLL_PUBLIC Mat16f {
        Vec4f r0, r1, r2, r3;
        Mat16f();
        Mat16f(const float* m);
        Mat16f(const Vec4f& r0, const Vec4f& r1, const Vec4f& r2, const Vec4f& r3);
        Mat16f(const Mat16f& m);
        void load(const float* m);
        void store(float * m) const;
        void trans();
        Mat16f cofactors() const;
        float det() const;
        Mat16f inv(float& det) const;
        Mat16f inv_left(float& det) const;
        Mat16f inv_right(float& det) const;
    };

    DLL_PUBLIC Mat16f dot(const Mat16f& m1, const Mat16f& m2);
    DLL_PUBLIC Vec4f dot(const Mat16f& m, const Vec4f& v);
    DLL_PUBLIC Vec4f dot(const Vec4f& v, const Mat16f& m);
    DLL_PUBLIC float dot(const Vec4f& v1, const Vec4f& v2);

    
    DLL_PUBLIC uint8_t encode_pattern(int p00, int p01, int p10, int p11);
    DLL_PUBLIC void decode_pattern(uint8_t pattern, int& p00, int& p01, int& p10, int& p11);

    //packs an image's superpixels into vectors according to pattern
    //rearranges a (h,w) image into (h/2,w/2,4) image
    //the pattern is specified by PIX0(r,c) to PIX3(r,c) macros (or PIXN(r,c,n))
    //PIX0(r,c) means that pixel 0 of the superpixel vector appears in row,col = r,c of the 2x2 super pixel.
    template<typename raw_type>
    void pack_superpixels(const size_t rows, const size_t cols, const raw_type* raw, float* out, uint8_t pattern = default_pattern) {
        size_t r,c;
        #pragma omp parallel for private(r,c)
        for(size_t r = 0; r < rows; r += 2) {
            for(size_t c = 0; c < cols; c += 2) {
                out[IDX(r,c,cols)]   = float(raw[IDX(r | PATR(pattern, 0), c | PATC(pattern,0), cols)]);
                out[IDX(r,c,cols)+1] = float(raw[IDX(r | PATR(pattern, 1), c | PATC(pattern,1), cols)]);
                out[IDX(r,c,cols)+2] = float(raw[IDX(r | PATR(pattern, 2), c | PATC(pattern,2), cols)]);
                out[IDX(r,c,cols)+3] = float(raw[IDX(r | PATR(pattern, 3), c | PATC(pattern,3), cols)]);
            }
        }
    }

    //reverse the operation of pack_superpixels
    DLL_PUBLIC void unpack_superpixels(const size_t rows, const size_t cols, const float* packed, float* out, uint8_t pattern = default_pattern);
    
    //expands the superpixel vector 4x1 into a 2x2x4 shape according to pattern.
    //e.g. {a,b,c,d} -> {{{a,0,0,0},{0,0,c,0}},{{0,0,0,d},{0,b,0,0}}} for pattern = default_pattern
    DLL_PUBLIC void expand_superpixels(const size_t rows, const size_t cols, const float* packed, float* out, uint8_t pattern = default_pattern);

    DLL_PUBLIC void repack_superpixels(const size_t n, const float* packed, float* repacked, uint8_t old_pattern, uint8_t new_pattern);


    DLL_PUBLIC void mask_low_high(const size_t n, const float* raw, float low, float high, float* out);
    DLL_PUBLIC void mask_low(const size_t n, const float* raw, float low, float* out);
    DLL_PUBLIC void mask_high(const size_t n, const float* raw, float high, float* out);

    //packs super-pixels and does dot(gains[i], raw[i] - darks[i])
    template<typename raw_type>
    void calibrate_matrix(const size_t rows, const size_t cols, const raw_type* raw, const float* darks, const float* gains, float* out, uint8_t pattern = default_pattern) {
        size_t r,c;
        Vec4f v, d;
        Mat16f g;
        #pragma omp parallel for private(r,c,v,d,g)
        for(size_t r = 0; r < rows; r += 2) {
            for(size_t c = 0; c < cols; c += 2) {
                v = Vec4f(
                        float(raw[IDX(r | PATR(pattern, 0), c | PATC(pattern,0), cols)]),
                        float(raw[IDX(r | PATR(pattern, 1), c | PATC(pattern,1), cols)]),
                        float(raw[IDX(r | PATR(pattern, 2), c | PATC(pattern,2), cols)]),
                        float(raw[IDX(r | PATR(pattern, 3), c | PATC(pattern,3), cols)])
                    );
                d.load(darks + IDX(r,c,cols));
                g.load(gains + IDX(r,c,cols)*4);
                dot(g, v-d).store(out + IDX(r,c,cols));
            }
        }
    }

    //does dot(gains[i], raw[i] - darks[i])
    DLL_PUBLIC void calibrate_matrix(const size_t n, const float* raw, const float* darks, const float* gains, float* out);

    DLL_PUBLIC struct edge_mode {
        enum t {
            ZERO,
            WRAP,
            REFLECT
        };
    };

    //2D filtering (with 2D kernel)
    DLL_PUBLIC void filter(const size_t rows, const size_t cols, const float* in, float* out, const size_t filt_rows, const size_t filt_cols, const float* filt, const edge_mode::t mode);
    DLL_PUBLIC void filter_zero(const size_t rows, const size_t cols, const float* in, float* out, const size_t filt_rows, const size_t filt_cols, const float* filt);
    DLL_PUBLIC void filter_wrap(const size_t rows, const size_t cols, const float* in, float* out, const size_t filt_rows, const size_t filt_cols, const float* filt);
    DLL_PUBLIC void filter_reflect(const size_t rows, const size_t cols, const float* in, float* out, const size_t filt_rows, const size_t filt_cols, const float* filt);

    //2D filtering (with 1D kernel applied on both axes)
    DLL_PUBLIC void filter(const size_t rows, const size_t cols, const float *in, float *out, const size_t filt_size, const float *filt, const edge_mode::t mode);
    DLL_PUBLIC void filter_zero(const size_t rows, const size_t cols, const float *in, float *out, const size_t filt_size, const float *filt);
    DLL_PUBLIC void filter_wrap(const size_t rows, const size_t cols, const float *in, float *out, const size_t filt_size, const float *filt);
    DLL_PUBLIC void filter_reflect(const size_t rows, const size_t cols, const float *in, float *out, const size_t filt_size, const float *filt);

    //compute stokes vector
    //assumes 
    DLL_PUBLIC void stokes(const size_t n, const float* img, float* out);
    DLL_PUBLIC void stokes(const size_t n, const float* img, const float* R, float* out);
    DLL_PUBLIC void stokes_r(const size_t n, const float* img, const float* R, float* out);

    //extract a single element from the float
    DLL_PUBLIC void element(const size_t param, const size_t n, const float* simg, float* out);

    //degree of polarization: hypot(s1,s2,s3)/s0
    DLL_PUBLIC void dop(const size_t n, const float* simg, float* out);

    //degree of linear polarization: hypot(s1,s2)/s0
    DLL_PUBLIC void dolp(const size_t n, const float* simg, float* out);

    //degree of circular polarization: abs(s3)/s0
    DLL_PUBLIC void docp(const size_t n, const float* simg, float* out);

    //angle of polarization: 0.5*atan(s2/s1)
    DLL_PUBLIC void aop(const size_t n, const float* simg, float* out);

    //2x angle of polarization: atan(s2/s1)
    DLL_PUBLIC void aopx2(const size_t n, const float* simg, float* out);

    //ellipticity angle: 0.5*asin(s3/s0)
    DLL_PUBLIC void ella(const size_t n, const float* simg, float* out);

    //TODO:
    //decompose stokes vector?

}


#endif // POLARIZATION_HPP
