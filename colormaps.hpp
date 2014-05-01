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
#pragma once
#ifndef COLORMAPS_HPP
#define COLORMAPS_HPP

#include <vector>
#include <string>
#include <functional>
#include <algorithm>
#include <utility>
#include <unordered_map>
#include <cstdint>
#include <limits>
using namespace std;

typedef std::numeric_limits<float> flimits;

#ifdef MSVC_ISNAN
int isnan(double x);
int isnan(float x);
#endif
//A map that translates floats to uint32s

struct Colormap {
public:
    static uint32_t make_format(int red_byte=0, int green_byte=1, int blue_byte=2, int alpha_byte=3, uint8_t alpha_value=0xff);
    static uint32_t format_color(uint32_t format, uint8_t r, uint8_t g, uint8_t b);
    static uint32_t format_color(uint32_t format, uint8_t r, uint8_t g, uint8_t b, uint8_t a);
    static void get_rgb(uint32_t color, uint32_t format, uint8_t& r, uint8_t& g, uint8_t& b);
    static void get_rgba(uint32_t color, uint32_t format, uint8_t& r, uint8_t& g, uint8_t& b, uint8_t& a);
    static uint32_t reformat_color(uint32_t color, uint32_t old_format, uint32_t new_format);
    static const int min_idx, max_idx, nan_idx;
protected:
    uint32_t _apply(float val, float min, float max, float scale) const;
public:
    vector<uint32_t> map;
    uint32_t format;
    uint32_t min_color, max_color, nan_color;
    float min, max;


    //constructors:
    Colormap();
    Colormap(const Colormap& cm); //copy construct
    Colormap& operator=(const Colormap& cm); //copy assign
    void swap(Colormap& cm);
    Colormap(Colormap&& cm); //move
    Colormap& operator=(Colormap&& cm); //move assign

    size_t size() const;

    //copy each member:
    Colormap(const vector<uint32_t>& map, uint32_t format, uint32_t min_color = 0, uint32_t max_color = 0, uint32_t nan_color = 0, float min = flimits::quiet_NaN(), float max = flimits::quiet_NaN());
    //interpolate the map:
    Colormap(const size_t n, const vector<float>& vals, const vector<float>& gray, uint32_t format);
    Colormap(const size_t n, const vector<float>& vals, const vector<float>& red, const vector<float>& green, const vector<float>& blue, uint32_t format);
    Colormap(const size_t n, const vector<pair<float, float>> &red, const vector<pair<float, float>> &green, const vector<pair<float, float>> &blue, uint32_t format);

    //apply the map to a float:
    //if min, max = NAN, then we use the member min, max; if member min, max = NAN, then we use 0,1
    uint32_t apply(float val, float min = flimits::quiet_NaN(), float max = flimits::quiet_NaN()) const;
    void apply(float val, uint8_t &r, uint8_t &g, uint8_t &b, float min = flimits::quiet_NaN(), float max = flimits::quiet_NaN()) const;
    void apply(float val, uint8_t &r, uint8_t &g, uint8_t &b, uint8_t &a, float min = flimits::quiet_NaN(), float max = flimits::quiet_NaN()) const;
    uint32_t operator()(float val, float min = flimits::quiet_NaN(), float max = flimits::quiet_NaN()) const {
        return apply(val, min, max);
    }
    uint32_t operator[](float val) const {
        return apply(val);
    }

    uint32_t* data();
    const uint32_t* data() const;
    const uint32_t& get(int idx) const;
    void get(int idx, uint8_t& r, uint8_t& g, uint8_t& b) const;
    void get(int idx, uint8_t& r, uint8_t& g, uint8_t& b, uint8_t& a) const;

    void set(int idx, uint8_t r, uint8_t g, uint8_t b);
    void set(int idx, uint8_t r, uint8_t g, uint8_t b, uint8_t a);

    void reformat(uint32_t new_format);

    //apply to an iterable dataset
    //if min, max are not specified, then we use member min, max; if member min, max = NAN, then we find the min, max of the dataset
    template<typename InputIt, typename OutputIt>
    void apply(InputIt first, InputIt last, OutputIt out_first, float min = flimits::quiet_NaN(), float max = flimits::quiet_NaN()) const {
        //get min and max set up:
        bool find_min = false, find_max = false;
        if (isnan(min)) {
            min = this->min;
            if (isnan(min)) {
                min = flimits::infinity();
                find_min = true;
            }
        }
        if (isnan(max)) {
            max = this->max;
            if (isnan(max)) {
                max = -flimits::infinity();
                find_max = true;
            }
        }
        if (find_min || find_max) {
            InputIt it = first;
            while (it != last) {
                float val = *it++;
                if (find_min && val < min) min = val;
                if (find_max && val > max) max = val;
            }
        }
        //apply the colormap
        float scale = float(map.size()) / (max - min);
        while (first != last) {
            *out_first++ = _apply(*first++, min, max, scale);
        }
    }
    //apply to iterable collections (using std::begin and std::end)
    template<typename InputT, typename OutputT>
    void apply(const InputT& in, OutputT& out, const float& min = 0.0f, const float& max = 1.0f) const {
        apply(std::begin(in), std::end(in), std::begin(out), min, max);
    }
};

struct Colormaps {
    Colormap gray, hsv, jet;
    unordered_map<string, Colormap> maps;
    Colormaps(const std::string& filename, uint32_t format, size_t n = 256);

    void parse(const std::string& filename, uint32_t format, size_t n=256);

    bool has(const string& name) const;
    const Colormap& get(const string& name) const;
    Colormap& get(const string& name);
    Colormap& operator[](const string& name);
    Colormap& operator[](string&& name);

    Colormap& add(const string& name, const Colormap& cmap);
    Colormap& add(const string& name, Colormap&& cmap);

    unordered_map<string, Colormap>::iterator begin() {
        return maps.begin();
    }
    unordered_map<string, Colormap>::iterator end() {
        return maps.end();
    }
    unordered_map<string, Colormap>::const_iterator begin() const {
        return maps.begin();
    }
    unordered_map<string, Colormap>::const_iterator end() const {
        return maps.end();
    }
};

template<typename value_t>
vector<value_t> interpolate(size_t n, const vector<float>& positions, const vector<value_t> values) {
    vector<value_t> out(n);
    float step = 1.0f / (n - 1);
    if (values.size() != positions.size())
        throw runtime_error("positions and values must have same size!");
    size_t left_idx = 0, end_idx = positions.size();
    size_t right_idx = (left_idx + 1 == end_idx) ? left_idx : left_idx + 1;

    float left_pos = positions[left_idx], right_pos = positions[right_idx];
    value_t left_val = values[left_idx], right_val = values[right_idx];

    for (size_t i = 0; i < n; ++i) {
        float pos = i*step;
        //increment the left/right points until they straddle the current position
        while (right_pos <= pos && left_idx + 1 != end_idx) {
            ++left_idx;
            right_idx = (left_idx + 1 == end_idx) ? left_idx : left_idx + 1;
            left_pos = positions[left_idx], right_pos = positions[right_idx];
            left_val = values[left_idx], right_val = values[right_idx];
        }
        //take right or left values if below or above the position:
        if (pos >= right_pos) out[i] = right_val;
        else if (pos <= left_pos) out[i] = left_val;
        //otherwise linearly interpolate between the values:
        else out[i] = left_val + (right_val - left_val)*((pos - left_pos) / (right_pos - left_pos));
    }
    return out;
}

template<typename value_t>
vector<value_t> interpolate(size_t n, const vector<pair<float, value_t>> &pts) {
    vector<value_t> out(n);
    float step = 1.0f / (n - 1);
    auto left = pts.begin(), end = pts.end();
    auto right = left + 1 == end ? left : left + 1;
    float left_pos = left->first, right_pos = right->first;
    value_t left_val = left->second, right_val = right->second;
    for (size_t i = 0; i < n; ++i) {
        float pos = i*step;
        while (right_pos <= pos && left + 1 != end) {
            ++left;
            right = left + 1 == end ? left : left + 1;
            left_pos = left->first, left_val = left->second;
            right_pos = right->first, right_val = right->second;
        }
        if (pos >= right_pos) out[i] = right_val;
        else if (pos <= left_pos) out[i] = left_val;
        else out[i] = left_val + (right_val - left_val)*((pos - left_pos) / (right_pos - left_pos));
    }
    return out;
}

#endif // COLORMAPS_HPP
