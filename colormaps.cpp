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
#include "Colormaps.hpp"
#include <algorithm>
#include <fstream>
#include <locale>

#ifdef MSVC_ISNAN
int isnan(double x) {
    return _isnan(x);
}
int isnan(float x) {
    return _isnanf(x);
}
#endif

#undef min
#undef max

template<typename T, size_t N>
static vector<T> vec(T(&arr)[N]) {
    return vector<T>(begin(arr), end(arr));
}

uint32_t Colormap::make_format(int red_byte, int green_byte, int blue_byte, int alpha_byte, uint8_t alpha_value) {
    return (red_byte & 3) | ((green_byte & 3) << 2) | ((blue_byte & 3) << 4) | ((alpha_byte & 3) << 6) | (uint32_t(alpha_value) << 8);
}

uint32_t Colormap::format_color(uint32_t format, uint8_t r, uint8_t g, uint8_t b) {
    uint32_t a = ((format >> 8) & 0xff);
    return format_color(format,r,g,b,a);
}

uint32_t Colormap::format_color(uint32_t format, uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
    uint32_t rshift = (format & 3) * 8;
    uint32_t gshift = ((format >> 2) & 3) * 8;
    uint32_t bshift = ((format >> 4) & 3) * 8;
    uint32_t ashift = ((format >> 6) & 3) * 8;
    return (a << ashift) | (uint32_t(r) << rshift) | (uint32_t(g) << gshift) | (uint32_t(b) << bshift);
}

void Colormap::get_rgb(uint32_t color, uint32_t format, uint8_t& r, uint8_t& g, uint8_t& b) {
    uint8_t a;
    get_rgba(color,format,r,g,b,a);
}

void Colormap::get_rgba(uint32_t color, uint32_t format, uint8_t& r, uint8_t& g, uint8_t& b, uint8_t& a) {
    uint32_t rshift = (format & 3) * 8;
    uint32_t gshift = ((format >> 2) & 3) * 8;
    uint32_t bshift = ((format >> 4) & 3) * 8;
    uint32_t ashift = ((format >> 6) & 3) * 8;
    r = (color >> rshift)&0xff;
    g = (color >> gshift)&0xff;
    b = (color >> bshift)&0xff;
    a = (color >> ashift)&0xff;
}

uint32_t Colormap::reformat_color(uint32_t color, uint32_t old_format, uint32_t new_format) {
    uint8_t r,g,b,a;
    get_rgba(color, old_format, r, g, b, a);
    return format_color(new_format, r, g, b, a);
}

Colormap::Colormap() : map(1, 0), format(0), min_color(0), max_color(0), nan_color(0), min(flimits::quiet_NaN()), max(flimits::quiet_NaN()) {}

Colormap::Colormap(const Colormap& cm) : map(cm.map), format(cm.format), max_color(cm.max_color), min_color(cm.min_color), nan_color(cm.nan_color), min(cm.min), max(cm.max) {}

uint32_t Colormap::_apply(float val, float min, float max, float scale) const {
    if (isnan(val))
        return nan_color;
    else if (val < min)
        return min_color;
    else if (val >= max)
        return max_color;
    else {
        int idx((val - min)*scale);
        return map[idx];
    }
}

Colormap& Colormap::operator=(const Colormap& cm) {
    Colormap tmp(cm);
    swap(tmp);
    return *this;
}

void Colormap::swap(Colormap& cm) {
    std::swap(map, cm.map);
    std::swap(format, cm.format);
    std::swap(min_color, cm.min_color);
    std::swap(max_color, cm.max_color);
    std::swap(nan_color, cm.nan_color);
    std::swap(min, cm.min);
    std::swap(max, cm.max);
}

Colormap::Colormap(Colormap&& cm) : map(std::move(cm.map)), format(cm.format), max_color(cm.max_color), min_color(cm.min_color), nan_color(cm.nan_color), min(cm.min), max(cm.max) {}

Colormap& Colormap::operator=(Colormap&& cm) {
    Colormap tmp(std::move(cm));
    swap(tmp);
    return *this;
}

Colormap::Colormap(const vector<uint32_t>& map, uint32_t format, uint32_t min_color, uint32_t max_color, uint32_t nan_color, float min, float max) :
map(map), format(format), min_color(min_color), max_color(max_color), nan_color(nan_color), min(min), max(max) {}

Colormap::Colormap(const size_t n, const vector<float>& vals, const vector<float>& gray, uint32_t format) : map(n), format(format), min_color(0), max_color(0), nan_color(0), min(flimits::quiet_NaN()), max(flimits::quiet_NaN()) {
    auto g = interpolate(n, vals, gray);
    for (size_t i = 0; i < n; ++i) {
        uint8_t x = uint8_t(g[i] * 0xff);
        map[i] = format_color(format, x, x, x);
    }
    min_color = map[0];
    max_color = map[n - 1];
}
Colormap::Colormap(const size_t n, const vector<float>& vals, const vector<float>& red, const vector<float>& green, const vector<float>& blue, uint32_t format) : map(n), format(format), min_color(0), max_color(0), nan_color(0), min(flimits::quiet_NaN()), max(flimits::quiet_NaN()) {
    auto r = interpolate(n, vals, red);
    auto g = interpolate(n, vals, green);
    auto b = interpolate(n, vals, blue);
    for (size_t i = 0; i < n; ++i) {
        map[i] = format_color(format, uint8_t(r[i] * 0xff), uint8_t(g[i] * 0xff), uint8_t(b[i] * 0xff));
    }
    min_color = map[0];
    max_color = map[n - 1];
}
Colormap::Colormap(const size_t n, const vector<pair<float, float>> &red, const vector<pair<float, float>> &green, const vector<pair<float, float>> &blue, uint32_t format) : map(n), format(format), min_color(0), max_color(0), nan_color(0), min(flimits::quiet_NaN()), max(flimits::quiet_NaN()) {
    auto r = interpolate(n, red);
    auto g = interpolate(n, green);
    auto b = interpolate(n, blue);
    for (size_t i = 0; i < n; ++i) {
        map[i] = format_color(format, uint8_t(r[i] * 0xff), uint8_t(g[i] * 0xff), uint8_t(b[i] * 0xff));
    }
    min_color = map[0];
    max_color = map[n - 1];
}

size_t Colormap::size() const {
    return map.size();
}

void Colormap::reformat(uint32_t new_format) {
    if(new_format != format) {
        Colormap tmp(*this);
        tmp.format = new_format;
        tmp.min_color = reformat_color(min_color, format, tmp.format);
        tmp.max_color = reformat_color(max_color, format, tmp.format);
        tmp.nan_color = reformat_color(nan_color, format, tmp.format);
        for (uint32_t& c : tmp.map) {
            c = reformat_color(c, format, tmp.format);
        }
        swap(tmp);
    }
}

uint32_t Colormap::apply(float val, float min, float max) const {
    if (isnan(min)) {
        min = this->min;
        if (isnan(min))
            min = 0.0f;
    }
    if (isnan(max)) {
        max = this->max;
        if (isnan(max))
            max = 1.0f;
    }
    float scale = float(map.size()) / (max - min);
    return _apply(val, min, max, scale);
}

void Colormap::apply(float val, uint8_t &r, uint8_t &g, uint8_t &b, float min, float max) const {
    get_rgb(apply(val,min,max),format,r,g,b);
}
void Colormap::apply(float val, uint8_t &r, uint8_t &g, uint8_t &b, uint8_t &a, float min, float max) const {
    get_rgba(apply(val,min,max),format,r,g,b,a);
}


const int Colormap::min_idx = -1;
const int Colormap::max_idx = -2;
const int Colormap::nan_idx = -3;

uint32_t* Colormap::data() {
    return map.data();
}

const uint32_t* Colormap::data() const {
    return map.data();
}

const uint32_t& Colormap::get(int idx) const {
    if(idx == min_idx) return min_color;
    else if(idx == max_idx) return max_color;
    else if(idx == nan_idx) return nan_color;
    else return map[idx];
}
void Colormap::get(int idx, uint8_t& r, uint8_t& g, uint8_t& b) const {
    Colormap::get_rgb(get(idx),format,r,g,b);
}
void Colormap::get(int idx, uint8_t& r, uint8_t& g, uint8_t& b, uint8_t& a) const {
    Colormap::get_rgba(get(idx),format,r,g,b,a);
}

void Colormap::set(int idx, uint8_t r, uint8_t g, uint8_t b) {
    uint32_t a = ((format >> 8) & 0xff);
    set(idx,r,g,b,a);
}

void Colormap::set(int idx, uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
    if(idx == min_idx) min_color = format_color(format,r,g,b,a);
    else if(idx == max_idx) max_color = format_color(format,r,g,b,a);
    else if(idx == nan_idx) nan_color = format_color(format,r,g,b,a);
    else map[idx] = format_color(format,r,g,b,a);
}

Colormaps::Colormaps(const std::string& filename, uint32_t format, size_t n) {
    parse(filename, format, n);
    ////grayscale:
    //float zo[] = { 0.0f, 1.0f };
    //gray = Colormap(n, vec(zo), vec(zo), format);
    //add("gray",gray);
    ////jet:
    //float jv[] = { 0.000, 0.125, 0.375, 0.625, 0.875, 1.000 };
    //float jr[] = { 0, 0, 0, 1, 1, 0.5 };
    //float jg[] = { 0, 0, 1, 1, 0, 0 };
    //float jb[] = { 0.5, 1, 1, 0, 0, 0 };
    //jet = Colormap(n, vec(jv), vec(jr), vec(jg), vec(jb), format);
    //add("jet",jet);
    ////hsv:
    //float hv[] = { 0, 1. / 6, 2. / 6, 3. / 6, 4. / 6, 5. / 6, 1 };
    //float hr[] = { 1, 1, 0, 0, 0, 1, 1 };
    //float hg[] = { 0, 1, 1, 1, 0, 0, 0 };
    //float hb[] = { 0, 0, 0, 1, 1, 1, 0 };
    //hsv = Colormap(n, vec(hv), vec(hr), vec(hg), vec(hb), format);
    //add("hsv", hsv);
}

static bool parse_name(std::istream& in, std::string& name, char delim) {
    name = "";
    std::istream::sentry s(in);
    if (s) while (in.good()) {
        char c = in.get();
        if (std::isspace(c,in.getloc())) continue; //skip space
        if (c == delim) return true;
        else name += c;
    }
    return false;
}

static bool parse_list(std::istream& in, std::vector<float>& vals) {
    vals.clear();
    std::istream::sentry s(in);
    float v;
    bool open = false;
    if (s) while (in.good()) {
        char c = in.get();
        if (std::isspace(c, in.getloc())) continue;
        if (!open && c == '{') {
            open = true;
            in >> v;
            vals.push_back(v);
        }
        else if (open && c == ',') {
            //number comes next
            in >> v;
            vals.push_back(v);
        }
        else if (open && c == '}') {
            //end list
            return true;
        }
        else {
            //syntax error
            return false;
        }
    }
    return false;
}

static bool parse_pair(std::istream& in, std::pair<float,float>& p) {
    std::istream::sentry s(in);
    bool open=false;
    int done = 0;
    if (s) while (in.good()) {
        char c = in.get();
        if (std::isspace(c, in.getloc())) continue; //ignore whitespace
        if (!open && c == '(') {
            open = true;
            in >> p.first;
            done = 1;
        }
        else if (open && done == 1 && c == ',') {
            in >> p.second;
            done = 2;
        }
        else if (open && done == 2 && c == ')') {
            return true;
        }
        else return false;
    }
    return false;
}

static bool parse_list(std::istream& in, std::vector<std::pair<float,float>>& vals) {
    vals.clear();
    std::istream::sentry s(in);
    std::pair<float,float> val;
    bool open = false;
    if (s) while (in.good()) {
        char c = in.get();
        if (std::isspace(c, in.getloc())) continue; //ignore spaces
        if (!open && c == '{') {
            open = true;
            if (!parse_pair(in, val)) return false;
            vals.push_back(val);
        }
        else if (open && c == ',') {
            if (!parse_pair(in, val)) return false;
            vals.push_back(val);
        }
        else if (open && c == '}') {
            return true;
        }
        else {
            return false;
        }
    }
    return false;
}

void Colormaps::parse(const std::string& filename, uint32_t format, size_t n) {
    ifstream in(filename);
    while (true) {
        std::string name, cname;
        if (!parse_name(in, name, ':')) return;
        if (!parse_name(in, cname, '=')) return;
        if (cname == "value") {
            //we're in value, red, green, blue mode
            std::vector<float> vals, red, green, blue;
            if (!parse_list(in, vals)) return;

            if (!parse_name(in, cname, '=')) return;
            if (cname != "red") return;
            if (!parse_list(in, red)) return;

            if (!parse_name(in, cname, '=')) return;
            if (cname != "green") return;
            if (!parse_list(in, green)) return;

            if (!parse_name(in, cname, '=')) return;
            if (cname != "blue") return;
            if (!parse_list(in, blue)) return;

            add(name, Colormap(n, vals, red, green, blue, format));
        }
        else if (cname == "red") {
            //we're in pairs mode
            std::vector<std::pair<float,float>> red, green, blue;
            if (!parse_list(in, red)) return;

            if (!parse_name(in, cname, '=')) return;
            if (cname != "green") return;
            if (!parse_list(in, green)) return;

            if (!parse_name(in, cname, '=')) return;
            if (cname != "blue") return;
            if (!parse_list(in, blue)) return;

            add(name, Colormap(n, red, green, blue, format));
        }
        else {
            return;
        }
    }
}

bool Colormaps::has(const string& name) const {
    return (maps.find(name) != maps.end());
}

const Colormap& Colormaps::get(const string& name) const {
    try {
        return maps.at(name);
    } catch(std::out_of_range&) {
        throw std::runtime_error("Colormap not found: " + name);
    }
}

Colormap& Colormaps::get(const string& name) {
    try {
        return maps.at(name);
    } catch(std::out_of_range&) {
        throw std::runtime_error("Colormap not found: " + name);
    }
}

Colormap& Colormaps::operator[](const string& name) {
    return maps[name];
}

Colormap& Colormaps::operator[](string&& name) {
    return maps[move(name)];
}

Colormap& Colormaps::add(const string& name, const Colormap& cmap) {
    if (name == "gray") gray = cmap;
    else if (name == "jet") jet = cmap;
    else if (name == "hsv") hsv = cmap;
    return maps[name] = cmap;
}

Colormap& Colormaps::add(const string& name, Colormap&& cmap) {
    if (name == "gray") gray = cmap;
    else if (name == "jet") jet = cmap;
    else if (name == "hsv") hsv = cmap;
    return maps[name] = std::move(cmap);
}
