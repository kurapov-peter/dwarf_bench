struct PolynomialHasher {
    PolynomialHasher(size_t sz) {
        _sz = sz;
        p = possible_p[rand() % 14];
        int k = possible_p[rand() % 14];
    }

    size_t operator()(const uint32_t &v) const {
        uint32_t v_copy = v;
        int res = 0;
        int pow_p = p;
        while(v_copy > 0){
            res += ((v_copy % 10) * pow_p) % _sz;
            res = res % _sz;
            pow_p *= p;
            v_copy /= 10;
        }
        return res;
    }
    private:
        size_t _sz;
        int p;
        const int possible_p[14] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43};
};

template <size_t Size> struct StaticSimpleHasher {
  size_t operator()(const uint32_t &v) const { return v % Size; }
};

template <size_t Size, size_t Offset> struct StaticSimpleHasherWithOffset {
  size_t operator()(const uint32_t &v) const { return (v % Size + Offset) % Size; }
};

template <class Integral> struct SimpleHasher {
  SimpleHasher(size_t sz) : _sz(sz) {}
  size_t operator()(const Integral &v) const { return v % _sz; }

private:
  size_t _sz;
};
