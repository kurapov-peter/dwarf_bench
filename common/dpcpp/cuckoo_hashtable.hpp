#include <time.h> 
#include <CL/sycl.hpp>

// #define EMPTY_KEY 2147483647

struct Hasher {

    Hasher(size_t sz) {
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

template <class Key, class Val, class Hasher1, class Hasher2> class CuckooHashtable{
    private:
        sycl::global_ptr<Key> _keys;
        sycl::global_ptr<Val> _vals;

        const size_t _size;
        Hasher1 _hasher1;
        Hasher2 _hasher2;
        const Key _EMPTY_KEY;
        sycl::global_ptr<uint32_t> _bitmask;
        static constexpr uint32_t elem_sz = CHAR_BIT * sizeof(uint32_t);
        
    public:
        explicit CuckooHashtable(const size_t size, sycl::global_ptr<Key> keys, sycl::global_ptr<Val> vals, sycl::global_ptr<uint32_t> bitmask, Hasher1 hasher1, Hasher2 hasher2, const Key EMPTY_KEY):
            _size(size), _keys(keys), _vals(vals), _bitmask(bitmask), _hasher1(hasher1), _hasher2(hasher2), _EMPTY_KEY(EMPTY_KEY){}
        
        bool at(Key key) {
            if (_keys[_hasher1(key)] == key || _keys[_hasher2(key)] == key)
                return true;
            return false;
        }

        bool insert(Key key, Val value, size_t cnt) {
            while (cnt < _size) {

                uint32_t mask, present, major_idx;

                size_t pos[] = {_hasher1(key), _hasher2(key)};

                for (size_t i = 0; i < 2; i++) {

                    major_idx = pos[i] / elem_sz;
                    uint8_t minor_idx = pos[i] % elem_sz;
                    mask = uint32_t(1) << minor_idx;

                    lock(present, _bitmask + major_idx, mask);

                    if (_keys[pos[i]] == _EMPTY_KEY) {
                        _keys[pos[i]] = key;
                        _vals[pos[i]] = value;

                        unlock(_bitmask + major_idx, mask);
                        return true;
                    }

                    unlock(_bitmask + major_idx, mask);
                }
                
                lock(present, _bitmask + major_idx, mask);

                std::swap(key, _keys[pos[1]]);
                std::swap(value, _vals[pos[1]]);

                unlock(_bitmask + major_idx, mask);

                cnt++;
            }
            return false;
        }

        void lock(uint32_t &present, sycl::global_ptr<uint32_t> atomic_var, uint32_t &mask) {
            do {
                present = sycl::atomic<uint32_t>(atomic_var).fetch_or(mask);
            } while (present & mask);
        }

        void unlock(sycl::global_ptr<uint32_t> atomic_var, uint32_t &mask) {
            sycl::atomic<uint32_t>(atomic_var).fetch_and(~mask);

        }
};