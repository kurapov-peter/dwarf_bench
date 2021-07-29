#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <CL/sycl.hpp>

using namespace std;
using namespace sycl;

#define EMPTY_KEY 2147483647

class join_build;
class join_probe;

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

template <class Key, class Hash> class CuckooHashtable{
    private:
        sycl::global_ptr<Key> _keys;
        //sycl::global_ptr<T> _vals;
        size_t _size;
        Hash _hasher1;
        Hash _hasher2;
        //static constexpr uint32_t elem_sz = CHAR_BIT * sizeof(uint32_t);
        static const sycl::ONEAPI::memory_order default_memory_order = sycl::ONEAPI::memory_order::relaxed;
        static const sycl::memory_scope default_memory_scope = sycl::memory_scope::system;
        
    public:
        explicit CuckooHashtable(size_t size, sycl::global_ptr<Key> keys, Hash hasher1, Hash hasher2):
            _size(size), _keys(keys), _hasher1(hasher1), _hasher2(hasher2) {}
        
        bool at(Key key) {
            if (_keys[_hasher1(key)] == key || _keys[_hasher2(key)] == key)
                return true;
            return false;
        }

        bool insert(Key key, size_t cnt) {
            while (cnt < _size) {
                size_t pos[] = {_hasher1(key), _hasher2(key)};

                for (size_t i = 0; i < 2; i++){
                    Key expected = EMPTY_KEY;
                    sycl::ONEAPI::atomic_ref<Key, default_memory_order, default_memory_scope,
                                    access::address_space::global_space> atomic_data(_keys[pos[i]]);
                
                    if(atomic_data.compare_exchange_strong(expected, key, default_memory_order, default_memory_scope)) {
                        return true;    
                    }
                }

                sycl::ONEAPI::atomic_ref<Key, default_memory_order, default_memory_scope,
                                access::address_space::global_space> atomic_data(_keys[pos[0]]);
            
                key = atomic_data.exchange(key, default_memory_order, default_memory_scope);
            
                cnt++;
            }

            return false;
        }
};