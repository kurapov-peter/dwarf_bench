#include "cuckoo_hash_build.hpp"
#include "common/dpcpp/cuckoo_hashtable.hpp"

CuckooHashBuild::CuckooHashBuild() : Dwarf("CuckooHashBuild") {}

void CuckooHashBuild::_run(const size_t buf_size, Meter &meter) {
  
}

void CuckooHashBuild::run(const RunOptions &opts) {
  for (auto size : opts.input_size) {
    _run(size, meter());
  }
}
void CuckooHashBuild::init(const RunOptions &opts) {
  meter().set_opts(opts);
  DwarfParams params = {{"device_type", to_string(opts.device_ty)}};
  meter().set_params(params);
}
