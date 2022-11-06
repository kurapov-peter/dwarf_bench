#include "registry.hpp"

std::unique_ptr<Registry> Registry::instance_;
Registry::Registry() {}

Registry *Registry::instance() {
  if (!instance_) {
    instance_.reset(new Registry());
  }
  return instance_.get();
}

void Registry::clear() {
  dwarfs_.clear();
}

void Registry::registerd(Dwarf *dw) { dwarfs_.emplace(dw->name(), dw); }

void Registry::set_root(const std::string &root) { root_path_ = root; }

Dwarf *Registry::find(const std::string &name) const {
  auto it = dwarfs_.find(name);
  return (it != dwarfs_.end()) ? it->second.get() : nullptr;
}

Registry::const_iterator Registry::begin() const { return dwarfs_.begin(); }

Registry::const_iterator Registry::end() const { return dwarfs_.end(); }