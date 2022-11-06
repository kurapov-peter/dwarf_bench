#pragma once
#include <map>
#include <memory>

#include "dwarf.hpp"

class Registry {
public:
  using const_iterator =
      std::map<std::string, std::unique_ptr<Dwarf>>::const_iterator;
  static Registry *instance();
  void registerd(Dwarf *dw);

  Dwarf *find(const std::string &name) const;
  void set_root(const std::string &root);

  const_iterator begin() const;
  const_iterator end() const;

  void clear();

private:
  Registry();
  std::map<std::string, std::unique_ptr<Dwarf>> dwarfs_;
  static std::unique_ptr<Registry> instance_;
  std::string root_path_;
};