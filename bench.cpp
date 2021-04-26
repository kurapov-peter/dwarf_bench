#include "common/common.hpp"
#include "common/registry.hpp"
#include "register_dwarfs.hpp"
#include <boost/program_options.hpp>
#include <iostream>

int main(int argc, char *argv[]) {
  populate_registry();

  auto registry = Registry::instance();
  namespace po = boost::program_options;
  RunOptions opts;
  opts.root_path = helpers::get_kernels_root_env(argv[0]);
  std::cout
      << "DWARF_BENCH_ROOT is set to " << opts.root_path << std::endl
      << "You can change that with 'export DWARF_BENCH_ROOT=/your/path'\n";

  Dwarf *dwarf;
  std::string dwarf_name;
  po::options_description desc("Dwarf bench");
  desc.add_options()("help", "Show help message");
  desc.add_options()("dwarf", po::value<std::string>(&dwarf_name),
                     "Dwarf to run. List all with 'list' option.");
  desc.add_options()(
      "input_size",
      po::value<std::vector<size_t>>(&opts.input_size)->multitoken(),
      "Data array size, ususally a column size in elements");
  desc.add_options()("iterations", po::value<size_t>(&opts.iterations),
                     "Number of iterations to run a bmark.");
  po::positional_options_description pos_opts;
  pos_opts.add("dwarf", 1);

  po::variables_map vm;

  try {
    po::store(po::command_line_parser(argc, argv)
                  .options(desc)
                  .positional(pos_opts)
                  .run(),
              vm);
    vm.notify();

    if (dwarf_name == "list") {
      std::cout << "Supported dwarfs:\n";
      for (const auto &dw : *registry) {
        std::cout << "\t" << dw.first << std::endl;
      }
      return 0;
    } else {
      dwarf = registry->find(dwarf_name);
      if (!dwarf) {
        std::cerr << "Wrong dwarf name: '" << dwarf_name << "'" << std::endl;
        std::cerr << "To list all available dwarfs, run '" << argv[0]
                  << " list'" << std::endl;
        return 1;
      }
    }
    if (vm.count("help")) {
      std::cout << desc;
      return 0;
    } else if (!dwarf) {
      std::cerr << "List supported dwarfs to run with '" << argv[0] << " list'"
                << std::endl;
      return 1;
    }

    if (opts.input_size.empty()) {
      opts.input_size.push_back(1);
    }

    dwarf->init(opts);
    dwarf->run(opts);
    dwarf->report(opts);
  } catch (std::exception &e) {
    std::cerr << "Caught exception: " << e.what() << std::endl;
  }
  return 0;
}