#include "common/common.hpp"
#include "common/registry.hpp"
#include "register_dwarfs.hpp"
#include <boost/program_options.hpp>
#include <iostream>

bool isGroupBy(const std::string &dwarfName) {
  return (dwarfName.find("GroupBy") != std::string::npos);
}

int main(int argc, char *argv[]) {
  populate_registry();

  auto registry = Registry::instance();
  namespace po = boost::program_options;

  std::unique_ptr<RunOptions> opts = std::make_unique<RunOptions>();
  size_t groups_count = 1;
  size_t threads_count = 1;
  size_t work_group_size = 1;

  opts->root_path = helpers::get_kernels_root_env(argv[0]);
  std::cout
      << "DWARF_BENCH_ROOT is set to " << opts->root_path << std::endl
      << "You can change that with 'export DWARF_BENCH_ROOT=/your/path'\n";

  Dwarf *dwarf;
  std::string dwarf_name, device_type;
  po::options_description desc("Dwarf bench");
  desc.add_options()("help", "Show help message");
  desc.add_options()("dwarf", po::value<std::string>(&dwarf_name),
                     "Dwarf to run. List all with 'list' option.");
  desc.add_options()(
      "input_size",
      po::value<std::vector<size_t>>(&opts->input_size)->multitoken(),
      "Data array size, ususally a column size in elements");
  desc.add_options()("iterations", po::value<size_t>(&opts->iterations),
                     "Number of iterations to run a bmark.");
  desc.add_options()("device",
                     po::value<RunOptions::DeviceType>(&opts->device_ty),
                     "Device to run on.");
  desc.add_options()("report_path", po::value<std::string>(&opts->report_path),
                     "Full/Relative path to a report file.");
  desc.add_options()(
      "groups_count", po::value<size_t>(&groups_count),
      "Number of unique keys for dwarfs with keys (groupby, hash build etc.).");
  desc.add_options()("threads_count", po::value<size_t>(&threads_count),
                     "Number of threads for GroupBy dwarfs.");
  desc.add_options()("work_group_size", po::value<size_t>(&work_group_size),
                     "Work group size for GroupBy dwarfs. threads_count must "
                     "be divisible by work_group_size.");
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
    }
    if (vm.count("help")) {
      std::cout << desc;
      return 0;
    } else if (!dwarf) {
      std::cerr << "List supported dwarfs to run with '" << argv[0] << " list'"
                << std::endl;
      return 1;
    }

    if (opts->input_size.empty()) {
      opts->input_size.push_back(1);
    }

    helpers::set_dpcpp_filter_env(*opts);

    if (isGroupBy(dwarf_name)) {
      std::unique_ptr<GroupByRunOptions> tmpPtr =
          std::make_unique<GroupByRunOptions>(*opts, groups_count,
                                              threads_count, work_group_size);
      opts.reset();
      opts = std::move(tmpPtr);
    }

    dwarf->init(*opts);
    dwarf->run(*opts);
    dwarf->report(*opts);
  } catch (std::exception &e) {
    std::cerr << "Caught exception: " << e.what() << std::endl;
  }
  return 0;
}