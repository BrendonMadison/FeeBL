#include "adaptiveGridScheduleCPP.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(_WIN32)
#include <process.h>
#else
#include <unistd.h>
#endif

namespace {

using feebl::adaptive_grid::AdaptiveGridEncounterKey;
using feebl::adaptive_grid::AdaptiveGridEncounterRecord;
using feebl::adaptive_grid::AdaptiveGridSchedule;
using feebl::adaptive_grid::AdaptiveGridScheduleError;
using feebl::adaptive_grid::AdaptiveGridScheduleHeader;

int failures = 0;

void check(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
  }
}

template <typename Callable>
void expect_error(Callable&& callable,
                  const std::string& expected_fragment,
                  const std::string& message) {
  try {
    callable();
    check(false, message + " (no exception)");
  } catch (const AdaptiveGridScheduleError& error) {
    check(std::string(error.what()).find(expected_fragment) != std::string::npos,
          message + " (unexpected error: " + error.what() + ")");
  } catch (const std::exception& error) {
    check(false, message + " (wrong exception: " + error.what() + ")");
  }
}

std::uint64_t test_process_id() noexcept {
#if defined(_WIN32)
  return static_cast<std::uint64_t>(_getpid());
#else
  return static_cast<std::uint64_t>(getpid());
#endif
}

class TestFiles {
 public:
  TestFiles() = default;
  TestFiles(const TestFiles&) = delete;
  TestFiles& operator=(const TestFiles&) = delete;
  ~TestFiles() {
    for (const auto& path : paths_) {
      std::error_code ignored;
      std::filesystem::remove(path, ignored);
    }
  }
  std::filesystem::path make(const std::string& suffix) {
    std::filesystem::path path = std::filesystem::temp_directory_path() /
        ("feebl-adaptive-schedule-test-" + std::to_string(test_process_id()) +
         "-" + suffix);
    paths_.push_back(path);
    return path;
  }

 private:
  std::vector<std::filesystem::path> paths_;
};

std::string repeat(char value, std::size_t count) {
  return std::string(count, value);
}

AdaptiveGridScheduleHeader make_header(
    const std::string& build_id = "release-cpu") {
  return AdaptiveGridScheduleHeader(
      repeat('a', 64U), 0x0123456789abcdefULL, "source-commit", build_id,
      "uniform-window-v1", repeat('b', 64U));
}

AdaptiveGridEncounterRecord make_first_record() {
  return AdaptiveGridEncounterRecord(
      AdaptiveGridEncounterKey{0U, 0U, 0U, 0U, 0U}, 3, -4, 5, 6, 64U,
      128U, 128U, 256U, 0x1111222233334444ULL, "64x128", 100U, 90U,
      false);
}

AdaptiveGridEncounterRecord make_second_record(
    std::uint64_t target_count = 91U) {
  return AdaptiveGridEncounterRecord(
      AdaptiveGridEncounterKey{0U, 0U, 0U, 0U, 1U}, 4, -5, 7, 8, 128U,
      128U, 256U, 256U, 0xaabbccddeeff0011ULL, "128x128", 110U,
      target_count, true);
}

AdaptiveGridSchedule make_schedule() {
  AdaptiveGridSchedule result(make_header());
  result.append(make_first_record());
  result.append(make_second_record());
  return result;
}

std::string read_text(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) {
    throw std::runtime_error("test could not open input file");
  }
  const std::streampos end = input.tellg();
  if (end < std::streampos(0)) {
    throw std::runtime_error("test could not measure input file");
  }
  std::string result(static_cast<std::size_t>(end), '\0');
  input.seekg(0, std::ios::beg);
  if (!result.empty()) {
    input.read(result.data(), static_cast<std::streamsize>(result.size()));
  }
  if (!input || input.gcount() != static_cast<std::streamsize>(result.size())) {
    throw std::runtime_error("test input file was truncated");
  }
  return result;
}

void write_text(const std::filesystem::path& path, const std::string& value) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  output.write(value.data(), static_cast<std::streamsize>(value.size()));
  if (!output) {
    throw std::runtime_error("test could not write output file");
  }
}

std::string fixed_header_line() {
  return "H\t1\t" + repeat('a', 64U) +
      "\t0123456789abcdef"
      "\t736f757263652d636f6d6d6974"
      "\t72656c656173652d637075"
      "\t756e69666f726d2d77696e646f772d7631\t" +
      repeat('b', 64U);
}

std::string first_record_line() {
  return "E\t0\t0\t0\t0\t0\t3\t-4\t5\t6\t64\t128\t128\t256"
         "\t1111222233334444\t363478313238\t100\t90\t0";
}

std::string second_record_line() {
  return "E\t0\t0\t0\t0\t1\t4\t-5\t7\t8\t128\t128\t256\t256"
         "\taabbccddeeff0011\t31323878313238\t110\t91\t1";
}

std::string malformed_document(const std::string& first,
                               const std::string& second,
                               const std::string& checksum) {
  return "FEEBL_ADAPTIVE_GRID_SCHEDULE\t1\n" + fixed_header_line() +
      "\nN\t2\n" + first + "\n" + second + "\nC\t" + checksum + "\n";
}

void test_round_trip_and_golden_hash(TestFiles& files) {
  const AdaptiveGridSchedule schedule = make_schedule();
  check(schedule.stable_hash() ==
            "5b959ac75f066c898459d5314aa1ba52bc6627f3aa92f9061f3d1db4311c4b52",
        "canonical schedule SHA-256 matches the pinned cross-platform value");

  AdaptiveGridSchedule independently_built(make_header());
  independently_built.append(make_first_record());
  independently_built.append(make_second_record());
  check(independently_built.stable_hash() == schedule.stable_hash(),
        "independent canonical append produces an identical schedule hash");

  const std::filesystem::path path = files.make("round-trip.txt");
  schedule.save_atomic(path);
  const AdaptiveGridSchedule loaded = AdaptiveGridSchedule::load(path);
  check(loaded == schedule, "strict save/load round trip preserves every field");
  check(loaded.stable_hash() == schedule.stable_hash(),
        "round-trip schedule hash is stable");
  loaded.validate_exact(schedule);

  AdaptiveGridSchedule replacement(make_header());
  replacement.append(make_first_record());
  replacement.save_atomic(path);
  check(AdaptiveGridSchedule::load(path) == replacement,
        "atomic save replaces an existing schedule completely");
}

void test_append_order_and_duplicates() {
  AdaptiveGridSchedule duplicate(make_header());
  duplicate.append(make_first_record());
  expect_error([&duplicate]() { duplicate.append(make_first_record()); },
               "duplicate", "append rejects a duplicate canonical key");

  AdaptiveGridSchedule reversed(make_header());
  reversed.append(make_second_record());
  expect_error([&reversed]() { reversed.append(make_first_record()); },
               "canonical order", "append rejects a decreasing canonical key");
}

void test_canonical_empty_encounter() {
  AdaptiveGridSchedule empty(make_header());
  empty.append(AdaptiveGridEncounterRecord(
      AdaptiveGridEncounterKey{0U, 0U, 0U, 0U, 0U}, 0, 0, 0, 0,
      0U, 0U, 0U, 0U, 0x1234U, "empty", 0U, 0U, false));
  check(empty.records().size() == 1U,
        "canonical all-zero empty encounter is serializable");
  expect_error(
      []() {
        (void)AdaptiveGridEncounterRecord(
            AdaptiveGridEncounterKey{}, 0, 0, 1, 0, 0U, 0U, 0U, 0U,
            1U, "empty", 0U, 0U, false);
      },
      "all-zero", "noncanonical empty encounter is rejected");
}

void test_reader_failure_modes(TestFiles& files) {
  const AdaptiveGridSchedule schedule = make_schedule();
  const std::filesystem::path valid_path = files.make("valid.txt");
  schedule.save_atomic(valid_path);
  const std::string valid = read_text(valid_path);

  const std::filesystem::path corrupt_path = files.make("corrupt.txt");
  std::string corrupt = valid;
  const std::size_t class_position = corrupt.find("363478313238");
  check(class_position != std::string::npos, "test found serialized class ID");
  if (class_position != std::string::npos) {
    corrupt[class_position] = corrupt[class_position] == '3' ? '4' : '3';
  }
  write_text(corrupt_path, corrupt);
  expect_error([&corrupt_path]() { (void)AdaptiveGridSchedule::load(corrupt_path); },
               "checksum mismatch", "reader rejects payload corruption");

  const std::filesystem::path checksum_path = files.make("checksum.txt");
  std::string bad_checksum = valid;
  const std::size_t trailer = bad_checksum.rfind("C\t");
  check(trailer != std::string::npos, "test found checksum trailer");
  if (trailer != std::string::npos) {
    const std::size_t digit = trailer + 2U;
    bad_checksum[digit] = bad_checksum[digit] == '0' ? '1' : '0';
  }
  write_text(checksum_path, bad_checksum);
  expect_error(
      [&checksum_path]() { (void)AdaptiveGridSchedule::load(checksum_path); },
      "checksum mismatch", "reader rejects an incorrect checksum trailer");

  const std::filesystem::path truncated_path = files.make("truncated.txt");
  std::string truncated = valid;
  truncated.pop_back();
  write_text(truncated_path, truncated);
  expect_error(
      [&truncated_path]() { (void)AdaptiveGridSchedule::load(truncated_path); },
      "truncated", "reader rejects a truncated final line");

  const std::filesystem::path incompatible_path = files.make("incompatible.txt");
  std::string incompatible = valid;
  const std::string magic = "FEEBL_ADAPTIVE_GRID_SCHEDULE\t1";
  check(incompatible.compare(0U, magic.size(), magic) == 0,
        "test found versioned magic");
  incompatible[magic.size() - 1U] = '2';
  write_text(incompatible_path, incompatible);
  expect_error(
      [&incompatible_path]() {
        (void)AdaptiveGridSchedule::load(incompatible_path);
      },
      "unsupported", "reader rejects an incompatible format version");

  // These fixtures carry valid SHA-256 trailers.  They prove that ordering is
  // checked independently of corruption detection, including hostile files
  // whose checksum has been recomputed after reordering or duplication.
  const std::filesystem::path order_path = files.make("order.txt");
  write_text(order_path,
             malformed_document(
                 second_record_line(), first_record_line(),
                 "83adc8616a317e97c8172a076ea9bced8c0651294f503edc80829906d94cdd57"));
  expect_error([&order_path]() { (void)AdaptiveGridSchedule::load(order_path); },
               "canonical order", "reader rejects a checksummed reordered file");

  const std::filesystem::path duplicate_path = files.make("duplicate.txt");
  write_text(duplicate_path,
             malformed_document(
                 first_record_line(), first_record_line(),
                 "421626ea2d3dfefc28bbc1e1684744001a6471541ed842d3055136ff27ad47f1"));
  expect_error(
      [&duplicate_path]() { (void)AdaptiveGridSchedule::load(duplicate_path); },
      "duplicate", "reader rejects a checksummed duplicate record");
}

void test_exact_replay_failure_modes() {
  const AdaptiveGridSchedule expected = make_schedule();

  AdaptiveGridSchedule record_mismatch(make_header());
  record_mismatch.append(make_first_record());
  record_mismatch.append(make_second_record(92U));
  expect_error(
      [&record_mismatch, &expected]() { record_mismatch.validate_exact(expected); },
      "record mismatch", "exact replay rejects a changed plan record");

  AdaptiveGridSchedule header_mismatch(make_header("release-gpu"));
  header_mismatch.append(make_first_record());
  header_mismatch.append(make_second_record());
  expect_error(
      [&header_mismatch, &expected]() { header_mismatch.validate_exact(expected); },
      "header mismatch", "exact replay rejects a changed build identity");

  AdaptiveGridSchedule missing(make_header());
  missing.append(make_first_record());
  expect_error([&missing, &expected]() { missing.validate_exact(expected); },
               "encounter-count mismatch",
               "exact replay rejects missing encounters");
}

void test_constructor_validation() {
  expect_error(
      []() {
        (void)AdaptiveGridScheduleHeader(
            repeat('A', 64U), 1U, "source", "build", "controller",
            repeat('b', 64U));
      },
      "lowercase", "header rejects a noncanonical card hash");

  expect_error(
      []() {
        (void)AdaptiveGridEncounterRecord(
            AdaptiveGridEncounterKey{}, 0, 0, 0, 0, 0U, 4U, 8U, 8U, 1U,
            "class", 0U, 0U, false);
      },
      "extent", "record rejects a zero physical dimension");

  expect_error(
      []() {
        (void)AdaptiveGridEncounterRecord(
            AdaptiveGridEncounterKey{}, 0, 0, 0, 0, 8U, 8U, 4U, 8U, 1U,
            "class", 0U, 0U, false);
      },
      "padded", "record rejects a padded dimension smaller than its window");

  expect_error(
      []() {
        (void)AdaptiveGridEncounterRecord(
            AdaptiveGridEncounterKey{}, 0, 0,
            std::numeric_limits<std::int64_t>::max(), 0, 2U, 8U, 4U, 8U,
            1U, "class", 0U, 0U, false);
      },
      "overflows", "record rejects an overflowing canonical endpoint");
}

}  // namespace

int main() {
  try {
    TestFiles files;
    test_round_trip_and_golden_hash(files);
    test_append_order_and_duplicates();
    test_canonical_empty_encounter();
    test_reader_failure_modes(files);
    test_exact_replay_failure_modes();
    test_constructor_validation();
  } catch (const std::exception& error) {
    std::cerr << "UNCAUGHT: " << error.what() << '\n';
    return 2;
  }

  if (failures != 0) {
    std::cerr << failures << " adaptive-grid schedule test(s) failed\n";
    return 1;
  }
  std::cout << "adaptive-grid schedule tests passed\n";
  return 0;
}
