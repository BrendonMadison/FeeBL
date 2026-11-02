#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <streambuf>
#include <string>
#include <vector>

#include "lumiCPP.h"

namespace sampling = feebl::electron_lumi_sampling;

namespace {

const char* const kNativeColumns =
    "e1 e2 x y z t vx1 vy1 vz1 vx2 vy2 vz2 "
    "sx1 sy1 sz1 sx2 sy2 sz2 index";

void fail(const std::string& message) {
  std::cerr << "FAIL: " << message << std::endl;
  std::exit(1);
}

void check(bool condition, const std::string& message) {
  if (!condition) fail(message);
}

std::string read_bytes(const std::string& path) {
  std::ifstream input(path.c_str(), std::ios::in | std::ios::binary);
  if (!input) fail("could not read direct-stream output: " + path);
  return std::string(std::istreambuf_iterator<char>(input),
                     std::istreambuf_iterator<char>());
}

void check_bytes(const std::string& expected, const std::string& actual,
                 const std::string& label) {
  if (expected == actual) return;
  const std::size_t common = std::min(expected.size(), actual.size());
  std::size_t mismatch = 0U;
  while (mismatch < common && expected[mismatch] == actual[mismatch])
    ++mismatch;
  std::ostringstream message;
  message << label << " differs at byte " << mismatch
          << " (expected size " << expected.size()
          << ", actual size " << actual.size() << ")";
  fail(message.str());
}

void check_persistent_terminator(const std::string& bytes,
                                 const std::string& label) {
  check(bytes.size() >= 2U &&
            bytes[bytes.size() - 2U] == '\n' && bytes.back() == '\n',
        label + " did not preserve the historical final blank line");
}

void check_native_terminator(const std::string& bytes,
                             const std::string& label) {
  check(!bytes.empty() && bytes.back() == '\n',
        label + " did not end with a newline");
  check(bytes.size() < 2U || bytes[bytes.size() - 2U] != '\n',
        label + " unexpectedly appended a blank line after the footer");
}

class CountingNullBuffer : public std::streambuf {
 public:
  CountingNullBuffer() : bytes_(0U) {}

  std::size_t bytes() const { return bytes_; }

 protected:
  std::streamsize xsputn(const char*, std::streamsize count) override {
    bytes_ += static_cast<std::size_t>(count);
    return count;
  }

  int_type overflow(int_type character) override {
    if (!traits_type::eq_int_type(character, traits_type::eof())) ++bytes_;
    return traits_type::not_eof(character);
  }

  int sync() override { return 0; }

 private:
  std::size_t bytes_;
};

LUMI_PAIR make_pair(unsigned int ordinal) {
  LUMI_PAIR record(45.625f + 0.03125f * ordinal,
                   44.875f - 0.015625f * ordinal);
  record.set_position(-12.75f + ordinal, 0.03125f * ordinal,
                      453.125f - 2.5f * ordinal);
  return record;
}

LUMI_PAIR_EE make_ee_pair(unsigned int ordinal) {
  LUMI_PAIR_EE record(45.625f + 0.03125f * ordinal,
                      44.875f - 0.015625f * ordinal,
                      0.000125f * (ordinal + 1U),
                      -0.00025f * (ordinal + 2U),
                      -0.000375f * (ordinal + 1U),
                      0.0000625f * (ordinal + 3U),
                      static_cast<int>(17U + 3U * ordinal));
  record.set_position(-12.75f + ordinal, 0.03125f * ordinal,
                      453.125f - 2.5f * ordinal);
  record.set_spins(TRIDVECTOR(0.125 + 0.01 * ordinal,
                             -0.25 + 0.02 * ordinal,
                             0.75 - 0.03 * ordinal),
                   TRIDVECTOR(-0.375 + 0.01 * ordinal,
                              0.5 - 0.02 * ordinal,
                              -0.625 + 0.03 * ordinal));
  return record;
}

sampling::PairIdentity identity(unsigned int ordinal) {
  sampling::PairIdentity value;
  value.run_seed = 0x1020304050607080ull;
  value.crossing = 11U;
  value.diagonal = ordinal / 3U;
  value.wave = ordinal / 2U;
  value.substep = ordinal % 5U;
  value.encounter = ordinal + 7U;
  value.parent_a = 0x1100000000000000ull + ordinal;
  value.parent_b = 0x2200000000000000ull + 3U * ordinal;
  value.interaction = ordinal % 2U;
  return value;
}

NativeRunMeta metadata() {
  NativeRunMeta meta;
  meta.feebl_version = "direct-stream-regression";
  meta.accelerator = "fccz byte oracle";
  meta.parameters = "deterministic and legacy heaps";
  meta.half_crossing_angle_rad = 0.015;
  meta.momenta_carry_crossing_angle = false;
  meta.spin_present = true;
  meta.rndm_seed = 0x314159;
  return meta;
}

class PairHeapHarness : public LUMI_HEAP {
 public:
  PairHeapHarness(int capacity, RNDM* random)
      : LUMI_HEAP(capacity, 1.0f, random) {}

  void append(const LUMI_PAIR& record) {
    data_.push_back(record);
    nb_pairs_ = static_cast<int>(data_.size());
  }

  const std::vector<LUMI_PAIR>& records() const { return data_; }
};

class EeHeapHarness : public LUMI_HEAP_EE {
 public:
  EeHeapHarness(int capacity, RNDM* random, int deterministic_mode)
      : LUMI_HEAP_EE(capacity, 1.0f, random, deterministic_mode) {}

  void append_legacy(const LUMI_PAIR_EE& record) {
    check(deterministic_sampling_mode_ == 0,
          "legacy record appended to a deterministic heap");
    data_.push_back(record);
    nb_pairs_ = static_cast<int>(data_.size());
  }

  void offer_deterministic(const LUMI_PAIR_EE& record,
                           const sampling::PairIdentity& pair_identity,
                           double priority, std::uint64_t payload_token) {
    check(deterministic_sampling_mode_ == 1,
          "deterministic record offered to a legacy heap");
    const sampling::OfferResult offered =
        deterministic_sampler_.offer_precomputed(pair_identity, priority,
                                                 payload_token);
    if (offered.status == sampling::OfferStatus::Rejected) return;
    check(offered.accepted(), "deterministic test offer failed");
    if (offered.status == sampling::OfferStatus::Inserted) {
      check(offered.storage_slot == data_.size(),
            "deterministic insertion returned a non-append slot");
      data_.push_back(record);
    } else {
      check(offered.storage_slot < data_.size(),
            "deterministic replacement returned an invalid slot");
      data_[offered.storage_slot] = record;
    }
    nb_pairs_ = static_cast<int>(data_.size());
  }

  const std::vector<LUMI_PAIR_EE>& records() const { return data_; }

  std::string deterministic_native_reference(const NativeRunMeta& meta) const {
    check(deterministic_sampling_mode_ == 1,
          "canonical native reference requested for a legacy heap");
    const std::vector<std::size_t> selected = canonical_storage_order();
    check(selected.size() == data_.size(),
          "deterministic reference has mismatched sampler/data sizes");
    std::ostringstream output;
    output << native_header(meta, "lumi_ee", kNativeColumns);
    for (std::size_t index = 0; index < selected.size(); ++index)
      output << data_[selected[index]].native_flow()
             << " " << (index + 1U) << std::endl;
    output << native_footer("lumi_ee",
                            static_cast<unsigned long>(selected.size()));
    return output.str();
  }
};

template <typename Record>
std::string shuffled_native_reference(const std::vector<Record>& records,
                                      RNDM* random,
                                      const NativeRunMeta& meta) {
  std::vector<unsigned long int> order;
  random->getShuffledIntegerSequence(static_cast<int>(records.size()), order);
  std::ostringstream output;
  output << native_header(meta, "lumi_ee", kNativeColumns);
  for (std::size_t index = 0; index < order.size(); ++index)
    output << records[order[index] - 1U].native_flow()
           << " " << order[index] << std::endl;
  output << native_footer("lumi_ee",
                          static_cast<unsigned long>(records.size()));
  return output.str();
}

void test_record_streams() {
  const LUMI_PAIR base = make_pair(3U);
  std::ostringstream base_persistent;
  base.write_persistent(base_persistent);
  check_bytes(base.persistent_flow(), base_persistent.str(),
              "LUMI_PAIR persistent record");
  std::ostringstream base_native;
  base.write_native(base_native);
  check_bytes(base.native_flow(), base_native.str(),
              "LUMI_PAIR native record");

  const LUMI_PAIR_EE electron = make_ee_pair(5U);
  std::ostringstream ee_persistent;
  electron.write_persistent(ee_persistent);
  check_bytes(electron.persistent_flow(), ee_persistent.str(),
              "LUMI_PAIR_EE persistent record");
  std::ostringstream ee_native;
  electron.write_native(ee_native);
  check_bytes(electron.native_flow(), ee_native.str(),
              "LUMI_PAIR_EE native record");
}

void test_legacy_pair_heap(const std::string& stem) {
  const unsigned long seed = 0x12345678ul;
  RNDM random(seed);
  PairHeapHarness heap(8, &random);
  for (unsigned int item = 0; item < 6U; ++item) heap.append(make_pair(item));

  random.reseed_encounter(seed);
  const std::string persistent_reference = heap.persistent_flow() + "\n";
  random.reseed_encounter(seed);
  const std::string persistent_path = stem + "-pair-legacy.dat";
  heap.saveLumi(persistent_path);
  const std::string persistent_actual = read_bytes(persistent_path);
  check_bytes(persistent_reference, persistent_actual,
              "legacy LUMI_HEAP persistent file");
  check_persistent_terminator(persistent_actual,
                              "legacy LUMI_HEAP persistent file");

  const NativeRunMeta meta = metadata();
  random.reseed_encounter(seed);
  const std::string native_reference =
      shuffled_native_reference(heap.records(), &random, meta);
  random.reseed_encounter(seed);
  const std::string native_path = stem + "-pair-legacy.native";
  heap.saveLumiNative(native_path, meta);
  const std::string native_actual = read_bytes(native_path);
  check_bytes(native_reference, native_actual,
              "legacy LUMI_HEAP native file");
  check_native_terminator(native_actual, "legacy LUMI_HEAP native file");

  std::remove(persistent_path.c_str());
  std::remove(native_path.c_str());
}

void test_legacy_ee_heap(const std::string& stem) {
  const unsigned long seed = 0x87654321ul;
  RNDM random(seed);
  EeHeapHarness heap(8, &random, 0);
  for (unsigned int item = 0; item < 6U; ++item)
    heap.append_legacy(make_ee_pair(item));

  random.reseed_encounter(seed);
  const std::string persistent_reference = heap.persistent_flow() + "\n";
  random.reseed_encounter(seed);
  const std::string persistent_path = stem + "-ee-legacy.dat";
  heap.saveLumi(persistent_path);
  const std::string persistent_actual = read_bytes(persistent_path);
  check_bytes(persistent_reference, persistent_actual,
              "legacy LUMI_HEAP_EE persistent file");
  check_persistent_terminator(persistent_actual,
                              "legacy LUMI_HEAP_EE persistent file");

  const NativeRunMeta meta = metadata();
  random.reseed_encounter(seed);
  const std::string native_reference =
      shuffled_native_reference(heap.records(), &random, meta);
  random.reseed_encounter(seed);
  const std::string native_path = stem + "-ee-legacy.native";
  heap.saveLumiNative(native_path, meta);
  const std::string native_actual = read_bytes(native_path);
  check_bytes(native_reference, native_actual,
              "legacy LUMI_HEAP_EE native file");
  check_native_terminator(native_actual,
                          "legacy LUMI_HEAP_EE native file");

  std::remove(persistent_path.c_str());
  std::remove(native_path.c_str());
}

void test_deterministic_ee_heap(const std::string& stem) {
  RNDM unused_random(0xabcdeful);
  EeHeapHarness heap(4, &unused_random, 1);
  const double priorities[] = {9.0, 1.0, 7.0, 2.0, 0.5, 12.0, 1.5};
  for (unsigned int item = 0; item < 7U; ++item)
    heap.offer_deterministic(make_ee_pair(item), identity(item),
                             priorities[item], 1000U + item);
  check(heap.nb_pairs() == 4,
        "deterministic heap did not retain its bounded top K");

  const std::string persistent_reference = heap.persistent_flow() + "\n";
  const std::string persistent_path = stem + "-ee-deterministic.dat";
  heap.saveLumi(persistent_path);
  const std::string persistent_actual = read_bytes(persistent_path);
  check_bytes(persistent_reference, persistent_actual,
              "deterministic LUMI_HEAP_EE persistent file");
  check_persistent_terminator(persistent_actual,
                              "deterministic LUMI_HEAP_EE persistent file");

  const NativeRunMeta meta = metadata();
  const std::string native_reference =
      heap.deterministic_native_reference(meta);
  const std::string native_path = stem + "-ee-deterministic.native";
  heap.saveLumiNative(native_path, meta);
  const std::string native_actual = read_bytes(native_path);
  check_bytes(native_reference, native_actual,
              "deterministic LUMI_HEAP_EE native file");
  check_native_terminator(native_actual,
                          "deterministic LUMI_HEAP_EE native file");

  std::remove(persistent_path.c_str());
  std::remove(native_path.c_str());
}

void test_empty_heap_newlines(const std::string& stem) {
  RNDM random(0x424242ul);
  PairHeapHarness legacy(3, &random);
  const std::string legacy_path = stem + "-empty-legacy.dat";
  legacy.saveLumi(legacy_path);
  check_bytes("\n", read_bytes(legacy_path),
              "empty legacy heap persistent file");

  EeHeapHarness deterministic(3, &random, 1);
  const std::string deterministic_path = stem + "-empty-deterministic.dat";
  deterministic.saveLumi(deterministic_path);
  check_bytes("\n", read_bytes(deterministic_path),
              "empty deterministic heap persistent file");

  const NativeRunMeta meta = metadata();
  const std::string native_path = stem + "-empty-deterministic.native";
  deterministic.saveLumiNative(native_path, meta);
  const std::string expected_native =
      native_header(meta, "lumi_ee", kNativeColumns) +
      native_footer("lumi_ee", 0U);
  const std::string actual_native = read_bytes(native_path);
  check_bytes(expected_native, actual_native,
              "empty deterministic heap native file");
  check_native_terminator(actual_native,
                          "empty deterministic heap native file");

  std::remove(legacy_path.c_str());
  std::remove(deterministic_path.c_str());
  std::remove(native_path.c_str());
}

void benchmark_record_streaming() {
  constexpr std::size_t records_count = 100000U;
  std::vector<LUMI_PAIR_EE> records;
  records.reserve(records_count);
  for (std::size_t item = 0; item < records_count; ++item)
    records.push_back(make_ee_pair(static_cast<unsigned int>(item % 997U)));

  CountingNullBuffer string_buffer;
  std::ostream string_output(&string_buffer);
  const std::chrono::steady_clock::time_point string_begin =
      std::chrono::steady_clock::now();
  for (std::size_t item = 0; item < records.size(); ++item)
    string_output << records[item].persistent_flow()
                  << " " << (item + 1U) << std::endl;
  const std::chrono::steady_clock::time_point string_end =
      std::chrono::steady_clock::now();

  CountingNullBuffer direct_buffer;
  std::ostream direct_output(&direct_buffer);
  const std::chrono::steady_clock::time_point direct_begin =
      std::chrono::steady_clock::now();
  for (std::size_t item = 0; item < records.size(); ++item) {
    records[item].write_persistent(direct_output);
    direct_output << " " << (item + 1U) << '\n';
  }
  const std::chrono::steady_clock::time_point direct_end =
      std::chrono::steady_clock::now();

  check(string_output.good() && direct_output.good(),
        "informational serialization benchmark stream failed");
  check(string_buffer.bytes() == direct_buffer.bytes(),
        "informational serialization paths emitted different byte counts");

  const double string_ms = std::chrono::duration<double, std::milli>(
      string_end - string_begin).count();
  const double direct_ms = std::chrono::duration<double, std::milli>(
      direct_end - direct_begin).count();
  const double observed_speedup =
      direct_ms > 0.0 ? string_ms / direct_ms : 0.0;
  // Timings deliberately are not correctness gates: they are local diagnostic
  // evidence, while every serialization comparison above remains byte-exact.
  std::cout << "lumi_direct_stream_benchmark records=" << records_count
            << " bytes=" << direct_buffer.bytes()
            << " string_ms=" << string_ms
            << " direct_ms=" << direct_ms
            << " observed_speedup=" << observed_speedup << std::endl;
}

}  // namespace

int main(int argc, char** argv) {
  const std::string stem =
      argc > 1 ? argv[1] : "lumi-direct-stream-regression";
  test_record_streams();
  test_legacy_pair_heap(stem);
  test_legacy_ee_heap(stem);
  test_deterministic_ee_heap(stem);
  test_empty_heap_newlines(stem);
  benchmark_record_streaming();
  std::cout << "PASS: direct luminosity streams are byte-identical to "
               "string representations"
            << std::endl;
  return 0;
}
