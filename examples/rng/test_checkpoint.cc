#include "rndmCPP.h"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>

namespace {
std::uint32_t bits(float value) {
  std::uint32_t result;
  std::memcpy(&result, &value, sizeof(result));
  return result;
}
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
void write(const std::string& path, const std::string& content) {
  std::ofstream output(path, std::ios::binary);
  output << content;
  output.close();
  require(!output.fail(), "test fixture write failed");
}
std::string read(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  return std::string(std::istreambuf_iterator<char>(input), {});
}
void same_sequence(RNDM left, RNDM right) {
  for (int index = 0; index < 32; ++index) {
    require(bits(left.rndm()) == bits(right.rndm()), "uniform replay differs");
    require(bits(left.gasdev()) == bits(right.gasdev()), "normal replay differs");
  }
}
void rejected_without_mutation(RNDM& generator, const std::string& content) {
  const RNDM before = generator;
  write("rng_checkpoint_invalid.save", content);
  bool rejected = false;
  try {
    generator.rndm_load("rng_checkpoint_invalid.save");
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  require(rejected, "malformed checkpoint was accepted");
  same_sequence(before, generator);
}
}  // namespace

// A checkpoint must preserve every public draw and reject a partial record
// without advancing either the uniform cursor or the Box--Muller cache.
int main() {
  try {
    RNDM generator(2731);
    generator.gasdev();
    generator.rndm_save("rng_checkpoint_legacy.save");
    RNDM restored(99);
    restored.rndm_load("rng_checkpoint_legacy.save");
    same_sequence(generator, restored);
    const std::string valid = read("rng_checkpoint_legacy.save");
    for (std::size_t length = 0; length < valid.size(); ++length)
      rejected_without_mutation(restored, valid.substr(0, length));
    std::string corrupt = valid;
    corrupt.back() = '4';
    rejected_without_mutation(restored, corrupt);
    rejected_without_mutation(restored, valid + "0");
    rejected_without_mutation(restored, std::string(1236, '\0'));

    feebl::PhysicalRandomKeyV2 key;
    key.run_seed = 1732;
    key.domain = feebl::RandomDomain::Beamstrahlung;
    key.encounter = 81;
    key.parent_a = 27;
    generator.begin_counter_stream(key);
    generator.gasdev();
    generator.rndm_save("rng_checkpoint_counter.save");
    restored.rndm_load("rng_checkpoint_counter.save");
    require(restored.counter_stream_active(), "counter mode was lost");
    same_sequence(generator, restored);

    // Replacing an existing checkpoint is a complete old-or-new operation.
    generator.gasdev();
    generator.rndm_save("rng_checkpoint_counter.save");
    restored.rndm_load("rng_checkpoint_counter.save");
    same_sequence(generator, restored);
    bool missing_rejected = false;
    const RNDM before = restored;
    try {
      restored.rndm_load("rng_checkpoint_missing_directory/file");
    } catch (const std::runtime_error&) {
      missing_rejected = true;
    }
    require(missing_rejected, "explicit missing checkpoint must fail");
    same_sequence(before, restored);
    bool write_rejected = false;
    try {
      generator.rndm_save("rng_checkpoint_missing_directory/file");
    } catch (const std::runtime_error&) {
      write_rejected = true;
    }
    require(write_rejected, "failed write must be reported");
    std::cout << "RNG checkpoint round-trip, truncation and transaction checks passed\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
