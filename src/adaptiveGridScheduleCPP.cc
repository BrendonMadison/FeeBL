#include "adaptiveGridScheduleCPP.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <tuple>
#include <utility>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <process.h>
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace feebl {
namespace adaptive_grid {
namespace {

constexpr std::size_t kSha256Bytes = 32U;
constexpr std::size_t kSha256HexCharacters = 2U * kSha256Bytes;
constexpr std::size_t kMaximumIdentifierBytes = 4096U;
constexpr std::size_t kMaximumClassIdBytes = 256U;
constexpr std::uintmax_t kMaximumScheduleFileBytes =
    static_cast<std::uintmax_t>(256U) * 1024U * 1024U;

std::uint32_t rotate_right(std::uint32_t value, unsigned int count) noexcept {
  return (value >> count) | (value << (32U - count));
}

class Sha256 {
 public:
  Sha256() noexcept
      : state_{{0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
                0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U}},
        buffer_{},
        buffered_(0U),
        total_bytes_(0U) {}

  void update(std::string_view input) {
    if (input.size() >
        static_cast<std::size_t>(std::numeric_limits<std::uint64_t>::max() -
                                 total_bytes_)) {
      throw AdaptiveGridScheduleError("schedule hash input is too large");
    }
    total_bytes_ += static_cast<std::uint64_t>(input.size());

    std::size_t cursor = 0U;
    if (buffered_ != 0U) {
      const std::size_t take = std::min(buffer_.size() - buffered_, input.size());
      for (std::size_t i = 0U; i < take; ++i) {
        buffer_[buffered_ + i] =
            static_cast<std::uint8_t>(static_cast<unsigned char>(input[i]));
      }
      buffered_ += take;
      cursor += take;
      if (buffered_ == buffer_.size()) {
        transform(buffer_.data());
        buffered_ = 0U;
      }
    }

    while (input.size() - cursor >= buffer_.size()) {
      std::array<std::uint8_t, 64U> block{};
      for (std::size_t i = 0U; i < block.size(); ++i) {
        block[i] = static_cast<std::uint8_t>(
            static_cast<unsigned char>(input[cursor + i]));
      }
      transform(block.data());
      cursor += block.size();
    }

    while (cursor < input.size()) {
      buffer_[buffered_] = static_cast<std::uint8_t>(
          static_cast<unsigned char>(input[cursor]));
      ++buffered_;
      ++cursor;
    }
  }

  std::array<std::uint8_t, kSha256Bytes> finish() {
    if (total_bytes_ > std::numeric_limits<std::uint64_t>::max() / 8U) {
      throw AdaptiveGridScheduleError("schedule hash bit length overflow");
    }
    const std::uint64_t bit_length = total_bytes_ * 8U;

    buffer_[buffered_++] = 0x80U;
    if (buffered_ > 56U) {
      while (buffered_ < buffer_.size()) {
        buffer_[buffered_++] = 0U;
      }
      transform(buffer_.data());
      buffered_ = 0U;
    }
    while (buffered_ < 56U) {
      buffer_[buffered_++] = 0U;
    }
    for (unsigned int shift = 56U; shift <= 56U; shift -= 8U) {
      buffer_[buffered_++] =
          static_cast<std::uint8_t>((bit_length >> shift) & 0xffU);
      if (shift == 0U) {
        break;
      }
    }
    transform(buffer_.data());
    buffered_ = 0U;

    std::array<std::uint8_t, kSha256Bytes> result{};
    for (std::size_t i = 0U; i < state_.size(); ++i) {
      result[4U * i] = static_cast<std::uint8_t>(state_[i] >> 24U);
      result[4U * i + 1U] = static_cast<std::uint8_t>(state_[i] >> 16U);
      result[4U * i + 2U] = static_cast<std::uint8_t>(state_[i] >> 8U);
      result[4U * i + 3U] = static_cast<std::uint8_t>(state_[i]);
    }
    return result;
  }

 private:
  void transform(const std::uint8_t* block) noexcept {
    static constexpr std::array<std::uint32_t, 64U> constants{{
        0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U,
        0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
        0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
        0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
        0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
        0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
        0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
        0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
        0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
        0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
        0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U,
        0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
        0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U,
        0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
        0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
        0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U}};

    std::array<std::uint32_t, 64U> words{};
    for (std::size_t i = 0U; i < 16U; ++i) {
      words[i] = (static_cast<std::uint32_t>(block[4U * i]) << 24U) |
                 (static_cast<std::uint32_t>(block[4U * i + 1U]) << 16U) |
                 (static_cast<std::uint32_t>(block[4U * i + 2U]) << 8U) |
                 static_cast<std::uint32_t>(block[4U * i + 3U]);
    }
    for (std::size_t i = 16U; i < words.size(); ++i) {
      const std::uint32_t s0 = rotate_right(words[i - 15U], 7U) ^
                               rotate_right(words[i - 15U], 18U) ^
                               (words[i - 15U] >> 3U);
      const std::uint32_t s1 = rotate_right(words[i - 2U], 17U) ^
                               rotate_right(words[i - 2U], 19U) ^
                               (words[i - 2U] >> 10U);
      words[i] = words[i - 16U] + s0 + words[i - 7U] + s1;
    }

    std::uint32_t a = state_[0];
    std::uint32_t b = state_[1];
    std::uint32_t c = state_[2];
    std::uint32_t d = state_[3];
    std::uint32_t e = state_[4];
    std::uint32_t f = state_[5];
    std::uint32_t g = state_[6];
    std::uint32_t h = state_[7];

    for (std::size_t i = 0U; i < words.size(); ++i) {
      const std::uint32_t upper_sigma1 =
          rotate_right(e, 6U) ^ rotate_right(e, 11U) ^ rotate_right(e, 25U);
      const std::uint32_t choose = (e & f) ^ ((~e) & g);
      const std::uint32_t temporary1 =
          h + upper_sigma1 + choose + constants[i] + words[i];
      const std::uint32_t upper_sigma0 =
          rotate_right(a, 2U) ^ rotate_right(a, 13U) ^ rotate_right(a, 22U);
      const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
      const std::uint32_t temporary2 = upper_sigma0 + majority;

      h = g;
      g = f;
      f = e;
      e = d + temporary1;
      d = c;
      c = b;
      b = a;
      a = temporary1 + temporary2;
    }

    state_[0] += a;
    state_[1] += b;
    state_[2] += c;
    state_[3] += d;
    state_[4] += e;
    state_[5] += f;
    state_[6] += g;
    state_[7] += h;
  }

  std::array<std::uint32_t, 8U> state_;
  std::array<std::uint8_t, 64U> buffer_;
  std::size_t buffered_;
  std::uint64_t total_bytes_;
};

char lower_hex_digit(unsigned int value) noexcept {
  return static_cast<char>(value < 10U ? ('0' + value) : ('a' + value - 10U));
}

std::string bytes_to_lower_hex(const std::uint8_t* data, std::size_t size) {
  std::string result;
  result.resize(2U * size);
  for (std::size_t i = 0U; i < size; ++i) {
    result[2U * i] = lower_hex_digit(static_cast<unsigned int>(data[i] >> 4U));
    result[2U * i + 1U] =
        lower_hex_digit(static_cast<unsigned int>(data[i] & 0x0fU));
  }
  return result;
}

std::string sha256_hex_impl(std::string_view input) {
  Sha256 hash;
  hash.update(input);
  const auto digest = hash.finish();
  return bytes_to_lower_hex(digest.data(), digest.size());
}

bool is_lower_hex(char value) noexcept {
  return (value >= '0' && value <= '9') || (value >= 'a' && value <= 'f');
}

unsigned int lower_hex_value(char value) {
  if (value >= '0' && value <= '9') {
    return static_cast<unsigned int>(value - '0');
  }
  if (value >= 'a' && value <= 'f') {
    return 10U + static_cast<unsigned int>(value - 'a');
  }
  throw AdaptiveGridScheduleError("noncanonical hexadecimal digit");
}

void validate_sha256(const std::string& value, const char* field) {
  if (value.size() != kSha256HexCharacters ||
      !std::all_of(value.begin(), value.end(), is_lower_hex)) {
    throw AdaptiveGridScheduleError(std::string(field) +
                                    " must be 64 lowercase hexadecimal digits");
  }
}

void validate_identifier(const std::string& value,
                         std::size_t maximum,
                         const char* field) {
  if (value.empty() || value.size() > maximum ||
      value.find('\0') != std::string::npos) {
    throw AdaptiveGridScheduleError(std::string(field) +
                                    " is empty, oversized, or contains NUL");
  }
}

std::string encode_identifier(const std::string& value) {
  return bytes_to_lower_hex(reinterpret_cast<const std::uint8_t*>(value.data()),
                            value.size());
}

std::string decode_identifier(const std::string& encoded,
                              std::size_t maximum,
                              const char* field) {
  if (encoded.empty() || (encoded.size() % 2U) != 0U ||
      encoded.size() / 2U > maximum ||
      !std::all_of(encoded.begin(), encoded.end(), is_lower_hex)) {
    throw AdaptiveGridScheduleError(std::string(field) +
                                    " has invalid canonical hex encoding");
  }
  std::string result;
  result.resize(encoded.size() / 2U);
  for (std::size_t i = 0U; i < result.size(); ++i) {
    const unsigned int upper = lower_hex_value(encoded[2U * i]);
    const unsigned int lower = lower_hex_value(encoded[2U * i + 1U]);
    result[i] = static_cast<char>((upper << 4U) | lower);
  }
  validate_identifier(result, maximum, field);
  return result;
}

std::string uint64_hex(std::uint64_t value) {
  std::array<std::uint8_t, 8U> bytes{};
  for (std::size_t i = 0U; i < bytes.size(); ++i) {
    const unsigned int shift = static_cast<unsigned int>(8U * (7U - i));
    bytes[i] = static_cast<std::uint8_t>(value >> shift);
  }
  return bytes_to_lower_hex(bytes.data(), bytes.size());
}

std::uint64_t parse_uint64_hex(const std::string& value, const char* field) {
  if (value.size() != 16U ||
      !std::all_of(value.begin(), value.end(), is_lower_hex)) {
    throw AdaptiveGridScheduleError(std::string(field) +
                                    " must be 16 lowercase hexadecimal digits");
  }
  std::uint64_t result = 0U;
  const auto parsed =
      std::from_chars(value.data(), value.data() + value.size(), result, 16);
  if (parsed.ec != std::errc() || parsed.ptr != value.data() + value.size()) {
    throw AdaptiveGridScheduleError(std::string(field) + " is not a uint64");
  }
  return result;
}

template <typename Unsigned>
Unsigned parse_unsigned_decimal(const std::string& value, const char* field) {
  static_assert(std::numeric_limits<Unsigned>::is_integer,
                "integer parser requires an integer type");
  static_assert(!std::numeric_limits<Unsigned>::is_signed,
                "unsigned parser requires an unsigned type");
  if (value.empty() || (value.size() > 1U && value.front() == '0')) {
    throw AdaptiveGridScheduleError(std::string(field) +
                                    " is not canonical unsigned decimal");
  }
  Unsigned result = 0;
  const auto parsed =
      std::from_chars(value.data(), value.data() + value.size(), result, 10);
  if (parsed.ec != std::errc() || parsed.ptr != value.data() + value.size()) {
    throw AdaptiveGridScheduleError(std::string(field) +
                                    " is outside its unsigned range");
  }
  return result;
}

std::int64_t parse_signed_decimal(const std::string& value, const char* field) {
  if (value.empty() || value == "-0" ||
      (value.front() == '0' && value.size() > 1U) ||
      (value.front() == '-' && value.size() > 2U && value[1] == '0')) {
    throw AdaptiveGridScheduleError(std::string(field) +
                                    " is not canonical signed decimal");
  }
  std::int64_t result = 0;
  const auto parsed =
      std::from_chars(value.data(), value.data() + value.size(), result, 10);
  if (parsed.ec != std::errc() || parsed.ptr != value.data() + value.size()) {
    throw AdaptiveGridScheduleError(std::string(field) +
                                    " is outside the int64 range");
  }
  return result;
}

std::vector<std::string> split_tabs(const std::string& line) {
  std::vector<std::string> fields;
  std::size_t start = 0U;
  while (true) {
    const std::size_t separator = line.find('\t', start);
    if (separator == std::string::npos) {
      fields.emplace_back(line.substr(start));
      break;
    }
    fields.emplace_back(line.substr(start, separator - start));
    start = separator + 1U;
  }
  return fields;
}

std::vector<std::string> strict_lines(const std::string& data) {
  if (data.empty() || data.back() != '\n') {
    throw AdaptiveGridScheduleError("schedule is empty or truncated before newline");
  }
  if (data.find('\r') != std::string::npos ||
      data.find('\0') != std::string::npos) {
    throw AdaptiveGridScheduleError("schedule contains forbidden CR or NUL bytes");
  }
  std::vector<std::string> lines;
  std::size_t start = 0U;
  while (start < data.size()) {
    const std::size_t end = data.find('\n', start);
    if (end == std::string::npos) {
      throw AdaptiveGridScheduleError("schedule line scan failed");
    }
    if (end - start > 65536U) {
      throw AdaptiveGridScheduleError("schedule contains an oversized line");
    }
    lines.emplace_back(data.substr(start, end - start));
    start = end + 1U;
  }
  return lines;
}

std::string join_payload_lines(const std::vector<std::string>& lines) {
  if (lines.empty()) {
    throw AdaptiveGridScheduleError("schedule has no payload lines");
  }
  std::string result;
  for (std::size_t i = 0U; i + 1U < lines.size(); ++i) {
    result += lines[i];
    result.push_back('\n');
  }
  return result;
}

void validate_extent(std::int64_t origin,
                     std::uint64_t extent,
                     const char* axis) {
  if (extent == 0U ||
      extent > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
    throw AdaptiveGridScheduleError(std::string(axis) +
                                    " window extent is zero or too large");
  }
  const std::int64_t offset = static_cast<std::int64_t>(extent - 1U);
  if (origin > std::numeric_limits<std::int64_t>::max() - offset) {
    throw AdaptiveGridScheduleError(std::string(axis) +
                                    " window endpoint overflows int64");
  }
}

std::uint64_t process_id() noexcept {
#if defined(_WIN32)
  return static_cast<std::uint64_t>(_getpid());
#else
  return static_cast<std::uint64_t>(getpid());
#endif
}

class TemporaryFileGuard {
 public:
  explicit TemporaryFileGuard(std::filesystem::path path)
      : path_(std::move(path)), armed_(true) {}
  TemporaryFileGuard(const TemporaryFileGuard&) = delete;
  TemporaryFileGuard& operator=(const TemporaryFileGuard&) = delete;
  ~TemporaryFileGuard() {
    if (armed_) {
      std::error_code ignored;
      std::filesystem::remove(path_, ignored);
    }
  }
  void release() noexcept { armed_ = false; }

 private:
  std::filesystem::path path_;
  bool armed_;
};

void replace_file(const std::filesystem::path& temporary,
                  const std::filesystem::path& destination) {
#if defined(_WIN32)
  if (MoveFileExW(temporary.c_str(), destination.c_str(),
                  MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) == 0) {
    throw AdaptiveGridScheduleError(
        "atomic schedule replacement failed with Windows error " +
        std::to_string(static_cast<unsigned long>(GetLastError())));
  }
#else
  std::error_code error;
  std::filesystem::rename(temporary, destination, error);
  if (error) {
    throw AdaptiveGridScheduleError("atomic schedule rename failed: " +
                                    error.message());
  }
#endif
}

}  // namespace

std::string sha256_hex(std::string_view input) {
  return sha256_hex_impl(input);
}

AdaptiveGridScheduleHeader::AdaptiveGridScheduleHeader(
    std::string card_sha256,
    std::uint64_t canonical_lattice_hash,
    std::string source_id,
    std::string build_id,
    std::string controller_id,
    std::string particle_ordering_sha256)
    : card_sha256_(std::move(card_sha256)),
      canonical_lattice_hash_(canonical_lattice_hash),
      source_id_(std::move(source_id)),
      build_id_(std::move(build_id)),
      controller_id_(std::move(controller_id)),
      particle_ordering_sha256_(std::move(particle_ordering_sha256)) {
  validate_sha256(card_sha256_, "card_sha256");
  validate_identifier(source_id_, kMaximumIdentifierBytes, "source_id");
  validate_identifier(build_id_, kMaximumIdentifierBytes, "build_id");
  validate_identifier(controller_id_, kMaximumIdentifierBytes, "controller_id");
  validate_sha256(particle_ordering_sha256_, "particle_ordering_sha256");
}

bool AdaptiveGridScheduleHeader::operator==(
    const AdaptiveGridScheduleHeader& other) const noexcept {
  return card_sha256_ == other.card_sha256_ &&
         canonical_lattice_hash_ == other.canonical_lattice_hash_ &&
         source_id_ == other.source_id_ && build_id_ == other.build_id_ &&
         controller_id_ == other.controller_id_ &&
         particle_ordering_sha256_ == other.particle_ordering_sha256_;
}

bool AdaptiveGridEncounterKey::operator==(
    const AdaptiveGridEncounterKey& other) const noexcept {
  return crossing == other.crossing && diagonal == other.diagonal &&
         wave == other.wave && substep == other.substep &&
         encounter == other.encounter;
}

bool AdaptiveGridEncounterKey::operator<(
    const AdaptiveGridEncounterKey& other) const noexcept {
  return std::tie(crossing, diagonal, wave, substep, encounter) <
         std::tie(other.crossing, other.diagonal, other.wave, other.substep,
                  other.encounter);
}

AdaptiveGridEncounterRecord::AdaptiveGridEncounterRecord(
    AdaptiveGridEncounterKey key,
    std::int64_t beam1_slice,
    std::int64_t beam2_slice,
    std::int64_t origin_x,
    std::int64_t origin_y,
    std::uint64_t n_x,
    std::uint64_t n_y,
    std::uint64_t padded_n_x,
    std::uint64_t padded_n_y,
    std::uint64_t plan_hash,
    std::string class_id,
    std::uint64_t source_count,
    std::uint64_t target_count,
    bool full_window)
    : key_(key),
      beam1_slice_(beam1_slice),
      beam2_slice_(beam2_slice),
      origin_x_(origin_x),
      origin_y_(origin_y),
      n_x_(n_x),
      n_y_(n_y),
      padded_n_x_(padded_n_x),
      padded_n_y_(padded_n_y),
      plan_hash_(plan_hash),
      class_id_(std::move(class_id)),
      source_count_(source_count),
      target_count_(target_count),
      full_window_(full_window) {
  const bool empty = n_x_ == 0U && n_y_ == 0U;
  if (empty) {
    if (n_x_ != 0U || n_y_ != 0U || padded_n_x_ != 0U ||
        padded_n_y_ != 0U || origin_x_ != 0 || origin_y_ != 0 ||
        source_count_ != 0U || target_count_ != 0U || full_window_ ||
        class_id_ != "empty") {
      throw AdaptiveGridScheduleError(
          "empty encounter requires the canonical all-zero empty encoding");
    }
  } else {
    validate_extent(origin_x_, n_x_, "x");
    validate_extent(origin_y_, n_y_, "y");
  }
  if (padded_n_x_ < n_x_ || padded_n_y_ < n_y_) {
    throw AdaptiveGridScheduleError(
        "padded dimensions must contain the physical window");
  }
  validate_identifier(class_id_, kMaximumClassIdBytes, "class_id");
}

bool AdaptiveGridEncounterRecord::operator==(
    const AdaptiveGridEncounterRecord& other) const noexcept {
  return key_ == other.key_ && beam1_slice_ == other.beam1_slice_ &&
         beam2_slice_ == other.beam2_slice_ && origin_x_ == other.origin_x_ &&
         origin_y_ == other.origin_y_ && n_x_ == other.n_x_ &&
         n_y_ == other.n_y_ && padded_n_x_ == other.padded_n_x_ &&
         padded_n_y_ == other.padded_n_y_ && plan_hash_ == other.plan_hash_ &&
         class_id_ == other.class_id_ && source_count_ == other.source_count_ &&
         target_count_ == other.target_count_ && full_window_ == other.full_window_;
}

AdaptiveGridSchedule::AdaptiveGridSchedule(AdaptiveGridScheduleHeader header)
    : header_(std::move(header)), records_() {}

void AdaptiveGridSchedule::append(const AdaptiveGridEncounterRecord& record) {
  if (!records_.empty() && !(records_.back().key() < record.key())) {
    if (records_.back().key() == record.key()) {
      throw AdaptiveGridScheduleError("duplicate adaptive-grid encounter key");
    }
    throw AdaptiveGridScheduleError(
        "adaptive-grid encounter records are not in canonical order");
  }
  records_.push_back(record);
}

std::string AdaptiveGridSchedule::canonical_payload() const {
  std::ostringstream output;
  output.imbue(std::locale::classic());
  output << "FEEBL_ADAPTIVE_GRID_SCHEDULE\t"
         << kAdaptiveGridScheduleFormatVersion << '\n';
  output << "H\t" << kAdaptiveGridScheduleFormatVersion << '\t'
         << header_.card_sha256() << '\t'
         << uint64_hex(header_.canonical_lattice_hash()) << '\t'
         << encode_identifier(header_.source_id()) << '\t'
         << encode_identifier(header_.build_id()) << '\t'
         << encode_identifier(header_.controller_id()) << '\t'
         << header_.particle_ordering_sha256() << '\n';
  output << "N\t" << records_.size() << '\n';
  for (const AdaptiveGridEncounterRecord& record : records_) {
    const AdaptiveGridEncounterKey& key = record.key();
    output << "E\t" << key.crossing << '\t' << key.diagonal << '\t'
           << key.wave << '\t' << key.substep << '\t' << key.encounter << '\t'
           << record.beam1_slice() << '\t' << record.beam2_slice() << '\t'
           << record.origin_x() << '\t' << record.origin_y() << '\t'
           << record.n_x() << '\t' << record.n_y() << '\t'
           << record.padded_n_x() << '\t' << record.padded_n_y() << '\t'
           << uint64_hex(record.plan_hash()) << '\t'
           << encode_identifier(record.class_id()) << '\t'
           << record.source_count() << '\t' << record.target_count() << '\t'
           << (record.full_window() ? 1 : 0) << '\n';
  }
  if (!output) {
    throw AdaptiveGridScheduleError("failed to serialize adaptive-grid schedule");
  }
  return output.str();
}

std::string AdaptiveGridSchedule::stable_hash() const {
  return sha256_hex_impl(canonical_payload());
}

void AdaptiveGridSchedule::save_atomic(const std::filesystem::path& path) const {
  if (path.empty() || path.filename().empty()) {
    throw AdaptiveGridScheduleError("schedule destination path is empty");
  }
  const std::filesystem::path parent =
      path.has_parent_path() ? path.parent_path() : std::filesystem::path(".");
  std::error_code status_error;
  if (!std::filesystem::is_directory(parent, status_error) || status_error) {
    throw AdaptiveGridScheduleError("schedule destination directory is unavailable");
  }

  const std::string payload = canonical_payload();
  const std::string document =
      payload + "C\t" + sha256_hex_impl(payload) + "\n";
  if (document.size() >
      static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max())) {
    throw AdaptiveGridScheduleError("serialized schedule exceeds stream limits");
  }
  static std::atomic<std::uint64_t> sequence{0U};
  const std::uint64_t ticket = sequence.fetch_add(1U, std::memory_order_relaxed);
  std::filesystem::path temporary = path;
  temporary += ".tmp." + std::to_string(process_id()) + "." +
               std::to_string(ticket);
  TemporaryFileGuard guard(temporary);

  {
    std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
    if (!output) {
      throw AdaptiveGridScheduleError("cannot create temporary schedule file");
    }
    output.write(document.data(), static_cast<std::streamsize>(document.size()));
    output.flush();
    if (!output) {
      throw AdaptiveGridScheduleError("cannot write complete temporary schedule");
    }
  }
  replace_file(temporary, path);
  guard.release();
}

AdaptiveGridSchedule AdaptiveGridSchedule::load(
    const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) {
    throw AdaptiveGridScheduleError("cannot open adaptive-grid schedule");
  }
  const std::streampos end = input.tellg();
  if (end < std::streampos(0)) {
    throw AdaptiveGridScheduleError("cannot determine schedule size");
  }
  const auto size = static_cast<std::uintmax_t>(end);
  if (size > kMaximumScheduleFileBytes ||
      size > static_cast<std::uintmax_t>(
                 std::numeric_limits<std::streamsize>::max())) {
    throw AdaptiveGridScheduleError("schedule exceeds the reader size limit");
  }
  std::string data(static_cast<std::size_t>(size), '\0');
  input.seekg(0, std::ios::beg);
  if (!data.empty()) {
    input.read(data.data(), static_cast<std::streamsize>(data.size()));
  }
  if (!input || input.gcount() != static_cast<std::streamsize>(data.size())) {
    throw AdaptiveGridScheduleError("schedule read was truncated");
  }

  const std::vector<std::string> lines = strict_lines(data);
  if (lines.size() < 4U) {
    throw AdaptiveGridScheduleError("schedule is missing required records");
  }
  if (lines[0] != "FEEBL_ADAPTIVE_GRID_SCHEDULE\t1") {
    throw AdaptiveGridScheduleError("unsupported adaptive-grid schedule version");
  }

  const std::vector<std::string> checksum_fields = split_tabs(lines.back());
  if (checksum_fields.size() != 2U || checksum_fields[0] != "C") {
    throw AdaptiveGridScheduleError("schedule checksum trailer is malformed");
  }
  validate_sha256(checksum_fields[1], "schedule checksum");
  const std::string payload = join_payload_lines(lines);
  if (sha256_hex_impl(payload) != checksum_fields[1]) {
    throw AdaptiveGridScheduleError("adaptive-grid schedule checksum mismatch");
  }

  const std::vector<std::string> header_fields = split_tabs(lines[1]);
  if (header_fields.size() != 8U || header_fields[0] != "H" ||
      parse_unsigned_decimal<std::uint32_t>(header_fields[1],
                                            "header version") !=
          kAdaptiveGridScheduleFormatVersion) {
    throw AdaptiveGridScheduleError("schedule header is incompatible");
  }
  AdaptiveGridScheduleHeader header(
      header_fields[2],
      parse_uint64_hex(header_fields[3], "canonical_lattice_hash"),
      decode_identifier(header_fields[4], kMaximumIdentifierBytes, "source_id"),
      decode_identifier(header_fields[5], kMaximumIdentifierBytes, "build_id"),
      decode_identifier(header_fields[6], kMaximumIdentifierBytes,
                        "controller_id"),
      header_fields[7]);

  const std::vector<std::string> count_fields = split_tabs(lines[2]);
  if (count_fields.size() != 2U || count_fields[0] != "N") {
    throw AdaptiveGridScheduleError("schedule record-count line is malformed");
  }
  const std::uint64_t count =
      parse_unsigned_decimal<std::uint64_t>(count_fields[1], "record count");
  if (count > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) ||
      count > static_cast<std::uint64_t>(lines.size())) {
    throw AdaptiveGridScheduleError("schedule record count overflows host size");
  }
  const std::size_t record_count = static_cast<std::size_t>(count);
  if (record_count > std::numeric_limits<std::size_t>::max() - 4U ||
      lines.size() != record_count + 4U) {
    throw AdaptiveGridScheduleError("schedule record count does not match file");
  }

  AdaptiveGridSchedule result(std::move(header));
  result.records_.reserve(record_count);
  for (std::size_t index = 0U; index < record_count; ++index) {
    const std::vector<std::string> fields = split_tabs(lines[3U + index]);
    if (fields.size() != 19U || fields[0] != "E") {
      throw AdaptiveGridScheduleError("encounter record is malformed at index " +
                                      std::to_string(index));
    }
    AdaptiveGridEncounterKey key;
    key.crossing = parse_unsigned_decimal<std::uint64_t>(fields[1], "crossing");
    key.diagonal = parse_unsigned_decimal<std::uint64_t>(fields[2], "diagonal");
    key.wave = parse_unsigned_decimal<std::uint64_t>(fields[3], "wave");
    key.substep = parse_unsigned_decimal<std::uint64_t>(fields[4], "substep");
    key.encounter = parse_unsigned_decimal<std::uint64_t>(fields[5], "encounter");
    const std::uint64_t full =
        parse_unsigned_decimal<std::uint64_t>(fields[18], "full_window");
    if (full > 1U) {
      throw AdaptiveGridScheduleError("full_window must be exactly 0 or 1");
    }
    result.append(AdaptiveGridEncounterRecord(
        key, parse_signed_decimal(fields[6], "beam1_slice"),
        parse_signed_decimal(fields[7], "beam2_slice"),
        parse_signed_decimal(fields[8], "origin_x"),
        parse_signed_decimal(fields[9], "origin_y"),
        parse_unsigned_decimal<std::uint64_t>(fields[10], "n_x"),
        parse_unsigned_decimal<std::uint64_t>(fields[11], "n_y"),
        parse_unsigned_decimal<std::uint64_t>(fields[12], "padded_n_x"),
        parse_unsigned_decimal<std::uint64_t>(fields[13], "padded_n_y"),
        parse_uint64_hex(fields[14], "plan_hash"),
        decode_identifier(fields[15], kMaximumClassIdBytes, "class_id"),
        parse_unsigned_decimal<std::uint64_t>(fields[16], "source_count"),
        parse_unsigned_decimal<std::uint64_t>(fields[17], "target_count"),
        full == 1U));
  }
  return result;
}

void AdaptiveGridSchedule::validate_exact(
    const AdaptiveGridSchedule& expected) const {
  if (header_ != expected.header_) {
    throw AdaptiveGridScheduleError("adaptive-grid replay header mismatch");
  }
  if (records_.size() != expected.records_.size()) {
    throw AdaptiveGridScheduleError("adaptive-grid replay encounter-count mismatch");
  }
  for (std::size_t i = 0U; i < records_.size(); ++i) {
    if (records_[i] != expected.records_[i]) {
      throw AdaptiveGridScheduleError(
          "adaptive-grid replay record mismatch at canonical index " +
          std::to_string(i));
    }
  }
}

bool AdaptiveGridSchedule::operator==(
    const AdaptiveGridSchedule& other) const noexcept {
  return header_ == other.header_ && records_ == other.records_;
}

}  // namespace adaptive_grid
}  // namespace feebl
