#include <cstdio>
#include <iostream>
#include <cmath>
#include <limits>
#include <climits>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <locale>
#include <sstream>
#include <string>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <io.h>
#include <windows.h>
#else
#include <unistd.h>
#endif
#include "rndmCPP.h"
#include "mathconst.h"

RNDM::RNDM(unsigned long seed)
{
  counter_mode_ = false;
  counter_draw_ = 0U;
  counter_exhausted_ = false;
  rndmst0(1);
  rndmst1(1);
  rndmst2(1);
  //  rndmst3(1);
  rndmst5(12,34,56,78);
  rndmst6(1);
  rndmst7(seed);
  rndmst8(1);
  iset_ = 0;
  v1_ = 0.0, v2_ = 0.0;
  counterRndm7_ = 0;
}

void  RNDM::rndmst0(int i)
{
  rndm0_store.i=i;
}

float RNDM::rndm0()
{
  const int m=2147483647,a=16807,q=127773,r=2836;
  const float m_inv=(1.0-RNDM_EPS)/(float)m;
  int k;

  k=rndm0_store.i/q;
  if((rndm0_store.i=a*(rndm0_store.i-k*q)-r*k)<0) rndm0_store.i+=m;
  return m_inv*rndm0_store.i;
}


void RNDM::rndmst1(int i)
{
  //    const long m=2147483647,m1=m-1,a=16807,q=127773,r=2836,n=32,nd=1+m1/n;
  //   const float m_inv=(1.0-RNDM_EPS)/(float)m;
  const long m=2147483647, a=16807,q=127773,r=2836,n=32;
  int k,j;

  for (j=n+7;j>=0;j--)
    {
      k=i/q;
      if((i=a*(i-k*q)-r*k)<0) i+=m;
      if (j<n) rndm1_store.is[j]=i;
    }
  rndm1_store.i=i;
  rndm1_store.p=rndm1_store.is[0];
}

float RNDM::rndm1()
{
  const long m=2147483647,m1=m-1,a=16807,q=127773,r=2836,n=32,nd=1+m1/n;
  const float m_inv=(1.0-RNDM_EPS)/(float)m;
  int k,j;

  k=rndm1_store.i/q;
  if((rndm1_store.i=a*(rndm1_store.i-k*q)-r*k)<0) rndm1_store.i+=m;
  j=rndm1_store.p/nd;
  rndm1_store.p=rndm1_store.is[j];
  rndm1_store.is[j]=rndm1_store.i;
  return rndm1_store.p<m1?m_inv*rndm1_store.p:m_inv*m1;
/*  return (m_inv*min(rndm1_store.p,m1));*/
}


void RNDM::rndmst2(int i)
{
  //   const int m1=2147483563,m2=2147483399,m1_1=m1-1;
  //    const int a1=40014,a2=40692,q1=53668,q2=52774,r1=12211,r2=3791,n=32, nd=1+m1_1/n;
  const int m1=2147483563;
  const int a1=40014,q1=53668,r1=12211,n=32;
/*  const float m1_inv=(1.0-RNDM_EPS)/(float)m1,m2_inv=(1.0-RNDM_EPS)/(float)m2;*/
  int k,j;

  for (j=n+7;j>=0;j--)
    {
      k=i/q1;
      if((i=a1*(i-k*q1)-r1*k)<0) i+=m1;
      if (j<n) rndm2_store.is[j]=i;
    }
  rndm2_store.i1=i;
  rndm2_store.i2=12345678;
  rndm2_store.p=rndm2_store.is[0];
}

float RNDM::rndm2()
{
  const int m1=2147483563,m2=2147483399,m1_1=m1-1;
  const int a1=40014,a2=40692,q1=53668,q2=52774,r1=12211,r2=3791,n=32,
            nd=1+m1_1/n;
  const float m_inv=(1.0-RNDM_EPS)/(float)m1;
  int k,j;

  k=rndm2_store.i1/q1;
  if((rndm2_store.i1=a1*(rndm2_store.i1-k*q1)-r1*k)<0) rndm2_store.i1+=m1;
  k=rndm2_store.i2;
  if((rndm2_store.i2=a2*(rndm2_store.i2-k*q2)-r2*k)<0) rndm2_store.i2+=m2;
  j=rndm2_store.p/nd;
  if((rndm2_store.p=rndm2_store.is[j]-rndm2_store.i2)<1) rndm2_store.p+=m1_1;
  rndm2_store.is[j]=rndm2_store.i2;
  return rndm2_store.p*m_inv;
/*  return rndm2_store.p<m1_1?m_inv*rndm2_store.p:m_inv*m1_1;*/
/*  return m_inv*min(rndm2_store.p,m1_1);*/
}

void RNDM::rndmst5(int na1,int na2,int na3, int nb1)
{
  int i,j,nat;
  float s,t;
  rndm5_store.i=96;
  rndm5_store.j=32;
  for (i=0;i<97;i++)
    {
      s=0.0;
      t=0.5;
      for (j=0;j<24;j++)
	{
	  nat=(((na1*na2) % 179)*na3) % 179;
	  na1=na2;
	  na2=na3;
	  na3=nat;
	  nb1=(53*nb1+1) % 169;
	  if ((nb1*nat) % 64 >= 32)
	    {
	      s+=t;
	    }
	  t*=0.5;
	}
      rndm5_store.u[i]=s;
    }
  rndm5_store.c=    362436.0/16777216.0;
  rndm5_store.cd=  7654321.0/16777216.0;
  rndm5_store.cm= 16777213.0/16777216.0;
}

float RNDM::rndm5()
{
  float temp;

/* for (;;){*/
  temp=rndm5_store.u[rndm5_store.i]-rndm5_store.u[rndm5_store.j];
  if (temp<0.0)
    {
      temp+=1.0;
    }
  rndm5_store.u[rndm5_store.i]=temp;
  if (--rndm5_store.i<0) rndm5_store.i=96;
  if (--rndm5_store.j<0) rndm5_store.j=96;
  rndm5_store.c-=rndm5_store.cd;
  if (rndm5_store.c<0.0)
    {
      rndm5_store.c+=rndm5_store.cm;
    }
  temp-=rndm5_store.c;
  if (temp<0.0)
    {
      return temp+1.0;
    }
  else
    {
/*      if (temp>0.0) */
        return temp;
    }
/*}*/
}


void RNDM::rndmst6(int i)
{
  rndm6_store.i=i;
}

float RNDM::rndm6()
{

  if((rndm6_store.i*=48828125)<0)
    {
      rndm6_store.i-=2147483647;
      rndm6_store.i--;
    }
  return (float)rndm6_store.i*0.46566129e-9*(1.0-RNDM_EPS);
}


void RNDM::rndmst7(unsigned long i)
{
  //  std::cout << " test of the random generator rndm7 " << std::endl;

  rndm7_store.i= 0L;
  unsigned long a_coeff;
  unsigned long c_coeff;
  unsigned long test7[11];

  // 32-bits machine 
#ifndef COMPUTER64b

  test7[0] = 0x3c6ef35f;
  test7[1] = 0x47502932;
  test7[2] = 0xd1ccf6e9;
  test7[3] = 0xaaf95334;
  test7[4] = 0x6252e503;
  test7[5] = 0x9f2ec686;
  test7[6] = 0x57fe6c2d;
  test7[7] = 0xa3d95fa8;
  test7[8] = 0x81fdbee7;
  test7[9] = 0x94f0af1a;
  test7[10] = 0xcbf633b1;

  a_coeff = 1664525L;
  c_coeff = 1013904223L;
  if ( rndm_test7(a_coeff , c_coeff, test7) )
    {
      std::cout << " random generator successful test : 32-bits computer " << std::endl;      
      // following factor is 1/(2.exp(32))
      rndm7_modulo_dividing_factor_ = 0.232830643654e-9;
    }
  else
    {
      std::cerr << " RNDM::rndmst7 : test for random generator failed, are we really working on a  32-bits computer ? " << std::endl;
      exit(1);
    }
#else
  test7[0] = 0x1;
  test7[1] = 0x5851f42d4c957f2e;
  test7[2] = 0xc0b18ccf4e252d17;
  test7[3] = 0xcbb5f646404a560c;
  test7[4] = 0xc7033129d2bd141d;
  test7[5] = 0x30705b042917ec1a;
  test7[6] = 0x20fd5db43a776693;
  test7[7] = 0x9a8b7f78da6ef4d8;
  test7[8] = 0x502959d812b031f9;
  test7[9] = 0xab894868b3b04fc6;
  test7[10]= 0x6c0356a743cf3fcf;


  a_coeff = 6364136223846793005L;
  c_coeff = 1L;
  if ( rndm_test7(a_coeff , c_coeff, test7) )
    {
      std::cout << " random generator successful test : 64-bits computer " << std::endl;
      // following factor is 1/(2.exp(64))
      rndm7_modulo_dividing_factor_ = 0.54210108624e-19;
    }
  else
    {
      std::cerr << " RNDM::rndmst7 : test for random generator failed, are we really working on a  64-bits computer ? maybe you should desactive the define COMPUTER64b (CMT requirements) " << std::endl;
      exit(1);
    }
#endif

  rndm7_a_coeff_ = a_coeff;
  rndm7_c_coeff_ = c_coeff;
  rndm7_store.i=i;
}

bool RNDM::rndm_test7(unsigned long a_coeff, unsigned long c_coeff, const unsigned long* test_values)
{
  int k;
  unsigned long test_i = 0;
  bool test = true;
  for (k=0; k< 11; k++)
    {
       test_i =a_coeff* test_i + c_coeff;
       if (test_i != test_values[k])
	 {
	   test = false;
	   printf(" k= %d random value= %lx should be %lx \n", k, test_i, test_values[k]);
	   break;
	 }
    }
  return test;
}

float RNDM::rndm7()
{
  if (counterRndm7_ == std::numeric_limits<std::uint64_t>::max())
    throw std::overflow_error("RNDM draw counter exhausted");
  ++counterRndm7_;
  if (counter_mode_)
    {
      if (counter_exhausted_)
        throw std::overflow_error("CounterRngV2 stream exhausted");
      const std::uint64_t draw = counter_draw_;
      if (counter_draw_ == std::numeric_limits<std::uint64_t>::max())
        counter_exhausted_ = true;
      else
        ++counter_draw_;
      return feebl::CounterRngV2::uniform_float(counter_key_, draw);
    }
  rndm7_store.i=rndm7_a_coeff_*rndm7_store.i+rndm7_c_coeff_;
/* additional factor to ensure rndm7!=1.0 if float */
  return  (float)rndm7_store.i*rndm7_modulo_dividing_factor_*(1.0-RNDM_EPS);
}


void RNDM::rndmst8(int idummy)
{

  m1_ =259200;
  ia1_ =7141;
  ic1_ =54773;
  rm1_ =1.0/259200.0;
  m2_ =134456;
  ia2_ =8121;
  ic2_ =28411;
  rm2_ =1.0/134456.0;
  m3_ =243000;
  ia3_ =4561;
  ic3_ =51349;
  iff_ =0;
  rm3_ =1.0/243000.0;


  if (idummy>0) idummy=-idummy;
  ix1_=(ic1_-idummy) % m1_;
  ix1_=(ia1_*ix1_+ic1_) % m1_;
  ix2_=ix1_ % m2_;
  ix1_=(ia1_*ix1_+ic1_) % m1_;
  ix3_=ix1_ % m3_;
  int k;
  for (k=0;k<97;k++)
    {
      ix1_=(ia1_*ix1_+ic1_) % m1_;
      ix2_=(ia2_*ix2_+ic2_) % m2_;
      rndm8_store.r[k]=(ix1_+ix2_*rm2_)*rm1_;
    }
  rndm8_store.ix1=ix1_;
  rndm8_store.ix2=ix2_;
  rndm8_store.ix3=ix3_;
}

float RNDM::rndm8()
{
  float help;

  rndm8_store.ix1=(ia1_*rndm8_store.ix1+ic1_) % m1_;
  rndm8_store.ix2=(ia2_*rndm8_store.ix2+ic2_) % m2_;
  rndm8_store.ix3=(ia3_*rndm8_store.ix3+ic3_) % m3_;
  int k =(int)((float)(97*rndm8_store.ix3)*rm3_);
  help=rndm8_store.r[k];
  rndm8_store.r[k]=(rndm8_store.ix1+rndm8_store.ix2*rm2_)*rm1_;
  return help;
}

float RNDM::expdev()
{
  return -log(1.0-rndm());
}

float RNDM::exp_dev()
{
    return -log(1.0-rndm2());
}

float RNDM::gasdev()
{
  float r;
  if (iset_ == 0)
    {
      for (;;)
	{
          v1_ = 2.0*rndm()-1.0;
          v2_ = 2.0*rndm()-1.0;
          r=v1_*v1_+v2_*v2_;
          if ((r<=1.0) && (r!=0.0))
	    {
	      break;
	    }
        }
      iset_ = 1;

               r=sqrt(-2.0*log((double)r)/r);
      v1_ *= r;
      v2_ *= r;
      return v1_;
    }
  else
    {
      iset_ = 0;
      return v2_;
    }
}

float RNDM::rndm_sincos(float *c)
{
    const float twopi=2.0*PI;
    float tmp;
    tmp=rndm();
    *c=cos(twopi*tmp);
    if (tmp>0.5)
	return sqrt(1.0- *c * *c);
    else
	return -sqrt(1.0- *c * *c);
}

// get a vector containing the sequence of integers from 1 to maxInt 
// randomly shuffled 
void RNDM::getShuffledIntegerSequence(int maxInt, std::vector<unsigned long int>& vec)
{
  int k, j;
  unsigned long int itemp;
  float u;
  vec.clear();
  vec.resize(maxInt);
  for (k = 0; k < maxInt; k++) vec[k] = k + 1;
  j = maxInt;
  while (j > 1)
    {
      u = rndm();
      k = (int)floor(j*u) +1;
      itemp = vec[k-1];
      vec[k-1] = vec[j-1];
      vec[j-1] = itemp;
      j--;
    }
}

namespace {

// FNV-1a detects accidental truncation/corruption; this is not authentication.
std::uint64_t checkpoint_checksum(const std::string& bytes)
{
  std::uint64_t hash = 14695981039346656037ULL;
  for (unsigned char byte : bytes) {
    hash = (hash ^ byte) * 1099511628211ULL;
  }
  return hash;
}

std::uint32_t float_bits(float value)
{
  static_assert(sizeof(float) == sizeof(std::uint32_t), "32-bit float required");
  std::uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

float float_from_bits(std::uint32_t bits)
{
  float value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

// A run owns its checkpoint path. Concurrent writers to one path are invalid.
// Failed writes leave the previous checkpoint intact and retain the temporary
// file for diagnosis. The same-directory rename is atomic on POSIX filesystems.
void write_checkpoint(const char* path, const std::string& bytes)
{
  const std::string temporary = std::string(path) + ".tmp";
  FILE* file = std::fopen(temporary.c_str(), "wb");
  if (!file) throw std::runtime_error("Cannot open RNG checkpoint temporary file: " + temporary);
  bool ok = std::fwrite(bytes.data(), 1, bytes.size(), file) == bytes.size();
  ok = (std::fflush(file) == 0) && ok;
#ifdef _WIN32
  ok = (_commit(_fileno(file)) == 0) && ok;
#else
  ok = (fsync(fileno(file)) == 0) && ok;
#endif
  ok = (std::fclose(file) == 0) && ok;
  if (!ok) throw std::runtime_error("Cannot write complete RNG checkpoint: " + temporary);
#ifdef _WIN32
  const bool replaced = MoveFileExA(temporary.c_str(), path,
      MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) != 0;
#else
  const bool replaced = std::rename(temporary.c_str(), path) == 0;
#endif
  if (!replaced) throw std::runtime_error("Cannot replace RNG checkpoint: " + std::string(path));
}

}  // namespace

int RNDM::rndm_save(const char* path) const
{
  // Only rndm7/CounterRngV2 are reachable through the public drawing API.
  // Decimal integers encode exact float bits, avoiding locale and rounding.
  std::ostringstream payload;
  payload.imbue(std::locale::classic());
  payload << sizeof(unsigned long) * CHAR_BIT << ' '
          << rndm7_a_coeff_ << ' ' << rndm7_c_coeff_ << ' '
          << float_bits(rndm7_modulo_dividing_factor_) << ' '
          << rndm7_store.i << ' ' << counterRndm7_ << ' '
          << iset_ << ' ' << float_bits(v1_) << ' ' << float_bits(v2_) << ' '
          << counter_mode_ << ' ' << counter_draw_ << ' ' << counter_exhausted_ << '\n';
  const auto& key = counter_key_;
  payload << key.schema_version << ' ' << key.run_seed << ' '
          << feebl::random_domain_number(key.domain) << ' ' << key.crossing << ' '
          << key.diagonal << ' ' << key.wave << ' ' << key.substep << ' '
          << key.encounter << ' ' << key.beam << ' ' << key.parent_a << ' '
          << key.parent_b << ' ' << key.interaction << ' ' << key.emission << ' '
          << key.rejection_attempt << '\n';
  std::ostringstream record;
  record.imbue(std::locale::classic());
  record << "FEEBL_RNG_V2\n" << checkpoint_checksum(payload.str()) << '\n' << payload.str();
  write_checkpoint(path, record.str());
  return 0;
}

int RNDM::rndm_load(const char* path)
{
  // Read a bounded record, then parse into a copy. No failure path may advance
  // a cursor, replace the seed or discard a cached normal variate.
  std::ifstream file(path, std::ios::binary);
  if (!file) throw std::runtime_error("Cannot open requested RNG checkpoint: " + std::string(path));
  char buffer[4097];
  file.read(buffer, sizeof(buffer));
  const auto count = file.gcount();
  if (file.bad() || count == sizeof(buffer))
    throw std::runtime_error("RNG checkpoint is unreadable or oversized");
  const std::string bytes(buffer, static_cast<std::size_t>(count));
  std::istringstream record(bytes);
  record.imbue(std::locale::classic());
  std::string magic, checksum_line;
  std::getline(record, magic);
  std::getline(record, checksum_line);
  if (magic != "FEEBL_RNG_V2")
    throw std::runtime_error("Unsupported RNG checkpoint; legacy raw files lack Gaussian state. Rerun from rndm_seed.");
  const auto payload_start = bytes.find('\n', bytes.find('\n') + 1);
  if (payload_start == std::string::npos)
    throw std::runtime_error("Truncated RNG checkpoint header");
  const std::string payload = bytes.substr(payload_start + 1);
  // All serialized numbers are unsigned decimal integers, including float bits.
  const auto decimal_record = [](const std::string& text) {
    return !text.empty() && text.find_first_not_of("0123456789 \n\t") == std::string::npos;
  };
  std::uint64_t expected_checksum = 0;
  std::istringstream checksum_stream(checksum_line);
  checksum_stream.imbue(std::locale::classic());
  if (!decimal_record(checksum_line) || !(checksum_stream >> expected_checksum) ||
      !(checksum_stream >> std::ws).eof() || !decimal_record(payload) ||
      checkpoint_checksum(payload) != expected_checksum)
    throw std::runtime_error("RNG checkpoint checksum or format mismatch");

  RNDM candidate = *this;
  unsigned width, cached, mode, exhausted;
  std::uint32_t factor_bits, v1_bits, v2_bits, domain;
  std::istringstream input(payload);
  input.imbue(std::locale::classic());
  auto& key = candidate.counter_key_;
  if (!(input >> width >> candidate.rndm7_a_coeff_ >> candidate.rndm7_c_coeff_
      >> factor_bits >> candidate.rndm7_store.i >> candidate.counterRndm7_
      >> cached >> v1_bits >> v2_bits >> mode >> candidate.counter_draw_ >> exhausted
      >> key.schema_version >> key.run_seed >> domain >> key.crossing
      >> key.diagonal >> key.wave >> key.substep >> key.encounter >> key.beam
      >> key.parent_a >> key.parent_b >> key.interaction >> key.emission
      >> key.rejection_attempt) || !(input >> std::ws).eof())
    throw std::runtime_error("Truncated or malformed RNG checkpoint payload");
  key.domain = static_cast<feebl::RandomDomain>(domain);
  candidate.iset_ = static_cast<int>(cached);
  candidate.v1_ = float_from_bits(v1_bits);
  candidate.v2_ = float_from_bits(v2_bits);
  candidate.counter_mode_ = mode != 0;
  candidate.counter_exhausted_ = exhausted != 0;
  if (width != sizeof(unsigned long) * CHAR_BIT ||
      candidate.rndm7_a_coeff_ != rndm7_a_coeff_ || candidate.rndm7_c_coeff_ != rndm7_c_coeff_ ||
      factor_bits != float_bits(rndm7_modulo_dividing_factor_) || cached > 1 || mode > 1 || exhausted > 1 ||
      !std::isfinite(candidate.v1_) || !std::isfinite(candidate.v2_) ||
      !feebl::physical_random_key_is_valid_v2(key) ||
      (exhausted && candidate.counter_draw_ != std::numeric_limits<std::uint64_t>::max()))
    throw std::runtime_error("RNG checkpoint state or generator ABI is incompatible");
  candidate.loaded_checkpoint_checksum_ = std::to_string(expected_checksum);
  *this = candidate;
  std::cerr << "RNDM: restored FEEBL_RNG_V2 checkpoint " << path
            << "; payload_fnv1a64=" << loaded_checkpoint_checksum_
            << "; checkpoint state supersedes rndm_seed for this stream.\n";
  return 1;
}
