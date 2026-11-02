// Deterministic HDF5 input for the canonical-reader contract. Generation uses
// the HDF5 C API so testing a C++ reader does not require Python bindings.
#include <hdf5.h>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace {

void require(bool ok, const char* operation)
{
  if (!ok) throw std::runtime_error(operation);
}

void string_attribute(hid_t file, const char* name, const char* value)
{
  const hid_t space = H5Screate(H5S_SCALAR);
  const hid_t type = H5Tcopy(H5T_C_S1);
  require(space >= 0 && type >= 0, "create string attribute type");
  require(H5Tset_size(type, std::strlen(value) + 1) >= 0, "set string size");
  const hid_t attr = H5Acreate2(file, name, type, space, H5P_DEFAULT, H5P_DEFAULT);
  require(attr >= 0 && H5Awrite(attr, type, value) >= 0, "write string attribute");
  H5Aclose(attr); H5Tclose(type); H5Sclose(space);
}

void double_attribute(hid_t file, const char* name, double value)
{
  const hid_t space = H5Screate(H5S_SCALAR);
  require(space >= 0, "create double attribute space");
  const hid_t attr = H5Acreate2(file, name, H5T_IEEE_F64LE, space,
                              H5P_DEFAULT, H5P_DEFAULT);
  require(attr >= 0 && H5Awrite(attr, H5T_NATIVE_DOUBLE, &value) >= 0,
          "write double attribute");
  H5Aclose(attr); H5Sclose(space);
}

void column(hid_t group, const char* name, const std::vector<double>& values)
{
  const hsize_t count = values.size();
  const hid_t space = H5Screate_simple(1, &count, nullptr);
  require(space >= 0, "create dataset space");
  const hid_t data = H5Dcreate2(group, name, H5T_IEEE_F64LE, space,
                              H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  require(data >= 0 && H5Dwrite(data, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
                              H5P_DEFAULT, values.data()) >= 0, "write column");
  H5Dclose(data); H5Sclose(space);
}
}

int main(int argc, char** argv)
{
  if (argc != 2) { std::fprintf(stderr, "usage: make_fixture OUTPUT.h5\n"); return 2; }
  try
    {
      const hid_t file = H5Fcreate(argv[1], H5F_ACC_EXCL, H5P_DEFAULT, H5P_DEFAULT);
      require(file >= 0, "create new fixture");
      string_attribute(file, "schema", "gp++-arbitrary-bunch/1.0");
      string_attribute(file, "coordinate_system", "beam_local_canonical");
      string_attribute(file, "longitudinal_convention", "zeta>0 toward bunch head");
      string_attribute(file, "species", "electron");
      string_attribute(file, "weight_normalization", "sum = 1");
      double_attribute(file, "reference_energy_GeV", 62.5);
      double_attribute(file, "bunch_population", 2e10);
      const hid_t group = H5Gcreate2(file, "particles", H5P_DEFAULT,
                                    H5P_DEFAULT, H5P_DEFAULT);
      require(group >= 0, "create particles group");
      const int count = 20000;
      const char* names[] = {"x", "px_over_p0", "y", "py_over_p0", "zeta", "delta", "weight"};
      const double scales[] = {1e-10, 1e-9, 2e-12, 2e-10, 1e-8, 1e-7};
      std::vector<double> values(count);
      for (int col = 0; col < 7; ++col)
        {
          for (int i = 0; i < count; ++i)
            values[i] = col == 6 ? 2.0*(i+1)/(count*(count+1.0))
                                 : (i-count/2)*scales[col];
          column(group, names[col], values);
        }
      require(H5Gclose(group) >= 0 && H5Fclose(file) >= 0, "close fixture");
      std::printf("created deterministic 20000-particle fixture: %s\n", argv[1]);
      return 0;
    }
  catch (const std::exception& error)
    {
      std::fprintf(stderr, "fixture generation failed: %s\n", error.what());
      return 1;
    }
}
