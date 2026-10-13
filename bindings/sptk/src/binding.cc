// A Python binding of the pitch extraction of SPTK.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include <cstddef>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "SPTK/analysis/pitch_extraction.h"

namespace nb = nanobind;

namespace {

const std::map<std::string, sptk::PitchExtraction::Algorithms> kAlgorithms{
    {"rapt", sptk::PitchExtraction::kRapt},
    {"swipe", sptk::PitchExtraction::kSwipe},
    {"reaper", sptk::PitchExtraction::kReaper},
    {"dio", sptk::PitchExtraction::kDio},
    {"harvest", sptk::PitchExtraction::kHarvest},
};

using Waveform = nb::ndarray<const double, nb::ndim<1>, nb::c_contig,
                             nb::device::cpu>;
using Output = nb::ndarray<nb::numpy, double, nb::ndim<1>>;

Output ExtractF0(Waveform waveform, double sample_rate, int frame_shift,
                 double lower_f0, double upper_f0, double voicing_threshold,
                 const std::string& algorithm) {
  const auto it(kAlgorithms.find(algorithm));
  if (kAlgorithms.end() == it) {
    throw std::invalid_argument("Unsupported algorithm: " + algorithm);
  }

  const sptk::PitchExtraction pitch_extraction(frame_shift, sample_rate,
                                               lower_f0, upper_f0,
                                               voicing_threshold, it->second);
  if (!pitch_extraction.IsValid()) {
    throw std::invalid_argument(
        "Invalid arguments for the pitch extraction of SPTK.");
  }

  const std::vector<double> input(waveform.data(),
                                  waveform.data() + waveform.shape(0));
  // The result is owned by the unique pointer until it is handed to the capsule,
  // so that it is released even if the extraction throws an exception.
  auto f0(std::make_unique<std::vector<double>>());
  bool success(false);
  {
    nb::gil_scoped_release release;
    success = pitch_extraction.Run(input, f0.get(), nullptr, nullptr);
  }
  if (!success) {
    throw std::runtime_error("Failed to extract pitch.");
  }

  double* data(f0->data());
  const std::size_t size(f0->size());
  nb::capsule owner(f0.get(), [](void* p) noexcept {
    delete static_cast<std::vector<double>*>(p);
  });
  f0.release();
  return Output(data, {size}, owner);
}

}  // namespace

NB_MODULE(_core, m) {
  m.doc() = "A binding of the pitch extraction of SPTK.";
  m.def("extract_f0", &ExtractF0, nb::arg("waveform"), nb::arg("sample_rate"),
        nb::arg("frame_shift"), nb::arg("lower_f0"), nb::arg("upper_f0"),
        nb::arg("voicing_threshold"), nb::arg("algorithm"),
        "Extract F0 in Hz, where unvoiced frames are 0.");
}
