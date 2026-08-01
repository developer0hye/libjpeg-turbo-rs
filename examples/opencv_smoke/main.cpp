#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <cstdint>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

namespace {

constexpr double kMinimumPsnrDb = 49.22;
constexpr std::uint64_t kExpectedGraySum = 6299471;

cv::Mat make_fixture() {
  constexpr int width = 257;
  constexpr int height = 193;
  cv::Mat image(height, width, CV_8UC3);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const int tile_bias = ((x / 32 + y / 24) & 1) == 0 ? 0 : 24;
      image.at<cv::Vec3b>(y, x) = cv::Vec3b{
          static_cast<std::uint8_t>((x * 231) / (width - 1) + tile_bias),
          static_cast<std::uint8_t>((y * 231) / (height - 1) + tile_bias),
          static_cast<std::uint8_t>(
              ((x + y) * 231) / (width + height - 2) + tile_bias)};
    }
  }
  return image;
}

bool has_marker(const std::vector<std::uint8_t>& jpeg, std::uint8_t marker) {
  for (std::size_t i = 0; i + 1 < jpeg.size(); ++i) {
    if (jpeg[i] == 0xff && jpeg[i + 1] == marker) {
      return true;
    }
  }
  return false;
}

bool has_dri_interval(const std::vector<std::uint8_t>& jpeg,
                      std::uint16_t interval) {
  for (std::size_t i = 0; i + 5 < jpeg.size(); ++i) {
    if (jpeg[i] == 0xff && jpeg[i + 1] == 0xdd && jpeg[i + 2] == 0x00 &&
        jpeg[i + 3] == 0x04 && jpeg[i + 4] == (interval >> 8) &&
        jpeg[i + 5] == (interval & 0xff)) {
      return true;
    }
  }
  return false;
}

bool has_restart_marker(const std::vector<std::uint8_t>& jpeg) {
  for (std::uint8_t marker = 0xd0; marker <= 0xd7; ++marker) {
    if (has_marker(jpeg, marker)) {
      return true;
    }
  }
  return false;
}

bool write_mat(const std::string& path, const cv::Mat& matrix) {
  if (!matrix.isContinuous()) {
    return false;
  }
  std::ofstream output(path, std::ios::binary);
  const auto bytes =
      static_cast<std::streamsize>(matrix.total() * matrix.elemSize());
  output.write(reinterpret_cast<const char*>(matrix.data), bytes);
  return output.good();
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2 && argc != 3) {
    std::cerr << "usage: opencv_smoke <output.jpg> [cross-decode.jpg]\n";
    return 2;
  }

  try {
    const cv::Mat original = make_fixture();
    const std::vector<int> encode_params{
        cv::IMWRITE_JPEG_QUALITY, 90, cv::IMWRITE_JPEG_PROGRESSIVE, 1,
        cv::IMWRITE_JPEG_OPTIMIZE, 1, cv::IMWRITE_JPEG_RST_INTERVAL, 4};
    if (!cv::imwrite(argv[1], original, encode_params)) {
      std::cerr << "cv::imwrite returned false\n";
      return 3;
    }
    std::ifstream jpeg_input(argv[1], std::ios::binary);
    const std::vector<std::uint8_t> jpeg{
        std::istreambuf_iterator<char>(jpeg_input),
        std::istreambuf_iterator<char>()};
    if (!has_marker(jpeg, 0xc2) || !has_dri_interval(jpeg, 4) ||
        !has_restart_marker(jpeg)) {
      std::cerr << "encoded JPEG lacks SOF2, DRI=4, or an RST marker\n";
      return 13;
    }

    const cv::Mat color = cv::imread(argv[1], cv::IMREAD_COLOR);
    const cv::Mat gray = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    if (color.empty() || gray.empty()) {
      std::cerr << "cv::imread returned an empty image\n";
      return 4;
    }
    if (color.size() != original.size() || color.type() != original.type() ||
        gray.size() != original.size() || gray.type() != CV_8UC1) {
      std::cerr << "decoded geometry/type mismatch\n";
      return 5;
    }

    const double psnr = cv::PSNR(original, color);
    if (psnr < kMinimumPsnrDb) {
      std::cerr << "PSNR below " << kMinimumPsnrDb << " dB: " << psnr
                << '\n';
      return 6;
    }
    const auto gray_sum = static_cast<std::uint64_t>(cv::sum(gray)[0]);
    if (gray_sum != kExpectedGraySum) {
      std::cerr << "grayscale decode checksum mismatch: expected "
                << kExpectedGraySum << ", got " << gray_sum << '\n';
      return 7;
    }
    const std::string output_path = argv[1];
    if (!write_mat(output_path + ".color.raw", color) ||
        !write_mat(output_path + ".gray.raw", gray)) {
      std::cerr << "could not persist self-decode matrices\n";
      return 14;
    }

    std::cout << "OpenCV=" << CV_VERSION << " width=" << color.cols
              << " height=" << color.rows << " PSNR=" << std::fixed
              << std::setprecision(3) << psnr << " gray_sum=" << gray_sum;

    if (argc == 3) {
      const cv::Mat cross_color = cv::imread(argv[2], cv::IMREAD_COLOR);
      const cv::Mat cross_gray = cv::imread(argv[2], cv::IMREAD_GRAYSCALE);
      if (cross_color.empty() || cross_gray.empty()) {
        std::cerr << "cross cv::imread returned an empty image\n";
        return 9;
      }
      if (cross_color.size() != original.size() ||
          cross_color.type() != original.type() ||
          cross_gray.size() != original.size() ||
          cross_gray.type() != CV_8UC1) {
        std::cerr << "cross-decoded geometry/type mismatch\n";
        return 10;
      }
      const double cross_psnr = cv::PSNR(original, cross_color);
      if (cross_psnr < kMinimumPsnrDb) {
        std::cerr << "cross-decoded PSNR below " << kMinimumPsnrDb
                  << " dB: " << cross_psnr << '\n';
        return 11;
      }
      const auto cross_gray_sum =
          static_cast<std::uint64_t>(cv::sum(cross_gray)[0]);
      if (cross_gray_sum != kExpectedGraySum) {
        std::cerr << "cross-decoded grayscale checksum mismatch: expected "
                  << kExpectedGraySum << ", got " << cross_gray_sum << '\n';
        return 12;
      }
      if (!write_mat(output_path + ".cross-color.raw", cross_color) ||
          !write_mat(output_path + ".cross-gray.raw", cross_gray)) {
        std::cerr << "could not persist cross-decode matrices\n";
        return 15;
      }
      std::cout << " cross_PSNR=" << cross_psnr
                << " cross_gray_sum=" << cross_gray_sum;
    }
    std::cout << '\n';
  } catch (const std::exception& error) {
    std::cerr << "OpenCV smoke exception: " << error.what() << '\n';
    return 8;
  }
  return 0;
}
