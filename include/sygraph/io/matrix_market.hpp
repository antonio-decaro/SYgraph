/*
 * Copyright (c) 2025 University of Salerno
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <sstream>

#include <sycl/sycl.hpp>

#include <sygraph/formats/coo.hpp>
#include <sygraph/formats/csr.hpp>

namespace sygraph {
namespace io {
namespace detail {
namespace mm {

enum class mm_object {
  matrix,
  vector,
};

enum class mm_format {
  coordinate,
  array,
};

enum class mm_field {
  real,
  integer,
  complex,
  pattern,
};

enum class mm_symmetry {
  general,
  symmetric,
  skew_symmetric,
  hermitian,
};

struct Banner {
  std::string banner;
  mm_object object;
  mm_format format;
  mm_field field;
  mm_symmetry symmetry;

  void read(const std::string& line) {
    std::istringstream iss(line);
    std::string banner;
    mm_object object;
    mm_format format;
    mm_field field;
    mm_symmetry symmetry;

    iss >> banner;

    if (banner == "%%MatrixMarket") {
      std::string object_str;
      std::string format_str;
      std::string field_str;
      std::string symmetry_str;

      iss >> object_str >> format_str >> field_str >> symmetry_str;
      // Keywords are case-insensitive in the Matrix Market format.
      for (auto* str : {&object_str, &format_str, &field_str, &symmetry_str}) {
        std::transform(str->begin(), str->end(), str->begin(), [](unsigned char c) { return std::tolower(c); });
      }

      if (object_str == "matrix") {
        object = mm_object::matrix;
      } else if (object_str == "vector") {
        object = mm_object::vector;
      } else {
        throw std::runtime_error("Invalid object type in MatrixMarket banner");
      }

      if (format_str == "coordinate") {
        format = mm_format::coordinate;
      } else if (format_str == "array") {
        format = mm_format::array;
      } else {
        throw std::runtime_error("Invalid format type in MatrixMarket banner");
      }

      if (field_str == "real") {
        field = mm_field::real;
      } else if (field_str == "integer") {
        field = mm_field::integer;
      } else if (field_str == "complex") {
        field = mm_field::complex;
      } else if (field_str == "pattern") {
        field = mm_field::pattern;
      } else {
        throw std::runtime_error("Invalid field type in MatrixMarket banner");
      }

      if (symmetry_str == "general") {
        symmetry = mm_symmetry::general;
      } else if (symmetry_str == "symmetric") {
        symmetry = mm_symmetry::symmetric;
      } else if (symmetry_str == "skew-symmetric") {
        symmetry = mm_symmetry::skew_symmetric;
      } else if (symmetry_str == "hermitian") {
        symmetry = mm_symmetry::hermitian;
      } else {
        throw std::runtime_error("Invalid symmetry type in MatrixMarket banner");
      }
    } else {
      throw std::runtime_error("Invalid MatrixMarket banner");
    }

    this->banner = banner;
    this->object = object;
    this->format = format;
    this->field = field;
    this->symmetry = symmetry;
  }

  Banner() {
    banner = "";
    object = mm_object::matrix;
    format = mm_format::coordinate;
    field = mm_field::real;
    symmetry = mm_symmetry::general;
  };

  bool isMatrix() const { return object == mm_object::matrix; }
  bool isVector() const { return object == mm_object::vector; }

  bool isCoordinate() const { return format == mm_format::coordinate; }
  bool isArray() const { return format == mm_format::array; }

  bool isReal() const { return field == mm_field::real; }
  bool isInteger() const { return field == mm_field::integer; }
  bool isComplex() const { return field == mm_field::complex; }
  bool isPattern() const { return field == mm_field::pattern; }

  bool isGeneral() const { return symmetry == mm_symmetry::general; }
  bool isSymmetric() const { return symmetry == mm_symmetry::symmetric; }

  template<typename ValueT, typename IndexT, typename OffsetT>
  void validate() {
    if (this->object != mm_object::matrix) { throw std::runtime_error("Unsupported MatrixMarket object: only 'matrix' is supported"); }

    if (this->format != mm_format::coordinate) { throw std::runtime_error("Unsupported MatrixMarket format: only 'coordinate' is supported"); }

    if (this->field == mm_field::complex) { throw std::runtime_error("Unsupported MatrixMarket field: 'complex'"); }

    if (this->symmetry == mm_symmetry::skew_symmetric || this->symmetry == mm_symmetry::hermitian) {
      throw std::runtime_error("Unsupported MatrixMarket symmetry: only 'general' and 'symmetric' are supported");
    }

    if (this->field == mm_field::real && !std::is_floating_point<ValueT>::value) { throw std::runtime_error("Invalid MatrixMarket field type"); }
    if (this->field == mm_field::integer && !std::is_integral<ValueT>::value) { throw std::runtime_error("Invalid MatrixMarket field type"); }
  }
};

} // namespace mm
} // namespace detail
} // namespace io
} // namespace sygraph
