#pragma once

#include <sygraph/frontier/frontier_settings.hpp>
#include <sygraph/frontier/impls/bitmap_frontier.hpp>
#include <sygraph/frontier/impls/bitvec_frontier.hpp>
#include <sygraph/frontier/impls/vector_frontier.hpp>

namespace sygraph {
inline namespace v0 {
namespace frontier {
namespace detail {
template<typename type_t, FrontierType _type>
class frontier_impl_t;

template<typename type_t>
class frontier_impl_t<type_t, FrontierType::bitmap> : public frontier_bitmap_t<type_t> {
  using frontier_bitmap_t<type_t>::frontier_bitmap_t;
};

template<typename type_t>
class frontier_impl_t<type_t, FrontierType::vector> : public frontier_vector_t<type_t> {
  using frontier_vector_t<type_t>::frontier_vector_t;
};

template<typename type_t>
class frontier_impl_t<type_t, FrontierType::bitvec> : public frontier_bitvec_t<type_t> {
  using frontier_bitvec_t<type_t>::frontier_bitvec_t;
};
} // namespace detail

template<typename T, FrontierView view = FrontierView::vertex, FrontierType type = FrontierType::bitmap>
class Frontier : public detail::frontier_impl_t<T, type> {
public:
  using detail::frontier_impl_t<T, type>::frontier_impl_t;
  using type_t = T;
};

template<FrontierView view, FrontierType type, typename GraphType>
auto makeFrontier(sycl::queue& q, const GraphType& graph) {
  size_t frontier_size = 0;
  if constexpr (view == FrontierView::vertex) {
    frontier_size = graph.getVertexCount() * 2;
    return Frontier<typename GraphType::vertex_t, view, type>(q, frontier_size);
  } else {
    frontier_size = graph.getEdgeCount();
    return Frontier<typename GraphType::edge_t, view, type>(q, frontier_size);
  }
}

template<typename type_t, FrontierView view, FrontierType type>
void swap(Frontier<type_t, view, type>& a, Frontier<type_t, view, type>& b) {
  if constexpr (type == FrontierType::bitmap) {
    detail::frontier_bitmap_t<type_t>::swap(a, b);
  } else if constexpr (type == FrontierType::vector) {
    detail::frontier_vector_t<type_t>::swap(a, b);
  } else if constexpr (type == FrontierType::bitvec) {
    detail::frontier_bitvec_t<type_t>::swap(a, b);
  }
}

} // namespace frontier
} // namespace v0
} // namespace sygraph