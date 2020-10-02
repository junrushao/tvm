/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
#ifndef SRC_META_SCHEDULE_UTILS_H_
#define SRC_META_SCHEDULE_UTILS_H_

#include <tvm/arith/analyzer.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/expr.h>

#include <set>
#include <unordered_set>
#include <vector>

#include "./schedule.h"

namespace tvm {
namespace meta_schedule {

/*!
 * \brief Compute mean of a FloatImm array.
 * Taken from Ansor
 * \param float_array The array of floating point numbers to be averaged
 * \return The mean of the given array
 */
inline double FloatArrayMean(const Array<PrimExpr>& float_array) {
  double sum = 0;
  if (float_array.empty()) {
    return 0.0;
  }
  for (const auto& x : float_array) {
    const auto* float_imm = x.as<tir::FloatImmNode>();
    CHECK(float_imm != nullptr);
    sum += float_imm->value;
  }
  return sum / float_array.size();
}

/*!
 * \brief An empty output stream
 * Taken from Ansor
 */
class NullStream : public std::ostream {
 public:
  NullStream() : std::ostream(nullptr) {}
  NullStream(const NullStream&) : std::ostream(nullptr) {}
  static NullStream& Global();
};

template <class T>
NullStream& operator<<(NullStream& os, const T& value) {
  return os;
}

/*!
 * \brief Get std cout with verbose control
 * Taken from Ansor
 */
inline std::ostream& StdCout(int verbose, int setting = 1) {
  return verbose >= setting ? std::cout : NullStream::Global();
}

/*!
 * \brief Find all positions that the specific char occurs in the string
 * \param str The string to be examined
 * \param c The specific char
 * \return A list of integers indicating the occurrence position
 */
inline std::vector<int> FindCharPos(const String& str, char c) {
  std::vector<int> result;
  const char* data = str.data();
  int n = str.length();
  for (int i = 0; i < n; ++i) {
    if (data[i] == c) {
      result.push_back(i);
    }
  }
  return result;
}

/*!
 * \brief Concatenate the nested vector into a flattened vector
 * \tparam T The element type of the nested vector
 * \param source The nested vector
 * \return The flattened vector
 */
template <class T>
inline std::vector<T> ConcatArray(const std::vector<std::vector<T> >& source) {
  std::vector<T> result;
  for (const std::vector<T>& item : source) {
    result.insert(result.end(), item.begin(), item.end());
  }
  return result;
}

/*!
 * \brief Compare two domains and check if they are equal
 * \param lhs One domain
 * \param rhs The other domain
 * \return A boolean indicating if the two domains are proved to be equal
 */
inline bool DomainEqual(const Array<Range>& lhs, const Array<Range>& rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  arith::Analyzer analyzer;
  int n = lhs.size();
  for (int i = 0; i < n; ++i) {
    const Range& l = lhs[i];
    const Range& r = rhs[i];
    if (!analyzer.CanProve(l->min == r->min)) {
      return false;
    }
    if (!analyzer.CanProve(l->extent == r->extent)) {
      return false;
    }
  }
  return true;
}

/*!
 * \brief Get the string representation of a schedule
 * \param sch The schedule to be stringified
 * \return The string representation of a schedule
 */
inline String Repr(const Schedule& sch) {
  const auto* f = runtime::Registry::Get("hybrid.AsHybrid");
  CHECK(f) << "IndexError: global function \"hybrid.AsHybrid\" not found";
  String s = (*f)(sch->sch->func, false);
  return s;
}

/*!
 * \brief A heap with a size up-limit. If out-growth happens, it evicted the worst items
 * \tparam ItemType Type of the items in the heap. ItemType::KeyType is the type of its key, which
 * the container can access using ItemType::key
 */
template <class ItemType>
class SizedHeap {
  using KeyType = typename ItemType::KeyType;

 public:
  /*!
   * \brief Constructor
   * \param size_limit The up-limit of the heap
   */
  explicit SizedHeap(int size_limit) : size_limit_(size_limit) { heap_.reserve(size_limit_); }

  /*!
   * \brief Push the specific item to the heap if its key did not appears in the heap
   * \param item The item to be pushed
   */
  void Push(const ItemType& item) {
    if (in_heap_.count(item.key)) {
      return;
    }
    int size = heap_.size();
    if (size < size_limit_) {
      // Heap is not full, just push
      heap_.emplace_back(item);
      std::push_heap(heap_.begin(), heap_.end());
      in_heap_.insert(item.key);
    } else if (item < heap_.front()) {
      // if the item is better than the worst one in the heap, we can safely kick it out
      in_heap_.erase(heap_.front().key);
      in_heap_.insert(item.key);
      std::pop_heap(heap_.begin(), heap_.end());
      heap_.back() = item;
      std::push_heap(heap_.begin(), heap_.end());
    }
    // Otherwise, the item is worse than any other element in the heap
  }

  /*!
   * \brief Add the specific key to the heap to avoid it being pushed
   * \param key The key to be inserted
   */
  void AddKey(const KeyType& key) { in_heap_.insert(key); }

 private:
  /*! \brief Up-limit of the heap size */
  int size_limit_;
  /*! \brief The heap, the worse the topper */
  std::vector<ItemType> heap_;
  /*! \brief Collection of keys in th heap */
  std::unordered_set<KeyType> in_heap_;
};

/*!
 * \brief A table containing keys for de-duplication and sorted values
 * \tparam KeyType Type of the keys
 * \tparam ValueType Type of the values
 */
template <class KeyType, class ValueType>
class SortedTable {
 public:
  /*!
   * \brief Check if a key is in the table
   * \param key The key to be checked
   * \return A boolean indicating if it is in the table
   */
  bool Has(const KeyType& key) const { return keys_.count(key); }

  /*!
   * \brief Add a key to the table
   * \param key The key to be added
   */
  void Add(const KeyType& key) { keys_.insert(key); }

  /*!
   * \brief Add a value to the table
   * \param value The value to be added
   */
  void Add(const ValueType& value) { values_.push_back(value); }

  /*!
   * \brief Get the top-k values, the smaller the better
   * \param top_k The number of top-k values to be retrieved
   * \return A vector of values whose length is at most `top_k`
   */
  std::vector<ValueType> GetTopK(int top_k) const {
    std::vector<ValueType> result;
    result.reserve(top_k);
    int i = 0;
    for (const ValueType& value : values_) {
      result.push_back(value);
      if (++i >= top_k) {
        break;
      }
    }
    return result;
  }

 private:
  /*! \brief The table to store keys */
  std::unordered_set<KeyType> keys_;
  /*! \brief The table to store values */
  std::multiset<ValueType> values_;
};

}  // namespace meta_schedule
}  // namespace tvm

#endif  // SRC_META_SCHEDULE_UTILS_H_
