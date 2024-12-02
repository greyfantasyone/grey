#ifndef HEAPSORT_H
#define HEAPSORT_H

#include <vector>
#include <algorithm>
#include <random> 

template <typename T>
void heapSort(std::vector<T>& vec) {
    // 将向量转换为堆
    std::make_heap(vec.begin(), vec.end());

    // 逐个将最大元素移到末尾，并调整堆
    for (auto it = vec.end(); it != vec.begin(); --it) {
        // 将堆顶元素移到当前范围的末尾
        std::pop_heap(vec.begin(), it);
        // 此时 it-1 是当前范围的最大元素
    }
}

// 检查排序是否正确
template <typename T>
bool check(const std::vector<T>& vec) {
    for (size_t i = 1; i < vec.size(); ++i) {
        if (vec[i - 1] > vec[i]) {
            return false;
        }
    }
    return true;
}
std::vector<int> generateRandomSequence(size_t length) {
    std::vector<int> vec(length);
    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<int> dist(1, 1000000);
    for (size_t i = 0; i < length; ++i) {
        vec[i] = dist(rng);
    }
    return vec;
}
std::vector<int> generateSortedSequence(size_t length) {
    std::vector<int> vec(length);
    for (size_t i = 0; i < length; ++i) {
        vec[i] = i;
    }
    return vec;
}
std::vector<int> generateReverseSequence(size_t length) {
    std::vector<int> vec(length);
    for (size_t i = 0; i < length; ++i) {
        vec[i] = length - i;
    }
    return vec;
}
std::vector<int> generatePartiallyRepeatedSequence(size_t length) {
    std::vector<int> vec(length);
    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<int> dist(1, 100);
    for (size_t i = 0; i < length; ++i) {
        vec[i] = dist(rng);
    }
    return vec;
}

#endif