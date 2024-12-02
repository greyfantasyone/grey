#include "HeapSort.h"
#include <iostream>
#include <chrono>

void testHeapSort() {
    const size_t length = 1000000;
    auto randomSeq = generateRandomSequence(length);
    auto start = std::chrono::high_resolution_clock::now();
    heapSort(randomSeq);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "HeapSort on random sequence took: " << duration.count() << " seconds\n";
    std::cout << "Check: " << (check(randomSeq) ? "Passed" : "Failed") << "\n";
    auto randomSeqForSortHeap = generateRandomSequence(length);
    std::make_heap(randomSeqForSortHeap.begin(), randomSeqForSortHeap.end());
    start = std::chrono::high_resolution_clock::now();
    std::sort_heap(randomSeqForSortHeap.begin(), randomSeqForSortHeap.end());
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "std::sort_heap on random sequence took: " << duration.count() << " seconds\n";
    std::cout << "Check: " << (check(randomSeqForSortHeap) ? "Passed" : "Failed") << "\n";
    auto sortedSeq = generateSortedSequence(length);
    start = std::chrono::high_resolution_clock::now();
    heapSort(sortedSeq);
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "HeapSort on sorted sequence took: " << duration.count() << " seconds\n";
    std::cout << "Check: " << (check(sortedSeq) ? "Passed" : "Failed") << "\n";
    auto sortedSeqForSortHeap = generateSortedSequence(length);
    std::make_heap(sortedSeqForSortHeap.begin(), sortedSeqForSortHeap.end());
    start = std::chrono::high_resolution_clock::now();
    std::sort_heap(sortedSeqForSortHeap.begin(), sortedSeqForSortHeap.end());
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "std::sort_heap on sorted sequence took: " << duration.count() << " seconds\n";
    std::cout << "Check: " << (check(sortedSeqForSortHeap) ? "Passed" : "Failed") << "\n";
    auto reverseSeq = generateReverseSequence(length);
    start = std::chrono::high_resolution_clock::now();
    heapSort(reverseSeq);
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "HeapSort on reverse sequence took: " << duration.count() << " seconds\n";
    std::cout << "Check: " << (check(reverseSeq) ? "Passed" : "Failed") << "\n";
    auto reverseSeqForSortHeap = generateReverseSequence(length);
    std::make_heap(reverseSeqForSortHeap.begin(), reverseSeqForSortHeap.end());
    start = std::chrono::high_resolution_clock::now();
    std::sort_heap(reverseSeqForSortHeap.begin(), reverseSeqForSortHeap.end());
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "std::sort_heap on reverse sequence took: " << duration.count() << " seconds\n";
    std::cout << "Check: " << (check(reverseSeqForSortHeap) ? "Passed" : "Failed") << "\n";
    auto partiallyRepeatedSeq = generatePartiallyRepeatedSequence(length);
    start = std::chrono::high_resolution_clock::now();
    heapSort(partiallyRepeatedSeq);
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "HeapSort on partially repeated sequence took: " << duration.count() << " seconds\n";
    std::cout << "Check: " << (check(partiallyRepeatedSeq) ? "Passed" : "Failed") << "\n";
    auto partiallyRepeatedSeqForSortHeap = generatePartiallyRepeatedSequence(length);
    std::make_heap(partiallyRepeatedSeqForSortHeap.begin(), partiallyRepeatedSeqForSortHeap.end());
    start = std::chrono::high_resolution_clock::now();
    std::sort_heap(partiallyRepeatedSeqForSortHeap.begin(), partiallyRepeatedSeqForSortHeap.end());
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "std::sort_heap on partially repeated sequence took: " << duration.count() << " seconds\n";
    std::cout << "Check: " << (check(partiallyRepeatedSeqForSortHeap) ? "Passed" : "Failed") << "\n";
}

int main() {
    testHeapSort();
    return 0;
}