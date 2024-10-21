#include <iostream>
#include "List.h"

int main() {

    List<int> list;

    list.push_front(10);
    list.push_back(20);
    list.push_front(5);
    list.push_back(25);

    std::cout << "List elements after push_front and push_back: ";
    for (List<int>::iterator iter = list.begin(); iter != list.end(); ++iter) {
        std::cout << *iter << " ";
    }
    std::cout << std::endl; 

    std::cout << "Size: " << list.size() << std::endl; 
    std::cout << "Is empty: " << (list.empty() ? "Yes" : "No") << std::endl; 

    std::cout << "Front: " << list.front() << std::endl; 
    std::cout << "Back: " << list.back() << std::endl; 

    
    std::cout << "List elements: ";
    for (auto it = list.begin(); it != list.end(); ++it) {
        std::cout << *it << " ";
    }
    std::cout << std::endl;

    list.pop_front();
    list.pop_back();
    std::cout << "After pop_front and pop_back, size: " << list.size() << std::endl; 

    std::cout << "List elements after pop_front and pop_back: ";
    for (List<int>::iterator iter = list.begin(); iter != list.end(); ++iter) {
        std::cout << *iter << " ";
    }
    std::cout << std::endl; 

    List<int>::iterator it = list.begin();
    list.insert(it, 15);
    std::cout << "After insert, size: " << list.size() << std::endl; 

    std::cout << "List elements after insert: ";
    for (List<int>::iterator iter = list.begin(); iter != list.end(); ++iter) {
        std::cout << *iter << " ";
    }
    std::cout << std::endl; 

    it = list.begin();
    list.erase(it);
    std::cout << "After erase, size: " << list.size() << std::endl; 

    std::cout << "List elements after erase: ";
    for (List<int>::iterator iter = list.begin(); iter != list.end(); ++iter) {
        std::cout << *iter << " ";
    }
    std::cout << std::endl; 

    list.clear();
    std::cout << "After clear, size: " << list.size() << std::endl; 

    List<int> list2 = {1, 2, 3, 4, 5};
    std::cout << "List2 size: " << list2.size() << std::endl; 

    std::cout << "List2 elements: ";
    for (List<int>::iterator iter = list2.begin(); iter != list2.end(); ++iter) {
        std::cout << *iter << " ";
    }
    std::cout << std::endl; 

    List<int> list3 = list2;
    std::cout << "List3 size (copy of list2): " << list3.size() << std::endl; 

    std::cout << "List3 elements (copy of list2): ";
    for (List<int>::iterator iter = list3.begin(); iter != list3.end(); ++iter) {
        std::cout << *iter << " ";
    }
    std::cout << std::endl; 

    List<int> list4;
    list4 = list3;
    std::cout << "List4 size (assigned from list3): " << list4.size() << std::endl; 

    std::cout << "List4 elements (assigned from list3): ";
    for (List<int>::iterator iter = list4.begin(); iter != list4.end(); ++iter) {
        std::cout << *iter << " ";
    }
    std::cout << std::endl; 

    List<int> list5 = std::move(list4);
    std::cout << "List5 size (moved from list4): " << list5.size() << std::endl; 
    std::cout << "List4 size after move: " << list4.size() << std::endl; 

    std::cout << "List5 elements (moved from list4): ";
    for (List<int>::iterator iter = list5.begin(); iter != list5.end(); ++iter) {
        std::cout << *iter << " ";
    }
    std::cout << std::endl; 

    List<int>::iterator iter = list2.begin();
    std::cout << "Testing increment operators:" << std::endl;
    std::cout << "Initial value: " << *iter << std::endl; 
    std::cout << "Value after ++iter: " << *++iter << std::endl; 
    std::cout << "Value after iter++: " << *iter++ << std::endl; 
    std::cout << "Value after iter++: " << *iter << std::endl; 

    std::cout << "Testing decrement operators:" << std::endl;
    std::cout << "Initial value: " << *iter << std::endl; 
    std::cout << "Value after --iter: " << *--iter << std::endl; 
    std::cout << "Value after iter--: " << *iter-- << std::endl; 
    std::cout << "Value after iter--: " << *iter << std::endl; 

    List<int> list6{5, 10, 15, 20, 25}; 
    List<int>::iterator itr1 = list6.begin(); 
    std::cout << *itr1 << std::endl; 
    *itr1 = -5; 
    std::cout << *(itr1++) << std::endl; 
    std::cout << *itr1 << std::endl; 
    List<int>::iterator itr2 = list6.end(); 
    std::cout << *(itr2--) << std::endl; 
    std::cout << *itr2 << std::endl; 
    std::cout << *(++itr1) << "" << *(--itr2) << std::endl; 
    std::cout << *itr1 << "" << *itr2 << std::endl; 
    std::cout << (itr1 == itr1) << std::endl; 
    std::cout << (itr1 == itr2) << std::endl; 
    std::cout << (itr1 != itr1) << std::endl; 
    std::cout << (itr1 != itr2) << std::endl; 
    std::cout << std::endl; 

    return 0;
}