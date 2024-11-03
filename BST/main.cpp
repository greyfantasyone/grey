#include <iostream>
#include "BinarySearchTree.h"

int main() {
    BinarySearchTree<int> bst;
    bst.insert(20);
    bst.insert(10);
    bst.insert(30);
    bst.insert(5);
    bst.insert(15);
    bst.insert(25);
    bst.insert(35);
    std::cout << "Initial tree:" << std::endl;
    bst.printTree();
    bst.remove(5);
    std::cout << "After removing 5:" << std::endl;
    bst.printTree();
    bst.remove(30);
    std::cout << "After removing 30:" << std::endl;
    bst.printTree();
    bst.remove(10);
    std::cout << "After removing 10:" << std::endl;
    bst.printTree();
    bst.remove(20);
    std::cout << "After removing 20:" << std::endl;
    bst.printTree();
    return 0;
}