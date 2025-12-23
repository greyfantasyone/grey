# Project Repository Summary

This repository contains a collection of C++ data structure and algorithm implementations, along with related exercises and reports.

## Overview

- **Author:** 郭子昊 (Guo Zihao)
- **Primary Language:** C++
- **Build System:** Makefile (present in most subdirectories)

## Directory Contents

### Data Structures
*   **BST (`/BST`)**: Implementation of a Binary Search Tree. Includes a `BinarySearchTree.h` header and report files.
*   **BSTsec (`/BSTsec`)**: An enhanced/secondary implementation of a Binary Search Tree (`BST.h`), featuring exception classes (e.g., `UnderflowException`, `IteratorMismatchException`) and likely iterator support.
*   **LinkedList (`/LinkedList`)**: Implementation of a **Singly Linked List**.
*   **List (`/List`)**: Implementation of a **Doubly Linked List** (nodes with `prev` and `next` pointers), following textbook structures.

### Algorithms & Applications
*   **HeapSort (`/HeapSort`)**: Implementation of the Heap Sort algorithm using standard library heap operations (`make_heap`, `pop_heap`).
*   **expression_evaluator (`/expression_evaluator`)**: A program to parse and evaluate mathematical expressions.

### Exercises & Miscellaneous
*   **chicken (`/chicken`)**: A C++ class design exercise focusing on memory management, copy constructors, assignment operators, and destructors (Rule of Three/Five).
*   **hello (`/hello`)**: A standard "Hello, World!" C++ program.
*   **BIS (`/BIS`)**: Contains a LaTeX report (seemingly related to "LIS" - Longest Increasing Subsequence, based on headers) and associated images.

## Building and Running

Most directories contain a `Makefile`. To build a project, navigate to the directory and run:

```bash
cd <DirectoryName>
make
./<ExecutableName>
```