#include "expression_evaluator.h"
#include <cctype>
#include <cmath>
#include <sstream>
#include <iostream>
#include <string>

double ExpressionEvaluator::evaluate(const std::string& expression) {
    size_t pos = 0;
    double result = parseExpression(expression, pos);
    skipWhitespace(expression, pos);
    if (pos != expression.length()) {
        throw std::runtime_error("ILLEGAL");
    }
    return result;
}

double ExpressionEvaluator::parseExpression(const std::string& expression, size_t& pos) {
    double result = parseTerm(expression, pos);
    while (true) {
        skipWhitespace(expression, pos);
        if (pos >= expression.length()) break;
        char op = expression[pos];
        if (op != '+' && op != '-') break;
        ++pos;
        skipWhitespace(expression, pos);
        double term = parseTerm(expression, pos);
        if (op == '+') {
            result += term;
        } else {
            result -= term;
        }
    }
    return result;
}

double ExpressionEvaluator::parseTerm(const std::string& expression, size_t& pos) {
    double result = parseFactor(expression, pos);
    while (true) {
        skipWhitespace(expression, pos);
        if (pos >= expression.length()) break;
        char op = expression[pos];
        if (op != '*' && op != '/') break;
        ++pos;
        skipWhitespace(expression, pos);
        double factor = parseFactor(expression, pos);
        if (op == '*') {
            result *= factor;
        } else {
              if (factor == 0) {
                throw std::runtime_error("ILLEGAL");
            }
            result /= factor;
        }
    }
    return result;
}

double ExpressionEvaluator::parseFactor(const std::string& expression, size_t& pos) {
    skipWhitespace(expression, pos);
    bool negative = false;
    if (expression[pos] == '-') {
        negative = true;
        ++pos;
        skipWhitespace(expression, pos);
    }
    double result;
    if (expression[pos] == '(') {
        ++pos;
        result = parseExpression(expression, pos);
        if (expression[pos] != ')') {
            throw std::runtime_error("ILLEGAL");
        }
        ++pos;
    } else {
        size_t startPos = pos;
        while (pos < expression.length() && (isdigit(expression[pos]) || expression[pos] == '.' || expression[pos] == 'e' || expression[pos] == 'E' || (expression[pos] == '-' && (expression[pos-1] == 'e' || expression[pos-1] == 'E')))) {
            ++pos;
        }
        if (startPos == pos) {
            throw std::runtime_error("ILLEGAL");
        }
        try {
            result = std::stod(expression.substr(startPos, pos - startPos));
        } catch (const std::invalid_argument&) {
            throw std::runtime_error("ILLEGAL");
        } catch (const std::out_of_range&) {
            throw std::runtime_error("ILLEGAL");
        }
    }
    if (negative) {
        result = -result;
    }
    return result;
}

void ExpressionEvaluator::skipWhitespace(const std::string& expression, size_t& pos) {
    while (pos < expression.length() && isspace(expression[pos])) {
        ++pos;
    }
}

int main() {
    ExpressionEvaluator evaluator;
    std::string expression;

    while (true) {
        std::cout << "输入 (输入 'exit' 退出): ";
        std::getline(std::cin, expression);

        if (expression == "exit") {
            break;
        }

        try {
            double result = evaluator.evaluate(expression);
            std::cout << "结果: " << result << std::endl;
        } catch (const std::runtime_error& e) {
            std::cout << "错误: " << e.what() << std::endl;
        }
    }

    return 0;
}