#include "expression_evaluator.h"
#include <cctype>
#include <cmath>
#include <sstream>
#include <iostream>

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
        if (pos < expression.length() && (expression[pos] == '+'  || expression[pos] == '*' || expression[pos] == '/')) {
            throw std::runtime_error("ILLEGAL");
        }
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
        if (pos < expression.length() && (expression[pos] == '+' || expression[pos] == '-')) {
            throw std::runtime_error("ILLEGAL");
        }
        double factor = parseFactor(expression, pos);
        if (op == '*') {
            result *= factor;
        } else {
            if (factor == 0) throw std::runtime_error("ILLEGAL");
            result /= factor;
        }
    }
    return result;
}

double ExpressionEvaluator::parseFactor(const std::string& expression, size_t& pos) {
    skipWhitespace(expression, pos);
    if (pos >= expression.length()) throw std::runtime_error("ILLEGAL");

    char c = expression[pos];
    if (c == '(') {
        ++pos;
        double result = parseExpression(expression, pos);
        skipWhitespace(expression, pos);
        if (pos >= expression.length() || expression[pos] != ')') {
            throw std::runtime_error("ILLEGAL");
        }
        ++pos;
        return result;
    } else if (c == '+' || c == '-') {
        ++pos;
        double factor = parseFactor(expression, pos);
        return (c == '-') ? -factor : factor;
    } else {
        return parseNumber(expression, pos);
    }
}

double ExpressionEvaluator::parseNumber(const std::string& expression, size_t& pos) {
    skipWhitespace(expression, pos);
    size_t start = pos;
    bool hasDecimal = false;
    bool hasExponent = false;

    if (expression[pos] == '+' || expression[pos] == '-') {
        ++pos;
    }

    while (pos < expression.length() && (std::isdigit(expression[pos]) || expression[pos] == '.')) {
        if (expression[pos] == '.') {
            if (hasDecimal) throw std::runtime_error("ILLEGAL");
            hasDecimal = true;
        }
        ++pos;
    }

    if (pos < expression.length() && (expression[pos] == 'e' || expression[pos] == 'E')) {
        hasExponent = true;
        ++pos;
        if (pos < expression.length() && (expression[pos] == '+' || expression[pos] == '-')) {
            ++pos;
        }
        if (pos >= expression.length() || !std::isdigit(expression[pos])) {
            throw std::runtime_error("ILLEGAL");
        }
        while (pos < expression.length() && std::isdigit(expression[pos])) {
            ++pos;
        }
    }

    if (start == pos) {
        throw std::runtime_error("ILLEGAL");
    }

    return std::stod(expression.substr(start, pos - start));
}

void ExpressionEvaluator::skipWhitespace(const std::string& expression, size_t& pos) {
    while (pos < expression.length() && std::isspace(expression[pos])) {
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
            std::cout << e.what() << std::endl;
        }
    }

    return 0;
}