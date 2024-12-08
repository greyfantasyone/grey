#ifndef EXPRESSION_EVALUATOR_H
#define EXPRESSION_EVALUATOR_H

#include <string>
#include <stdexcept>

class ExpressionEvaluator {
public:
    double evaluate(const std::string& expression);

private:
    double parseExpression(const std::string& expression, size_t& pos);
    double parseTerm(const std::string& expression, size_t& pos);
    double parseFactor(const std::string& expression, size_t& pos);
    double parseNumber(const std::string& expression, size_t& pos);
    void skipWhitespace(const std::string& expression, size_t& pos);
};

#endif