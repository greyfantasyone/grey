#include <iostream>
#include <cstring>

class Chicken
{
    private:
        int age = 24;
        char *name = nullptr;

    public:
        Chicken() = default;

        Chicken(int _age) : age(_age) {}

        Chicken(int _age, const char *_name) : age(_age) 
        {
            int len = std::strlen(_name) + 1;
            name = new char[len];
            std::strcpy(name, _name);
        }

        Chicken(const Chicken &other)
        {
            age = other.age;
            if (other.name)
            {
                int len = std::strlen(other.name) + 1;
                name = new char[len];
                std::strcpy(name, other.name);
            }
            else
            {
                name = nullptr;
            }
        }

        Chicken& operator=(const Chicken &other)
        {
            if (this == &other)
                return *this;

            age = other.age;
            if (name)
                delete[] name;

            if (other.name)
            {
                int len = std::strlen(other.name) + 1;
                name = new char[len];
                std::strcpy(name, other.name);
            }
            else
            {
                name = nullptr;
            }

            return *this;
        }

        ~Chicken()
        {
            delete[] name;
        }

        void setAge(int _age)
        {
            age = _age;
        }

        void setName(const char *_name)
        {
            if (name)
                delete[] name;

            int len = std::strlen(_name) + 1;
            name = new char[len];
            std::strcpy(name, _name);
        }

        const char *getName() const
        {
            return name;
        }

        int getAge() const
        {
            return age;
        }
};

int main()
{
    auto print = [](const Chicken &c){
        std::cout << "Hi, everyone! My name is " << c.getName()
                  << ", I am " << c.getAge() << " years old." << std::endl;
    };

    Chicken c(24, "Kunkun");
    print(c);

    Chicken d;
    d = c;
    print(d);

    Chicken a = c;
    print(a); 

    c.setName("Xukun Cai");
    print(c);
    print(a);
    print(d);

    Chicken b;
    b = d = c;
    print(b);
    print(d);

    return 0;
}