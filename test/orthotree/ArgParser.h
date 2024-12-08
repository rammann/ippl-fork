#include "Ippl.h"

#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace ippl {

    /**
     * @brief simple argument parser to make our runs simpler to repeat etc etc
     */
    class ArgParser {
        static const std::string prefix;  // prefix for the arguments

    public:
        template <typename T>
        static void add_argument(const std::string& name, const T& default_value,
                                 const std::string& description) {
            arguments()[name] = {to_string(default_value), description};
        }

        static void parse(int argc, char* argv[]) {
            args().assign(argv, argv + argc);
            parsed_args().clear();
            for (const auto& arg : args()) {
                // if help is found we print it and abort
                if (arg == prefix + "help") {
                    if (Comm->rank() == 0) {
                        print_help();
                    }
                    exit(0);
                }

                auto pos = arg.find('=');
                if (pos != std::string::npos && arg.substr(0, prefix.size()) == prefix) {
                    std::string key   = arg.substr(prefix.size(), pos - prefix.size());
                    std::string value = arg.substr(pos + 1);

                    if (arguments().find(key) == arguments().end()) {
                        throw std::runtime_error("Unknown argument: " + key);
                    }

                    parsed_args()[key] = value;
                }
            }
        }

        /**
         * @brief Get the value of an argument. If none was given we use the default.
         * @tparam the type of the value we want to get
         */
        template <typename T>
        static T get(const std::string& name) {
            auto it = parsed_args().find(name);
            if (it != parsed_args().end()) {
                return convert<T>(it->second);
            }

            auto def_it = arguments().find(name);
            if (def_it == arguments().end()) {
                throw std::runtime_error("Argument not found: " + name);
            }

            return convert<T>(def_it->second.default_value);
        }

        static void print_help() {
            std::cout << "Usage: program [options]\nOptions:\n";
            for (const auto& [name, info] : arguments()) {
                std::cout << "  " << prefix << std::setw(20) << std::left << name
                          << info.description << " (default: " << info.default_value << ")\n";
            }
        }

    private:
        struct ArgumentInfo {
            std::string default_value;
            std::string description;
        };

        static std::vector<std::string>& args() {
            static std::vector<std::string> instance;
            return instance;
        }

        static std::unordered_map<std::string, std::string>& parsed_args() {
            static std::unordered_map<std::string, std::string> instance;
            return instance;
        }

        static std::unordered_map<std::string, ArgumentInfo>& arguments() {
            static std::unordered_map<std::string, ArgumentInfo> instance;
            return instance;
        }

        template <typename T>
        static T convert(const std::string& value);

        template <typename T>
        static std::string to_string(const T& value);
    };

    inline const std::string ArgParser::prefix = "-";

    // ================
    // SPECIALISATIONS
    // ================

    template <>
    inline int ArgParser::convert<int>(const std::string& value) {
        return std::stoi(value);
    }

    template <>
    inline size_t ArgParser::convert<size_t>(const std::string& value) {
        return static_cast<size_t>(std::stoul(value));
    }

    template <>
    inline double ArgParser::convert<double>(const std::string& value) {
        return std::stod(value);
    }

    template <>
    inline std::string ArgParser::convert<std::string>(const std::string& value) {
        return value;
    }

    template <>
    inline std::string ArgParser::to_string<int>(const int& value) {
        return std::to_string(value);
    }

    template <>
    inline std::string ArgParser::to_string<size_t>(const size_t& value) {
        return std::to_string(value);
    }

    template <>
    inline std::string ArgParser::to_string<double>(const double& value) {
        std::ostringstream oss;
        oss << value;
        return oss.str();
    }

    template <>
    inline std::string ArgParser::to_string<std::string>(const std::string& value) {
        return value;
    }

}  // namespace ippl
