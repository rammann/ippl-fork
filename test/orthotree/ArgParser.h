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
    public:
        static const std::string prefix;  // prefix for the arguments
        static const std::string random_command;
        static const std::string help_command;

        template <typename T>
        static void add_argument(const std::string& name, const T& default_value,
                                 const std::string& description) {
            // min and max are not enabled for this
            arguments()[name] = {to_string(default_value), "", "", description};
        }

        template <typename T>
        static void add_argument(const std::string& name, const T& default_value,
                                 const T& min_value, const T& max_value,
                                 const std::string& description) {
            // min and max are enabled for this
            arguments()[name] = {to_string(default_value), to_string(min_value),
                                 to_string(max_value), description};
        }

        static void parse(int argc, char* argv[]) {
            args().assign(argv, argv + argc);
            parsed_args().clear();
            for (const auto& arg : args()) {
                // if help is found we print it and abort
                if (arg == prefix + ArgParser::help_command) {
                    if (Comm->rank() == 0) {
                        print_help();
                    }
                    exit(1);  // aborting because the visualise.sh script would open an empty
                              // window, which is annoying
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
        static T get(const std::string& key) {
            // did we pass a value for this key?
            auto it = parsed_args().find(key);
            if (it != parsed_args().end()) {
                return convert<T>(it->second);
            }

            // does the key even exist?
            auto def_it = arguments().find(key);
            if (def_it == arguments().end()) {
                throw std::runtime_error("Argument not found: " + key);
            }

            // generate a random value for this key if (enabled and min/max is given)
            if constexpr (std::is_integral_v<T> || std::is_floating_point_v<T>) {
                if (get<bool>(ArgParser::random_command) && !def_it->second.min_value.empty()
                    && !def_it->second.max_value.empty()) {
                    T min_val = convert<T>(def_it->second.min_value);
                    T max_val = convert<T>(def_it->second.max_value);

                    if (min_val > max_val) {
                        throw std::runtime_error("Invalid range for argument: " + key);
                    }

                    T random_val = generate_random<T>(min_val, max_val);
                    // store the random value in case there is a second call
                    parsed_args()[key] = to_string(random_val);
                    return random_val;
                }
            }

            // return the default value
            return convert<T>(def_it->second.default_value);
        }

        static void print_help() {
            std::cout << "Usage: program [options]\nOptions:\n";
            std::cout << "  " << prefix << std::setw(20) << std::left << ArgParser::random_command
                      << "Enables random arguments for arguments that provide min/max values and "
                         "are not defined by the user\n";

            for (const auto& [name, info] : arguments()) {
                std::cout << "  " << prefix << std::setw(20) << std::left << name
                          << info.description << " (default: " << info.default_value << ")\n";
            }
        }

        static std::string get_args() {
            std::ostringstream oss;
            for (const auto& argument : arguments()) {
                const std::string arg_name = argument.first;
                oss << arg_name << '=' << get<std::string>(arg_name) << " ";
            }
            return oss.str();
        }

        template <typename T>
        static T generate_random(T min_val, T max_val) {
            static std::random_device rd;
            static std::mt19937 gen(rd());
            if constexpr (std::is_integral_v<T>) {
                std::uniform_int_distribution<T> dis(min_val, max_val);
                return dis(gen);
            } else if constexpr (std::is_floating_point_v<T>) {
                std::uniform_real_distribution<T> dis(min_val, max_val);
                return dis(gen);
            } else {
                throw std::runtime_error("Unsupported type for random generation");
            }
        }

    private:
        struct ArgumentInfo {
            std::string default_value;
            std::string min_value;
            std::string max_value;
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
    const std::string ArgParser::random_command = "rand";
    const std::string ArgParser::help_command   = "help";

    // ================
    // SPECIALISATIONS
    // ================

    template <>
    inline int ArgParser::convert<int>(const std::string& value) {
        return std::stoi(value);
    }

    template <>
    inline bool ArgParser::convert<bool>(const std::string& value) {
        return value == "true";
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
