///////////////////////////////////////////////////////////////////////////////
// File: CmdLine.hh                                                          //
// Part of the CmdLine library
//                                                                           //
// Copyright (c) 2007-2023 Gavin Salam with contributions from               //
// Gregory Soyez and Rob Verheyen                                            //
//                                                                           //
// This program is free software; you can redistribute it and/or modify      //
// it under the terms of the GNU General Public License as published by      //
// the Free Software Foundation; either version 2 of the License, or         //
// (at your option) any later version.                                       //
//                                                                           //
// This program is distributed in the hope that it will be useful,           //
// but WITHOUT ANY WARRANTY; without even the implied warranty of            //
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the             //
// GNU General Public License for more details.                              //
//                                                                           //
// You should have received a copy of the GNU General Public License         //
// along with this program; if not, write to the Free Software               //
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301, USA //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////


#ifndef __CMDLINE__
#define __CMDLINE__

#include<string>
#include<sstream>
#include<map>
#include<vector>
#include<ctime>
#include<memory>
#include<typeinfo> 

/// Class designed to deal with command-line arguments.
///
/// Basic usage:
///
/// \code
///
/// #include "CmdLine.hh"
/// 
/// int main(int argc, char** argv) {
///   CmdLine cmdline(argc,argv);
///   cmdline.help("Overall help for your program");
///   
///   // required argument, no help string
///   double x = cmdline.value<double>("-x");
/// 
///   // optional argument, with default value, and help string
///   double y = cmdline.value("-y",1.0).help("sets the value of y");
/// 
///   //
///   bool b_is_present = cmdline.present("-b").help("sets b_is_present to true");
/// 
///   // makes sure that all provided command-line options have been used
///   // (also triggers printout of help if -h was present)
///   cmdline.assert_all_options_used();
/// }
///
/// \endcode
class CmdLine {
 public :

  enum class OptKind {
    present,
    required_value,
    value_with_default,
    optional_value,
    undefined
  };

  /// base class for holding results
  class ResultBase {
  public:
    virtual ~ResultBase() {}
    virtual bool present() const = 0;
    virtual bool has_value() const = 0;
    virtual std::string value_as_string() const = 0;
  };

  /// class to store help related to an option
  class OptionHelp {
  public:
    std::string option;
    std::vector<std::string> aliases;
    std::string default_value, help, argname="val";
    std::string type;
    std::vector<std::string> choices;
    std::vector<std::string> range_strings;
    bool required;
    bool takes_value;
    bool has_default;
    bool no_dump = false;
    OptKind kind;

    std::shared_ptr<ResultBase> result_ptr;

    std::string section;
    /// returns a short summary of the option (suitable for
    /// placing in the command-line summary
    std::string summary() const; 
    /// returns a longer description of the option (suitable for
    /// placing in the more extended help)
    std::string description(const std::string & prefix="  ", int wrap_column = 80) const;
    /// returns an attempt at a human readable typename
    std::string type_name() const;
    /// returns a string with a comma-separated list of choices
    std::string choice_list() const;
    /// returns the string with the allowed range
    std::string range_string() const;
  };

  /// class that contains the result of an option.
  /// Can be implicitly converted to type T, and can also be used
  /// to add help information to the option.
  template<class T>
  class Result : public ResultBase {
  public:
    Result(const T & t) : _t(t), _opthelp(0), _is_present(true) {}
    Result(const T & t, OptionHelp * opthelp_ptr, bool is_present) : 
               _t(t), _opthelp(opthelp_ptr), _is_present(is_present) {}

    /// this allows for implicit conversion to type T in assignments
    operator T() const;

    /// this allows the user to do the conversion to the argument's value manually
    T operator()() const;

    /// an alternative member name for getting the value manually
    T value() const;

    /// returns true if the argument was present on the command-line
    bool present() const override {return _is_present;}

    /// returns true if the result can be converted to a value
    bool has_value() const override {return kind() != OptKind::optional_value || _is_present;}

    /// returns the value of the option, or val if the option is not present
    T value_or(T val) const {
      if (has_value()) return _t;
      else return val;
    }

    /// returns the value of the option, as a string (with 16 digits precision)
    std::string value_as_string() const override;

    /// for adding help to an option
    const Result & help(const std::string & help_string) const {
      opthelp().help = help_string;
      return *this;
    }

    /// for adding an argument name to an option
    const Result & argname(const std::string & argname_string) const {
      opthelp().argname = argname_string;
      return *this;
    }    

    /// @brief sets the allowed choices
    /// @param allowed_choices 
    /// @return the Result object
    const Result & choices(const std::vector<T> allowed_choices) const; 

    /// sets the allowed range: minval  <= arg <= maxval
    const Result & range(T minval, T maxval) const; 

    const Result & no_dump() const {
      opthelp().no_dump = true;
      return *this;
    }

    /// returns a reference to the option help, and throws an error if
    /// there is no help
    OptionHelp & opthelp() const;    

    /// sets a pointer to the help instance for this argument.
    void set_opthelp(OptionHelp * opthelp) {_opthelp = opthelp;}

    /// returns the OptKind enum indicating what kind of option this is
    OptKind kind() const {return _opthelp ? _opthelp->kind : OptKind::undefined;}


  protected:
    void throw_value_not_available() const;

    T _t;
    mutable OptionHelp * _opthelp;
    bool _is_present;
  };
  
  CmdLine() {};
  /// initialise a CmdLine from a C-style array of command-line arguments
  CmdLine(const int argc, char** argv, bool enable_help = true, const std::string & file_option=_default_argfile_option );
  /// initialise a CmdLine from a C++ std::vector of arguments 
  CmdLine(const std::vector<std::string> & args, bool enable_help = true, const std::string & file_option=_default_argfile_option );

  /// Add an overall help string
  CmdLine & help(const std::string & help_str);

  /// @name Member functions to add and classify command-line options
  ///@{

  /// return true if the option is present
  Result<bool> present(const std::string & opt) const {return any_present(std::vector<std::string>{opt});}

  /// returns the value of the argument following opt, converted to type Result<T>
  template<class T> Result<T> value(const std::string & opt) const {
    return any_value<T>(std::vector<std::string>{opt});}

  /// returns the value of the option, or defval if the option is not present
  template<class T> Result<T> value(const std::string & opt, const T & defval) const {
    return any_value<T>(std::vector<std::string>{opt}, defval);
  }

  /// returns a Result<T> for the option; the result.present() should be queried to
  /// see if it was present before trying to use the value
  template<class T> Result<T> optional_value(const std::string & opt) const {
    return any_optional_value<T>(std::vector<std::string>{opt});
  }

  /// returns the value of the argument, prefixed with prefix (NB: 
  /// require different function name to avoid confusion with 
  /// 2-arg template). This can be useful when the option is
  /// automatically converted from a string to some other type via
  /// the << operator, but the conversion needs a prefix to be
  /// applied to the argument for the conversion to work.
  template<class T> Result<T> value_prefix(const std::string & opt, const std::string & prefix) const {
    return any_value_prefix<T>(std::vector<std::string>{opt}, prefix);
  }

  /// returns the value of the argument, prefixed with prefix, with defval returned
  /// if the option is not present.
  template<class T> Result<T> value(const std::string & opt, const T & defval, 
                                    const std::string & prefix) const {
    return any_value<T>(std::vector<std::string>{opt}, defval, prefix);
  }


  /// return true if any of the options in the option vector is present
  /// (at most one of the options should be present)
  Result<bool> any_present(const std::vector<std::string> & opts) const;

  /// returns the value of the argument following any of the (mutually
  /// exclusive) opts, converted to type Result<T> 
  template<class T> Result<T> any_value(const std::vector<std::string> & opts) const;

  /// returns the value following any of the (mutually exclusive)
  /// options, or defval if none is present
  template<class T> Result<T> any_value(const std::vector<std::string> & opt, const T & defval) const;

  /// like optional_value, but for a (mutually exclusive) vector of options
  template<class T> Result<T> any_optional_value(const std::vector<std::string> & opts) const;

  /// like value_prefix, but for a (mutually exclusive) vector of options
  template<class T> Result<T> any_value_prefix(const std::vector<std::string> & opts, 
                                               const std::string & prefix) const;

  /// like value (with prefix), but for a (mutually exclusive) vector of options
  template<class T> Result<T> any_value(const std::vector<std::string> & opt, const T & defval, 
                                    const std::string & prefix) const;

  /// start a section of the help
  void start_section(const std::string & section_name) {
    __current_section = section_name;
  }

  /// end a section of the help
  void end_section() {__current_section = "";}

  /// end a section of the help, with the given name (the code will check it matches)
  void end_section(const std::string & section_name);

  ///@}


  /// @name General helpers
  ///@{

  /// when an option is missing but help has been asked for, we will
  /// still return a value, specified by this function, which can
  /// be specialised by the user if they want to extend it to
  /// further types (currently defaults to 0, except for strings
  /// where it defaults to the empty string).
  template<class T> T value_for_missing_option() const;
  
  /// return a reference to the std::vector of command-line arguments (0 is
  /// command).
  inline const std::vector<std::string> & arguments() const {return __arguments;}

  /// return the full command line
  std::string command_line() const;

  /// return the command (i.e. program) name
  std::string command_name() const {return __arguments[0];}

  /// print the help std::string that has been deduced from all the options called
  void print_help() const;
  
  /// return a std::string in argfile format that contains all
  /// options queried and, where relevant, their values
  /// - if an option is optional (no default), it is printed commented-out
  /// - if an option was not supplied but has a default, it is printed out with its default
  ///
  /// @param prefix is the string the precedes each description line (default is "# ")
  /// @param absence_prefix is the string that precedes each line for an option that was not present
  std::string dump(const std::string & prefix = "# ", const std::string & absence_prefix = "// ") const;

  /// return true if all options have been asked for at some point or other
  bool all_options_used() const;

  /// gives an error if there are unused options
  void assert_all_options_used() const;

  /// return a time stamp (UTC) corresponding to now
  std::string time_stamp(bool utc = false) const;

  /// return a time stamp (UTC) corresponding to time of object construction
  std::string time_stamp_at_start(bool utc = false) const;

  /// return the elapsed time in seconds since the CmdLine object was
  /// created
  double time_elapsed_since_start() const;

  /// return output similar to that from uname -a on unix
  std::string unix_uname() const;

  /// return the username
  std::string unix_username() const;

  /// In C++17 we don't need this, we can instead use std::filesystem::current_path();
  /// But for compatibility with older system it is useful to have
  std::string current_path() const;

  /// enable/disable git info support (on by default)
  CmdLine & set_git_info_enabled(bool enable=true) {__git_info_enabled = enable; return *this;}

  /// return true if git info support is enabled
  bool git_info_enabled() const {return __git_info_enabled;}

  /// sets fussy behaviour, which means that an option that
  /// is re-declared with a different kind, or different default
  /// is considered an error.  This is useful for catching inconsistencies
  CmdLine & set_fussy(bool fussy = true) {__fussy = fussy; return *this;}

  /// return true if fussy behaviour is enabled
  bool fussy() const {return __fussy;}

  /// returns a string with basic info about the git
  std::string git_info() const;

  /// return a multiline header that contains
  /// - the command line
  /// - the current directory path
  /// - the start time
  /// - the user
  /// - the system name
  /// The header includes a final newline
  std::string header(const std::string & prefix = "# ") const;
  
  ///@}


  class Error;

  /// take a string and return a wrapped version of it (with the given prefix on each line).
  /// - \n triggers newline preceded by prefix
  /// - no end of line is added to the final line
  /// - if first_line_prefix is false, no prefix is added to the first line 
  ///   (user's responsibility to sort that out if needed)
  static std::string wrap(const std::string & str, int wrap_column = 80, 
                          const std::string & prefix = "", bool first_line_prefix = true);


  /// @name Deprecated functions
  ///@{

  /// true if the option is present and corresponds to a value
  [[deprecated("use CmdLine::optional_value instead, and that query whether it is present")]]
  bool present_and_set(const std::string & opt) const;

  /// return the integer value corresponding to the given option
  [[deprecated]]
  int     int_val(const std::string & opt) const;
  /// return the integer value corresponding to the given option or default if option is absent
  [[deprecated]]
  int     int_val(const std::string & opt, const int & defval) const;

  /// return the double value corresponding to the given option
  [[deprecated]]
  double  double_val(const std::string & opt) const;
  /// return the double value corresponding to the given option or default if option is absent
  [[deprecated]]
  double  double_val(const std::string & opt, const double & defval) const;

  /// return the std::string value corresponding to the given option
  [[deprecated("use value<std::string>(opt) instead")]]
  std::string  string_val(const std::string & opt) const;

  /// return the std::string value corresponding to the given option or default if option is absent
  [[deprecated("use value<std::string>(opt, defval) instead")]]
  std::string  string_val(const std::string & opt, const std::string & defval) const;
  ///@} 

 private:

  /// returns the stdout (and stderr) from the command
  std::string stdout_from_command(std::string cmd) const;

  /// check if the option is present --  for internal use only (does not set help)
  /// returns
  /// - (-1,-1) if the option is not present
  /// - ( n,-1) if the option is present (at index n) but cannot be associated with a value
  /// - ( n, m) if the option is present (at index n) and can be associated with a value (at index m)
  std::pair<int,int> internal_present(const std::string & opt) const;

  /// same as the scalar version of internal_present, but for a vector
  /// of options, returning similarly if none or one of the options is found
  /// and throwing an error if multiple options are found
  std::pair<int,int> internal_present(const std::vector<std::string> & opts) const;


  /// true if the option is present and corresponds to a value (internal use only)
  bool         internal_present_and_set(const std::string & opt) const;

  /// returns string value of option (assumed to be present_and_set) 
  /// -- for internal use only (does not set help)
  std::string internal_string_val(const std::vector<std::string> & opt) const;
  std::string internal_string_val(const std::string & opt) const {
    return internal_string_val(std::vector<std::string>{opt});
  }

  /// returns converted value of option (assumed to be present_and_set) 
  /// -- for internal use only (does not set help)
  template<class T> T internal_value(const std::string & opt, const std::string & prefix = "") const {
    return internal_value<T>(std::vector<std::string>{opt}, prefix);
  }
  template<class T> T internal_value(const std::vector<std::string> & opt, const std::string & prefix = "") const;



  /// stores the command line arguments in a C++ friendly way
  std::vector<std::string> __arguments;

  /// a map of possible options found on the command line, referencing
  /// the index of the argument that might assign a value to that
  /// option (an option being anything starting with a dash)
  ///
  /// The first element of the pair is the location is the option,
  /// the second is the location of its value (or -1 if there is no value)
  mutable std::map<std::string,std::pair<int,int>> __options;
  

  /// whether a given option has been requested
  mutable std::map<std::string,bool> __options_used;
  /// whether a given argument has been used
  mutable std::vector<bool> __arguments_used;

  /// whether help functionality is enabled
  bool __help_enabled;
  /// whether the user has requested help with -h or --help
  bool __help_requested;
  /// whether the git info is included or not
  bool __git_info_enabled;

  //std::string __progname;
  std::string __command_line;
  std::time_t __time_at_start;
  std::string __overall_help_string;
  bool        __fussy = false;

  std::string __current_section = "";

  /// default option to tell CmdLine to read arguments 
  /// from a file
  static std::string _default_argfile_option;
  std::string __argfile_option = _default_argfile_option;
  

  template<class T>
  OptionHelp OptionHelp_value_with_default(const std::vector<std::string> & options, const T & default_value,
                                     const std::string & help_string = "") const {
    OptionHelp help;
    help.option        = options[0];
    help.aliases       = options;
    std::ostringstream defval_ostr;
    defval_ostr << default_value;
    help.default_value = defval_ostr.str();
    help.help          = help_string;
    help.type          = typeid(T).name();
    help.required      = false;
    help.takes_value   = true;
    help.has_default   = true;
    help.kind          = OptKind::value_with_default;
    help.section       = __current_section;
    return help;
  }
  template<class T>
  OptionHelp OptionHelp_value_required(const std::vector<std::string> & options,
                                       const std::string & help_string = "") const {
    OptionHelp help;
    help.option        = options[0];
    help.aliases       = options;
    help.default_value = "";
    help.help          = help_string;
    help.type          = typeid(T).name();
    help.required      = true;
    help.takes_value   = true;
    help.has_default   = false;
    help.kind          = OptKind::required_value;
    help.section       = __current_section;
    return help;
  }
  template<class T>
  OptionHelp OptionHelp_optional_value(const std::vector<std::string> & options,
                                       const std::string & help_string = "") const {
    OptionHelp help;
    help.option        = options[0];
    help.aliases       = options;
    help.default_value = "";
    help.help          = help_string;
    help.type          = typeid(T).name();
    help.required      = false;
    help.takes_value   = true;
    help.has_default   = false;
    help.kind          = OptKind::optional_value;
    help.section       = __current_section;
    return help;
  }
  OptionHelp OptionHelp_present(const std::vector<std::string> & options,
                                const std::string & help_string = "") const {
    OptionHelp help;
    help.option        = options[0];
    help.aliases       = options;
    help.default_value = "";
    help.help          = help_string;
    help.type          = "";
    help.required      = false;
    help.takes_value   = false;
    help.has_default   = true;  // not 100% sure what the right choice is here; "default" is that value is false
    help.kind          = OptKind::present;
    help.section       = __current_section;
    return help;
  }
  
  /// return a pointer to an existing opthelp is the option is present
  /// otherwise register the given opthelp and return a pointer to that
  /// (if help is disabled, return a null poiner)
  OptionHelp * opthelp_ptr(const OptionHelp & opthelp) const;

  /// a std::vector of the options queried (this may evolve)
  mutable std::vector<std::string> __options_queried;
  /// a map with help for each option that was queried
  mutable std::map<std::string, OptionHelp> __options_help;
  

  /// builds the internal structures needed to keep track of arguments and options
  void init();

  /// report failure of conversion
  void _report_conversion_failure(const std::string & opt, 
                                  const std::string & optstring) const;

  /// convert the time into a std::string (local by default -- utc if 
  /// utc=true).
  std::string _string_time(const time_t & time, bool utc) const;
};



//----------------------------------------------------------------------
/// class that deals with errors
class CmdLine::Error {
public:
  Error(const std::ostringstream & ostr);
  Error(const std::string & str);
  const char* what() throw() {return _message.c_str();}
  const std::string & message() throw() {return _message;}
  static void set_print_message(bool doprint) {_do_printout = doprint;}
private:
  std::string _message;
  static bool _do_printout;
};

template<class T> 
void CmdLine::Result<T>::throw_value_not_available() const {
  std::ostringstream ostr;
  ostr << "value of option ";
  if (_opthelp) ostr << _opthelp->option;
  else ostr << "[unknown -- because help disabled]";
  ostr << " requested, but that value is not available\n"
       << "because the option was not present on the command line and no default was supplied\n";
  throw Error(ostr.str());
}


template<class T>
inline CmdLine::Result<T>::operator T() const {
  return value();
}

template<class T>
inline T CmdLine::Result<T>::operator()() const {
  return value();
}

template<class T>
inline T CmdLine::Result<T>::value() const {
  if (!has_value()) throw_value_not_available();
  return _t;
}

template<class T>
CmdLine::OptionHelp & CmdLine::Result<T>::opthelp() const {
  if (_opthelp) {
    return *_opthelp;
  } else {
    throw CmdLine::Error("tried to access optionhelp for option where it does not exist\n"
                  "(e.g. because option help already set for an identical option earlier)");
  }
}

template<class T> inline T CmdLine::value_for_missing_option() const {return T(0);}
template<> inline std::string CmdLine::value_for_missing_option<std::string>() const {return "";}


template<class T> T CmdLine::internal_value(const std::vector<std::string> & opts, const std::string & prefix) const {
  std::string optstring = prefix+internal_string_val(opts);
  std::istringstream optstream(optstring);
  T result;
  optstream >> result;
  if (optstream.fail()) {
    std::string opt = __arguments[internal_present(opts).first];
    _report_conversion_failure(opt, optstring);
  }
  return result;
}

template<> std::string CmdLine::internal_value<std::string>(const std::string & opt, 
                                                      const std::string & prefix) const;

/// returns the value of the argument, convertible to type T
template<class T> CmdLine::Result<T> CmdLine::any_value(const std::vector<std::string> & opts) const {
  // we create the result from the (more general) value_prefix
  // function, with an empty prefix
  return any_value_prefix<T>(opts,"");
}

/// returns the value of the argument converted to type T
template<class T> 
CmdLine::Result<T> CmdLine::any_value_prefix(const std::vector<std::string> & opts, 
                                             const std::string & prefix) const {
  OptionHelp * opthelp = opthelp_ptr(OptionHelp_value_required<T>(opts, ""));

  T result;
  if (__help_requested && internal_present(opts).second < 0) {
    result = value_for_missing_option<T>();
  } else {
    result = internal_value<T>(opts, prefix);
  }
  Result<T> res(result, opthelp, true);
  opthelp->result_ptr = std::make_shared<Result<T>>(res);
  return res;
}


template<class T> CmdLine::Result<T> CmdLine::any_value(const std::vector<std::string> & opts, const T & defval) const {
  // construct help
  OptionHelp * opthelp = opthelp_ptr(OptionHelp_value_with_default(opts, defval, ""));

  std::shared_ptr<Result<T>> res;
  // return value
  auto pres = this->internal_present(opts);
  if (pres.second > 0) {
    auto result = internal_value<T>(opts);
    res = std::make_shared<Result<T>>(result,opthelp,true);
  } else if (pres.first > 0) {
    throw Error("option " + __arguments[pres.first] + " present, but expected value was absent");
  } else {
    res = std::make_shared<Result<T>>(defval,opthelp,false);
  }
  opthelp->result_ptr = res;
  return *res;
}

template<class T> CmdLine::Result<T> CmdLine::any_optional_value(const std::vector<std::string> & opts) const {
  // construct help
  OptionHelp * opthelp = opthelp_ptr(OptionHelp_optional_value<T>(opts));
  opthelp->default_value = "None";

  // return value
  std::shared_ptr<Result<T>> res;
  auto pres = this->internal_present(opts);
  if (pres.second > 0) {
    auto result = internal_value<T>(opts);
    res = std::make_shared<Result<T>>(result, opthelp, true);
  } else if (pres.first > 0) {
    throw Error("option " + __arguments[pres.first] + " present, but expected value was absent");
  } else {    
    res = std::make_shared<Result<T>>(value_for_missing_option<T>(), opthelp, false);
  }
  opthelp->result_ptr = res;
  return *res;
}


template<class T> CmdLine::Result<T> CmdLine::any_value(const std::vector<std::string> & opts, const T & defval, 
                                   const std::string & prefix) const {
  OptionHelp * opthelp = opthelp_ptr(OptionHelp_value_with_default(opts, defval, ""));

  // return value
  std::shared_ptr<Result<T>> res;
  auto pres = this->internal_present(opts);
  if (pres.second > 0) {
    auto result = internal_value<T>(opts, prefix);
    res = std::make_shared<Result<T>>(result, opthelp, true);
  } else if (pres.first > 0) {
    throw Error("option " + __arguments[pres.first] + " present, but expected value was absent");
  } else {
    res = std::make_shared<Result<T>>(defval, opthelp, false);
  }
  opthelp->result_ptr = res;
  return *res;
}

template<class T>
std::ostream & operator<<(std::ostream & ostr, const CmdLine::Result<T> & result) {
  ostr << result();
  return ostr;
}

template<class T>
std::string CmdLine::Result<T>::value_as_string() const {
  std::ostringstream ostr;
  ostr.precision(16);
  ostr << (*this)();
  return ostr.str();
}

template<class T>
const CmdLine::Result<T> & CmdLine::Result<T>::choices(const std::vector<T> allowed_choices) const {
  // register the choices with the help module
  for (const auto & choice: allowed_choices) {
    std::ostringstream ostr;
    ostr << choice;
    _opthelp->choices.push_back(ostr.str());
  }

  // check the choice actually made is valid
  bool valid = false;
  for (const auto & choice: allowed_choices) {
    if (_t == choice) {valid = true; break;}
  }
  if (!valid) {
    std::ostringstream ostr;
    ostr << "For option " << _opthelp->option << ", invalid option value " 
        << _t << ". Allowed choices are: " << _opthelp->choice_list();
    throw Error(ostr.str());
  }
  return *this;
}

template<class T>
const CmdLine::Result<T> & CmdLine::Result<T>::range(T minval, T maxval) const {
  std::ostringstream minstr, maxstr;
  minstr << minval;
  maxstr << maxval;
  _opthelp->range_strings.push_back(minstr.str());
  _opthelp->range_strings.push_back(maxstr.str());
  if (_t < minval || _t > maxval) {
    std::ostringstream errstr;
    errstr << "For option " << _opthelp->option << ", option value " << _t 
           << " out of allowed range: " 
           << _opthelp->range_string();
    throw Error(errstr.str());
  }
  return *this;
}

std::ostream & operator<<(std::ostream & ostr, CmdLine::OptKind optkind);
#endif
