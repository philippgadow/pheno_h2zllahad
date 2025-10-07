///////////////////////////////////////////////////////////////////////////////
// File: CmdLine.cc                                                          //
// Part of the CmdLine library                                               //
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



#include "CmdLine.hh"
#include<string>
#include<sstream>
#include<iostream> // testing
#include<fstream>
#include<vector>
#include<cstddef> // for size_t
#include <sys/utsname.h> // for getting uname
#include <unistd.h> // for getting current path
#include <stdlib.h> // for getting the environment (including username)
#include <cstdio>
using namespace std;

string CmdLine::_default_argfile_option = "-argfile";

std::ostream & operator<<(std::ostream & ostr, CmdLine::OptKind optkind) {
  if      (optkind == CmdLine::OptKind::present) ostr << "present";
  else if (optkind == CmdLine::OptKind::required_value) ostr << "required_value";
  else if (optkind == CmdLine::OptKind::optional_value) ostr << "optional_value";
  else if (optkind == CmdLine::OptKind::value_with_default) ostr << "value_with_default";
  else if (optkind == CmdLine::OptKind::undefined) ostr << "undefined";
  else ostr << "UNRECOGNISED";
  return ostr;
}


// initialise the various structures that we shall
// use to access the command-line options;
//
// If an option appears several times, it is its LAST value
// that will be used in searching for option values (opposite of f90)
CmdLine::CmdLine (const int argc, char** argv, bool enable_help, const string & file_option) : 
    __help_enabled(enable_help), __argfile_option(file_option) {

  __arguments.resize(argc);
  for(int iarg = 0; iarg < argc; iarg++){
    __arguments[iarg] = argv[iarg];
  }
  this->init();
}

/// constructor from a vector of strings, one argument per string
CmdLine::CmdLine (const vector<string> & args, bool enable_help, const string & file_option) : 
    __help_enabled(enable_help), __argfile_option(file_option) {

  __arguments = args;
  this->init();
}

/// Add an overall help string
CmdLine & CmdLine::help(const std::string & help_str) {
  __overall_help_string = help_str;
  __help_enabled = true;
  return *this;
}

//----------------------------------------------------------------------
void CmdLine::init (){
  // record time at start
  time(&__time_at_start);

  // this does not work...
  //__options_help[__argfile_option] = 
  //    OptionHelp_value_with_default<string>(__argfile_option, "filename", 
  //                          "if present, further arguments are read from the filename");

  // check first if a file option is passed
  for(size_t iarg = 0; iarg < __arguments.size(); iarg++) {
    const string & arg = __arguments[iarg];
    if (arg == __argfile_option) {
      // make sure a file is passed too
      bool found_file = true;
      ifstream file_in;
      if (iarg+1 == __arguments.size()) found_file = false;
      else {
        file_in.open(__arguments[iarg+1].c_str());
        found_file = file_in.good();
      }

      // error if no file found
      if (!found_file) {
        ostringstream ostr;
        ostr << "Option "<< __argfile_option
             <<" is passed but no file was found"<<endl;
        throw Error(ostr);
      }

      // remove the file options from the list of arguments
      __arguments.erase(__arguments.begin()+iarg, __arguments.begin()+iarg+2);

      string read_string = "";
      while (file_in >> read_string) {
        // skip the rest of the line if it's a comment;
        // allow both C++-style and shell-style comments
        if (read_string.find("//") != string::npos || read_string.find("#") != string::npos) {
          // read in the rest of this line, effectively discarding it
          getline(file_in, read_string);
        }
        else {
          __arguments.push_back(read_string);
        }
      }

      // start from the beginning of the argument list again again
      iarg = 0;
    }
  }

  // record whole command line so that it can be easily reused
  __command_line = "";
  for(size_t iarg = 0; iarg < __arguments.size(); iarg++){
    const string & arg = __arguments[iarg];
    // if an argument contains special characters, enclose it in
    // single quotes [NB: does not work if it contains a single quote
    // itself: treated below]
    if (arg.find(' ') != string::npos ||
        arg.find('|') != string::npos ||
        arg.find('<') != string::npos || 
        arg.find('>') != string::npos || 
        arg.find('"') != string::npos || 
        arg.find('#') != string::npos) {
      __command_line += "'"+arg+"'";
    } else if (arg.find("'") != string::npos) {
      // handle the case with single quotes in the argument
      // (NB: if there are single and double quotes, we are in trouble...)
      __command_line += '"'+arg+'"';
    } else {
      __command_line += arg;
    }
    __command_line += " ";
  }
  
  // group things into options
  bool next_may_be_val = false;
  string currentopt;
  __arguments_used.resize(__arguments.size(), false);
  __arguments_used[0] = true;
  for(size_t iarg = 1; iarg < __arguments.size(); iarg++){
    // if expecting an option value, then take it (even if
    // it is actually next option...)
    if (next_may_be_val) {__options[currentopt].second = iarg;}
    // now see if it might be an option itself
    string arg = __arguments[iarg];
    bool thisisopt = (arg.compare(0,1,"-") == 0);
    if (thisisopt) {
      // set option to a standard undefined value and say that 
      // we expect (possibly) a value on next round
      currentopt = arg;
      __options[currentopt] = make_pair(int(iarg),-1);
      __options_used[currentopt] = false;
      next_may_be_val = true;}
    else {
      // otherwise throw away the argument for now...
      next_may_be_val = false;
      currentopt = "";
    }
  }
  if (__help_enabled) {
    __help_requested = any_present({"-h","-help","--help"}).help("prints this help message").no_dump();
  }

  // by default, enabe the git info
  set_git_info_enabled(true);
}

// indicates whether an option is present
CmdLine::Result<bool> CmdLine::any_present(const vector<string> & opts) const {
  OptionHelp * opthelp = opthelp_ptr(OptionHelp_present(opts));\
  pair<int,int> result_pair = internal_present(opts);
  bool result = (result_pair.first > 0);
  Result<bool> res(result, opthelp, result);
  opthelp->result_ptr = std::make_shared<Result<bool>>(res);
  return res;
}

// indicates whether an option is present (for internal use only -- does not set help)
pair<int,int> CmdLine::internal_present(const string & opt) const {
  bool result = (__options.find(opt) != __options.end());
  if (result) {
    __options_used[opt] = true;
    __arguments_used[__options[opt].first] = true;
    return __options[opt];
  } else {
    return make_pair(-1,-1);
  }
}

// indicates whether an option is present (for internal use only -- does not set help)
pair<int,int> CmdLine::internal_present(const vector<string> & opts) const {
  vector<string> opts_present;
  for (const auto & opt: opts) {
    bool opt_present = (__options.find(opt) != __options.end());
    if (opt_present) opts_present.push_back(opt);
  }

  if      (opts_present.size() == 0) return make_pair(-1,-1);
  else if (opts_present.size() == 1) {
    __options_used[opts_present[0]] = true;
    __arguments_used[__options[opts_present[0]].first] = true;
    return __options[opts_present[0]];
  } else {
    // options are supposed to be mutually exclusive, so eliminate
    // them all
    ostringstream ostr;
    ostr << "Options " << opts_present[0];
    for (size_t i = 1; i < opts_present.size()-1; i++) {
      ostr << ", " << opts_present[i];
    }
    ostr << " and " << opts_present[opts_present.size()-1] << " are mutually exclusive";
    throw Error(ostr);
  }
}


// indicates whether an option is present and has a value associated
bool CmdLine::internal_present_and_set(const string & opt) const {
  pair<int,int> is_present = internal_present(opt);
  return (is_present.second > 0);

}


// return the string value corresponding to the specified option
string CmdLine::internal_string_val(const vector<string> & opts) const {
  pair<int,int> is_present = internal_present(opts);
  if (is_present.second < 0) {
    if (opts.size() == 1) {
      throw Error("Option " +opts[0]+ " requested but not present and set");
    } else {
      ostringstream ostr;
      ostr << "One of the options " << opts[0];
      for (size_t i = 1; i < opts.size()-1; i++) {
        ostr << ", " << opts[i];
      }
      ostr << " or " << opts[opts.size()-1] << " requested but none present and set";
      throw Error(ostr);
    }
  }
  string arg = __arguments[is_present.second];
  __arguments_used[is_present.second] = true;
  // this may itself look like an option -- if that is the case
  // declare the option to have been used
  if (arg.compare(0,1,"-") == 0) {__options_used[arg] = true;}
  return arg;
}

template<> std::string CmdLine::internal_value<std::string>(const std::string & opt, 
                                                      const std::string & prefix) const {
  if (internal_present_and_set(opt)) {
    return prefix+internal_string_val(opt);
  } else {
    throw Error("internal_value called for option " + opt + ", which is not present and set");
  }
}



void CmdLine::end_section(const std::string & section_name) {
  if (__current_section != section_name) {
    std::ostringstream ostr;
    ostr << "Tried to end section '" << section_name 
          << "' but current section is '" << __current_section << "'";
    throw Error(ostr.str());
  }
  __current_section = "";
}


// return true if all options have been asked for at some point or other
bool CmdLine::all_options_used() const {
  bool result = true;
  // do not use this version (which cekced only for unused options), but rather 
  // the one below, which checks for unused arguments
  // for(map<string,bool>::const_iterator opt = __options_used.begin();
  //     opt != __options_used.end(); opt++) {
  //   bool this_one = opt->second;
  //   if (! this_one) {cerr << "Option "<<opt->first<<" unused/unrecognized"<<endl;}
  //   result &= this_one;
  // }
  for (size_t iarg = 1; iarg < __arguments_used.size(); iarg++) {
    string arg = __arguments[iarg];
    bool this_one = __arguments_used[iarg];
    if (! this_one) {
      cerr << "Argument " << arg << " at position " << iarg << " unused/unrecognized";
      if (__options_used.count(arg) > 0 && __options_used[arg]) {
        cerr << "  (this could be because the same option already appeared";
        if (__options.count(arg) && __options[arg].first > 0) {
          cerr << " at position " << __options[arg].first << ")";
        } else {
          cerr << " elsewhere on the command line)";
        }
      }
      cerr << endl;
    }
    result &= this_one;
  }
  return result;
}

/// return a time stamp corresponding to now
string CmdLine::time_stamp(bool utc) const {
  time_t timenow;
  time(&timenow);
  return _string_time(timenow, utc);
}

/// return a time stamp corresponding to start time
string CmdLine::time_stamp_at_start(bool utc) const {
  return _string_time(__time_at_start, utc);
}

/// return the elapsed time in seconds since the CmdLine object was
/// created
double CmdLine::time_elapsed_since_start() const {
  time_t timenow;
  time(&timenow);
  return std::difftime(timenow, __time_at_start);
}


/// convert the time into a string (local by default -- utc if 
/// utc=true).
string CmdLine::_string_time(const time_t & time, bool utc) const {
  struct tm * timeinfo;
  if (utc) {
    timeinfo = gmtime(&time);
  } else {
    timeinfo = localtime(&time);
  }
  char timecstr[100];
  strftime (timecstr,100,"%Y-%m-%d %H:%M:%S (%Z)",timeinfo);
  //sprintf(timecstr,"%04d-%02d-%02d %02d:%02d:%02d",
  //        timeinfo->tm_year+1900,
  //        timeinfo->tm_mon+1,
  //        timeinfo->tm_mday,
  //        timeinfo->tm_hour,
  //        timeinfo->tm_min,
  //        timeinfo->tm_sec);
  //string timestr = timecstr;
  //if (utc) {
  //  timestr .= " (UTC)";
  //} else {
  //  timestr .= " (local)";
  //}
  return timecstr;
}

/// return a unix-style uname
string CmdLine::unix_uname() const {
  utsname utsbuf;
  int utsret = uname(&utsbuf);
  if (utsret != 0) {return "Error establishing uname";}
  ostringstream uname_result;
  uname_result << utsbuf.sysname << " " 
               << utsbuf.nodename << " "
               << utsbuf.release << " "
               << utsbuf.version << " "
               << utsbuf.machine;
  return uname_result.str();
}

string CmdLine::unix_username() const {
  char * logname;
  logname = getenv("LOGNAME");
  if (logname != nullptr) {return logname;}
  else {return "unknown-username";}
}

/// report failure of conversion
void CmdLine::_report_conversion_failure(const string & opt, 
                                         const string & optstring) const {
  ostringstream ostr;
  ostr << "could not convert option ("<<opt<<") value ("
       <<optstring<<") to requested type"<<endl; 
  throw Error(ostr);
}

void CmdLine::assert_all_options_used() const {
  // deal with the help part
  if (__help_enabled && __help_requested) {
    print_help();
    exit(0);
  }
  if (! all_options_used()) {
    ostringstream ostr;
    ostr <<"Unrecognised options on the command line" << endl;
    throw Error(ostr);
  }
}

string CmdLine::string_val(const string & opt) const {return value<std::string>(opt);}

// as above, but if opt is not present_and_set, return default
string CmdLine::string_val(const string & opt, const string & defval) const {
  return value<string>(opt,defval);
}

// Return the integer value corresponding to the specified option;
// Not too sure what happens if option is present_and_set but does not
// have string value...
int CmdLine::int_val(const string & opt) const { return value<int>(opt);}

// as above, but if opt is not present_and_set, return default
int CmdLine::int_val(const string & opt, const int & defval) const {
  return value<int>(opt,defval);
}


// Return the integer value corresponding to the specified option;
// Not too sure what happens if option is present_and_set but does not
// have string value...
double CmdLine::double_val(const string & opt) const {return value<double>(opt);}

// as above, but if opt is not present_and_set, return default
double CmdLine::double_val(const string & opt, const double & defval) const {
  return value<double>(opt,defval);
}

// return the full command line including the command itself
string CmdLine::command_line() const {
  return __command_line;
}



bool CmdLine::Error::_do_printout = true;
CmdLine::Error::Error(const std::ostringstream & ostr) 
  : _message(ostr.str()) {
  if (_do_printout) cerr << "CmdLine Error: " << _message << endl;;
}
CmdLine::Error::Error(const std::string & str) 
  : _message(str) {
  if (_do_printout) cerr << "CmdLine Error: " << _message << endl;;
}

string CmdLine::current_path() const {
  const size_t maxlen = 10000;
  char tmp[maxlen];
  char * result = getcwd(tmp,maxlen);
  if (result == nullptr) {
    return "error-getting-path";
  } else {
    return string(tmp);
  }
}

string CmdLine::header(const string & prefix) const {
  ostringstream ostr;
  ostr << prefix << "" << command_line() << endl;
  ostr << prefix << "from path: " << current_path() << endl;
  ostr << prefix << "started at: " << time_stamp_at_start() << endl;
  ostr << prefix << "by user: "    << unix_username() << endl;
  ostr << prefix << "running on: " << unix_uname() << endl;
  ostr << prefix << "git state (if any): " << git_info() << endl;
  return ostr.str();
}

/// return a pointer to an existing opthelp is the option is present
/// otherwise register the given opthelp and return a pointer to that
/// (if help is disabled, return a null poiner)
CmdLine::OptionHelp * CmdLine::opthelp_ptr(const CmdLine::OptionHelp & opthelp) const {
  if (!__help_enabled) return nullptr;

  OptionHelp * result;

  auto opthelp_iter = __options_help.find(opthelp.option);
  if (opthelp_iter == __options_help.end()) {
    __options_queried.push_back(opthelp.option);
    __options_help[opthelp.option] = opthelp;
    result = &__options_help[opthelp.option];
  } else {
    result = &opthelp_iter->second;
    // now create a lambda to help with checks that
    // - the option is not being redefined with a different kind
    // - the option is not being redefined with a different default value
    auto warn_or_fail = [&](const string & message) {
      if (fussy()) throw Error(message);
      else         cout << "********* CmdLine warning: " << message << endl;
    };
    if (result->kind != opthelp.kind) {
      ostringstream ostr;
      ostr << "Option " << opthelp.option << " has already been requested with kind '" 
           << result->kind << "' but is now being requested with kind '" << opthelp.kind << "'";
      warn_or_fail(ostr.str());
    }
    if (result->kind == OptKind::value_with_default && result->default_value != opthelp.default_value) {
      ostringstream ostr;
      ostr << "Option " << opthelp.option << " has already been requested with default value " 
           << result->default_value << " but is now being requested with default_value " << opthelp.default_value;
      warn_or_fail(ostr.str());      
    }
  }
  return result;

}

string CmdLine::OptionHelp::type_name() const {
  if      (type == typeid(int)   .name()) return "int"   ;
  else if (type == typeid(double).name()) return "double";
  else if (type == typeid(string).name()) return "string";
  else return type;
}

string CmdLine::OptionHelp::summary() const {
  ostringstream ostr;
  if (! required) ostr << "[";
  ostr << option;
  if (takes_value) ostr << " " << argname;
  if (! required) ostr << "]";
  return ostr.str();
}


string CmdLine::OptionHelp::description(const string & prefix, int wrap_column) const {
  ostringstream ostr;
  ostr << prefix << option;
  if (takes_value) {
    ostr << " " << argname << " (" << type_name() << ")";
    if (has_default) ostr << "     default: " << default_value;
    if (choices.size() != 0) {
      ostr << ", valid choices: {" << choice_list() << "}";
    }
    if (range_strings.size() != 0) {
      ostr << ", allowed range: " << range_string() << "";
    }
  }
  ostr << "\n";
  if (aliases.size() > 1) {
    ostr << prefix << "  aliases: ";
    for (unsigned i = 1; i < aliases.size(); i++) {
      ostr << aliases[i];
      if (i+1 != aliases.size()) ostr << ", ";
    }
    ostr << "\n";
  }
  if (help.size() > 0) {
    ostr << wrap(help, wrap_column, prefix + "  ");
  } 
  ostr << endl;
  return ostr.str();
}

std::string CmdLine::wrap(const std::string & str, int wrap_column, 
                          const std::string & prefix, bool first_line_prefix) {
  // start by separating the string into tokens: words or spaces or new line characters
  vector<string> tokens;
  size_t last_i = 0;
  size_t i = 0;
  size_t n = str.size();
  while (i < n) {
    if (str[i] == ' ' || str[i] == '\n') {
      tokens.push_back(str.substr(last_i,i-last_i));
      tokens.push_back(str.substr(i,1));
      last_i = i+1;
    }
    i++;
  }
  if (last_i < n) tokens.push_back(str.substr(last_i,n-last_i));

  // then loop over the tokens, printing them out, and wrapping if need be
  ostringstream ostr;
  size_t line_len = 0;
  if (first_line_prefix) {
    ostr << prefix;
    prefix.size();
  }
  for (const auto & token: tokens) {
    if (token == "\n") {
      ostr << endl << prefix;
      line_len = prefix.size();
    } else {
      if (int(line_len + token.size()) < wrap_column) {
        ostr << token;
        line_len += token.size();
      } else if (token == " ") {
        ostr << endl << prefix;
        line_len = prefix.size();
      } else {
        ostr << endl << prefix << token;
        line_len = prefix.size() + token.size();
      }
    }
  }
  return ostr.str();
}

string CmdLine::OptionHelp::choice_list() const {
  ostringstream ostr;
  for (unsigned i = 0; i < choices.size(); i++)   {
    ostr << choices[i];
    if (i+1 != choices.size()) ostr << ", ";
  }
  return ostr.str();
}

string CmdLine::OptionHelp::range_string() const {
  ostringstream ostr;
  if (range_strings.size() != 2) return "";
  ostr << range_strings[0] << " <= " << argname << " <= " << range_strings[1];
  return ostr.str();
}

void CmdLine::print_help() const {
  // First print a summary
  cout << "\nUsage: \n       " << __arguments[0];
  for (const auto & opt: __options_queried) {
    cout << " " << __options_help[opt].summary();
  }
  cout << endl << endl;

  if (__overall_help_string.size() != 0) {
    cout << wrap(__overall_help_string);
    cout << endl << endl;
    cout << "Detailed option help" << endl;
    cout << "====================" << endl << endl;
  }
  
  map<string,vector<const OptionHelp *> > opthelp_section_contents;
  vector<string> opthelp_sections;

  // Then print detailed usage for each option that is not in a section
  string prefix = "";
  for (const auto & opt: __options_queried) {
    const OptionHelp & opthelp = __options_help[opt];
    if (opthelp.section == "") {
      cout << opthelp.description(prefix) << endl;
    } else {
      // if an option is in a section, register it for later
      if (opthelp_section_contents.find(opthelp.section) == opthelp_section_contents.end()) {
        opthelp_sections.push_back(opthelp.section);
      }
      opthelp_section_contents[opthelp.section].push_back(&opthelp);
    }
  }

  // then print out the options that are in sections
  for (const auto & section: opthelp_sections) {
    cout << endl;
    cout << section << endl;
    cout << string(section.size(),'-') << endl << endl;
    for (const auto & opthelp: opthelp_section_contents[section]) {
      cout << opthelp->description(prefix) << endl;
    }
  }
}

//------------------------------------------------------------------------
string CmdLine::dump(const string & prefix, const string & absence_prefix) const {
  ostringstream ostr;
  map<string,vector<const OptionHelp *> > opthelp_section_contents;
  vector<string> opthelp_sections;

  ostr << prefix << "argfile for " << command_line() << endl;
  ostr << wrap(__overall_help_string, 80, prefix) << endl;
  ostr << prefix << "generated by CmdLine::dump() on " << time_stamp() << endl;

  auto print_option = [&](const OptionHelp & opthelp) {
    const ResultBase & res = *(opthelp.result_ptr);
    if (opthelp.kind == OptKind::present) {
      if (res.present()) ostr << opthelp.option << endl;
      else               ostr << absence_prefix << opthelp.option << endl;
    } else if (opthelp.kind == OptKind::optional_value) {
      if (res.present()) ostr << opthelp.option << " " << res.value_as_string() << endl;
      else               ostr << absence_prefix << opthelp.option << " " << opthelp.argname << endl;
    } else {      
      ostr << opthelp.option << " " << res.value_as_string() << endl;
    }
  };

  for (const auto & opt: __options_queried) {
    const OptionHelp & opthelp = __options_help[opt];
    if (opthelp.no_dump) continue;
    if (opthelp.section == "") {
      ostr << prefix << "\n" << opthelp.description(prefix);
      print_option(opthelp);
    } else {
      // if an option is in a section, register it for later
      if (opthelp_section_contents.find(opthelp.section) == opthelp_section_contents.end()) {
        opthelp_sections.push_back(opthelp.section);
      }
      opthelp_section_contents[opthelp.section].push_back(&opthelp);
    }
  }

  // then print out the options that are in sections
  for (const auto & section: opthelp_sections) {
    ostr << prefix << endl;
    ostr << prefix << string(section.size(),'-') << endl ;
    ostr << prefix << section << endl;
    ostr << prefix << string(section.size(),'-') << endl ;
    for (const auto & opthelp: opthelp_section_contents[section]) {
      ostr << prefix << "\n" << opthelp->description(prefix) << prefix << endl;
      print_option(*opthelp);
    }
  }
  
  return ostr.str();
}

// From https://www.jeremymorgan.com/tutorials/c-programming/how-to-capture-the-output-of-a-linux-command-in-c/
string CmdLine::stdout_from_command(string cmd) const {

  string data;
  FILE * stream;
  const int max_buffer = 1024;
  char buffer[max_buffer];
  cmd.append(" 2>&1");
  
  stream = popen(cmd.c_str(), "r");
  if (stream) {
    while (!feof(stream))
      if (fgets(buffer, max_buffer, stream) != NULL) data.append(buffer);
    pclose(stream);
  }
  return data;
}

//
string CmdLine::git_info() const {
  if (!__git_info_enabled) return "unknown (disabled)";

  string log_line = stdout_from_command("git log --pretty='%H %d of %cd' --decorate=short -1");
  for (auto & c : log_line) {if (c == 0x0a || c == 0x0d) c = ';';}
  
  if (log_line.substr(0,6) == "fatal:") {
    log_line = "no git info";
  } else {
    // add info about potentially modified files
    string modifications;
    string status_output = stdout_from_command("git status --porcelain --untracked-files=no");
    for (auto & c : status_output) {if (c == 0x0a || c == 0x0d) c = ',';}
    log_line += "; ";
    log_line += status_output;
  }
  return log_line;
}

